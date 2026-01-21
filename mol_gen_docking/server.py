import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

import ray
from fastapi import FastAPI
from tdc import Evaluator

from mol_gen_docking.data.meeko_process import ReceptorProcess
from mol_gen_docking.reward.rl_rewards import (
    RewardScorer,
)
from mol_gen_docking.server_utils.buffer import RewardBuffer
from mol_gen_docking.server_utils.utils import (
    MolecularVerifierQuery,
    MolecularVerifierResponse,
    MolecularVerifierSettings,
)

# Set logging level to info
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("molecular_verifier_server")
logger.setLevel(logging.INFO)

server_settings: MolecularVerifierSettings
RemoteRewardScorer: Any = ray.remote(RewardScorer)

server_settings_log = "Server settings:\n"
for field_name, field_value in MolecularVerifierSettings().model_dump().items():
    server_settings_log += f"  {field_name}: {field_value}\n"
logger.info(server_settings_log)

_reward_model = None
_valid_reward_model = None


def get_or_create_reward_actor() -> Any:
    global _reward_model
    global server_settings
    if _reward_model is None or _reward_model.__ray_terminated__:
        _reward_model = RemoteRewardScorer.remote(
            path_to_mappings=server_settings.data_path,
            parse_whole_completion=False,
            docking_concurrency_per_gpu=server_settings.docking_concurrency_per_gpu,
            reaction_matrix_path=server_settings.reaction_matrix_path,
            oracle_kwargs=dict(
                exhaustiveness=server_settings.scorer_exhaustiveness,
                n_cpu=server_settings.scorer_ncpus,
                docking_oracle=server_settings.docking_oracle,
                vina_mode=server_settings.vina_mode,
            ),
        )
    return _reward_model


def get_or_create_valid_actor() -> Any:
    global _valid_reward_model
    global server_settings
    if _valid_reward_model is None:
        _valid_reward_model = RewardScorer(
            path_to_mappings=server_settings.data_path,
            reward="valid_smiles",
            parse_whole_completion=False,
            reaction_matrix_path=server_settings.reaction_matrix_path,
        )
    return _valid_reward_model


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global server_settings
    server_settings = MolecularVerifierSettings()
    logger.info(
        f"Initialized molecular docking verifier lifespan with {server_settings}"
    )
    logger.info("Initializing socket")
    app.state.reward_buffer = RewardBuffer(
        app, buffer_time=server_settings.buffer_time, max_batch_size=1000000000
    )

    app.state.reward_model = get_or_create_reward_actor()
    app.state.reward_valid_smiles = get_or_create_valid_actor()

    app.state.receptor_processor = (
        ReceptorProcess(data_path=server_settings.data_path)
        if server_settings.docking_oracle == "autodock_gpu"
        else None
    )

    with open(server_settings.data_path + "/pockets_info.json") as f:
        pockets_info = json.load(f)
    app.state.receptors = list(pockets_info.keys())
    app.state.diversity_evaluator = Evaluator(name="Diversity")
    app.state.uniqueness_evaluator = Evaluator(name="Uniqueness")

    yield


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    @app.get("/liveness")  # type: ignore
    async def live_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/get_reward")  # type: ignore
    async def get_reward(query: MolecularVerifierQuery) -> MolecularVerifierResponse:
        t0 = time.time()
        prepare_res = await prepare_receptor(query)
        status = prepare_res.get("status", "")
        if status == "Error":
            return MolecularVerifierResponse(error="Error in preprocessing")

        result: MolecularVerifierResponse = await app.state.reward_buffer.add_query(
            query
        )
        t1 = time.time()
        logger.info(f"Processed batch in {t1 - t0:.2f} seconds")
        if result.meta is not None and len(result.meta) == 1:
            if (
                result.meta[0].all_smi_rewards is not None
                and result.meta[0].all_smi is not None
            ):
                result.next_turn_feedback = (
                    "The score of the provided molecules are:\n"
                    + "\n".join(
                        [
                            f"{smi}: {score:.4f}"
                            for smi, score in zip(
                                result.meta[0].all_smi, result.meta[0].all_smi_rewards
                            )
                        ]
                    )
                )
        return result

    @app.post("/prepare_receptor")  # type: ignore
    async def prepare_receptor(query: MolecularVerifierQuery) -> Dict[str, str]:
        metadata = query.metadata

        if app.state.receptor_processor is None:
            # No need to prepare receptors
            return {
                "status": "No need to prepare receptors for the selected docking oracle."
            }
        assert metadata is not None
        assert all(
            [
                "properties" in m and "objectives" in m and "target" in m
                for m in metadata
            ]
        )

        targets = [
            m["properties"][i]
            for m in metadata
            for i in range(len(m["properties"]))
            if m["properties"][i] in app.state.receptors
        ]
        targets = list(set(targets))
        if targets == []:
            return {"status": "Success"}

        missed_receptors_1, missed_receptors_2 = (
            app.state.receptor_processor.process_receptors(
                receptors=targets, allow_bad_res=True
            )
        )
        if len(missed_receptors_2) > 0:
            # Return error if some receptors could not be processed
            return {
                "status": "Error",
                "info": f"Receptors {missed_receptors_2} could not be processed.",
            }
        else:
            return {"status": "Success"}

    return app


app = create_app()
