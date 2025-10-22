import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict

import numpy as np
from fastapi import FastAPI
from tdc import Evaluator

import ray
from mol_gen_docking.data.meeko_process import ReceptorProcess
from mol_gen_docking.reward.rl_rewards import (
    RewardScorer,
)
from mol_gen_docking.reward.server_utils import (
    MolecularVerifierQuery,
    MolecularVerifierResponse,
    MolecularVerifierSettings,
)

logger = logging.getLogger("llm_verifier_server")

server_settings: MolecularVerifierSettings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global server_settings
    server_settings = MolecularVerifierSettings()
    logger.info(
        f"Initialized molecular docking verifier lifespan with {server_settings}"
    )

    logger.info("Initializing socket")

    RemoteRewardScorer = ray.remote(RewardScorer)
    app.state.reward_model = RemoteRewardScorer.remote(  # type: ignore
        path_to_mappings=server_settings.data_path,
        parse_whole_completion=False,
        gpu_utilization_gpu_docking=server_settings.gpu_utilization_gpu_docking,
        oracle_kwargs=dict(
            exhaustiveness=server_settings.scorer_exhaustivness,
            n_cpu=server_settings.scorer_ncpus,
            docking_oracle=server_settings.docking_oracle,
            vina_mode=server_settings.vina_mode,
        ),
    )

    app.state.reward_valid_smiles = RewardScorer(
        path_to_mappings=server_settings.data_path,
        reward="valid_smiles",
        parse_whole_completion=False,
    )

    app.state.receptor_processor = (
        ReceptorProcess(data_path=server_settings.data_path)
        if server_settings.docking_oracle != "soft_docking"
        else None
    )

    with open(server_settings.data_path + "/pockets_info.json") as f:
        pockets_info = json.load(f)
    app.state.receptors = list(pockets_info.keys())

    yield


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    @app.get("/liveness")  # type: ignore
    async def live_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/get_reward")  # type: ignore
    async def get_reward(query: MolecularVerifierQuery) -> MolecularVerifierResponse:
        prompts, queries, metadata = query.prompts, query.query, query.metadata

        if prompts is None:
            assert metadata is not None
            prompts = []
            for meta in metadata:
                assert all([k in meta for k in ["properties", "objectives", "target"]])
                prompts.append(
                    "|".join(
                        [
                            f"{p}, {o}, {t}"
                            for p, o, t in zip(
                                meta["properties"], meta["objectives"], meta["target"]
                            )
                        ]
                    )
                )

        rewards_job = app.state.reward_model.get_score.remote(
            prompts=prompts, completions=queries, metadata=metadata
        )
        valid_reward = app.state.reward_valid_smiles.get_score(
            prompts=prompts, completions=queries
        )
        final_smiles = app.state.reward_valid_smiles.get_all_completions_smiles(
            completions=queries
        )
        logger.info(f"Validity: {valid_reward}")
        logger.info(f"Final smiles: {final_smiles}")

        # Get the prompts level metrics
        unique_prompts = list(set(prompts))
        group_prompt_smiles = {
            p: [
                s[-1]
                for s, p_ in zip(final_smiles, prompts)
                if (p_ == p) and not s == []
            ]
            for p in unique_prompts
        }
        diversity_evaluator = Evaluator(name="Diversity")
        diversity_scores_dict = {
            p: diversity_evaluator(group_prompt_smiles[p])
            if len(group_prompt_smiles[p]) > 1
            else 0
            for p in unique_prompts
        }
        diversity_score = [float(diversity_scores_dict[p]) for p in prompts]
        diversity_score = [d if not np.isnan(d) else 0 for d in diversity_score]

        uniqueness_evaluator = Evaluator(name="Uniqueness")
        uniqueness_scores_dict = {
            p: uniqueness_evaluator(group_prompt_smiles[p])
            if len(group_prompt_smiles[p]) > 1
            else 0
            for p in unique_prompts
        }
        uniqueness_score = [float(uniqueness_scores_dict[p]) for p in prompts]
        uniqueness_score = [u if not np.isnan(u) else 0 for u in uniqueness_score]

        rewards = ray.get(rewards_job)
        max_per_prompt_dict = {
            p: max([float(r) for r, p_ in zip(rewards, prompts) if p_ == p])
            for p in unique_prompts
        }
        max_per_prompt = [max_per_prompt_dict[p] for p in prompts]

        response = MolecularVerifierResponse(
            reward=sum(rewards) / len(rewards),
            meta={
                "property_scores": rewards,
                "validity": valid_reward,
                "uniqueness": uniqueness_score,
                "diversity": diversity_score,
                "pass_at_n": max_per_prompt,
                "rewards": rewards,
                # "mol_filters": filter_reward,
            },
            error=None,
        )

        return response

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
            return {"status": "No need to prepare receptors for the given batch."}

        missed_receptors_1, missed_receptors_2 = (
            app.state.receptor_processor.process_receptors(
                receptors=targets, allow_bad_res=True
            )
        )
        if len(missed_receptors_2) > 0:
            # Return error if some receptors could not be processed
            return {"status": "Error"}
        else:
            return {"status": "Success"}

    return app


app = create_app()
