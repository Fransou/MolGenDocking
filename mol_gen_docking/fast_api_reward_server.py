import argparse
import json

import numpy as np
import ray
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openrlhf.utils.logging_utils import init_logger

from mol_gen_docking.data.meeko_process import ReceptorProcess
from mol_gen_docking.reward.rl_rewards import RewardScorer

logger = init_logger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
        description="Run a FastAPI server for molecular generation rewards scoring."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/mol_orz",
        help="Path to the dataset, notably where the pdb files are.",
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port number for the server"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    parser.add_argument(
        "--scorer-exhaustivness",
        type=int,
        default=2,
        help="Exhaustiveness for the docking computations.",
    )
    parser.add_argument(
        "--scorer-ncpus",
        type=int,
        default=2,
        help="Number of CPUs to use for the scoring computations.",
    )
    parser.add_argument(
        "--docking-oracle",
        type=str,
        default="pyscreener",
        help="The docking oracle.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    RemoteRewardScorer = ray.remote(RewardScorer)
    reward_model = RemoteRewardScorer.remote(  # type: ignore
        path_to_mappings=args.data_path,
        parse_whole_completion=False,
        oracle_kwargs=dict(
            exhaustiveness=args.scorer_exhaustivness,
            n_cpu=args.scorer_ncpus,
            docking_oracle=args.docking_oracle,
            vina_mode="autodock_gpu_256wi",
        ),
    )
    reward_valid_smiles = RewardScorer(
        path_to_mappings=args.data_path,
        reward="valid_smiles",
        parse_whole_completion=False,
    )

    receptor_processor = ReceptorProcess(data_path=args.data_path)

    with open(args.data_path + "/pockets_info.json") as f:
        pockets_info = json.load(f)
    receptors = list(pockets_info.keys())

    # reward_filters = RemoteRewardScorer.remote(
    #     path_to_mappings=args.data_path,
    #     reward="MolFilters",
    #     parse_whole_completion=False,
    # )

    app = FastAPI()

    @app.post("/get_reward")  # type: ignore
    async def get_reward(request: Request) -> JSONResponse:
        data = await request.json()
        queries = data.get("query")
        prompts = data.get("prompts", None)
        metadata = data.get("metadata", None)
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
        rewards_job = reward_model.get_score.remote(  # type: ignore
            prompts=prompts, completions=queries, metadata=metadata
        )
        valid_reward = reward_valid_smiles.get_score(
            prompts=prompts, completions=queries
        )
        final_smiles = reward_valid_smiles.get_all_completions_smiles(
            completions=queries
        )

        # filter_reward_job = reward_filters.get_score.remote(prompts=prompts, completions=queries)

        # filter_reward = ray.get(filter_reward_job)

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

        from tdc import Evaluator

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

        result = {
            "rewards": rewards,
            "scores": rewards,
            "extra_logs": {
                "property_scores": rewards,
                "validity": valid_reward,
                "uniqueness": uniqueness_score,
                "diversity": diversity_score,
                "pass_at_n": max_per_prompt,
                # "mol_filters": filter_reward,
            },
        }

        return JSONResponse(result)

    @app.post("/prepare_receptor")  # type: ignore
    async def prepare_receptor(request: Request) -> JSONResponse:
        data = await request.json()

        if args.docking_oracle != "soft_docking":
            # No need to prepare receptors
            return JSONResponse(
                {
                    "status": "No need to prepare receptors for the selected docking oracle."
                }
            )

        metadata = data.get("metadata", None)
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
            if m["properties"][i] in receptors
        ]
        targets = list(set(targets))
        if targets == []:
            return JSONResponse(
                {"status": "No need to prepare receptors for the given batch."}
            )
        missed_receptors_1, missed_receptors_2 = receptor_processor.process_receptors(
            receptors=targets, allow_bad_res=True
        )
        if len(missed_receptors_2) > 0:
            # Return error if some receptors could not be processed
            return JSONResponse(
                {"status": "Error", "missed_receptors_2": missed_receptors_2}
            )
        else:
            return JSONResponse(
                {"status": "Success", "missed_receptors_1": missed_receptors_1}
            )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
