import argparse

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openrlhf.utils.logging_utils import init_logger

import ray

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
        default=4,
        help="Exhaustiveness for the docking computations.",
    )
    parser.add_argument(
        "--scorer-ncpus",
        type=int,
        default=1,
        help="Number of CPUs to use for the scoring computations.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    RemoteRewardScorer = ray.remote(RewardScorer)
    reward_model = RemoteRewardScorer.remote(
        path_to_mappings=args.data_path,
        parse_whole_completion=False,
        oracle_kwargs=dict(
            exhaustiveness=args.scorer_exhaustivness,
            ncpu=args.scorer_ncpus,
        ),
    )
    reward_valid_smiles = RemoteRewardScorer.remote(
        path_to_mappings=args.data_path,
        reward="valid_smiles",
        parse_whole_completion=False,
    )
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
        prompts = data.get("prompts")
        
        rewards_job = reward_model.get_score.remote(prompts=prompts, completions=queries)
        valid_reward_job = reward_valid_smiles.get_score.remote(prompts=prompts, completions=queries)
        # filter_reward_job = reward_filters.get_score.remote(prompts=prompts, completions=queries)
        
        rewards = ray.get(rewards_job)
        valid_reward = ray.get(valid_reward_job)
        # filter_reward = ray.get(filter_reward_job)
        
        final_reward = rewards

        result = {
            "rewards": final_reward,
            "scores": final_reward,
            "extra_logs": {
                "property_scores": rewards,
                "valid_smiles_scores": valid_reward,
                # "mol_filters": filter_reward,
            },
        }
        
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
