import argparse

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openrlhf.utils.logging_utils import init_logger

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
        default=1,
        help="Exhaustiveness for the docking computations.",
    )
    parser.add_argument(
        "--scorer-ncpus",
        type=int,
        default=0.5,
        help="Number of CPUs to use for the scoring computations.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    reward_model = RewardScorer(
        path_to_mappings=args.data_path,
        parse_whole_completion=True,
        oracle_kwargs=dict(
            exhaustiveness=args.scorer_exhaustivness,
            ncpu=args.scorer_ncpus,
        ),
    )
    reward_valid_smiles = RewardScorer(
        path_to_mappings=args.data_path,
        reward="valid_smiles",
        parse_whole_completion=True,
    )
    reward_filters = RewardScorer(
        path_to_mappings=args.data_path,
        reward="MolFilters",
        parse_whole_completion=True,
    )

    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        print(data)
        queries = data.get("query")
        prompts = data.get("prompts")
        rewards = reward_model(prompts=prompts, completions=queries)
        valid_reward = reward_valid_smiles(prompts=prompts, completions=queries)
        filter_reward = reward_filters(prompts=prompts, completions=queries)
        final_reward = [(r + r_v) / 2 for r, r_v in zip(rewards, valid_reward)]

        result = {
            "rewards": final_reward,
            "scores": final_reward,
            "extra_logs": {
                "property_scores": rewards,
                "valid_smiles_scores": valid_reward,
                "mol_filters": filter_reward,
            },
        }
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
