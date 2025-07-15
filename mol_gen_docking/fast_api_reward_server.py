import argparse

import ray
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
    reward_model = RemoteRewardScorer.remote(  # type: ignore
        path_to_mappings=args.data_path,
        parse_whole_completion=False,
        oracle_kwargs=dict(
            exhaustiveness=args.scorer_exhaustivness,
            ncpu=args.scorer_ncpus,
        ),
    )
    reward_valid_smiles = RewardScorer(
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

        # rewards_job = reward_model.get_score.remote(  # type: ignore
        #     prompts=prompts, completions=queries
        # )
        valid_reward = reward_valid_smiles.get_score(
            prompts=prompts, completions=queries
        )
        final_smiles = reward_valid_smiles._get_smiles_list(completions=queries)

        # filter_reward_job = reward_filters.get_score.remote(prompts=prompts, completions=queries)

        # filter_reward = ray.get(filter_reward_job)

        # Get the prompts level metrics
        unique_prompts = list(set(prompts))
        group_prompt_smiles = {
            p:[s[-1] for s, p_ in zip(final_smiles, prompts) if (p_ == p) and not s == []] for p in unique_prompts
        }

        from tdc import Evaluator
        diversity_evaluator = Evaluator(name='Diversity')
        diversity_scores_dict = {
            p: diversity_evaluator(group_prompt_smiles[p]) for p in unique_prompts
        }
        diversity_score = [float(diversity_scores_dict[p]) for p in prompts]
        print(diversity_score)

        validity_evaluator = Evaluator(name='Validity')
        validity_scores_dict = {
            p: validity_evaluator(group_prompt_smiles[p]) for p in unique_prompts
        }
        validity_score = [float(validity_scores_dict[p]) for p in prompts]

        uniqueness_evaluator = Evaluator(name='Uniqueness')
        uniqueness_scores_dict = {
            p: uniqueness_evaluator(group_prompt_smiles[p]) for p in unique_prompts
        }
        uniqueness_score = [float(uniqueness_scores_dict[p]) for p in prompts]


        # rewards = ray.get(rewards_job)
        rewards = valid_reward

        print(validity_score, uniqueness_score, diversity_score)
        result = {
            "rewards": rewards,
            "scores": rewards,
            "extra_logs": {
                "property_scores": rewards,
                "valid_smiles_scores": valid_reward,
                "validity": validity_score,
                "uniqueness": uniqueness_score,
                "diversity": diversity_score,
                # "mol_filters": filter_reward,
            },
        }

        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
