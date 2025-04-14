import os
import argparse
import json

from mol_gen_docking.data.grpo_dataset import MolGenerationInstructionsDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=409600,
        help="The number of prompts to generate",
    )
    parser.add_argument("--vina", action="store_true", dest="vina")
    parser.add_argument("--no-vina", action="store_false", dest="vina")

    parser.add_argument(
        "--max-n-props",
        type=int,
        default=2,
        help="The maximum number of properties to optimize",
    )

    parser.add_argument(
        "--data-path", type=str, default="data/mol_orz", help="Path to the dataset"
    )
    args = parser.parse_args()

    dataset = MolGenerationInstructionsDataset(
        vina=args.vina, max_n_props=args.max_n_props
    ).generate_prompt_json(args.n_prompts + args.n_prompts // 8, format="orz")

    os.makedirs(args.data_path, exist_ok=True)

    with open(os.path.join(args.data_path, "train_prompts.json"), "w") as f:
        json.dump(dataset[args.n_prompts // 8 :], f)
    with open(os.path.join(args.data_path, "eval_prompts.json"), "w") as f:
        json.dump(dataset[: args.n_prompts // 8], f)
