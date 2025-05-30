import os
import argparse
import json

from mol_gen_docking.data.rl_dataset import MolGenerationInstructionsDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=128,
        help="The number of prompts to generate",
    )
    parser.add_argument("--vina", action="store_true", dest="vina")
    parser.add_argument("--no-vina", action="store_false", dest="vina")

    parser.add_argument(
        "--max-n-props",
        type=int,
        default=3,
        help="The maximum number of properties to optimize",
    )

    parser.add_argument(
        "--data-path", type=str, default="data/mol_orz", help="Path to the dataset"
    )
    parser.add_argument(
        "--split-docking",
        nargs="+",
        default=[0.7, 0.1, 0.2],
    )
    args = parser.parse_args()

    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(os.path.join(args.data_path, "eval_data"), exist_ok=True)
    os.makedirs(os.path.join(args.data_path, "test_data"), exist_ok=True)

    dataset = MolGenerationInstructionsDataset(
        vina=args.vina, max_n_props=args.max_n_props, split_docking=args.split_docking
    )
    n_valid_prompts = int(args.n_prompts / 0.7 * 0.1)  # 10% of the training set
    n_test_prompts = int(args.n_prompts / 0.7 * 0.2)  # 20% of the training set

    # Generate the training set
    train_dataset = dataset.generate_prompt_json(args.n_prompts, docking_split=0)
    with open(os.path.join(args.data_path, "train_prompts.json"), "w") as f:
        json.dump(train_dataset, f, indent=4)

    # Generate the in-domain validation set
    eval_dataset = dataset.generate_prompt_json(
        n_valid_prompts,
        eval_name=os.path.join(args.data_path, "eval_data", "eval_prompts.json"),
        docking_split=0,
    )
    with open(os.path.join(args.data_path, "eval_data", "eval_prompts.json"), "w") as f:
        json.dump(eval_dataset, f, indent=4)

    # Generate the out-of-domain validation set
    eval_dataset += dataset.generate_prompt_json(
        n_valid_prompts,
        eval_name=os.path.join(args.data_path, "eval_data", "eval_prompts_ood.json"),
        docking_split=1,
    )
    with open(
        os.path.join(args.data_path, "eval_data", "eval_prompts_ood.json"), "w"
    ) as f:
        json.dump(eval_dataset, f, indent=4)

    # Generate the in-domain test set
    eval_dataset += dataset.generate_prompt_json(
        n_test_prompts,
        eval_name=os.path.join(args.data_path, "test_data", "test_prompts.json"),
        docking_split=0,
    )
    with open(os.path.join(args.data_path, "test_data", "test_prompts.json"), "w") as f:
        json.dump(eval_dataset, f, indent=4)

    # Generate the out-of-domain test set
    eval_dataset += dataset.generate_prompt_json(
        n_test_prompts,
        eval_name=os.path.join(args.data_path, "test_data", "test_prompts_ood.json"),
        docking_split=2,
    )
    with open(
        os.path.join(args.data_path, "test_data", "test_prompts_ood.json"), "w"
    ) as f:
        json.dump(eval_dataset, f, indent=4)
