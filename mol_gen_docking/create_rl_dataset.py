import argparse
import json
import os

from mol_gen_docking.data.docking_target_extract import PocketExtractor
from mol_gen_docking.data.rl_dataset import (
    DatasetConfig,
    MolGenerationInstructionsDataset,
)
from mol_gen_docking.data.target_naming import get_names_mapping


def get_rl_data_parser() -> argparse.Namespace:
    """Get the parser for the RL dataset generation script."""
    parser = argparse.ArgumentParser(
        description="Generate the RL dataset for molecular generation with docking instructions"
    )
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

    parser.add_argument(
        "--t-pocket-score",
        type=float,
        default=0.5,
        help="Threshold for pocket score to consider a pocket.",
    )
    parser.add_argument(
        "--t-drug-score",
        type=float,
        default=0.5,
        help="Threshold for drug score to consider a pocket.",
    )

    args = parser.parse_args()

    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(os.path.join(args.data_path, "eval_data"), exist_ok=True)
    os.makedirs(os.path.join(args.data_path, "test_data"), exist_ok=True)
    return args


def generate_prompts(config: DatasetConfig, args: argparse.Namespace):
    dataset = MolGenerationInstructionsDataset(config)

    n_valid_prompts = int(args.n_prompts / 0.7 * 0.1)  # 10% of the training set
    n_test_prompts = int(args.n_prompts / 0.7 * 0.2)  # 20% of the training set

    variables = [
        ["train_prompts.json", 0, args.n_prompts],
        ["eval_data/eval_prompts.json", 0, n_valid_prompts],
        ["eval_data/eval_prompts_ood.json", 1, n_valid_prompts],
        ["test_data/test_prompts.json", 0, n_test_prompts],
        ["test_data/test_prompts_ood.json", 2, n_test_prompts],
    ]

    for name, docking_split, n_prompts in variables:
        if "train" not in name:
            prompts = dataset.generate_prompt_json(
                n_prompts,
                eval_name=os.path.join(args.data_path, name),
                docking_split=docking_split,
            )
        else:
            prompts = dataset.generate_prompt_json(
                n_prompts,
                docking_split=docking_split,
            )
        with open(os.path.join(args.data_path, name), "w") as f:
            json.dump(prompts, f, indent=4)


if __name__ == "__main__":
    args = get_rl_data_parser()
    config = DatasetConfig(
        data_path=args.data_path,
        vina=args.vina,
        max_n_props=args.max_n_props,
        split_docking=args.split_docking,
    )

    # Extract pockets from PDB files using fpocket
    if not os.path.exists(os.path.join(args.data_path, "pockets_info.json")):
        print("Extracting pockets from PDB files using fpocket...")
        extractor = PocketExtractor(
            save_path=args.data_path,
            t_pocket_score=args.t_pocket_score,
            t_drug_score=args.t_drug_score,
        )
        target_info = extractor.download_pdb()
        all_pockets_info = extractor.process_fpockets()
        df_pockets = extractor.get_pocket_df(all_pockets_info)

        with open(os.path.join(extractor.save_path, "pockets_info.json"), "w") as f:
            json.dump(all_pockets_info, f, indent=4)
        df_pockets.to_csv(
            os.path.join(extractor.save_path, "pockets_info.csv"), index=False
        )
        docking_targets = df_pockets.pdb_id.unique().tolist() + [
            "3pbl_docking",
            "1iep_docking",
            "2rgp_docking",
            "3eml_docking",
            "3ny8_docking",
            "4rlu_docking",
            "4unn_docking",
            "5mo4_docking",
            "7l11_docking",
        ]
        with open(os.path.join(extractor.save_path, "docking_targets.json"), "w") as f:
            json.dump(docking_targets, f, indent=4)

    # Generate names mapping
    if not os.path.exists(os.path.join(args.data_path, "names_mapping.json")):
        print("Generating names mapping for docking targets...")
        docking_targets = json.load(
            open(os.path.join(args.data_path, "docking_targets.json"))
        )
        names_mapping = get_names_mapping(docking_targets, n_proc=8)
        with open(os.path.join(args.data_path, "names_mapping.json"), "w") as f:
            json.dump(names_mapping, f, indent=4)

    # Finally generates prompt
    print("Generating prompts...")
    generate_prompts(config, args)
