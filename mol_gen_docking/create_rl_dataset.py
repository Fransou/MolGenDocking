import argparse
import json
import logging
import os

from tqdm import tqdm

from mol_gen_docking.data.docking_target_extract import PocketExtractor
from mol_gen_docking.data.featurize_pockets import ProteinStructureEmbeddingExtractor
from mol_gen_docking.data.rl_dataset import (
    DatasetConfig,
    MolGenerationInstructionsDataset,
)
from mol_gen_docking.data.target_naming import get_names_mapping

logger = logging.getLogger(__name__)
# Set up logging to INFO level
logging.basicConfig(level=logging.INFO)


def get_rl_data_parser() -> argparse.Namespace:
    """Get the parser for the RL dataset generation script."""
    parser = argparse.ArgumentParser(
        description="Generate the RL dataset for molecular generation with docking instructions"
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=512,
        help="The number of prompts to generate",
    )
    parser.add_argument("--vina", action="store_true", dest="vina")
    parser.add_argument("--no-vina", action="store_false", dest="vina")

    parser.add_argument("--min-n-pocket-infos", type=int, default=-1)

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
        default=0.2,
        help="Threshold for pocket score to consider a pocket.",
    )
    parser.add_argument(
        "--t-drug-score",
        type=float,
        default=0.5,
        help="Threshold for drug score to consider a pocket.",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--fill-missing-targets", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.data_path, exist_ok=True)
    return args


def generate_prompts(config: DatasetConfig, args: argparse.Namespace) -> None:
    dataset = MolGenerationInstructionsDataset(config)

    n_valid_prompts = int(args.n_prompts / 0.7 * 0.1)  # 10% of the training set
    n_test_prompts = int(args.n_prompts / 0.7 * 0.2)  # 20% of the training set

    variables = [
        ["train_prompts", 0, args.n_prompts],
        ["eval_data/eval_prompts", 0, n_valid_prompts],
        ["eval_data/eval_prompts_ood", 1, n_valid_prompts],
        ["test_data/test_prompts", 0, n_test_prompts],
        ["test_data/test_prompts_ood", 2, n_test_prompts],
    ]

    for name, docking_split, n_prompts in variables:
        data = dataset.generate_hf_dataset(
            n_prompts,
            docking_split=docking_split,
        )
        data.save_to_disk(os.path.join(args.data_path, name))
        for i in range(10):
            print(data[i]["prompt"][1]["content"])

        # dataset.save_sim_matrices()


if __name__ == "__main__":
    args = get_rl_data_parser()
    config = DatasetConfig(
        data_path=args.data_path,
        vina=args.vina,
        max_n_props=args.max_n_props,
        split_docking=args.split_docking,
        min_n_pocket_infos=args.min_n_pocket_infos,
        chat_template={"user": "role", "content": "content"},
    )
    # Download pdb diles
    extractor = PocketExtractor(
        save_path=args.data_path,
        t_pocket_score=args.t_pocket_score,
        t_drug_score=args.t_drug_score,
        download_siu=args.download,
    )
    extractor.download_pdb()

    if not args.download:
        # Extract pockets from PDB files after using fpocket
        if not os.path.exists(os.path.join(args.data_path, "pockets_info.json")):
            print("Extracting pockets from PDB files using fpocket...")

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
            print(f"Extracted {len(docking_targets)} targets")

            with open(
                os.path.join(extractor.save_path, "docking_targets.json"), "w"
            ) as f:
                json.dump(docking_targets, f, indent=4)

        # Generate names mapping and get sequences
        if not os.path.exists(os.path.join(args.data_path, "names_mapping.json")):
            print("Generating names mapping for docking targets...")
            docking_targets = json.load(
                open(os.path.join(args.data_path, "docking_targets.json"))
            )
            names_mapping = get_names_mapping(docking_targets, n_proc=8)
            with open(os.path.join(args.data_path, "names_mapping.json"), "w") as f:
                json.dump(names_mapping, f, indent=4)
        elif args.fill_missing_targets:
            print("Filling missing targets...")
            docking_targets = json.load(
                open(os.path.join(args.data_path, "docking_targets.json"))
            )
            with open(os.path.join(args.data_path, "names_mapping.json")) as f:
                names_mapping = json.load(f)
            to_requery = []
            for target in docking_targets:
                if target not in names_mapping.values():
                    to_requery.append(target)
            additional_names_mapping = get_names_mapping(to_requery, n_proc=8)
            additional_names_mapping.update(names_mapping)
            with open(os.path.join(args.data_path, "names_mapping.json"), "w") as f:
                json.dump(additional_names_mapping, f, indent=4)

        # Featurize all pockets
        embedding_extractor = ProteinStructureEmbeddingExtractor(
            data_dir=args.data_path
        )
        with open(os.path.join(args.data_path, "docking_targets.json")) as f:
            docking_targets = json.load(f)
        os.makedirs(os.path.join(args.data_path, "pockets_embeddings"), exist_ok=True)
        for pdb_id in tqdm(docking_targets, desc="Extracting embeddings"):
            pdb_path = os.path.join(
                args.data_path, "pdb_files", f"{pdb_id}_processed.pdb"
            )
            output_path = os.path.join(
                args.data_path, "pockets_embeddings", f"{pdb_id}_embeddings.pt"
            )
            if not os.path.exists(output_path):
                embedding_extractor.extract_embeddings(pdb_path, output_path)
            else:
                print(f"Embeddings for {pdb_id} already exist, skipping...")

        # Finally generates prompt
        print("Generating prompts...")
        generate_prompts(config, args)
