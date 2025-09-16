import argparse
import json
import logging
import os

from mol_gen_docking.data.fpocket_utils import PocketExtractor

logger = logging.getLogger(__name__)
# Set up logging to INFO level
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the RL dataset for molecular generation with docking instructions"
    )

    parser.add_argument(
        "--data-path", type=str, default="data/mol_orz", help="Path to the dataset"
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
    args = parser.parse_args()

    # Download pdb diles
    extractor = PocketExtractor(
        save_path=args.data_path,
        t_pocket_score=args.t_pocket_score,
        t_drug_score=args.t_drug_score,
    )

    # Extract pockets from PDB files after using fpocket
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

    with open(os.path.join(extractor.save_path, "docking_targets.json"), "w") as f:
        json.dump(docking_targets, f, indent=4)
