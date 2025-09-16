import argparse
import json
import logging
import os

from mol_gen_docking.data.pdb_uniprot.target_naming import get_names_mapping

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
    parser.add_argument("--fill-missing-targets", action="store_true")
    args = parser.parse_args()

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
