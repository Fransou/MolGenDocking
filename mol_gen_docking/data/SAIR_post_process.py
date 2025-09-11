import json
import os
from subprocess import DEVNULL, STDOUT, check_call

from mol_gen_docking.data.pdb_uniprot.target_naming import get_names_mapping_uniprot

# After all structures have been downloaded, and the pockets have been extracted, performs the last steps of the data processing:
# 1. Ensures all pdb files can be read byt the prepare_receptor script from AutoDockTools
# 2. Finds a textual description for each protein.


def check_pdb_file(path: str) -> bool:
    """Check that all pdb files in the given directory can be read by the prepare_receptor script from AutoDockTools.

    Args:
        pdb_dir (str): The directory containing the pdb files to check.
    """
    try:
        check_call(
            ["prepare_receptor", "-r", path, "-o", "tmp.pdbqt"],
            stdout=DEVNULL,
            stderr=STDOUT,
            timeout=60 * 5,
        )
        return True
    except Exception:
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Post-process the SAIR dataset after downloading and pocket extraction."
    )
    parser.add_argument(
        "--data-path", type=str, default="data/sair_rl", help="Path to the dataset"
    )
    parser.add_argument(
        "--sair-path",
        type=str,
        default="data/sair_pockets",
        help="Path to the SAIR raw data",
    )

    args = parser.parse_args()

    data_path = args.data_path
    data_pdb_path = os.path.join(data_path, "pdb_files")
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(data_pdb_path, exist_ok=True)

    pocket_info = json.load(open(os.path.join(args.sair_path, "pockets_info.json")))
    kept_pockets = []
    kept_pockets_uniprots = []
    for structure in pocket_info:
        pdb_path = os.path.join(args.sair_path, "structures", f"{structure}.pdb")
        if not check_pdb_file(pdb_path):
            print(f"[Warning] {structure} could not be processed by prepare_receptor.")
            continue
        pocket_info[structure]["pdb_path"] = pdb_path

        # Copy pdb file to the new location (pdb_path)
        os.system(
            f"cp {pdb_path} {os.path.join(data_pdb_path, f'{structure}_processed.pdb')}"
        )
        kept_pockets.append(structure)
        kept_pockets_uniprots.append(pocket_info[structure]["metadata"]["prot_id"])

    names_mapping_uniprot = get_names_mapping_uniprot(
        kept_pockets_uniprots, names_override=kept_pockets
    )

    docking_targets = list(
        [
            v
            for v in names_mapping_uniprot.values()
            if v in kept_pockets or "docking" in v
        ]
    )

    json.dump(
        pocket_info,
        open(os.path.join(data_path, "pockets_info.json"), "w"),
        indent=4,
    )

    json.dump(
        docking_targets,
        open(os.path.join(data_path, "docking_targets.json"), "w"),
        indent=4,
    )

    json.dump(
        names_mapping_uniprot,
        open(os.path.join(data_path, "names_mapping.json"), "w"),
        indent=4,
    )
    print(f"Kept {len(kept_pockets)} pockets after filtering.")
