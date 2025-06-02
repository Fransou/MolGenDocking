"""Script to extract pockets from the pdb files and preprocess them. Both are done in different process due to environement limitations."""

import argparse
import json
import os
import shutil
from typing import List

import pandas as pd
import psutil
from biopandas.pdb import PandasPdb
from openmm.app import PDBFile
from pdbfixer import PDBFixer
from prody.utilities import openFile


def read_pdb_to_dataframe(
    pdb_path: str,
    model_index: int = 1,
    parse_header: bool = True,
) -> pd.DataFrame:
    """
    Read a PDB file, and return a Pandas DataFrame containing the atomic coordinates and metadata.

    credits to: https://medium.com/@jgbrasier/working-with-pdb-files-in-python-7b538ee1b5e4
    Args:
        pdb_path (str, optional): Path to a local PDB file to read. Defaults to None.
        model_index (int, optional): Index of the model to extract from the PDB file, in case
            it contains multiple models. Defaults to 1.
        parse_header (bool, optional): Whether to parse the PDB header and extract metadata.
            Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the atomic coordinates and metadata, with one row
            per atom
    """
    atomic_df = PandasPdb().read_pdb(pdb_path)
    atomic_df = atomic_df.get_model(model_index)
    if len(atomic_df.df["ATOM"]) == 0:
        raise ValueError(f"No model found for index: {model_index}")
    return pd.concat([atomic_df.df["ATOM"], atomic_df.df["HETATM"]])


def process_pockets(file_list: List[str]):
    for f in file_list:
        processed_f = f.replace(".pdb", "_processed.pdb")
        if os.path.isfile(processed_f):
            os.system("fpocket -f " + processed_f)
            # Delete all files that do not end with .pdb in f_out
            out_path = f.replace(".pdb", "_out")
            for out_f in os.listdir(out_path):
                if not out_f == "pockets":
                    os.remove(os.path.join(out_path, out_f))

            out_path = os.path.join(out_path, "pockets")
            for out_f in os.listdir(out_path):
                id_pocket = int(out_f.replace("pocket", "").split("_")[0])
                if id_pocket >= 5:  # Keep top 5 pockets
                    os.remove(os.path.join(out_path, out_f))


def preprocess_pdb(file_list: List[str]):
    for i, f in enumerate(file_list):
        if i + 1 < len(file_list):
            f_amber_next = file_list[i + 1].replace(".pdb", "_amber.pdb")
            if os.path.exists(f_amber_next):
                continue

        f_amber = f.replace(".pdb", "_amber.pdb")
        f_fixed = f.replace(".pdb", "_fixed.pdb")
        f_out = f.replace(".pdb", "_processed.pdb")
        new_path = f.replace(".pdb", "")

        if not os.path.exists(f_amber):
            # First preprocess with pdb4amber
            os.system(f"pdb4amber -i {f} -o {f_amber} --model 1 -d")

        # Pass through pdbfixer
        if not os.path.exists(f_fixed):
            fixer = PDBFixer(filename=f_amber)
            fixer.removeHeterogens(False)

            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()

            fixer.findMissingResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()

            PDBFile.writeFile(fixer.topology, fixer.positions, open(f_fixed, "w"))

        if not os.path.exists(f_out):
            vargs = f"mk_prepare_receptor.py -i {f_fixed} -o {new_path} -p --write_pdb {f_out}"
            os.system(vargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a list of PDB files with fpocket."
    )
    parser.add_argument(
        "file_list",
        type=str,
        help="Path to the JSON file containing the list of PDB files.",
    )

    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--find-pockets", action="store_true")

    args = parser.parse_args()
    assert args.preprocess or args.find_pockets, (
        "At least one operation must be specified"
    )

    # Load the file list from the JSON file
    with open(args.file_list) as f:
        file_list = json.load(f)

    # Process the pockets
    if args.preprocess:
        preprocess_pdb(file_list)
    if args.find_pockets:
        process_pockets(file_list)
