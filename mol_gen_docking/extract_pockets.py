"""Script to extract pockets from the pdb files and preprocess them. Both are done in different process due to environement limitations."""

import argparse
import json
import os
from subprocess import DEVNULL, STDOUT, check_call
from typing import List

import pandas as pd
from biopandas.pdb import PandasPdb
from func_timeout import FunctionTimedOut, func_set_timeout


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
    df_final: pd.DataFrame = pd.concat([atomic_df.df["ATOM"], atomic_df.df["HETATM"]])
    return df_final


def process_pockets(file_list: List[str]) -> None:
    for f in file_list:
        processed_f = f.replace(".pdb", "_processed.pdb")
        if os.path.isfile(processed_f):
            check_call(["fpocket", "-f", processed_f], stdout=DEVNULL, stderr=STDOUT)
            # Delete all files that do not end with .pdb in f_out
            out_path = processed_f.replace(".pdb", "_out")
            for out_f in os.listdir(out_path):
                if not out_f == "pockets":
                    os.remove(os.path.join(out_path, out_f))

            out_path = os.path.join(out_path, "pockets")
            for out_f in os.listdir(out_path):
                id_pocket = int(out_f.replace("pocket", "").split("_")[0])
                if id_pocket >= 3 or not out_f.endswith(".pdb"):  # Keep top 3 pockets
                    os.remove(os.path.join(out_path, out_f))


@func_set_timeout(15 * 60)  # type: ignore
def preprocess_file(f: str) -> None:
    f_amber = f.replace(".pdb", "_processed.pdb")
    if not os.path.exists(f_amber):
        # First preprocess with pdb4amber
        check_call(
            [
                "pdb4amber",
                "-i",
                f,
                "-o",
                f_amber,
                "-d",
                "--prot",
                "--model",
                "1",
                "--reduce",
                "--add-missing-atoms",
            ],
            stdout=DEVNULL,
            stderr=STDOUT,
        )


def preprocess_pdb(file_list: List[str]) -> None:
    for i, f in enumerate(file_list):
        print("Processing: " + f)
        if i + 1 < len(file_list):
            f_amber_next = file_list[i + 1].replace(".pdb", "_amber.pdb")
            if os.path.exists(f_amber_next):
                print("This file was excluded for a timeout")
                continue
        try:
            preprocess_file(f)
        except FunctionTimedOut as e:
            print(e)


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
