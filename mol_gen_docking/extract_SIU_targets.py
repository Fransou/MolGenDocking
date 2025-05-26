import os
import pickle

import json
import numpy as np
from tqdm import tqdm

import urllib.request


import pandas as pd
from biopandas.pdb import PandasPdb
from typing import Optional, Dict, Tuple


def read_pdb_to_dataframe(
    pdb_path: Optional[str] = None,
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


PATH = "data/SIU"

with open(os.path.join(PATH, "final_dic.pkl"), "rb") as f:
    data = pickle.load(f)

target_info: Dict[
    str, Dict[str, Tuple[float, float, float]]
] = {}  # pdb_id: {"center": (x, y, z), "size": (x, y, z)}

for uniprot_id in tqdm(data):
    if len(data[uniprot_id]) == 0:
        print("No data for this uniprot id")
        continue
    pdb_id = data[uniprot_id][0]["source_data"].split(",")[1].split("_")[0]

    try:
        # Download the PDB file
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        path = os.path.join(PATH, "pdb_files", f"{pdb_id}.pdb")
        urllib.request.urlretrieve(url, path)

        df = read_pdb_to_dataframe(path)

    except urllib.error.HTTPError as e:
        print(f"Failed to download {pdb_id}: {e}")
        continue

    pocket_coords = np.concatenate(data[uniprot_id][0]["pocket_coordinates"]).reshape(
        -1, 3
    )
    df_stored_coords = df[["x_coord", "y_coord", "z_coord"]].values
    diff = (np.expand_dims(df_stored_coords, 0) - np.expand_dims(pocket_coords, 1)) ** 2
    diff = diff.mean(-1).min(-1)

    if diff.sum() > 1e-8:
        print(f"Coordinates are not matching for {pdb_id}\n DIFF: {diff}")
        continue

    center = np.mean(pocket_coords, axis=0)
    size = np.max(pocket_coords - center.reshape(1, -1), axis=0)
    size = np.round(size)

    target_info[pdb_id] = {
        "center": tuple(center),
        "size": tuple(size),
    }

# Save as a json file
with open(os.path.join(PATH, "target_info.json"), "w") as f:  # type: ignore
    json.dump(target_info, f)  # type: ignore
