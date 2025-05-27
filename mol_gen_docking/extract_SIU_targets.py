import os
import pickle
from typing import Any, Dict, List

import json
import numpy as np
from tqdm import tqdm

import urllib.request


import pandas as pd
from biopandas.pdb import PandasPdb
from typing import Optional, Tuple


class PocketExtractor:
    def __init__(self, save_path: str = "data/SIU"):
        self.save_path: str = save_path
        self.target_info: Dict[
            str, Dict[str, Tuple[float, float, float]]
        ] = {}  # pdb_id: {"center": (x, y, z), "size": (x, y, z)}
        self.pdb_dfs: Dict[
            str, pd.DataFrame
        ] = {}  # pdb_id: DataFrame with atomic coordinates

        with open(os.path.join(self.save_path, "final_dic.pkl"), "rb") as f:
            self.data: Dict[str, Any] = pickle.load(f)

    def read_pdb_to_dataframe(
        self,
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

    def _get_pdb_file(self, pdb_id: str) -> pd.DataFrame | None:
        if pdb_id in self.pdb_dfs:
            return self.pdb_dfs[pdb_id]
        try:
            # Download the PDB file
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            path = os.path.join(self.save_path, "pdb_files", f"{pdb_id}.pdb")
            urllib.request.urlretrieve(url, path)

            df = self.read_pdb_to_dataframe(path)
            self.pdb_dfs[pdb_id] = df
            return df
        except urllib.error.HTTPError as e:
            print(f"Failed to download {pdb_id}: {e}")
            self.pdb_dfs[pdb_id] = None
            return None

    def __call__(self):
        for uniprot_id in tqdm(self.data):
            if len(self.data[uniprot_id]) == 0:
                print("No data for this uniprot id")
                continue
            center: List[np.ndarray] = []

            for j in range(len(self.data[uniprot_id])):
                data_row = self.data[uniprot_id][j]
                pdb_id = data_row["source_data"].split(",")[1].split("_")[0]
                df = self._get_pdb_file(pdb_id)
                if df is None:
                    continue

                # Check the coordinates of the pocket match the pdb file
                pocket_coords = np.concatenate(
                    data_row["pocket_coordinates"], 0
                ).reshape(-1, 3)

                df_stored_coords = df[["x_coord", "y_coord", "z_coord"]].values
                diff = (
                    np.abs(
                        np.expand_dims(df_stored_coords, 0)
                        - np.expand_dims(pocket_coords, 1)
                    )
                    .mean(-1)
                    .min(-1)
                )
                if diff.sum() > 1e-8:
                    print(f"Coordinates are not matching for {pdb_id}\n DIFF: {diff}")
                    continue

                center.append(np.concatenate(data_row["coordinates"]).reshape(-1, 3))
            if len(center) == 0:
                print(f"No coordinates found for {uniprot_id}")
                continue
            center = np.mean(np.concatenate(center), axis=0)
            size = np.max(np.abs(center.reshape(1, -1) - df_stored_coords), axis=0) + 2
            size = np.clip(size, 15, 25)  # Ensure size is within a reasonable range
            size = np.round(size)
            self.target_info[pdb_id] = {
                "center": tuple(center),
                "size": tuple(size),
            }

            with open(os.path.join(self.save_path, "target_info.json"), "w") as f:
                json.dump(self.target_info, f)

        return self.target_info


extractor = PocketExtractor()
target_info = extractor()
