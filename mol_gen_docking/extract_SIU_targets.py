import os
import pickle
from typing import Any, Dict, List
import json
import numpy as np
from tqdm import tqdm

import urllib.request

from prody.utilities import openFile
import re

import pandas as pd
from biopandas.pdb import PandasPdb
from typing import Optional


class PocketExtractor:
    def __init__(self, save_path: str = "data/SIU"):
        self.save_path: str = save_path
        self.processed_pdb_ids: List[str] = []
        self.data: Dict[str, Any] = {}

    @staticmethod
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

    def _get_pdb_file(self, pdb_id: str) -> pd.DataFrame | None:
        path = os.path.join(self.save_path, "pdb_files", f"{pdb_id}.pdb")

        if os.path.exists(path):
            print(f"{pdb_id} already exists, skipping download.")
            self.processed_pdb_ids.append(pdb_id)
            return self.read_pdb_to_dataframe(path)

        if self.data == {}:
            with open(os.path.join(self.save_path, "final_dic.pkl"), "rb") as f:
                self.data = pickle.load(f)
        try:
            # Download the PDB file
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            urllib.request.urlretrieve(url, path)
            df = self.read_pdb_to_dataframe(path)
            return df
        except Exception as e:
            print(f"Failed to download {pdb_id}: {e}")
            return None

    def download_pdb(self):
        for uniprot_id in tqdm(self.data):
            if len(self.data[uniprot_id]) == 0:
                print("No data for this uniprot id")
                continue
            for j in range(len(self.data[uniprot_id])):
                data_row = self.data[uniprot_id][j]
                pdb_id = data_row["source_data"].split(",")[1].split("_")[0]
                if pdb_id in self.processed_pdb_ids:
                    continue

                _ = self._get_pdb_file(pdb_id)
                self.processed_pdb_ids.append(pdb_id)

    def process_fpockets(self):
        """
        Process the downloaded PDB files with fpocket.
        """
        all_pockets_info = {}
        pdb_ids = [
            f.replace(".pdb", "")
            for f in os.listdir(os.path.join(self.save_path, "pdb_files"))
            if f.endswith(".pdb")
        ]

        for pdb_id in tqdm(pdb_ids):
            pocket_path = os.path.join(
                self.save_path, "pockets", f"{pdb_id}_out", "pockets", "pocket1_atm.pdb"
            )
            if not os.path.exists(pocket_path):
                print(
                    f"File {pocket_path} does not exist, run an analysis with fpocket first."
                )
                continue
            df_pocket = self.read_pdb_to_dataframe(pocket_path)
            metadata = self.extract_fpocket_metadata(pocket_path)
            coords = df_pocket[["x_coord", "y_coord", "z_coord"]]
            center = coords.mean().values
            size = np.clip((coords - coords.mean()).abs().max().values + 3, 8, 25)
            pocket_info = {
                "size": tuple(size.tolist()),
                "center": tuple(center.tolist()),
                "metadata": metadata,
            }
            all_pockets_info[pdb_id] = pocket_info
        print(all_pockets_info)
        with open(os.path.join(self.save_path, "pockets_info.json"), "w") as f:
            json.dump(all_pockets_info, f, indent=4)

    def extract_fpocket_metadata(self, path):
        pdb = openFile(path, "rt")
        metadata = {}
        for loc, line in enumerate(pdb):
            startswith = line[0:6]
            if startswith == "HEADER":
                # Match the pattern "HEADER {n} - {property}" where n is a number
                pattern = r"HEADER\s+(\d+)\s+-\s+(?P<prop>.+)"
                match = re.search(pattern, line)
                if match:
                    # Extract the property name and value
                    prop = match.group("prop")
                    key = "".join(
                        [
                            word + " "
                            for word in prop.split(":")[0].split(" ")
                            if not word == ""
                        ]
                    ).lower()[:-1]
                    value = float(prop.split(":")[1].replace(" ", ""))
                    metadata[key] = value
        return metadata


if __name__ == "__main__":
    extractor = PocketExtractor()
    target_info = extractor.download_pdb()
    extractor.process_fpockets()
