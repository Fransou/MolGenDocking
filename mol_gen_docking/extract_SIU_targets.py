import os
import pickle
from typing import Any, Dict, List

from tqdm import tqdm

import urllib.request


import pandas as pd
from biopandas.pdb import PandasPdb
from typing import Optional


class PocketExtractor:
    def __init__(self, save_path: str = "data/SIU"):
        self.save_path: str = save_path
        self.processed_pdb_ids: List[str] = []

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
        try:
            # Download the PDB file
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            path = os.path.join(self.save_path, "pdb_files", f"{pdb_id}.pdb")
            urllib.request.urlretrieve(url, path)

            df = self.read_pdb_to_dataframe(path)
            return df
        except Exception as e:
            print(f"Failed to download {pdb_id}: {e}")
            return None

    def __call__(self):
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


extractor = PocketExtractor()
target_info = extractor()
