import os
import pickle
import re
import urllib.request
from multiprocessing import Pool
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from prody.utilities import openFile
from tqdm import tqdm


class PocketExtractor:
    def __init__(
        self,
        save_path: str = "data/SIU",
        t_pocket_score: float = 0.5,
        t_drug_score: float = 0.5,
    ):
        self.save_path: str = save_path
        self.t_pocket_score: float = t_pocket_score
        self.t_drug_score: float = t_drug_score
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

    @staticmethod
    def extract_pockets_coords(df: pd.DataFrame, pdb_id: str) -> pd.DataFrame:
        """
        Extract the coordinates of the pocket atoms from the DataFrame.
        Multiple pockets can appear in the same PDB file, we cluster atoms
        so the minimum distance between two atoms in a pocket is at most 15 angstroms.

        Args:
            df (pd.DataFrame): DataFrame containing atomic coordinates and metadata.

        Returns:
            pd.DataFrame: DataFrame containing only the coordinates of the pocket atoms.
        """
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        coords = df[["x_coord", "y_coord", "z_coord"]].values
        dist_matrix = pdist(coords)
        lkg = linkage(dist_matrix, method="single")
        clusters = fcluster(lkg, t=5, criterion="distance")

        # Select the largest cluster
        largest_cluster = np.argmax(np.bincount(clusters))
        pocket_coords = coords[clusters == largest_cluster]

        pocket_df = pd.DataFrame(
            pocket_coords, columns=["x_coord", "y_coord", "z_coord"]
        )

        if pocket_df.shape[0] != coords.shape[0]:
            print(
                f"[{pdb_id}] Deflating by {np.round(pocket_df.shape[0] / coords.shape[0], 3) * 100}% atoms "
            )
        return pocket_df

    @staticmethod
    def extract_fpocket_metadata(path: str) -> Dict[str, float]:
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
        return dict(metadata)

    @staticmethod
    def check_prepare_receptor(path: str) -> bool:
        out_path = os.path.join(os.path.dirname(path), "tmp.pdbqt")
        status = os.system(f"prepare_receptor -r {path} -o {out_path}")
        return status == 0

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

    def process_fpockets(self, n_cpus: int = 8) -> Dict[str, Dict[str, Any]]:
        """
        Process the downloaded PDB files with fpocket.
        """
        all_pockets_info = {}
        pdb_ids = [
            f.replace(".pdb", "")
            for f in os.listdir(os.path.join(self.save_path, "pdb_files"))
            if f.endswith(".pdb")
        ]

        if n_cpus == 1:
            for pdb_id in tqdm(pdb_ids):
                infos = self.process_pocket_pdb_id(pdb_id)
                if infos is not None:
                    all_pockets_info[pdb_id] = infos
        else:
            pool = Pool(n_cpus)
            results = list(
                tqdm(
                    pool.imap(self.process_pocket_pdb_id, pdb_ids),
                    total=len(pdb_ids),
                    desc="Processing pockets",
                )
            )
            for pdb_id, pocket_info in zip(pdb_ids, results):
                if pocket_info is not None:
                    assert pdb_id == pocket_info["pdb_id"]
                    all_pockets_info[pdb_id] = pocket_info
            pool.close()
        return all_pockets_info

    def get_pocket_df(
        self, all_pockets_info: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Convert the processed pockets information into a DataFrame.
        """
        pocket_data: List[Dict[str, Any]] = []
        for pdb_id, info in all_pockets_info.items():
            data = info["metadata"]
            for k in info:
                if not k == "metadata":
                    data[k] = info[k]
            pocket_data.append(
                data,
            )

        return pd.DataFrame(pocket_data)

    def process_pocket_pdb_id(self, pdb_id: str) -> Dict[str, Any] | None:
        pocket_path = os.path.join(
            self.save_path, "pdb_files", f"{pdb_id}_out", "pockets", "pocket1_atm.pdb"
        )
        if not os.path.exists(pocket_path):
            print(
                f"File {pocket_path} does not exist, run an analysis with fpocket first."
            )
            return None
        df_pocket = self.read_pdb_to_dataframe(pocket_path)
        metadata = self.extract_fpocket_metadata(pocket_path)
        # Filter out pockets with low scores
        pocket_score = metadata.get("pocket score", 0)
        drug_score = metadata.get("drug score", 0)

        if (
            pocket_score < self.t_pocket_score
            or drug_score < self.t_drug_score
            or not self.check_prepare_receptor(pocket_path)
        ):
            return None

        coords = self.extract_pockets_coords(df_pocket, pdb_id)

        center = (coords.max(0) + coords.min(0)) / 2
        size = np.round(
            np.clip((coords - coords.mean()).abs().max().values + 3, 10, 25)
        )
        return {
            "size": tuple(size.tolist()),
            "center": tuple(center.tolist()),
            "pdb_id": pdb_id,
            "metadata": metadata,
        }
