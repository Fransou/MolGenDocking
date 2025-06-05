import logging
import os
import pickle
import re
import urllib.request
from multiprocessing import Pool
from subprocess import DEVNULL, STDOUT, check_call
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from prody.utilities import openFile
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PocketExtractor:
    def __init__(
        self,
        save_path: str = "data/SIU",
        t_pocket_score: float = 0.5,
        t_drug_score: float = 0.5,
        download_siu: bool = False,
    ):
        self.save_path: str = save_path
        self.t_pocket_score: float = t_pocket_score
        self.t_drug_score: float = t_drug_score
        self.processed_pdb_ids: List[str] = []
        self.data: Dict[str, Any] = {}
        if download_siu:
            self.load_siu_data()

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
            logger.info(
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

    def load_siu_data(self):
        with open(os.path.join(self.save_path, "final_dic.pkl"), "rb") as f:
            self.data = pickle.load(f)

    def _get_pdb_file(self, pdb_id: str) -> pd.DataFrame | None:
        path = os.path.join(self.save_path, "pdb_files", f"{pdb_id}.pdb")

        if os.path.exists(path):
            logger.info(f"{pdb_id} already exists, skipping download.")
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
            logger.info(f"Failed to download {pdb_id}: {e}")
            return None

    def download_pdb(self):
        for uniprot_id in tqdm(self.data):
            if len(self.data[uniprot_id]) == 0:
                logger.info("No data for this uniprot id")
                continue
            for j in range(len(self.data[uniprot_id])):
                data_row = self.data[uniprot_id][j]
                pdb_id = data_row["source_data"].split(",")[1].split("_")[0]
                if pdb_id in self.processed_pdb_ids:
                    continue

                _ = self._get_pdb_file(pdb_id)
                self.processed_pdb_ids.append(pdb_id)

    @staticmethod
    def check_prepare_receptor(path: str) -> bool:
        try:
            check_call(
                ["prepare_receptor", "-r", path, "-o", "tmp.pdbqt"],
                stdout=DEVNULL,
                stderr=STDOUT,
                timeout=60 * 5,
            )
            return True
        except Exception as e:
            logger.info(e)
            return False

    def process_fpockets(self, n_cpus: int = 16) -> Dict[str, Dict[str, Any]]:
        """
        Process the downloaded PDB files with fpocket.
        """
        all_pockets_info: Dict[str, Dict[str, Any]] = {}
        pdb_ids = sorted(
            [
                f.replace("_processed.pdb", "")
                for f in os.listdir(os.path.join(self.save_path, "pdb_files"))
                if f.endswith("_processed.pdb")
            ]
        )
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
                    desc="Checking pocket metadata: ",
                )
            )
            for r, pdb_id in zip(results, pdb_ids):
                if r is not None:
                    all_pockets_info[pdb_id] = r

        pdb_ids = list(all_pockets_info.keys())
        prepared_mask = []
        pdb_paths = [
            os.path.join(self.save_path, "pdb_files", f"{pdb_id}_processed.pdb")
            for pdb_id in pdb_ids
        ]

        if n_cpus == 1:
            # remove pdb_ids that can be prepared
            for pdb_path in tqdm(pdb_paths, desc="Checking if can be prepared: "):
                mask = self.check_prepare_receptor(pdb_path)
                prepared_mask.append(mask)
        else:
            pool = Pool(n_cpus)
            prepared_mask = list(
                tqdm(
                    pool.imap(self.check_prepare_receptor, pdb_paths),
                    total=len(pdb_paths),
                    desc="Checking if can be prepared: ",
                )
            )
        valid_pdb_ids = []
        for pdb_id, mask in zip(pdb_ids, prepared_mask):
            if mask:
                valid_pdb_ids.append(pdb_id)
        logger.info(
            f"After checking which targets can be prepared, ended up with {len(valid_pdb_ids)} files ({len(pdb_ids)} before)"
        )
        pdb_ids = valid_pdb_ids
        return {k: v for k, v in all_pockets_info.items() if k in valid_pdb_ids}

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
        for pocket_id in range(1, 6):
            pocket_path = os.path.join(
                self.save_path,
                "pdb_files",
                f"{pdb_id}_processed_out",
                "pockets",
                f"pocket{pocket_id}_atm.pdb",
            )
            if not os.path.exists(pocket_path):
                logger.info(
                    f"File {pocket_path} does not exist, run an analysis with fpocket first."
                )
                continue
            df_pocket = self.read_pdb_to_dataframe(pocket_path)
            metadata = self.extract_fpocket_metadata(pocket_path)
            # Filter out pockets with low scores
            pocket_score = metadata.get("pocket score", 0)
            drug_score = metadata.get("drug score", 0)

            if pocket_score < self.t_pocket_score or drug_score < self.t_drug_score:
                logger.info(
                    f"[{pdb_id}] Pocket {pocket_id} does not fit the score criteria ({pocket_score} , {drug_score})"
                )
                continue

            coords = self.extract_pockets_coords(df_pocket, pdb_id)

            center = (coords.max(0) + coords.min(0)) / 2
            size = np.round(
                np.clip((coords - coords.mean()).abs().max().values + 3, 10, 25)
            )

            metadata["pocket_id"] = pocket_id
            return {
                "size": tuple(size.tolist()),
                "center": tuple(center.tolist()),
                "pdb_id": pdb_id,
                "metadata": metadata,
            }
        return None
