import argparse
import logging
import os
import pickle
import urllib.request
from typing import Optional

from tqdm import tqdm

from mol_gen_docking.data.fpocket_utils import PocketExtractor

logger = logging.getLogger(__name__)
# Set up logging to INFO level
logging.basicConfig(level=logging.INFO)


def _get_pdb_file(
    pdb_id: str,
    save_path: str,
    data: Optional[dict] = None,
) -> None:
    """Download a PDB file from the RCSB PDB database.

    Args:
        pdb_id (str): The PDB ID of the structure to download.

    Returns:
        str: The path to the downloaded PDB file.
    """
    path = os.path.join(save_path, "pdb_files", f"{pdb_id}.pdb")

    if os.path.exists(path):
        logger.info(f"{pdb_id} already exists, skipping download.")
        _ = PocketExtractor.read_pdb_to_dataframe(path)

    try:
        # Download the PDB file
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        urllib.request.urlretrieve(url, path)
        _ = PocketExtractor.read_pdb_to_dataframe(path)
    except Exception as e:
        logger.info(f"Failed to download {pdb_id}: {e}")


def download_siu_pdb(save_path: str) -> None:
    with open(os.path.join(save_path, "final_dic.pkl"), "rb") as f:
        data = pickle.load(f)
    processed_pdb_ids: list = []

    for uniprot_id in tqdm(data):
        if len(data[uniprot_id]) == 0:
            logger.info("No data for this uniprot id")
            continue
        for j in range(len(data[uniprot_id])):
            data_row = data[uniprot_id][j]
            pdb_id = data_row["source_data"].split(",")[1].split("_")[0]
            if pdb_id in processed_pdb_ids:
                continue

            _get_pdb_file(pdb_id, save_path, data)
            processed_pdb_ids.append(pdb_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the RL dataset for molecular generation with docking instructions"
    )

    parser.add_argument(
        "--data-path", type=str, default="data/mol_orz", help="Path to the dataset"
    )
    args = parser.parse_args()

    os.makedirs(args.data_path, exist_ok=True)
    # Download pdb diles
    download_siu_pdb(args.data_path)
