import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import ray
from Bio.PDB import PDBIO, MMCIFParser, PPBuilder
from ray.experimental import tqdm_ray

logging.basicConfig()
logger = logging.getLogger(__name__)

if not ray.is_initialized():
    ray.init()

Pocket = Tuple[str, str, str]  # (chain_id, residue_number, residue_name)


@ray.remote
def get_pocket_sequence_cif(
    cif_path: str, padding: float = 5.0, bar: Any = None
) -> Tuple[List[Pocket], str]:
    """
    Opens a CIF file annd identifies the position of the ligand.
    Returns the list of residues in the pocket, defined as a box around the ligand and the sequence of the protein.
    Args:
        cif_path: Path to the cif file
        padding: Padding around the ligand to define the pocket (in Å)

    Returns:
        List of residues in the pocket, each represented as a string in the format "CHAIN:RESIDUE_NUMBER:RESIDUE_NAME"
        Sequence of the protein as a string of one-letter amino acid codes
    """
    parser = MMCIFParser(QUIET=True)
    ppb = PPBuilder()
    structure = parser.get_structure("structure", cif_path)
    model = structure[0]  # Get the first model
    ligand_residue = None

    # Identify the ligand residue (assuming it's the first HETATM found)
    for chain in model:
        for residue in chain:
            if residue.id[0] != " " and residue.id[0] != "W":  # HETATM and not water
                ligand_residue = residue
                break
        if ligand_residue:
            break

    if not ligand_residue:
        raise ValueError("No ligand found in the CIF file.")

    # Get the coordinates of the ligand atoms
    ligand_coords = [atom.coord for atom in ligand_residue]

    # Define a box around the ligand (e.g., 5 Å padding)
    min_coords = [min(coord[i] for coord in ligand_coords) - padding for i in range(3)]
    max_coords = [max(coord[i] for coord in ligand_coords) + padding for i in range(3)]

    pocket_residues = []

    # Identify residues within the box
    for chain in model:
        for residue in chain:
            if residue.id[0] == " ":  # Only consider standard residues
                for atom in residue:
                    if all(
                        min_coords[i] <= atom.coord[i] <= max_coords[i]
                        for i in range(3)
                    ):
                        pocket_residues.append(residue)
                        break  # No need to check other atoms in this residue

    # Remove duplicates by converting to a set and back to a list
    pocket_residues = list(set(pocket_residues))

    for pp in ppb.build_peptides(model):
        sequences = pp.get_sequence()

    if bar is not None:
        bar.update.remote(1)

    return pocket_residues, sequences


def get_seq_to_pocket_residues(
    cif_files: List[str], padding: float = 5.0
) -> pd.DataFrame:
    """
    Converts a list of residues to a list of strings in the format "CHAIN:RESIDUE_NUMBER:RESIDUE_NAME"
    Args:
        pocket_residues: List of residues
        padding: Padding around the ligand to define the pocket (in Å)
        num_cpus: Number of CPUs to use for parallel processing. If 0, process sequentially.
    Returns:
        Dictionary mapping each CIF file path to a list of residue strings
    """
    df = pd.DataFrame(columns=["cif_file", "pocket_residues", "sequence"])
    res = []
    remote_tqdm = ray.remote(tqdm_ray.tqdm)
    bar = remote_tqdm.remote(total=len(cif_files))  # type: ignore

    for cif_file in cif_files:
        res.append(get_pocket_sequence_cif.remote(cif_file, padding, bar))
    results = ray.get(res)
    for idx, cif_file in enumerate(cif_files):
        pocket_residues, sequence = results[idx]
        df.loc[len(df)] = [cif_file, pocket_residues, sequence]

    bar.close.remote()  # type: ignore
    return df


def intersection_over_union(pocketA: List[Pocket], pocketB: List[Pocket]) -> float:
    """
    Computes the Intersection over Union (IoU) between two lists of residues.
    Args:
        pocketA: List of residues in pocket A
        pocketB: List of residues in pocket B

    Returns:
        IoU value as a float
    """
    setA = set(pocketA)
    setB = set(pocketB)
    intersection = len(setA.intersection(setB))
    union = len(setA.union(setB))
    if union == 0:
        return 0.0
    return intersection / union


def process_df_and_aggregate_pockets(
    df: pd.DataFrame, iou_threshold: float = 0.4
) -> Tuple[pd.DataFrame, Dict[str, Dict[int, List[Pocket]]]]:
    """
    Processes the DataFrame to aggregate pocket residues by sequence, and give them a cluster ID based on IoU.
    Args:
        df: DataFrame with columns "cif_file", "pocket_residues", and "sequence"

    Returns:
        DataFrame with an additional column "cluster" indicating the cluster ID for each pocket
        Dictionary mapping each sequence to a dictionary of cluster IDs and their corresponding aggregated pocket residues
    """
    sequences = df["sequence"].unique().tolist()
    # Get the IoU matrix for each sequence of the pockets
    for seq in sequences:
        seq_df = df[df["sequence"] == seq]
        if len(seq_df) == 1:
            df.loc[seq_df.index, "cluster"] = 1
            continue
        num_pockets = len(seq_df)
        iou_matrix = pd.DataFrame(index=seq_df.index, columns=seq_df.index, dtype=float)

        for i in range(num_pockets):
            for j in range(i, num_pockets):
                pocketA = seq_df.iloc[i]["pocket_residues"]
                pocketB = seq_df.iloc[j]["pocket_residues"]
                iou = intersection_over_union(pocketA, pocketB)
                iou_matrix.iat[i, j] = iou
                iou_matrix.iat[j, i] = iou

        # Cluster pockets based on IoU threshold
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        distance_matrix = 1 - iou_matrix.fillna(0).values
        condensed_distance = squareform(distance_matrix)
        Z = linkage(condensed_distance, method="single")
        clusters = fcluster(Z, t=1 - iou_threshold, criterion="distance")

        df.loc[seq_df.index, "cluster"] = clusters

    seq_pocketid_pocket: Dict[str, Dict[int, List[Pocket]]] = {}
    for seq in sequences:
        seq_df = df[df["sequence"] == seq]
        seq_pocketid_pocket[seq] = {}
        clusters = seq_df["cluster"].tolist()
        # For each cluster, aggregate the pocket residues by taking the residues that appear in at least 50% of the pockets
        for cluster_id in set(clusters):
            cluster_df = seq_df[seq_df["cluster"] == cluster_id]
            all_residues = []
            for residues in cluster_df["pocket_residues"]:
                all_residues.extend(residues)
            residue_counts = pd.Series(all_residues).value_counts()
            threshold = len(cluster_df) * 8 / 10
            aggregated_residues = residue_counts[
                residue_counts >= threshold
            ].index.tolist()
            seq_pocketid_pocket[seq][cluster_id] = aggregated_residues

            logger.info(
                f"Sequence: {seq}, Cluster ID: {cluster_id}, Number of pockets: {len(cluster_df)}, Aggregated residues: {len(aggregated_residues)}, Typical pocket size: {cluster_df.pocket_residues.map(len).median()}"
            )
    return df, seq_pocketid_pocket


def find_best_conf_pocket(df: pd.DataFrame, seq: str, pocket: List[Pocket]) -> str:
    """
    Finds the best conformation of a pocket based on the conformation with the lowest RMSD for the concerned residues.
    Args:
        df: DataFrame with columns "cif_file", "pocket_residues", "sequence", and "cluster"
        seq: Sequence to filter the DataFrame
        pocket: List of residues in the pocket to compare against
    Returns:
        CIF file path of the best conformation
    """
    # First open all concerned cif files
    parser = MMCIFParser(QUIET=True)
    structures = {}
    for cif_file in df["cif_file"]:
        structure = parser.get_structure(cif_file, cif_file)
        structures[cif_file] = structure
    # Now compute the RMSD for each structure against all others for the given pocket residues
    from Bio.PDB.Superimposer import Superimposer

    rmsd_values = {}

    for cif_fileA in df["cif_file"]:
        structureA = structures[cif_fileA]
        modelA = structureA[0]
        atomsA = []
        for chain in modelA:
            for residue in chain:
                if residue in pocket:
                    atomsA.extend(residue.get_atoms())
        if not atomsA:
            continue
        total_rmsd = 0.0
        count = 0

        for cif_fileB in df["cif_file"]:
            if cif_fileA == cif_fileB:
                continue
            structureB = structures[cif_fileB]
            modelB = structureB[0]
            atomsB = []
            for chain in modelB:
                for residue in chain:
                    if residue in pocket:
                        atomsB.extend(residue.get_atoms())
            if not atomsB or len(atomsA) != len(atomsB):
                continue
            # Superimpose atomsA onto atomsB
            sup = Superimposer()
            sup.set_atoms(atomsA, atomsB)
            total_rmsd += sup.rms
            count += 1
        if count > 0:
            rmsd_values[cif_fileA] = total_rmsd / count
    if not rmsd_values:
        raise ValueError("No valid RMSD values computed.")

    best_cif: str = min(rmsd_values, key=rmsd_values.get)  # type: ignore

    logger.info(
        f"Best conformation: {best_cif} with RMSD: {rmsd_values[best_cif]} (median of {np.median(list(rmsd_values.values()))})"
    )
    return best_cif


@ray.remote
def get_best_conf_pocket_center_width(
    df: pd.DataFrame,
    sequence: str,
    pocket: List[Pocket],
    output_dir: str,
    min_pocket_size: int = 8,
    max_pocket_size: int = 30,
    bar: Any = None,
) -> Tuple[str, np.ndarray, float]:
    """
    Finds the center and width of the pocket residues in the given CIF file.
    Args:
        df: DataFrame with columns "cif_file", "pocket_residues", "sequence", and "cluster"
        sequence: Sequence of the protein
        pocket: List of residues in the pocket
        output_dir: Directory to save the best conformation PDB file
        min_pocket_size: Minimum number of residues in the pocket to consider
        max_pocket_size: Maximum number of residues in the pocket to consider
    Returns:
        Best CIF file path,
        Center of the pocket as a numpy array of shape (3,)
        Width of the pocket as a float (maximum distance from the center to any pocket residue)
    """

    best_cif = find_best_conf_pocket(df, sequence, pocket)
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(best_cif, best_cif)
    model = structure[0]
    pocket_coords: List[List[float]] = []

    for chain in model:
        for residue in chain:
            if residue in pocket:
                for atom in residue:
                    pocket_coords.append(atom.coord)
    pocket_coords_array: np.ndarray = np.array(pocket_coords)

    center = pocket_coords_array.mean(axis=0)
    width = pocket_coords_array.max(axis=0) - pocket_coords_array.min(axis=0)
    width = np.clip(width, a_min=min_pocket_size, a_max=max_pocket_size)

    # Save to PDB file in output_dir
    file_path = os.path.join(output_dir, best_cif.replace(".cif", ".pdb"))
    io = PDBIO()
    io.set_structure(structure)
    io.save(file_path)
    if bar is not None:
        bar.update.remote(1)
    return best_cif, center, width


# Steps to generate the dataset.
# 1. Download the .tar.gz files from the SAIR dataset on Hugging Face.
# 2. Find the residues in the pocket around the ligand for each structure-ligand pair.
# 3. For each sequence, aggregate the pocket residues from all structures to find the intersection of residues.
# 4. Align the pocket for all conformations, and find the best conformation (the least different from the others).
# 5. Save the best conformation of the pocket residues as a PDB file.
# 6. From the list of all residues in the pocket, aggregate the list of residues to determine the center of the pocket.


def process_sair_dataset(
    cif_files: List[str],
    output_dir: str,
    padding: float = 5.0,
    iou_threshold: float = 0.4,
    num_cpus: int = 4,
) -> pd.DataFrame:
    """
    Processes the SAIR dataset to identify and aggregate pocket residues.
    Args:
        cif_files: List of paths to CIF files
        output_dir: Directory to save the best conformation PDB files
        padding: Padding around the ligand to define the pocket (in Å)
        iou_threshold: IoU threshold for clustering pockets
    Returns:
        DataFrame with columns "cif_file", "pocket_residues", "sequence", "cluster", "center", "width"
    """
    # Process the CIF files to get pocket residues and sequences
    df = get_seq_to_pocket_residues(cif_files, padding)
    # Aggregate pockets by sequence and cluster them based on IoU
    df, seq_pocketid_pocket = process_df_and_aggregate_pockets(df, iou_threshold)

    final_df = pd.DataFrame(
        columns=[
            "cif_file",
            "pocket_residues",
            "sequence",
            "cluster",
            "center",
            "width",
        ]
    )

    remote_tqdm = ray.remote(tqdm_ray.tqdm)
    bar = remote_tqdm.remote(total=len(seq_pocketid_pocket))  # type: ignore

    res = []
    for seq, pocket_dict in seq_pocketid_pocket.items():
        for cluster_id, pocket in pocket_dict.items():
            cluster_df = df[(df["sequence"] == seq) & (df["cluster"] == cluster_id)]
            # Only consider clusters with more than one conformation
            if len(cluster_df) <= 1:
                continue
            res.append(
                get_best_conf_pocket_center_width.remote(
                    cluster_df, seq, pocket, output_dir, 8, 30, bar
                )
            )
    results = ray.get(res)
    bar.close.remote()  # type: ignore

    idx = 0
    for seq, pocket_dict in seq_pocketid_pocket.items():
        for cluster_id, pocket in pocket_dict.items():
            cluster_df = df[(df["sequence"] == seq) & (df["cluster"] == cluster_id)]
            # Only consider clusters with more than one conformation
            if len(cluster_df) <= 1:
                continue
            try:
                best_cif, center, width = results[idx]
                logger.info(
                    f"Sequence: {seq}, Cluster ID: {cluster_id}, Best CIF: {best_cif}"
                )
                final_df.loc[len(final_df)] = [
                    best_cif,
                    pocket,
                    seq,
                    cluster_id,
                    center,
                    width,
                ]
            except ValueError as e:
                logger.warning(
                    f"Could not find best conformation for Sequence: {seq}, Cluster ID: {cluster_id}. Reason: {e}"
                )
            idx += 1

    return final_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process the SAIR dataset to identify and aggregate pocket residues."
    )
    parser.add_argument(
        "--cif-dir",
        type=str,
        required=True,
        help="Directory containing CIF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sair_pockets",
        help="Directory to save best conformation PDB files.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=5.0,
        help="Padding around the ligand to define the pocket (in Å).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.4,
        help="IoU threshold for clustering pockets.",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=4,
        help="Number of CPUs to use for parallel processing. Set to 0 for sequential processing.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cif_files = [
        os.path.join(args.cif_dir, f)
        for f in os.listdir(args.cif_dir)
        if f.endswith(".cif")
    ]
    final_df = process_sair_dataset(
        cif_files,
        args.output_dir,
        args.padding,
        args.iou_threshold,
        args.num_cpus,
    )
    final_df.to_csv(os.path.join(args.output_dir, "sair_pockets.csv"), index=False)
