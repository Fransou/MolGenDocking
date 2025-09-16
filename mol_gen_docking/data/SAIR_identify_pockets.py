import logging
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import ray
from Bio.PDB import PDBIO, MMCIFParser, Superimposer
from ray.experimental import tqdm_ray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger(__name__)

if not ray.is_initialized():
    ray.init()


def cif_file_to_entry_id(cif_file: str) -> int:
    """Extracts the entry ID from a CIF file name.
    Args:
        cif_file: Path to the CIF file
    Returns:
        Entry ID as a string
    """
    return int(cif_file.split("/")[-1].split("_")[1])


def filter_from_parquet(
    df_parquet: pd.DataFrame, cif_files: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Filters the list of CIF files to only include those that are in the top 50% most potent ligands
    for each sequence, and have a confidence score in the top 50%.
    Args:
        df_parquet: DataFrame containing the SAIR parquet data
        cif_files: List of paths to CIF files
    Returns:
        Filtered list of CIF files
        Kept entry IDs
    """
    entry_ids = [cif_file_to_entry_id(f) for f in cif_files]
    df_filtered = df_parquet[df_parquet["entry_id"].isin(entry_ids)]

    new_cif_files: List[str] = []
    kept_entry_ids: List[str] = []

    for protein in tqdm(
        df_filtered["protein"].unique(),
        desc="Filtering CIF files based on potency and confidence",
    ):
        df_prot = df_filtered[df_filtered["protein"] == protein]
        if len(df_prot) == 0:
            continue
        pIC50_threshold = df_prot["pIC50"].quantile(0.5)
        df_prot_filtered = df_prot[(df_prot["pIC50"] >= pIC50_threshold)]
        confidence_threshold = df_prot_filtered["confidence_score"].quantile(0.5)
        df_prot_filtered = df_prot_filtered[
            df_prot_filtered["confidence_score"] >= confidence_threshold
        ]
        df_prot_filtered = df_prot_filtered.sort_values("pIC50", ascending=False).iloc[
            : min(df_prot_filtered.shape[0], 20)
        ]
        entry_ids_prot = df_prot_filtered["entry_id"].unique().tolist()

        for entry_id in entry_ids_prot:
            matching_files = [
                f for f in cif_files if cif_file_to_entry_id(f) == entry_id
            ]
            new_cif_files.extend(matching_files)
            kept_entry_ids.append(entry_id)

    return new_cif_files, kept_entry_ids


@ray.remote(num_cpus=1)
def get_pocket_cif(
    cif_path: str, topk: int = 3, bar: Any = None
) -> Tuple[List[Any], Any]:
    """
    Opens a CIF file annd identifies the position of the ligand.
    Returns the list of residues in the pocket, defined as the top-k closest residues to the ligand + a padding.
    Args:
        cif_path: Path to the cif file
        topk: Number of closest residues to the ligand to include in the pocket

    Returns:
        List of residues in the pocket
        Structure object from Biopython (can be None if not needed)
    """
    parser = MMCIFParser(QUIET=True)
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

    # Find the closest residues to each ligand atom
    pocket_residues_set = set()
    for ligand_coord in ligand_coords:
        distances = []
        for chain in model:
            for residue in chain:
                if residue.id[0] == " ":
                    for atom in residue:
                        dist = np.linalg.norm(atom.coord - ligand_coord)
                        distances.append((dist, residue))
        distances.sort(key=lambda x: x[0])  # type: ignore
        closest_residues = [distances[i][1] for i in range(min(topk, len(distances)))]
        pocket_residues_set.update(closest_residues)
    pocket_residues = list(set(pocket_residues_set))

    if bar is not None:
        bar.update.remote(1)

    return pocket_residues, structure


def get_seq_to_pocket_residues(
    cif_files: List[str], df_parquet: pd.DataFrame, topk: int = 3
) -> pd.DataFrame:
    """
    Converts a list of residues to a list of strings in the format "CHAIN:RESIDUE_NUMBER:RESIDUE_NAME"
    Args:
        pocket_residues: List of residues
        topk: Number of closest residues to the ligand to include in the pocket
        num_cpus: Number of CPUs to use for parallel processing. If 0, process sequentially.
    Returns:
        Dictionary mapping each CIF file path to a list of residue strings
    """
    df = pd.DataFrame(
        columns=[
            "id",
            "pocket_residues",
            "structure",
            "prot_id",
            "sequence",
            "avg_pIC50",
            "avg_confidence",
        ]
    )
    res = []
    remote_tqdm = ray.remote(tqdm_ray.tqdm)
    bar = remote_tqdm.remote(total=len(cif_files))  # type: ignore

    for cif_file in cif_files:
        res.append(get_pocket_cif.remote(cif_file, topk, bar))
    results = ray.get(res)
    proteins = (
        df_parquet[
            df_parquet["entry_id"].isin([cif_file_to_entry_id(f) for f in cif_files])
        ]["protein"]
        .unique()
        .tolist()
    )
    sub_df_parquet = df_parquet[df_parquet["protein"].isin(proteins)]
    for idx, cif_file in enumerate(
        tqdm(cif_files, desc="Processing CIF files to get pocket residues")
    ):
        pocket_residues, structure = results[idx]
        entry_id = cif_file_to_entry_id(cif_file)
        df_entry = sub_df_parquet[sub_df_parquet["entry_id"] == entry_id]
        prot_id = df_entry["protein"].values[0]
        sequence = df_entry["sequence"].values[0]
        df_prot = sub_df_parquet[sub_df_parquet["protein"] == prot_id]
        avg_pIC50 = df_prot["pIC50"].mean()
        avg_confidence = df_prot["confidence_score"].mean()

        df.loc[len(df)] = [
            cif_file.split("/")[-1].replace(".cif", ""),
            pocket_residues,
            structure,
            prot_id,
            sequence,
            avg_pIC50,
            avg_confidence,
        ]

    bar.close.remote()  # type: ignore
    return df


def intersection_over_union(pocketA: List[Any], pocketB: List[Any]) -> float:
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


@ray.remote(num_cpus=4)
def aggregate_pocket(
    seq_df: pd.DataFrame, iou_threshold: float = 0.4, pbar: Any = None
) -> List[Any]:
    if len(seq_df) == 1:
        if pbar is not None:
            pbar.update.remote(1)
        return [1.0]

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

    distance_matrix = 1 - iou_matrix.fillna(0).values
    condensed_distance = squareform(distance_matrix)
    Z = linkage(condensed_distance, method="single")
    clusters = fcluster(Z, t=1 - iou_threshold, criterion="distance")
    if pbar is not None:
        pbar.update.remote(1)
    return clusters.tolist()  # type: ignore


def process_df_and_aggregate_pockets(
    df: pd.DataFrame, iou_threshold: float = 0.4
) -> Tuple[pd.DataFrame, Dict[str, Dict[int, List[Any]]]]:
    """
    Processes the DataFrame to aggregate pocket residues by sequence, and give them a cluster ID based on IoU.
    Args:
        df: DataFrame with columns "id", "pocket_residues", and "sequence"

    Returns:
        DataFrame with an additional column "cluster" indicating the cluster ID for each pocket
        Dictionary mapping each sequence to a dictionary of cluster IDs and their corresponding aggregated pocket residues
    """
    sequences = df["sequence"].unique().tolist()
    # Get the IoU matrix for each sequence of the pockets
    res = []
    remote_tqdm = ray.remote(tqdm_ray.tqdm)
    pbar = remote_tqdm.remote(
        total=len(sequences), desc="Clustering pockets by sequence: "
    )  # type: ignore

    for seq in sequences:
        seq_df = df[df["sequence"] == seq]
        res.append(
            aggregate_pocket.remote(seq_df, iou_threshold=iou_threshold, pbar=pbar)
        )

    all_clusters = ray.get(res)
    for seq, clusters in zip(sequences, all_clusters):
        seq_df = df[df["sequence"] == seq]
        df.loc[seq_df.index, "cluster"] = clusters

    seq_pocketid_pocket: Dict[str, Dict[int, List[Any]]] = {}
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
            threshold = len(cluster_df) * 7 / 10
            aggregated_residues = residue_counts[
                residue_counts >= threshold
            ].index.tolist()
            seq_pocketid_pocket[seq][cluster_id] = aggregated_residues

            logger.info(
                f"Sequence: {seq}, Cluster ID: {cluster_id}, Number of pockets: {len(cluster_df)}, Aggregated residues: {len(aggregated_residues)}, Typical pocket size: {cluster_df.pocket_residues.map(len).median()}"
            )
    return df, seq_pocketid_pocket


def find_best_conf_pocket(df: pd.DataFrame, seq: str, pocket: List[Any]) -> str:
    """
    Finds the best conformation of a pocket based on the conformation with the lowest RMSD for the concerned residues.
    Args:
        df: DataFrame with columns "id", "pocket_residues", "sequence", and "cluster"
        seq: Sequence to filter the DataFrame
        pocket: List of residues in the pocket to compare against
    Returns:
        CIF file path of the best conformation
    """
    # First open all concerned cif files
    import time

    t0 = time.time()

    structures = {}
    for id in df["id"]:
        structures[id] = df[df["id"] == id]["structure"].values[0]

    def get_rmsd_list(
        structA: Any, structBs: List[Any], pocket: List[Any]
    ) -> List[float] | None:
        sup = Superimposer()

        modelA = structA[0]
        rmsd_vals = []
        for structB in structBs:
            modelB = structB[0]
            atomsA = []
            atomsB = []
            for chain in modelA:
                for residue in chain:
                    if residue in pocket:
                        atomsA.extend(residue.get_atoms())
            for chain in modelB:
                for residue in chain:
                    if residue in pocket:
                        atomsB.extend(residue.get_atoms())
            if not atomsA or not atomsB or len(atomsA) != len(atomsB):
                return None
            sup.set_atoms(atomsA, atomsB)

            rmsd_vals.append(float(sup.rms))
        return rmsd_vals

    @ray.remote(num_cpus=4)
    def chunk_get_rmsd_list(
        structAs: List[Any], structBs: List[Any], pocket: List[Any]
    ) -> List[List[float] | None]:
        return [
            get_rmsd_list(structA, structBs[k + 1 :], pocket)
            for k, structA in enumerate(structAs)
        ]

    rmsd_values = []
    all_ids = list(df["id"])
    if len(all_ids) < 200:  # Do not parrallelize
        for i in range(len(all_ids)):
            idA = all_ids[i]
            structureA = structures[idA]
            structureBs = [structures[all_ids[j]] for j in range(i + 1, len(all_ids))]
            rmsd_values.append(get_rmsd_list(structureA, structureBs, pocket))
    else:
        chunk_size = 100
        chunks = [
            all_ids[i : i + chunk_size] for i in range(0, len(all_ids), chunk_size)
        ]
        res = []
        for i, chunk in enumerate(chunks):
            res.append(
                chunk_get_rmsd_list.remote(
                    [structures[id] for id in chunk],
                    [structures[id] for id in all_ids[i * chunk_size :]],
                    pocket,
                )
            )
        chunk_results = ray.get(res)
        rmsd_values = sum(chunk_results, [])

    rmsd_matrix = squareform(sum(rmsd_values, []))  # type: ignore
    rmsd_df = pd.DataFrame(rmsd_matrix, index=df["id"], columns=df["id"], dtype=float)
    best_struc = rmsd_df.mean(axis=1).idxmin()
    assert isinstance(best_struc, str)
    t1 = time.time()
    print(
        f"Found best conformation in {t1 - t0:.2f} second (for {len(all_ids)} structures). Best structure: {best_struc}"
    )

    return best_struc


@ray.remote(num_cpus=1)
def get_best_conf_pocket_center_width(
    df: pd.DataFrame,
    sequence: str,
    pocket: List[Any],
    output_dir: str,
    min_pocket_size: int = 8,
    max_pocket_size: int = 30,
    bar: Any = None,
) -> Tuple[str, np.ndarray, float]:
    """
    Finds the center and width of the pocket residues in the given CIF file.
    Args:
        df: DataFrame with columns "id", "pocket_residues", "sequence", and "cluster"
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

    import time

    t0 = time.time()

    best_struc = find_best_conf_pocket(df, sequence, pocket)

    structure = df[df["id"] == best_struc]["structure"].values[0]
    model = structure[0]
    pocket_coords: List[List[float]] = []

    for chain in model:
        for residue in chain:
            if residue in pocket:
                for atom in residue:
                    pocket_coords.append(atom.coord)
    pocket_coords_array: np.ndarray = np.array(pocket_coords)

    center = (pocket_coords_array.max(axis=0) + pocket_coords_array.min(axis=0)) / 2
    width = pocket_coords_array.max(axis=0) - pocket_coords_array.min(axis=0)
    width = np.clip(width, a_min=min_pocket_size, a_max=max_pocket_size)

    # Save to PDB file in output_dir without the ligand
    file_path = os.path.join(output_dir, best_struc + ".pdb")
    io = PDBIO()

    # Remove the ligand from the structure
    for chain in model:
        for residue in list(chain):
            if residue.id[0] != " " and residue.id[0] != "W":
                chain.detach_child(residue.id)
    io.set_structure(structure)
    io.save(file_path)

    if bar is not None:
        bar.update.remote(1)

    t1 = time.time()
    logger.info(f"Processed {best_struc} in {t1 - t0:.2f} seconds.")
    return best_struc, center, width


# Steps to generate the dataset.
# 1. Download the .tar.gz files from the SAIR dataset on Hugging Face.
# 2. Open the parquet sair file. Find the top-50% most potent ligands (based on pIC50) for each sequence, and select only the ones with a top 50% confidence score.
# 3. Find the residues in the pocket around the ligand for each structure-ligand pair.
# 4. For each sequence, aggregate the pocket residues from all structures to find the intersection of residues.
# 5. Align the pocket for all conformations, and find the best conformation (the least different from the others).
# 6. Save the best conformation of the pocket residues as a PDB file.
# 7. From the list of all residues in the pocket, aggregate the list of residues to determine the center of the pocket.


def process_sair_dataset(
    cif_files: List[str],
    df_parquet: pd.DataFrame,
    output_dir: str,
    topk: int = 3,
    iou_threshold: float = 0.4,
    num_cpus: int = 4,
) -> pd.DataFrame:
    """
    Processes the SAIR dataset to identify and aggregate pocket residues.
    Args:
        cif_files: List of paths to CIF files
        df_parquet: DataFrame containing the SAIR parquet data
        output_dir: Directory to save the best conformation PDB files
        topk: Number of closest residues to the ligand to include in the pocket
        iou_threshold: IoU threshold for clustering pockets
    Returns:
        DataFrame with columns "id", "pocket_residues", "cluster", "center", "width"
    """
    t0 = time.time()

    cif_files, kept_entry_ids = filter_from_parquet(df_parquet, cif_files)
    t1 = time.time()
    print(
        f"Kept {len(cif_files)} CIF files after filtering. Time taken: {t1 - t0:.2f} seconds."
    )

    # Process the CIF files to get pocket residues and sequences
    df = get_seq_to_pocket_residues(cif_files, df_parquet, topk)
    t2 = time.time()
    print(
        f"Processed CIF files to get pocket residues. Time taken: {t2 - t1:.2f} seconds."
    )

    # Aggregate pockets by sequence and cluster them based on IoU
    df, seq_pocketid_pocket = process_df_and_aggregate_pockets(df, iou_threshold)
    t3 = time.time()
    print(f"Aggregated pockets by sequence. Time taken: {t3 - t2:.2f} seconds.")

    final_df = pd.DataFrame(
        columns=[
            "id",
            "pocket_residues",
            "cluster",
            "center_x",
            "center_y",
            "center_z",
            "width_x",
            "width_y",
            "width_z",
            "n_ligand_poses",
            "sequence",
            "prot_id",
            "avg_pIC50",
            "avg_confidence",
        ]
    )

    remote_tqdm = ray.remote(tqdm_ray.tqdm)
    bar = remote_tqdm.remote(
        total=len(seq_pocketid_pocket),
        desc="Finding best configuration for each pocket: ",
    )  # type: ignore

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
                best_struc, center, width = results[idx]
                logger.info(
                    f"Sequence: {seq}, Cluster ID: {cluster_id}, Best Struc: {best_struc}"
                )
                final_df.loc[len(final_df)] = [
                    best_struc,
                    pocket,
                    cluster_id,
                    center[0],
                    center[1],
                    center[2],
                    width[0],
                    width[1],
                    width[2],
                    len(cluster_df),
                    seq,
                    cluster_df["prot_id"].values[0],
                    cluster_df["avg_pIC50"].values[0],
                    cluster_df["avg_confidence"].values[0],
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
        "--sair-dir",
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
        "--topk",
        type=int,
        default=3,
        help="Padding around the ligand to define the pocket (in Å).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.6,
        help="IoU threshold for clustering pockets.",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=2,
        help="Number of CPUs to use for parallel processing. Set to 0 for sequential processing.",
    )
    parser.add_argument(
        "--idx-to-download",
        type=int,
        default=-1,
        help="Index of the subset to download. If -1, do not download.",
    )
    args = parser.parse_args()

    df_parquet = pd.read_parquet(os.path.join(args.sair_dir, "sair.parquet"))

    if args.idx_to_download == -1:
        os.makedirs(args.output_dir, exist_ok=True)
        cif_dir = os.path.join(args.sair_dir, "structures")

        cif_files = [
            os.path.join(cif_dir, f) for f in os.listdir(cif_dir) if f.endswith("0.cif")
        ]
        final_df = process_sair_dataset(
            cif_files,
            df_parquet,
            args.output_dir,
            args.topk,
            args.iou_threshold,
            args.num_cpus,
        )
        final_df.to_csv(os.path.join(args.output_dir, "sair_pockets.csv"), index=False)

    else:
        from subprocess import check_call

        structure_folders = os.path.join(args.sair_dir, f"sair_{args.idx_to_download}")
        os.makedirs(structure_folders, exist_ok=True)
        os.makedirs(os.path.join(structure_folders, "structures"), exist_ok=True)
        check_call(
            [
                "python",
                "mol_gen_docking/data/SAIR_download.py",
                "--output-dir",
                structure_folders,
                "--start-subset",
                str(args.idx_to_download),
                "--end-subset",
                str(args.idx_to_download + 1),
            ]
        )
        os.makedirs(args.output_dir, exist_ok=True)
        cif_files = [
            os.path.join(structure_folders, "structures", f)
            for f in os.listdir(os.path.join(structure_folders, "structures"))
            if f.endswith("0.cif")
        ]
        final_df = process_sair_dataset(
            cif_files,
            df_parquet,
            args.output_dir,
            args.topk,
            args.iou_threshold,
            args.num_cpus,
        )
        final_df.to_csv(
            os.path.join(args.output_dir, f"sair_pockets_{args.idx_to_download}.csv"),
            index=False,
        )
        # Delete the .cif files to save space
        for f in tqdm(cif_files, desc="Deleting CIF files"):
            os.remove(f)
