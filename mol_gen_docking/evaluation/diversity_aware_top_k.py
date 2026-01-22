from typing import List, Optional, Sequence

import numpy as np
from rdkit import Chem
from scipy.spatial.distance import squareform

from mol_gen_docking.evaluation.fingeprints_utils import get_sim_matrix


def div_aware_top_k_from_dist(
    dist: np.ndarray[float],
    weights: np.ndarray[float],
    k: int,
    t: float,
) -> np.ndarray:
    """
    Finds at most k elements at distance of at least t from each other with highest weighted scores.
    :param dist: Condensed distance matrix
    :param weights: Weights
    :param k: Number of elements to select
    :param t: Minimum distance threshold
    :return: Indices of selected elements
    """

    n = len(weights)
    assert n * (n - 1) // 2 == len(dist), (
        "Distance matrix size does not match number of weights"
    )
    selected: List[int] = []
    sorted_indices = np.argsort(-weights)  # Sort indices by descending weights
    dist_mat = squareform(dist)
    dist_mat = dist_mat < t

    for idx in sorted_indices:
        if len(selected) >= k:
            break
        is_idx_too_close = dist_mat[idx, selected].any() if selected else False
        if not is_idx_too_close:
            selected.append(idx)
    return np.array(selected)


def diversity_aware_top_k(
    mols: List[Chem.Mol] | List[str] | np.ndarray,
    scores: Sequence[float | int],
    k: int,
    t: float,
    fingerprint_name: Optional[str] = "ecfp4-1024",
) -> float:
    dist_mat: np.ndarray[float]
    assert len(mols) == len(scores), "Mols and scores must have the same length."

    if isinstance(mols[0], str) or isinstance(mols[0], Chem.Mol):
        assert fingerprint_name is not None, (
            "Fingerprint name must be provided when mols are SMILES or Mol objects."
        )
        mols_list: List[Chem.Mol]
        if isinstance(mols[0], str):
            assert all(isinstance(smi, str) for smi in mols), (
                "All elements must be SMILES strings since the first is a string."
            )
            mols_list = [Chem.MolFromSmiles(smi) for smi in mols]
        else:
            assert all(isinstance(mol, Chem.Mol) for mol in mols), (
                "All elements must be RDKit Mol objects since the first is a Mol."
            )
            mols_list = mols  # type: ignore
        dist_mat = 1 - get_sim_matrix(mols_list, fingerprint_name=fingerprint_name)
    else:
        assert isinstance(mols, np.ndarray), "Unknown type for mols."
        assert mols.ndim == 2, (
            "Using distance matrix directly requires a 2D numpy array."
        )
        assert (mols.diagonal() == 1.0).all(), (
            "Similarity matrix diagonal must be all zeros."
        )
        dist_mat = squareform(1 - mols)

    idxs = div_aware_top_k_from_dist(dist_mat, np.array(scores), k, 1 - t)

    scores_arr = np.array(
        [scores[idx] for idx in idxs] + [0.0 for _ in range(len(idxs), k)]
    )
    out_val: float = np.mean(scores_arr)
    return out_val
