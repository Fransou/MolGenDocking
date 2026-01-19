from typing import List

import numpy as np
from scipy.spatial.distance import squareform


def diversity_aware_top_k(
    dist: np.ndarray, weights: np.ndarray, k: int, t: float
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
    dist = dist < t  # Convert to boolean matrix for distance thresholding
    dist_mat = squareform(dist)

    for idx in sorted_indices:
        if len(selected) >= k:
            break
        is_idx_too_close = dist_mat[idx, selected].any() if selected else False
        if not is_idx_too_close:
            selected.append(idx)
    return np.array(selected)
