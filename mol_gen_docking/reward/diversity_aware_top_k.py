import math
from typing import List

import numpy as np


def squareform_ij(n: int, i: int, j: int) -> int:
    """
    Returns the index in the condensed distance matrix for the pair (i, j)
    :param n: Number of elements
    :param i: First index
    :param j: Second index
    :return: Index in the condensed distance matrix
    """
    if i == j:
        return 0
    if i > j:
        i, j = j, i
    idx = math.comb(n, 2) - math.comb(n - i, 2) + (j - i - 1)
    return idx


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

    for idx in sorted_indices:
        if len(selected) >= k:
            break
        is_idx_too_close = any(dist[squareform_ij(n, idx, k)] < t for k in selected)
        if not is_idx_too_close:
            selected.append(idx)
    return np.array(selected)
