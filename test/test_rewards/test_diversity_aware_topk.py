import numpy as np
import pytest
from scipy.spatial.distance import squareform

from mol_gen_docking.reward.diversity_aware_top_k import diversity_aware_top_k

TEST_DIV_AWARE_TOPK = [
    {
        "dist": squareform(
            np.array(
                [
                    [0.0, 0.5, 0.9, 0.4],
                    [0.5, 0.0, 0.2, 0.8],
                    [0.9, 0.2, 0.0, 0.3],
                    [0.4, 0.8, 0.3, 0.0],
                ]
            )
        ),
        "weights": np.array([0.1, 0.4, 0.3, 0.2]),
        "results": {
            (1, 0.6): [1],
            (2, 0.6): [1, 3],
            (1, 0.1): [1],
            (2, 0.1): [1, 2],
            (3, 0.1): [1, 2, 3],
            (4, 0.1): [1, 2, 3, 0],
            (2, 0.25): [1, 3],
            (3, 0.25): [1, 3, 0],
            (4, 0.25): [1, 3, 0],
        },
    },
    {
        "dist": squareform(
            np.array(
                [
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.3],
                    [0.1, 0.0, 0.1, 0.1, 0.1, 0.1],
                    [0.2, 0.1, 0.0, 0.8, 0.5, 0.2],
                    [0.3, 0.1, 0.8, 0.0, 0.1, 0.6],
                    [0.4, 0.1, 0.5, 0.1, 0.0, 0.04],
                    [0.3, 0.1, 0.2, 0.6, 0.04, 0.0],
                ]
            )
        ),
        "weights": np.array([10, 5, 1, 7, 8, 9]),
        "results": {
            (1, 0.6): [0],
            (2, 0.6): [0],
            (1, 0.31): [0],
            (2, 0.31): [0, 4],
            (2, 0.21): [0, 5],
            (4, 0.21): [0, 5, 3],
            (2, 0.11): [0, 5],
            (3, 0.11): [0, 5, 3],
            (4, 0.05): [0, 5, 3, 1],
        },
    },
]


@pytest.mark.parametrize("data", TEST_DIV_AWARE_TOPK)
def test_diversity_aware_top_k(data):
    for k, t in data["results"].keys():
        results = diversity_aware_top_k(data["dist"], data["weights"], k=k, t=t)
        gt = data["results"][(k, t)]
        assert len(results) == len(gt) and all(results == gt), (
            f"Error with k,t={k, t}:\nresult: {results}\nexpected: {gt}"
        )
