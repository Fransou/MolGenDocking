import time as time

import numpy as np
import pytest
import requests
import torch

from .utils import DOCKING_PROP_LIST, SKIP_DOCKING_TEST, propeties_csv


@pytest.mark.skipif(SKIP_DOCKING_TEST, reason="No docking software installed")
@pytest.mark.parametrize("target", DOCKING_PROP_LIST[:16])
def test_docking(target, n_generations=16):
    """Test the reward function runs for vina targets."""
    # Launch the reward server
    port = "5001"

    smiles = [
        f"<answer> {smi} </answer>"
        for smi in propeties_csv.iloc[:n_generations]["smiles"].tolist()
    ]
    metadata = [
        {"properties": [target], "objectives": ["maximize"], "target": [0]}
        for k in range(n_generations)
    ]
    response = requests.post(
        f"http://0.0.0.0:{port}/prepare_receptor",
        json={"metadata": metadata},
    )
    assert response.status_code == 200, response.text
    print(response.json())
    # Request Server
    response = requests.post(
        f"http://0.0.0.0:{port}/get_reward",
        json={"query": smiles, "metadata": metadata},
    )

    assert response.status_code == 200, response.text
    rewards = response.json()["rewards"]

    assert isinstance(rewards, (np.ndarray, list, torch.Tensor))
    rewards = torch.Tensor(rewards)
    assert not rewards.isnan().any()


response = requests.post(
    "http://0.0.0.0:5001/prepare_receptor",
    json={
        "metadata": [
            {
                "properties": ["sample_720217_model_0"],
                "objectives": ["maximize"],
                "target": [0],
            }
        ]
    },
)
