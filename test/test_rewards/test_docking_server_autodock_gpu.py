import time as time

import numpy as np
import pytest
import requests
import torch

from .utils import DOCKING_PROP_LIST, propeties_csv


@pytest.mark.parametrize("target", DOCKING_PROP_LIST[:16])
def test_docking(target, has_gpu, n_generations=128):
    """Test the reward function runs for vina targets."""
    # Launch the reward server
    if not has_gpu:
        pytest.skip("Skipping test for gpu docking")
    port = "5001"

    smiles = [
        f"<answer> {smi} </answer>"
        for smi in propeties_csv.iloc[:n_generations]["smiles"].tolist()
    ]
    metadata = [
        {
            "properties": [target, "CalcPhi"],
            "objectives": ["maximize"] * 2,
            "target": [0] * 2,
        }
        for k in range(n_generations)
    ]
    response = requests.post(
        f"http://0.0.0.0:{port}/prepare_receptor",
        json={"metadata": metadata},
    )
    assert response.status_code == 200, response.text
    time.sleep(1)

    # Request Server
    response = requests.post(
        f"http://0.0.0.0:{port}/get_reward",
        json={"query": smiles, "metadata": metadata},
    )

    assert response.status_code == 200, response.text
    rewards = response.json()["reward_list"]

    assert isinstance(rewards, (np.ndarray, list, torch.Tensor))
    rewards = torch.Tensor(rewards)
    assert not rewards.isnan().any()
    assert not (rewards == 0).all()
