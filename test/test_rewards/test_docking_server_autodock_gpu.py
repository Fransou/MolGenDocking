import time as time

import numpy as np
import pytest
import requests
import torch

from .utils import DOCKING_PROP_LIST, fill_df_time, propeties_csv


@pytest.mark.parametrize("target", DOCKING_PROP_LIST[:16] * 3)
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
    t_pre = time.time()
    response = requests.post(
        f"http://0.0.0.0:{port}/prepare_receptor",
        json={"metadata": metadata},
    )
    print(response.json())
    # Request Server
    t0 = time.time()
    response = requests.post(
        f"http://0.0.0.0:{port}/get_reward",
        json={"query": smiles, "metadata": metadata},
    )

    assert response.status_code == 200, response.text
    rewards = response.json()["rewards"]
    t1 = time.time()

    assert isinstance(rewards, (np.ndarray, list, torch.Tensor))
    rewards = torch.Tensor(rewards)
    assert not rewards.isnan().any()

    fill_df_time(
        target,
        n_generations,
        t0=t0,
        t1=t1,
        method="autodock_gpu",
        server=True,
        scores=rewards.mean().item(),
        t_pre=t_pre,
    )
