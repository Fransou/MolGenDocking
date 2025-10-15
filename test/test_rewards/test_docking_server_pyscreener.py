import subprocess as sp

import numpy as np
import pytest
import requests
import torch

from .utils import (
    DATA_PATH,
    DOCKING_PROP_LIST,
    propeties_csv,
)


@pytest.mark.parametrize("target", DOCKING_PROP_LIST[:3])
def test_properties_single_prompt_vina_reward(target, n_generations=128):
    """Test the reward function runs for vina targets."""
    # Launch the reward server
    port = "5001"
    command = f"python mol_gen_docking/fast_api_reward_server.py --data-path {DATA_PATH} --port {port} --host 0.0.0.0"
    command += " --scorer-ncpus 8 --docking-oracle pyscreener --scorer-exhaustivness 8"
    process = sp.Popen(command, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    try:
        smiles = [
            f"<answer> {smi} </answer>"
            for smi in propeties_csv.iloc[:n_generations]["smiles"].tolist()
        ]

        metadata = [
            {"properties": [target], "objectives": ["maximize"], "target": [0]}
            for k in range(n_generations)
        ]
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
        # Kill the server
        process.terminate()

    except Exception as e:
        # Kill the server
        process.terminate()
        raise e
