"""Tests for the RL Rewards generation server functionality."""

import time
from typing import Any, Dict

import numpy as np
import pytest
import ray
import requests

# =============================================================================
# Ray Initialization
# =============================================================================

if not ray.is_initialized():
    ray.init()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def ensure_server(uvicorn_server, server_url: str) -> str:
    """
    Ensure the server is available for tests.

    This fixture depends on uvicorn_server to ensure automatic server management
    when --start-server is passed. It also checks if server is reachable.

    Args:
        uvicorn_server: The server process fixture from conftest.
        server_url: The server URL fixture from conftest.

    Returns:
        The server URL if available.

    Raises:
        pytest.skip: If server is not available.
    """
    try:
        response = requests.get(f"{server_url}/liveness", timeout=5)
        if response.status_code == 200:
            return server_url
    except requests.exceptions.RequestException:
        pass

    pytest.skip("Generation reward server is not available")


# =============================================================================
# Helper Functions
# =============================================================================


@ray.remote(num_cpus=0.1)
def get_reward_async(
    server_url: str, completions: str, metadata: Dict[str, Any]
) -> requests.Response:
    """
    Remote function to get reward from the server.

    Args:
        server_url: The server URL.
        smi: SMILES string to evaluate.
        metadata: Metadata dictionary containing properties, objectives, and target.

    Returns:
        Response from the reward server.
    """
    time.sleep(np.random.random() ** 2 * 2)
    r = requests.post(
        f"{server_url}/get_reward",
        json={
            "metadata": [metadata],
            "query": [completions],
            "prompts": [""],
        },
    )
    return r


# =============================================================================
# Server Availability Tests
# =============================================================================


class TestServerAvailability:
    """Tests for server availability checks."""

    def test_server_health_endpoint(self, ensure_server: str) -> None:
        """Test that the server liveness endpoint is accessible."""
        response = requests.get(f"{ensure_server}/liveness", timeout=5)
        assert response.status_code == 200


# =============================================================================
# Valid SMILES Server Tests
# =============================================================================


class TestValidSmilesServer:
    """Tests for valid SMILES reward scoring via the server."""

    def test_valid_smiles_with_fake_molecule(
        self,
        ensure_server: str,
    ) -> None:
        """Test that fake SMILES return zero reward."""
        completions = "<answer> FAKE </answer>"
        metadata = {
            "properties": ["qed"],
            "objectives": ["maximize"],
            "target": [0.0],
        }

        response = ray.get(
            get_reward_async.remote(ensure_server, completions, metadata)
        )
        assert response.status_code == 200
        rewards = np.array(response.json()["reward"])
        assert rewards.sum() == 0.0

    def test_valid_smiles_with_real_molecule(
        self,
        ensure_server: str,
    ) -> None:
        """Test that valid SMILES return non-zero reward."""
        valid_smiles = "CCC"
        completions = f"Here is a molecule: <answer> {valid_smiles} </answer>"
        metadata = {
            "properties": ["qed"],
            "objectives": ["maximize"],
            "target": [0.0],
        }

        response = ray.get(
            get_reward_async.remote(ensure_server, completions, metadata)
        )
        assert response.status_code == 200
        print(response.json())
        rewards = np.array(response.json()["reward"])
        assert rewards.sum() > 1.0
