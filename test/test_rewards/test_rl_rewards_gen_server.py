"""Tests for the RL Rewards generation server functionality."""

import time
from itertools import product
from typing import Any, Dict, List

import numpy as np
import pytest
import ray
import requests
import torch
from rdkit import Chem

from mol_gen_docking.reward.rl_rewards import has_bridged_bond
from mol_gen_docking.reward.verifiers.generation_reward.property_utils import (
    rescale_property_values,
)

from .utils import (
    OBJECTIVES_TO_TEST,
    PROP_LIST,
    get_unscaled_obj,
    propeties_csv,
)

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


@pytest.fixture
def sample_smiles_large() -> List[str]:
    """Get a large sample of SMILES from the properties CSV."""
    return propeties_csv.sample(16)["smiles"].tolist()


@pytest.fixture
def sample_smiles_small() -> List[str]:
    """Get a small sample of SMILES from the properties CSV."""
    return propeties_csv.sample(4)["smiles"].tolist()


@pytest.fixture
def sample_metadata() -> Dict[str, Any]:
    """Create sample metadata for testing."""
    return {
        "properties": [PROP_LIST[0]],
        "objectives": ["maximize"],
        "target": [0],
    }


# =============================================================================
# Helper Functions
# =============================================================================


def get_reward_sync(
    server_url: str, smi: str, metadata: Dict[str, Any]
) -> requests.Response:
    """
    Synchronous function to get reward from the server.

    Args:
        server_url: The server URL.
        smi: SMILES string to evaluate.
        metadata: Metadata dictionary containing properties, objectives, and targets.

    Returns:
        Response from the reward server.
    """
    r = requests.post(
        f"{server_url}/get_reward",
        json={
            "metadata": [metadata],
            "query": [f"<answer> {smi} </answer>"],
            "prompts": [""],
        },
    )
    return r


@ray.remote(num_cpus=0.1)
def get_reward_async(
    server_url: str, smi: str, metadata: Dict[str, Any]
) -> requests.Response:
    """
    Remote function to get reward from the server.

    Args:
        server_url: The server URL.
        smi: SMILES string to evaluate.
        metadata: Metadata dictionary containing properties, objectives, and targets.

    Returns:
        Response from the reward server.
    """
    time.sleep(np.random.random() ** 2 * 2)
    r = requests.post(
        f"{server_url}/get_reward",
        json={
            "metadata": [metadata],
            "query": [f"<answer> {smi} </answer>"],
            "prompts": [""],
        },
    )
    return r


def compute_expected_reward(
    objective: str,
    target: float,
    rewards_max: torch.Tensor,
    mask: torch.Tensor,
    prop: str,
) -> torch.Tensor:
    """
    Compute expected reward based on objective type.

    Args:
        objective: The optimization objective (maximize, minimize, below, above, equal).
        target: Target value for threshold objectives.
        rewards_max: Raw reward values from maximize objective.
        mask: Mask for valid molecules (no bridged bonds).
        prop: Property name for rescaling.

    Returns:
        Expected reward tensor.
    """
    if objective == "maximize":
        val = rewards_max
    elif objective == "minimize":
        val = 1 - rewards_max
    else:
        target = rescale_property_values(prop, target, False)
        if objective == "below":
            val = (rewards_max <= target).float()
        elif objective == "above":
            val = (rewards_max >= target).float()
        elif objective == "equal":
            val = torch.tensor(np.clip(1 - 100 * (rewards_max - target) ** 2, 0, 1))
        else:
            raise ValueError(f"Unknown objective: {objective}")
    return val * mask


def create_molecule_mask(smiles: List[str]) -> torch.Tensor:
    """
    Create a mask for valid molecules without bridged bonds.

    Args:
        smiles: List of SMILES strings.

    Returns:
        Boolean tensor mask.
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return torch.tensor(
        [not has_bridged_bond(m) if m is not None else False for m in mols]
    ).float()


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
# Standard Prompt Tests
# =============================================================================


class TestStandardPrompts:
    """Tests for standard prompt reward calculations via server."""

    @pytest.mark.parametrize(
        "prop, obj, smiles",
        [
            list(prod) + [smi]
            for smi, prod in zip(
                [
                    propeties_csv.sample(2)["smiles"].tolist()
                    for _ in range(len(PROP_LIST) * len(OBJECTIVES_TO_TEST[1:]))
                ],
                list(
                    product(
                        PROP_LIST,
                        OBJECTIVES_TO_TEST[1:],  # Skip "maximize" for this test
                    )
                ),
            )
        ],
    )
    def test_std_prompts_all_objectives(
        self,
        ensure_server: str,
        prop: str,
        obj: str,
        smiles: List[str],
    ) -> None:
        """
        Test the reward function with the optimization of one property.

        Assumes the value of the reward function when using maximize is correct.
        """
        mask = create_molecule_mask(smiles)

        obj_func, target = get_unscaled_obj(obj, prop)
        n_generations = len(smiles)

        # Create metadata for both objective and maximize baseline
        metadata = [
            {"properties": [prop], "objectives": [obj_func], "target": [target]}
        ] * n_generations + [
            {"properties": [prop], "objectives": ["maximize"], "target": [target]}
        ] * n_generations

        smiles_doubled = smiles * 2

        # Get rewards from server
        rewards = ray.get(
            [
                get_reward_async.remote(ensure_server, s, m)
                for s, m in zip(smiles_doubled, metadata)
            ]
        )
        rewards = [r.json()["reward"] for r in rewards]

        assert isinstance(rewards, (np.ndarray, list, torch.Tensor))
        rewards = torch.Tensor(rewards)
        assert not rewards.isnan().any()

        rewards_prop = rewards[:n_generations]
        rewards_max = rewards[n_generations:]

        objective = obj_func.split()[0]
        expected_val = compute_expected_reward(
            objective, target, rewards_max, mask, prop
        )

        assert torch.isclose(rewards_prop, expected_val, atol=1e-4).all()


# =============================================================================
# Feedback Prompt Tests
# =============================================================================


class TestFeedbackPrompts:
    """Tests for feedback prompt reward calculations via server."""

    @pytest.mark.parametrize(
        "prop, obj, smiles",
        list(
            product(
                PROP_LIST[:3],
                OBJECTIVES_TO_TEST[1:],  # Skip "maximize" for this test
                [propeties_csv.sample(4)["smiles"].tolist() for _ in range(2)],
            )
        ),
    )
    def test_feedback_prompts_all_objectives(
        self,
        ensure_server: str,
        prop: str,
        obj: str,
        smiles: List[str],
    ) -> None:
        """
        Test the reward function with feedback prompts.

        Assumes the value of the reward function when using maximize is correct.
        """
        mask = create_molecule_mask(smiles)

        obj_func, target = get_unscaled_obj(obj, prop)
        n_generations = len(smiles)

        # Create metadata for objective and maximize baseline
        metadata = [
            {"properties": [prop], "objectives": [obj_func], "target": [target]},
            {"properties": [prop], "objectives": ["maximize"], "target": [target]},
        ]

        # Join smiles for feedback format
        new_smiles = [" ".join(smiles)]
        smiles_doubled = new_smiles * 2

        # Get rewards from server
        rewards = ray.get(
            [
                get_reward_async.remote(ensure_server, s, m)
                for s, m in zip(smiles_doubled, metadata)
            ]
        )
        print([r.json()["meta"]["verifier_metadata_output"] for r in rewards])
        rewards = [
            r.json()["meta"]["verifier_metadata_output"][i]["all_smi_rewards"]
            for r in rewards
            for i in range(len(r.json()["meta"]["verifier_metadata_output"]))
        ]

        assert isinstance(rewards, (np.ndarray, list, torch.Tensor))
        rewards = torch.Tensor(rewards).view(-1)
        print(rewards)
        assert not rewards.isnan().any()

        rewards_prop = rewards[:n_generations]
        rewards_max = rewards[n_generations:]
        print(rewards_prop, rewards_max)
        objective = obj_func.split()[0]
        expected_val = compute_expected_reward(
            objective, target, rewards_max, mask, prop
        )
        print(rewards_prop, expected_val)
        assert torch.isclose(rewards_prop, expected_val, atol=1e-4).all()

    def test_feedback_multiple_smiles(self, ensure_server: str) -> None:
        """Test feedback with multiple SMILES in one request."""
        smiles = propeties_csv.sample(3)["smiles"].tolist()
        prop = PROP_LIST[0]

        metadata = {"properties": [prop], "objectives": ["maximize"], "target": [0]}
        joined_smiles = " ".join(smiles)

        response = get_reward_sync(ensure_server, joined_smiles, metadata)
        result = response.json()

        assert "meta" in result
        assert "verifier_metadata_output" in result["meta"]
        assert "all_smi_rewards" in result["meta"]["verifier_metadata_output"]


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_smiles_handling(self, ensure_server: str) -> None:
        """Test handling of invalid SMILES strings."""
        prop = PROP_LIST[0]
        metadata = {"properties": [prop], "objectives": ["maximize"], "target": [0]}

        response = get_reward_sync(ensure_server, "INVALID_SMILES", metadata)
        result = response.json()

        # Should handle invalid SMILES gracefully
        assert "reward" in result

    def test_empty_smiles_handling(self, ensure_server: str) -> None:
        """Test handling of empty SMILES string."""
        prop = PROP_LIST[0]
        metadata = {"properties": [prop], "objectives": ["maximize"], "target": [0]}

        response = get_reward_sync(ensure_server, "", metadata)
        result = response.json()

        # Should handle empty SMILES gracefully
        assert "reward" in result


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_reward_pipeline(self, ensure_server: str) -> None:
        """Test complete reward calculation pipeline via server."""
        smiles = propeties_csv.sample(5)["smiles"].tolist()
        prop = PROP_LIST[0]

        # Test with different objectives
        objectives = ["maximize", "minimize"]
        all_rewards = {}

        for obj in objectives:
            metadata = [
                {"properties": [prop], "objectives": [obj], "target": [0]}
                for _ in smiles
            ]

            rewards = ray.get(
                [
                    get_reward_async.remote(ensure_server, s, m)
                    for s, m in zip(smiles, metadata)
                ]
            )
            all_rewards[obj] = torch.Tensor([r.json()["reward"] for r in rewards])

        # Verify rewards are computed for all objectives
        for obj, rewards in all_rewards.items():
            assert len(rewards) == len(smiles)
            assert not rewards.isnan().any()

    def test_different_objectives_same_molecule(self, ensure_server: str) -> None:
        """Test different objectives on the same molecule."""
        smiles = propeties_csv.sample(1)["smiles"].tolist()[0]
        prop = PROP_LIST[0]

        objectives = ["maximize", "minimize", "above 0.5", "below 0.5"]
        all_rewards = []

        for obj in objectives:
            obj_func, target = get_unscaled_obj(obj, prop)
            metadata = {
                "properties": [prop],
                "objectives": [obj_func],
                "target": [target],
            }
            response = get_reward_sync(ensure_server, smiles, metadata)
            all_rewards.append(response.json()["reward"])

        # Check that rewards are valid numbers
        for reward in all_rewards:
            assert not np.isnan(reward)

    def test_multiple_properties(self, ensure_server: str) -> None:
        """Test reward calculation with multiple properties."""
        smiles = propeties_csv.sample(1)["smiles"].tolist()[0]
        props = PROP_LIST[:2] if len(PROP_LIST) >= 2 else PROP_LIST

        metadata = {
            "properties": props,
            "objectives": ["maximize"] * len(props),
            "target": [0] * len(props),
        }

        response = get_reward_sync(ensure_server, smiles, metadata)
        result = response.json()

        assert "reward" in result
        assert not np.isnan(result["reward"])
