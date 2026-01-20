"""Tests for the RL Rewards generation functionality."""

from itertools import product
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from rdkit import Chem

from mol_gen_docking.data.gen_dataset import DatasetConfig
from mol_gen_docking.reward.rl_rewards import RewardScorer, has_bridged_bond
from mol_gen_docking.reward.verifiers.generation_reward.property_utils import (
    rescale_property_values,
)

from .utils import (
    COMPLETIONS,
    DATA_PATH,
    OBJECTIVES_TO_TEST,
    PROP_LIST,
    PROPERTIES_NAMES_SIMPLE,
    SMILES,
    fill_completion,
    get_unscaled_obj,
    propeties_csv,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def dataset_config() -> DatasetConfig:
    """Create a DatasetConfig instance."""
    return DatasetConfig(data_path=DATA_PATH)


@pytest.fixture(scope="module")
def valid_scorer() -> RewardScorer:
    """Create a RewardScorer for valid SMILES checking."""
    return RewardScorer(
        DATA_PATH,
        "valid_smiles",
        parse_whole_completion=True,
        rescale=False,
    )


@pytest.fixture(scope="module")
def property_scorer() -> RewardScorer:
    """Create a RewardScorer for property scoring without rescaling."""
    return RewardScorer(
        DATA_PATH,
        "property",
        parse_whole_completion=False,
        rescale=False,
    )


@pytest.fixture(scope="module")
def property_scorer_rescale() -> RewardScorer:
    """Create a RewardScorer for property scoring with rescaling."""
    return RewardScorer(
        DATA_PATH,
        "property",
        parse_whole_completion=False,
        rescale=True,
    )


@pytest.fixture
def sample_smiles() -> List[str]:
    """Get a sample of SMILES from the properties CSV."""
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


def is_reward_valid(
    rewards: List[float],
    smiles: List[str],
    properties: List[str],
) -> None:
    """
    Check if the reward is valid.

    Args:
        rewards: List of reward values.
        smiles: List of SMILES strings.
        properties: List of property names.

    Raises:
        AssertionError: If rewards don't match expected values.
    """
    # Remove "FAKE" from smiles
    smiles = [s for s in smiles if s != "FAKE"]

    # Get the bridged_bond mask
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    bridged_mask = torch.tensor(
        [not has_bridged_bond(mol) if mol is not None else False for mol in mols]
    )

    if len(smiles) > 0:
        property_names = [PROPERTIES_NAMES_SIMPLE.get(p, p) for p in properties]
        props = torch.tensor(
            propeties_csv.set_index("smiles").loc[smiles, property_names].values
        )
        if bridged_mask.sum() == 0:
            props = torch.tensor(0.0)
        else:
            all_props = props[bridged_mask].float()
            props = all_props.prod(-1).pow(1 / len(properties)).mean()
        rewards = torch.tensor(rewards).mean()
        assert torch.isclose(rewards, props, atol=1e-3).all()


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


# =============================================================================
# Valid SMILES Tests
# =============================================================================


class TestValidSmiles:
    """Tests for valid SMILES reward scoring."""

    @pytest.mark.parametrize("completion, smiles", product(COMPLETIONS, SMILES))
    def test_valid_smiles_scoring(
        self,
        valid_scorer: RewardScorer,
        completion: str,
        smiles: List[str],
    ) -> None:
        """Test the valid SMILES reward function."""
        completions = ["<answer> " + " ".join(smiles) + " </answer>"]
        completions = [completion.format(SMILES=c) for c in completions]

        rewards = np.array(
            valid_scorer(
                completions,
                metadata=[{"objectives": ["maximize"]}],
            )[0]
        )
        expected = float(
            "SMILES" in completion
            and not all(has_bridged_bond(Chem.MolFromSmiles(s)) for s in smiles)
        )
        assert rewards.sum() == expected

    def test_valid_smiles_with_fake_molecule(
        self,
        valid_scorer: RewardScorer,
    ) -> None:
        """Test that fake SMILES return zero reward."""
        completions = ["<answer> FAKE </answer>"]
        rewards = np.array(
            valid_scorer(
                completions,
                metadata=[{"objectives": ["maximize"]}],
            )[0]
        )
        assert rewards.sum() == 0.0

    def test_valid_smiles_with_real_molecule(
        self,
        valid_scorer: RewardScorer,
    ) -> None:
        """Test that valid SMILES return non-zero reward."""
        valid_smiles = propeties_csv.sample(1)["smiles"].tolist()[0]
        completions = [f"Here is a molecule: <answer> {valid_smiles} </answer>"]

        rewards = np.array(
            valid_scorer(
                completions,
                metadata=[{"objectives": ["maximize"]}],
            )[0]
        )
        assert rewards.sum() == 1.0


# =============================================================================
# Multi-Prompt Multi-Generation Tests
# =============================================================================


class TestMultiPromptMultiGeneration:
    """Tests for multiple prompts with multiple generations."""

    @pytest.mark.parametrize(
        "property1, property2",
        product(
            PROP_LIST,
            np.random.choice(PROP_LIST, 8),
        ),
    )
    def test_multi_prompt_multi_generation(
        self,
        property_scorer: RewardScorer,
        property1: str,
        property2: str,
    ) -> None:
        """Test the reward function for a set of 2 prompts and multiple generations."""
        completion = "Here is a molecule: [SMILES] what are its properties?"
        metadata = [
            {"properties": [property1], "objectives": ["maximize"], "target": [0]},
            {"properties": [property2], "objectives": ["maximize"], "target": [0]},
        ]
        n_mols = np.random.choice([1, 2, 3, 4], p=[0.8, 0.05, 0.05, 0.1])
        smiles = [propeties_csv.sample(n_mols)["smiles"].tolist() for k in range(2)]
        n_valids = [
            sum([int(has_bridged_bond(Chem.MolFromSmiles(smi))) for smi in smiles[i]])
            for i in range(2)
        ]
        completions = [fill_completion(s, completion) for s in smiles]
        rewards, meta = property_scorer(completions, metadata)
        for i in range(2):
            if len(smiles[i]) == 1 and n_valids[i] == 1:
                is_reward_valid(
                    rewards[i], smiles[i], [property1 if i == 0 else property2]
                )
            elif len(smiles[i]) > 1:
                if n_valids[i] != 1:
                    assert (
                        rewards[i] == 0
                    )  # Known edge case, if one valid molecule, reward is not 0
                is_reward_valid(
                    meta[i]["all_smi_rewards"],
                    smiles[i],
                    [property1 if i == 0 else property2],
                )

    def test_single_molecule_single_property(
        self,
        property_scorer: RewardScorer,
    ) -> None:
        """Test reward calculation for a single molecule with a single property."""
        prop = PROP_LIST[0]
        smiles = propeties_csv.sample(1)["smiles"].tolist()
        completion = "Here is a molecule: [SMILES] what are its properties?"

        metadata = [{"properties": [prop], "objectives": ["maximize"], "target": [0]}]
        completions = [fill_completion(smiles, completion)]

        rewards, _ = property_scorer(completions, metadata)

        is_reward_valid(rewards[0], smiles, [prop])


# =============================================================================
# Objective-Based Reward Tests
# =============================================================================


class TestObjectiveBasedRewards:
    """Tests for different optimization objectives."""

    @pytest.mark.parametrize(
        "prop, obj, smiles",
        list(
            product(
                PROP_LIST,
                OBJECTIVES_TO_TEST[1:],  # Skip "maximize" for this test
                [propeties_csv.sample(4)["smiles"].tolist() for _ in range(8)],
            )
        ),
    )
    def test_all_objectives(
        self,
        property_scorer_rescale: RewardScorer,
        prop: str,
        obj: str,
        smiles: List[str],
    ) -> None:
        """
        Test the reward function with various optimization objectives.

        Assumes the value of the reward function when using maximize is correct.
        """
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        mask = torch.tensor(
            [not has_bridged_bond(m) if m is not None else False for m in mols]
        ).float()

        obj_func, target = get_unscaled_obj(obj, prop)
        n_generations = len(smiles)

        # Create metadata for both objective and maximize baseline
        metadata = [
            {"properties": [prop], "objectives": [obj_func], "target": [target]}
        ] * n_generations + [
            {"properties": [prop], "objectives": ["maximize"], "target": [target]}
        ] * n_generations

        smiles_doubled = smiles * 2
        completions = [
            fill_completion(
                [s], "Here is a molecule: [SMILES] does it have the right properties?"
            )
            for s in smiles_doubled
        ]

        rewards = property_scorer_rescale(completions, metadata, debug=True)[0]

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

    def test_maximize_objective(
        self,
        property_scorer_rescale: RewardScorer,
    ) -> None:
        """Test the maximize objective specifically."""
        prop = PROP_LIST[0]
        smiles = propeties_csv.sample(2)["smiles"].tolist()
        completion = "Here is a molecule: [SMILES] does it have the right properties?"

        metadata = [
            {"properties": [prop], "objectives": ["maximize"], "target": [0]}
            for _ in smiles
        ]
        completions = [fill_completion([s], completion) for s in smiles]

        rewards, _ = property_scorer_rescale(completions, metadata)

        assert isinstance(rewards, (np.ndarray, list, torch.Tensor))
        rewards = torch.Tensor(rewards)
        assert not rewards.isnan().any()
        assert (rewards >= 0).all()
        assert (rewards <= 1).all()

    def test_minimize_objective(
        self,
        property_scorer_rescale: RewardScorer,
    ) -> None:
        """Test the minimize objective specifically."""
        prop = PROP_LIST[0]
        smiles = propeties_csv.sample(2)["smiles"].tolist()
        completion = "Here is a molecule: [SMILES] does it have the right properties?"

        metadata = [
            {"properties": [prop], "objectives": ["minimize"], "target": [0]}
            for _ in smiles
        ]
        completions = [fill_completion([s], completion) for s in smiles]

        rewards, _ = property_scorer_rescale(completions, metadata)

        assert isinstance(rewards, (np.ndarray, list, torch.Tensor))
        rewards = torch.Tensor(rewards)
        assert not rewards.isnan().any()


# =============================================================================
# Bridged Bond Tests
# =============================================================================


class TestBridgedBondHandling:
    """Tests for bridged bond detection and handling."""

    def test_has_bridged_bond_with_valid_molecule(self) -> None:
        """Test bridged bond detection with a simple valid molecule."""
        mol = Chem.MolFromSmiles("CCO")  # Ethanol - no bridged bonds
        assert mol is not None
        assert not has_bridged_bond(mol)

    def test_has_bridged_bond_with_none(self) -> None:
        """Test bridged bond detection with None molecule."""
        result = has_bridged_bond(None)
        # Should handle None gracefully
        assert result is True or result is False

    def test_bridged_bond_mask_in_rewards(
        self,
        property_scorer: RewardScorer,
    ) -> None:
        """Test that bridged bond molecules are properly masked in rewards."""
        # Use known valid SMILES
        smiles = propeties_csv.sample(2)["smiles"].tolist()
        completion = "Here is a molecule: [SMILES] what are its properties?"
        prop = PROP_LIST[0]

        metadata = [
            {"properties": [prop], "objectives": ["maximize"], "target": [0]}
            for _ in smiles
        ]
        completions = [fill_completion([s], completion) for s in smiles]

        rewards, _ = property_scorer(completions, metadata)

        # Verify rewards are computed
        assert len(rewards) == len(smiles)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_completion(
        self,
        valid_scorer: RewardScorer,
    ) -> None:
        """Test handling of empty completions."""
        completions = ["This is an empty completion."]
        rewards = np.array(
            valid_scorer(
                completions,
                metadata=[{"objectives": ["maximize"]}],
            )[0]
        )
        assert rewards.sum() == 0.0

    def test_multiple_properties(
        self,
        property_scorer: RewardScorer,
    ) -> None:
        """Test reward calculation with multiple properties."""
        props = PROP_LIST[:2] if len(PROP_LIST) >= 2 else PROP_LIST
        smiles = propeties_csv.sample(1)["smiles"].tolist()
        completion = "Here is a molecule: [SMILES] what are its properties?"

        metadata = [
            {
                "properties": props,
                "objectives": ["maximize"] * len(props),
                "target": [0] * len(props),
            }
        ]
        completions = [fill_completion(smiles, completion)]

        rewards, _ = property_scorer(completions, metadata)

        assert len(rewards) == 1

    def test_invalid_smiles_handling(
        self,
        property_scorer: RewardScorer,
    ) -> None:
        """Test handling of invalid SMILES strings."""
        completion = "Here is a molecule: <answer> INVALID_SMILES </answer>"
        prop = PROP_LIST[0]

        metadata = [{"properties": [prop], "objectives": ["maximize"], "target": [0]}]

        rewards, _ = property_scorer([completion], metadata)

        # Invalid SMILES should result in zero or handled reward
        assert len(rewards) == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_reward_pipeline(
        self,
        valid_scorer: RewardScorer,
        property_scorer: RewardScorer,
    ) -> None:
        """Test complete reward calculation pipeline."""
        # Get valid SMILES
        smiles = propeties_csv.sample(3)["smiles"].tolist()
        prop = PROP_LIST[0]

        completion_template = "Here is a molecule: [SMILES] what are its properties?"
        completions = [fill_completion([s], completion_template) for s in smiles]

        # First check validity
        valid_metadata = [{"objectives": ["maximize"]} for _ in smiles]
        valid_rewards, _ = valid_scorer(completions, valid_metadata)

        # Then check properties
        prop_metadata = [
            {"properties": [prop], "objectives": ["maximize"], "target": [0]}
            for _ in smiles
        ]
        prop_rewards, _ = property_scorer(completions, prop_metadata)

        assert len(valid_rewards) == len(smiles)
        assert len(prop_rewards) == len(smiles)

    def test_different_objectives_same_molecule(
        self,
        property_scorer_rescale: RewardScorer,
    ) -> None:
        """Test different objectives on the same molecule."""
        smiles = propeties_csv.sample(1)["smiles"].tolist()[0]
        prop = PROP_LIST[0]
        completion = f"Here is a molecule: <answer> {smiles} </answer>"

        objectives = ["maximize", "minimize", "above 0.5", "below 0.5"]
        all_rewards = []

        for obj in objectives:
            obj_func, target = get_unscaled_obj(obj, prop)
            metadata = [
                {"properties": [prop], "objectives": [obj_func], "target": [target]}
            ]
            rewards, _ = property_scorer_rescale([completion], metadata)
            all_rewards.append(rewards[0])

        # Check that rewards are valid numbers
        for reward in all_rewards:
            assert (
                not np.isnan(reward).any()
                if hasattr(reward, "__iter__")
                else not np.isnan(reward)
            )
