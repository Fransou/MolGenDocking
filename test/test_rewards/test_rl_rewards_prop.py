"""Tests for property-based reward scoring."""

import numpy as np
import pytest

from mol_gen_docking.reward.molecular_verifier import MolecularVerifier


@pytest.fixture(scope="module")
def property_scorer(data_path):
    """Create a RewardScorer for property scoring without rescaling."""
    return MolecularVerifier(data_path, "property", rescale=False)


# =============================================================================
# Regression Tests
# =============================================================================


class TestRegression:
    """Tests for regression objective."""

    def test_regression(self, property_scorer):
        """Test that regression rewards are properly ordered by value."""
        target = np.random.random()
        completions = [
            "<answer> Here is an answer: {} </answer>".format(
                v * np.sign(np.random.random() - 0.5) + target
            )
            for v in [0, 0.01, 0.5, 1]
        ] + ["ksdjgf"]
        metadata = [
            {"objectives": ["regression"], "properties": [""], "target": [target]}
        ] * 5
        rewards = property_scorer(completions, metadata)[0]
        assert sorted(rewards)[::-1] == rewards
        assert sum(rewards) > 0.0


# =============================================================================
# Classification Tests
# =============================================================================


class TestClassification:
    """Tests for classification objective."""

    def test_classification_target_one(self, property_scorer):
        """Test classification with target=1."""
        completions = [
            "<answer> My answer is {} </answer>".format(v)
            for v in [1, 1, 0, 0, "bbfhdsbfsj"]
        ]
        metadata = [
            {"objectives": ["classification"], "properties": [""], "target": [1]}
        ] * 5
        rewards = property_scorer(completions, metadata)[0]
        assert rewards == [1.0, 1.0, 0.0, 0.0, 0.0]

    def test_classification_target_zero(self, property_scorer):
        """Test classification with target=0."""
        completions = [
            "<answer> My answer is {} </answer>".format(v)
            for v in [1, 1, 0, 0, "bbfhdsbfsj"]
        ]
        metadata = [
            {"objectives": ["classification"], "properties": [""], "target": [0]}
        ] * 5
        rewards = property_scorer(completions, metadata)[0]
        assert rewards == [0.0, 0.0, 1.0, 1.0, 0.0]


# =============================================================================
# Mixed Task Tests
# =============================================================================


class TestMixedGeneration:
    """Tests for mixed classification and generation tasks."""

    def test_with_generation_target_one(self, property_scorer):
        """Test mixed classification and generation with target=1."""
        completions = ["<answer> {} </answer>".format(v) for v in [1, 1, 0, 0, 0]] + [
            "<answer> CCC{} </answer>".format(smi)
            for smi in ["C" * i for i in range(5)]
        ]
        metadata = [
            {"objectives": ["classification"], "properties": [""], "target": [1]}
        ] * 5 + [
            {
                "objectives": ["maximize"],
                "properties": ["CalcNumRotatableBonds"],
                "target": [1],
            }
        ] * 5
        rewards = property_scorer(completions, metadata, debug=True)[0]
        assert rewards == [1.0, 1.0, 0.0, 0.0, 0.0] + [0, 1, 2, 3, 4]

    def test_with_generation_target_zero(self, property_scorer):
        """Test mixed classification and generation with target=0."""
        completions = ["<answer> {} </answer>".format(v) for v in [1, 1, 0, 0, 0]] + [
            "<answer> CCC{} </answer>".format(smi)
            for smi in ["C" * i for i in range(5)]
        ]
        metadata = [
            {"objectives": ["classification"], "properties": [""], "target": [0]}
        ] * 5 + [
            {
                "objectives": ["maximize"],
                "properties": ["CalcNumRotatableBonds"],
                "target": [1],
            }
        ] * 5
        rewards = property_scorer(completions, metadata)[0]
        assert rewards == [0.0, 0.0, 1.0, 1.0, 1.0] + [0, 1, 2, 3, 4]
