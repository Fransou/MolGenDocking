"""Tests for property-based reward scoring."""

from typing import Any, List

import numpy as np
import pytest

from mol_gen_docking.reward import (
    GenerationVerifierConfigModel,
    MolecularVerifier,
    MolecularVerifierConfigModel,
    MolPropVerifierConfigModel,
)
from mol_gen_docking.utils.property_utils import rescale_property_values


@pytest.fixture(scope="module")  # type: ignore
def property_scorer(data_path: str) -> MolecularVerifier:
    """Create a RewardScorer for property scoring."""
    return MolecularVerifier(
        MolecularVerifierConfigModel(
            mol_prop_verifier_config=MolPropVerifierConfigModel()
        )
    )


@pytest.fixture(scope="module")  # type: ignore
def property_scorer_mixed(data_path: str) -> MolecularVerifier:
    """Create a RewardScorer for property scoring."""
    return MolecularVerifier(
        MolecularVerifierConfigModel(
            mol_prop_verifier_config=MolPropVerifierConfigModel(),
            generation_verifier_config=GenerationVerifierConfigModel(
                path_to_mappings=data_path
            ),
        )
    )


# =============================================================================
# Regression Tests
# =============================================================================


class TestRegression:
    """Tests for regression objective."""

    def test_regression(self, property_scorer: MolecularVerifier) -> None:
        """Test that regression rewards are properly ordered by value."""
        target: float = np.random.random()
        values = [
            v * np.sign(np.random.random() - 0.5) + target for v in [0, 0.01, 0.5, 1]
        ]
        completions: List[str] = [
            "<answer> Here is an answer: {} </answer>".format(v) for v in values
        ]

        completions += [
            "<answer> Here is an answer: {} </answer>".format(v)
            for v in [
                "0.3x10<sup>1</sup>",
                "-0.3 x 10<sup>1</sup>",
                r"1.3 \times 10^{-1}",
                r"1.2 \times 10^{-1} - 1.4 \times 10^{-1}",
                r"-1.3 \times 10^{-1}",
                "-2.1 ± 0.5",
                "2.1 ± 0.5",
            ]
        ]
        values += [
            3,
            -3,
            0.13,
            0.13,
            -0.13,
            -2.1,
            2.1,
        ]

        metadata: List[dict[str, Any]] = [
            {"objectives": ["regression"], "properties": [""], "target": [target]}
        ] * len(completions)

        rewards: List[float] = property_scorer(completions, metadata).rewards
        verif_metadata: Any = property_scorer(completions, metadata).verifier_metadatas
        assert all(v == m.extracted_answer for v, m in zip(values, verif_metadata)), (
            f"extracted: {[m.extracted_answer for m in verif_metadata]}, expected: {values}"
        )
        assert sum(rewards) > 0.0


# =============================================================================
# Classification Tests
# =============================================================================


class TestClassification:
    """Tests for classification objective."""

    def test_classification_target_one(
        self, property_scorer: MolecularVerifier
    ) -> None:
        """Test classification with target=1."""
        completions: List[str] = [
            "<answer> My answer is {} </answer>".format(v)
            for v in [1, 1, 0, 0, "bbfhdsbfsj", "Y."]
        ]
        metadata: List[dict[str, Any]] = [
            {"objectives": ["classification"], "properties": [""], "target": [1]}
        ] * 6
        rewards: List[float] = property_scorer(completions, metadata).rewards
        assert rewards == [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    def test_classification_target_zero(
        self, property_scorer: MolecularVerifier
    ) -> None:
        """Test classification with target=0."""
        completions: List[str] = [
            "<answer> My answer is {} </answer>".format(v)
            for v in [1, 1, 0, 0, "bbfhdsbfsj"]
        ]
        metadata: List[dict[str, Any]] = [
            {"objectives": ["classification"], "properties": [""], "target": [0]}
        ] * 5
        rewards: List[float] = property_scorer(completions, metadata).rewards
        assert rewards == [0.0, 0.0, 1.0, 1.0, 0.0]


# =============================================================================
# Mixed Task Tests
# =============================================================================


class TestMixedGeneration:
    """Tests for mixed classification and generation tasks."""

    def test_with_generation(self, property_scorer_mixed: MolecularVerifier) -> None:
        """Test mixed classification and generation with target=1."""
        completions: List[str] = sum(
            [
                [
                    "<answer> {} </answer>".format(v),
                    "<answer> CCC{} </answer>".format(smi),
                ]
                for v, smi in zip([1, 1, 0, 0, 0], ["C" * i for i in range(5)])
            ],
            [],
        )
        metadata: List[dict[str, Any]] = sum(
            [
                [
                    {
                        "objectives": ["classification"],
                        "properties": [""],
                        "target": [1],
                    },
                    {
                        "objectives": ["maximize"],
                        "properties": ["CalcNumRotatableBonds"],
                        "target": [1],
                    },
                ]
            ]
            * 5,
            [],
        )
        rewards = property_scorer_mixed(completions, metadata, debug=True).rewards
        assert rewards[::2] == [1.0, 1.0, 0.0, 0.0, 0.0]
        assert rewards[1::2] == [
            rescale_property_values("CalcNumRotatableBonds", i) for i in range(5)
        ]

    def test_with_generation_no_gen_conf(
        self, property_scorer: MolecularVerifier
    ) -> None:
        """Test mixed classification and generation with target=0."""
        completions: List[str] = [
            "<answer> {} </answer>".format(v) for v in [1, 1, 0, 0, 0]
        ] + [
            "<answer> CCC{} </answer>".format(smi)
            for smi in ["C" * i for i in range(5)]
        ]
        metadata: List[dict[str, Any]] = [
            {"objectives": ["classification"], "properties": [""], "target": [0]}
        ] * 5 + [
            {
                "objectives": ["maximize"],
                "properties": ["CalcNumRotatableBonds"],
                "target": [1],
            }
        ] * 5
        with pytest.raises(AssertionError):
            property_scorer(completions, metadata, debug=True)
