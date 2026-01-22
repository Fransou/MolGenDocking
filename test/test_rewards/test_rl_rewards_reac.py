"""Tests for reaction-based reward scoring."""

import json
import os
from pathlib import Path

import pytest

from mol_gen_docking.data.pydantic_dataset import read_jsonl
from mol_gen_docking.reward.rl_rewards import RewardScorer

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def property_scorer(data_path):
    """Create a RewardScorer for property scoring without rescaling."""
    return RewardScorer(data_path, "property", rescale=False)


@pytest.fixture(scope="session")
def property_scorer_valid(data_path):
    """Create a RewardScorer for valid SMILES scoring without rescaling."""
    return RewardScorer(data_path, "valid_smiles", rescale=False)


@pytest.fixture(scope="session")
def dataset_reac():
    """Load the reaction test dataset."""
    current_path = os.path.dirname(os.path.abspath(__file__))
    return read_jsonl(
        Path(
            os.path.join(os.path.dirname(current_path), "data", "reactions_test.jsonl")
        )
    )


@pytest.fixture(scope="session")
def dataset_analog():
    """Load the analog test dataset."""
    current_path = os.path.dirname(os.path.abspath(__file__))
    return read_jsonl(
        Path(os.path.join(os.path.dirname(current_path), "data", "analog_test.jsonl"))
    )


@pytest.fixture(scope="session")
def refs_comp_analog():
    """Load the analog completion examples reference."""
    current_path = os.path.dirname(os.path.abspath(__file__))
    with open(
        os.path.join(
            os.path.dirname(current_path), "data", "analog_test_compl_example.jsonl"
        )
    ) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def add_sy_ex():
    """Load the additional synthesis full path examples."""
    current_path = os.path.dirname(os.path.abspath(__file__))
    with open(
        os.path.join(
            os.path.dirname(current_path),
            "data",
            "reaction_full_sythn_test_examples.json",
        )
    ) as f:
        return json.load(f)


# =============================================================================
# Reaction Tests
# =============================================================================


class TestReaction:
    """Tests for reaction reward scoring."""

    def test_reaction(
        self, dataset_reac, property_scorer, property_scorer_valid, properties_csv
    ):
        """Test reaction reward scoring for different objective types."""
        for line in dataset_reac:
            metadata = line.conversations[0].meta
            target = " + ".join(metadata["target"])

            if metadata["objectives"][0].startswith("full_path"):
                target = metadata["full_reaction"]
                fake0 = target.split(" + ")
                fake0[0] = properties_csv.smiles.sample(1).values[0]
                fake0 = " + ".join(fake0)

                fake1 = target.split(" -> ")
                fake1_first_p = fake1[1].split("\n")
                fake1_first_p[0] = properties_csv.smiles.sample(1).values[0]
                fake1 = " -> ".join(fake1_first_p)
                answers = [target, fake0, fake1] + ["impossible"]
            elif metadata["objectives"][0] in [
                "final_product",
                "reactant",
                "all_reactants",
                "all_reactants_bb_ref",
            ]:
                fake = [properties_csv.smiles.sample(1).values[0]]
                answers = (
                    [target]
                    + fake
                    + properties_csv.smiles.sample(3).tolist()
                    + [" and ".join(properties_csv.smiles.sample(3).tolist())]
                    + ["impossible"]
                )
            elif metadata["objectives"][0] in ["smarts"]:
                fakes = [
                    property_scorer.reaction_verifier.rxn_matrix._reactions[0].smarts,
                    property_scorer.reaction_verifier.rxn_matrix._reactions[10].smarts,
                    "[#6:1]-[N:5]=[N+:6]=[N-:7].[#6:2]-[C:3]>>[#6:2][cH0+0:3]1[cH1+0:4][nH0+0:5][nH0+0:6][nH0+0:7]1",
                ]
                answers = (
                    [target]
                    + fakes
                    + ["impossible", target.replace("O", "n"), "dfhdshjkh"]
                )
            completions = ["<answer>\n {} \n</answer>".format(v) for v in answers]

            rewards = property_scorer(completions, [metadata] * len(answers))[0]
            property_scorer_valid(completions, [metadata] * len(answers))[0]
            assert (rewards[0] == 1) or metadata.get("impossible", False)


# =============================================================================
# Additional Synthesis Tests
# =============================================================================


class TestAdditionalSynthesis:
    """Tests for additional synthesis full path examples."""

    def test_additional_synth_full_path(
        self, add_sy_ex, property_scorer, property_scorer_valid
    ):
        """Test additional synthesis examples."""
        for in_out in add_sy_ex:
            metadata = in_out["metadata"]
            completions = [in_out["output"]]
            property_scorer(completions, [metadata])[0]
            property_scorer_valid(completions, [metadata])[0]


# =============================================================================
# Reaction Analog Tests
# =============================================================================


class TestReactionAnalog:
    """Tests for reaction analog examples."""

    def test_reaction_analog(self, dataset_analog, refs_comp_analog, property_scorer):
        """Test reaction analog reward scoring."""
        for line, examples in zip(dataset_analog, refs_comp_analog):
            metadata = line.conversations[0].meta

            completions = ["<answer>\n {} \n</answer>".format(v[0]) for v in examples]

            rewards = property_scorer(completions, [metadata] * len(completions))[0]

            assert rewards == [v[1] for v in examples]
