import json
import os
from pathlib import Path

import pytest

from mol_gen_docking.data.pydantic_dataset import Sample, read_jsonl
from mol_gen_docking.reward.rl_rewards import RewardScorer

from .utils import DATA_PATH, propeties_csv

property_scorer = RewardScorer(DATA_PATH, "property", rescale=False)
property_scorer_valid = RewardScorer(DATA_PATH, "valid_smiles", rescale=False)

current_path = os.path.dirname(os.path.abspath(__file__))
DATASET_REAC: list[Sample] = read_jsonl(
    Path(os.path.join(os.path.dirname(current_path), "data", "reactions_test.jsonl"))
)
DATASET_ANALOG: list[Sample] = read_jsonl(
    Path(os.path.join(os.path.dirname(current_path), "data", "analog_test.jsonl"))
)
with open(
    os.path.join(
        os.path.dirname(current_path), "data", "analog_test_compl_example.jsonl"
    )
) as f:
    REFS_COMP_ANALOG: list[list[str, float]] = json.load(f)


with open(
    os.path.join(
        os.path.dirname(current_path), "data", "reaction_full_sythn_test_examples.json"
    )
) as f:
    ADD_SY_EX: list[dict[str, str]] = json.load(f)


@pytest.mark.parametrize("line", DATASET_REAC)
def test_reaction(line):
    metadata = line.conversations[0].meta
    target = " + ".join(metadata["target"])

    if metadata["objectives"][0].startswith("full_path"):
        target = metadata["full_reaction"]
        fake0 = target.split(" + ")
        fake0[0] = propeties_csv.smiles.sample(1).values[0]
        fake0 = " + ".join(fake0)

        fake1 = target.split(" -> ")
        fake1_first_p = fake1[1].split("\n")
        fake1_first_p[0] = propeties_csv.smiles.sample(1).values[0]
        fake1 = " -> ".join(fake1_first_p)
        answers = [target, fake0, fake1] + ["impossible"]
    elif metadata["objectives"][0] in [
        "final_product",
        "reactant",
        "all_reactants",
        "all_reactants_bb_ref",
    ]:
        fake = [propeties_csv.smiles.sample(1).values[0]]
        answers = (
            [target]
            + fake
            + propeties_csv.smiles.sample(3).tolist()
            + [" and ".join(propeties_csv.smiles.sample(3).tolist())]
            + ["impossible"]
        )
    elif metadata["objectives"][0] in ["smarts"]:
        fakes = [
            property_scorer.reaction_verifier.rxn_matrix._reactions[0].smarts,
            property_scorer.reaction_verifier.rxn_matrix._reactions[10].smarts,
            "[#6:1]-[N:5]=[N+:6]=[N-:7].[#6:2]-[C:3]>>[#6:2][cH0+0:3]1[cH1+0:4][nH0+0:5][nH0+0:6][nH0+0:7]1",
        ]
        answers = (
            [target] + fakes + ["impossible", target.replace("O", "n"), "dfhdshjkh"]
        )
    completions = ["<answer>\n {} \n</answer>".format(v) for v in answers]

    rewards = property_scorer(completions, [metadata] * len(answers))[0]
    property_scorer_valid(completions, [metadata] * len(answers))[0]
    assert (rewards[0] == 1) or metadata["impossible"]


@pytest.mark.parametrize("in_out", ADD_SY_EX)
def test_additional_synth_full_path(in_out):
    metadata = in_out["metadata"]
    completions = [in_out["output"]]
    property_scorer(completions, [metadata])[0]
    property_scorer_valid(completions, [metadata])[0]


@pytest.mark.parametrize("line, examples", zip(DATASET_ANALOG, REFS_COMP_ANALOG))
def test_reaction_analog(line, examples):
    metadata = line.conversations[0].meta

    completions = ["<answer>\n {} \n</answer>".format(v[0]) for v in examples]

    rewards = property_scorer(completions, [metadata] * len(completions))[0]

    assert rewards == [v[1] for v in examples]
