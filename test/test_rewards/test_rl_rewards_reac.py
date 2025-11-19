import json
import os
from pathlib import Path

import pytest
import ray

from mol_gen_docking.data.pydantic_dataset import Sample, read_jsonl
from mol_gen_docking.reward.rl_rewards import RewardScorer

from .utils import (
    DATA_PATH,
)

if not ray.is_initialized():
    ray.init(num_cpus=16)


property_scorer = RewardScorer(DATA_PATH, "property", rescale=False)

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


@pytest.mark.parametrize("line", DATASET_REAC)
def test_reaction(line):
    metadata = line.conversations[0].meta
    target = " + ".join(metadata["target"])
    if metadata["objectives"][0].startswith("full_path"):
        target = metadata["full_reaction"]

    answers = [target, "dsad"]
    completions = ["<answer>\n {} \n</answer>".format(v) for v in answers]

    rewards = property_scorer(completions, [metadata] * len(answers), use_pbar=False)
    assert rewards == [1.0 * float(not metadata["impossible"]), 0.0]


@pytest.mark.parametrize("line, examples", zip(DATASET_ANALOG, REFS_COMP_ANALOG))
def test_reaction_analog(line, examples):
    metadata = line.conversations[0].meta

    completions = ["<answer>\n {} \n</answer>".format(v[0]) for v in examples]

    rewards = property_scorer(
        completions, [metadata] * len(completions), use_pbar=False
    )

    assert rewards == [v[1] for v in examples]
