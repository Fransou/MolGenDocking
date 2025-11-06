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
DATASET: list[Sample] = read_jsonl(
    Path(os.path.join(os.path.dirname(current_path), "data", "reactions_test.jsonl"))
)


@pytest.mark.parametrize("line", DATASET[26:])
def test_reaction(line):
    metadata = line.conversations[0].meta
    target = " + ".join(metadata["target"])
    if metadata["objectives"][0].startswith("full_path"):
        target = metadata["full_reaction"]

    answers = [target, "dsad"]
    completions = ["<answer>\n {} \n</answer>".format(v) for v in answers]

    rewards = property_scorer(completions, [metadata] * len(answers), use_pbar=False)
    assert rewards == [1.0 * float(not metadata["impossible"]), 0.0]
