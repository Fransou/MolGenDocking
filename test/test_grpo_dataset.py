import os
import json

import pytest
from itertools import product

from mol_gen_docking.data.grpo_dataset import MolGenerationInstructionsDataset
from mol_gen_docking.reward.grpo_rewards import RewardScorer

from .utils import PROP_LIST, OBJECTIVES_TO_TEST


@pytest.mark.parametrize("props, obj", list(product(PROP_LIST, OBJECTIVES_TO_TEST)))
def test_fill_prompt(props, obj):
    """Tests if the prompt is generated correctly, i.e it can correctly be parsed."""
    dataset = MolGenerationInstructionsDataset()
    prompt = dataset.fill_prompt([props], [obj])
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    scorer = RewardScorer("properties")
    parsed = scorer.get_mol_props_from_prompt([prompt])[0]
    assert props in parsed
    assert parsed[props][0] == obj.split()[0]
    value = float(parsed[props][1])
    assert value == float(obj.split()[1] if len(obj.split()) > 1 else 0)


@pytest.mark.parametrize(
    "n, max_props",
    list(
        product(
            [i + 1 for i in range(5)],
            [1, 2, 3],
        )
    ),
)
def test_generate_with_rule(n, max_props):
    """Tests if the generation with rule is correct."""
    dataset = MolGenerationInstructionsDataset(max_n_props=max_props)
    dataset_chat = MolGenerationInstructionsDataset(max_n_props=max_props)
    d1 = dataset.generate_prompt_json(n, "chat_format")
    d2 = dataset_chat(n, "orz")
    assert len(d1) == len(d2)


def test_saved_train_dataset():
    path = os.path.join("data/mol_orz/train_prompts.json")
    with open(path) as f:
        data = json.load(f)

    scorer = RewardScorer("properties", parse_whole_completion=True)

    for dialogue in data:
        prompt = dialogue[0]["value"]
        assert isinstance(prompt, str)
        completion = "A molecule: CCCC"
        score = scorer([prompt], [completion])[0]
        assert isinstance(score, float)
