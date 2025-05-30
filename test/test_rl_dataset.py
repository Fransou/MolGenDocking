import json

import pytest
from itertools import product

from mol_gen_docking.data.rl_dataset import MolGenerationInstructionsDataset
from mol_gen_docking.reward.rl_rewards import RewardScorer

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


@pytest.mark.parametrize(
    "path",
    [
        "data/mol_orz/train_prompts.json",
        "data/mol_orz/eval_data/eval_prompts.json",
    ],
)
def test_saved_train_dataset(path):
    prompt_template_jinja = """\
    {{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
    The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}}
    Assistant: <think>\
    """
    prompt_instruction_template_jinja = """\
    You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>.
    This is the problem:
    {{prompt}}
    """

    with open(path) as f:
        data = json.load(f)

    scorer = RewardScorer(
        "properties",
        parse_whole_completion=True,
        oracle_kwargs=dict(
            ncpu=1,
            exhaustiveness=1,
        ),
    )

    for dialogue in data:
        if isinstance(dialogue, list):
            prompt = dialogue[0]["value"]
        else:
            prompt = dialogue["prompt"][0]["value"]
        assert isinstance(prompt, str)
        completion = "A molecule: "
        scorer([prompt], [completion])
        scorer([prompt_template_jinja.replace("{{prompt}}", prompt)], [completion])
        scorer(
            [prompt_instruction_template_jinja.replace("{{prompt}}", prompt)],
            [completion],
        )
