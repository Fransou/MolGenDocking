import os
from itertools import chain, product

import numpy as np
import pytest
from datasets import load_from_disk

from mol_gen_docking.data.rl_dataset import (
    DatasetConfig,
    MolGenerationInstructionsDataset,
)
from mol_gen_docking.reward.rl_rewards import RewardScorer

from .utils import DATA_PATH, OBJECTIVES_TO_TEST, PROP_LIST

cfg = DatasetConfig(data_path=DATA_PATH, vina=True, split_docking=[0.3, 0.3, 0.4])
templates = dict(
    none="{{prompt}}",
    prompt_template_jinja="""\
{{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}}
Assistant: <think>\
""",
    prompt_instruction_template_jinja="""\
You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>.
This is the problem:
{{prompt}}
""",
)


@pytest.fixture(scope="module", params=[True, False])
def build_metada_pocket(request):
    if not request.param:

        def wrapped_fn(props):
            return {}

    def wrapped_fn(props):
        out = {}
        for p in props:
            out[p] = {
                "number of alpha spheres": 10,
                "mean alpha-sphere radius": 0.561126,
                "mean alpha-sphere solvent acc.": 1.156,
                "mean b-factor of pocket residues": 1156.16546,
                "hydrophobicity score": 0.2,
                "polarity score": 0.1,
                "amino acid based volume score": 0.1,
                "pocket volume (monte carlo)": 0.1,
                "charge score": 0.1,
                "local hydrophobic density score": 0.1,
                "number of apolar alpha sphere": 1564614687684,
                "proportion of apolar alpha sphere": 0.1,
            }
        return out

    return wrapped_fn


@pytest.mark.parametrize("props, obj", list(product(PROP_LIST, OBJECTIVES_TO_TEST)))
def test_fill_prompt(props, obj, build_metada_pocket):
    """Tests if the prompt is generated correctly, i.e it can correctly be parsed."""
    dataset = MolGenerationInstructionsDataset(cfg)
    prompt = dataset.fill_prompt([props], [obj], build_metada_pocket(props))
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    scorer = RewardScorer(DATA_PATH, "properties")
    parsed = scorer.get_mol_props_from_prompt([prompt], scorer.search_patterns)[0]
    assert props in parsed
    assert parsed[props][0] == obj.split()[0]
    value = float(parsed[props][1])
    assert value == float(obj.split()[1] if len(obj.split()) > 1 else 0)


def test_generation_json():
    dataset = MolGenerationInstructionsDataset(cfg)
    metadatas = []
    for i in range(3):
        dialogues = dataset.generate_prompt_json(n=100, docking_split=i)
        metadatas.append([prompt[-1]["metadata"] for prompt in dialogues])

    docking_props = []
    for i in range(3):
        docking_props_list = []
        counts = {i: [] for i in range(1, 4)}

        for m in metadatas[i]:
            props = m["properties"]
            assert len(props) <= dataset.max_n_props

            dock_p = np.intersect1d(m["properties"], dataset.docking_properties)
            assert len(np.unique(dock_p)) == len(dock_p)
            assert len(dock_p) <= dataset.rule_set.max_docking_per_prompt

            for p in dock_p:
                counts[len(props)][p] = counts[len(props)].get(p, 0) + 1

            docking_props_list.append(dock_p)

        for n_props in counts:
            for p in counts[n_props]:
                assert counts[n_props][p] <= dataset.rule_set.max_occ

        docking_props.append(docking_props_list)

    assert (
        sum(
            len(
                np.intersect1d(
                    list(chain(*docking_props[i])),
                    list(chain(*docking_props[(i + 1) % 3])),
                )
            )
            for i in range(3)
        )
        == 0
    )


@pytest.mark.parametrize(
    "path, file, template",
    product(
        [DATA_PATH],
        [
            "train_prompts",
            "test_data/test_prompts",
            "test_data/test_prompts_ood",
            "eval_data/eval_prompts",
            "eval_data/eval_prompts_ood",
        ],
        [
            "none",
            "prompt_template_jinja",
            "prompt_instruction_template_jinja",
        ],
    ),
)
def test_saved_train_dataset(path, file, template):
    template = templates[template]
    file_path = os.path.join(DATA_PATH, file)
    dataset = load_from_disk(file_path)

    scorer = RewardScorer(path_to_mappings=path)

    for row in dataset:
        prompt = row["prompt"]
        gt_objectives = []
        for p, o, t in zip(row["properties"], row["objectives"], row["target"]):
            if isinstance(t, str):
                t = float(t)
            p = scorer.property_name_mapping.get(p, p)
            gt_objectives.append((p, (o, t)))
        prompt = prompt[0]["value"]
        prompt_fmt = template.replace("{{prompt}}", prompt)
        extracted = scorer.get_mol_props_from_prompt(
            [prompt_fmt], scorer.search_patterns
        )[0]
        objectives = []
        for p, obj in extracted.items():
            objectives.append((scorer.property_name_mapping.get(p, p), obj))
        assert objectives == gt_objectives
