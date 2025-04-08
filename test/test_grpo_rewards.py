from typing import List
from itertools import product
import pytest
import torch
import numpy as np

from mol_gen_docking.utils.grpo_rewards import RewardScorer
from mol_gen_docking.utils.molecular_properties import (
    propeties_csv,
    KNOWN_PROPERTIES,
    PROPERTIES_NAMES_SIMPLE,
)

SMILES = (
    [["FAKE"]]
    + [propeties_csv.sample(k)["smiles"].tolist() for k in range(1, 3)]
    + [propeties_csv.sample(1)["smiles"].tolist() + ["FAKE"]]
)

COMPLETIONS = [
    "Here is a molecule: [SMILES] what are its properties?",
    "This is an empty completion.",
]


def fill_completion(smiles: List[str], completion: str) -> str:
    """Fill the completion with the smiles."""
    smiles_joined: str = "".join(["<SMILES>{}</SMILES> ".format(s) for s in smiles])
    return completion.replace("[SMILES]", smiles_joined)


def fill_completion_no_flags(smiles: List[str], completion: str) -> str:
    """Fill the completion with the smiles."""
    smiles_joined: str = "".join([s + " " for s in smiles])
    return completion.replace("[SMILES]", smiles_joined)


def build_prompt(property: str) -> str:
    return property + " (maximize)"


def is_reward_valid(rewards, smiles, properties):
    """Check if the reward is valid."""
    # Remove "FAKE" from smiles
    smiles = [s for s in smiles if s != "FAKE"]
    if len(smiles) > 0:
        property_names = [PROPERTIES_NAMES_SIMPLE.get(p, p) for p in properties]
        props = (
            torch.tensor(
                propeties_csv.set_index("smiles").loc[smiles, property_names].values
            )
            .float()
            .mean()
        )
        assert torch.isclose(rewards, props, atol=1e-3).all()


@pytest.fixture(scope="module", params=[True, False])
def valid_smiles_scorer_filler(request):
    """Fixture to test the function molecular_properties."""
    if request.param:  # not parse_whole_completion
        return RewardScorer("valid_smiles"), fill_completion
    else:  # parse_whole_completion
        return RewardScorer(
            "valid_smiles", parse_whole_completion=True
        ), fill_completion_no_flags


@pytest.fixture(scope="module", params=[True, False])
def property_scorer_filler(request):
    """Fixture to test the function molecular_properties."""
    if request.param:
        return RewardScorer("property", rescale=False), fill_completion
    else:
        return RewardScorer(
            "property", rescale=False, parse_whole_completion=True
        ), fill_completion_no_flags


@pytest.mark.parametrize("completion, smiles", product(COMPLETIONS, SMILES))
def test_smiles_reward(completion, smiles):
    """Test the function molecular_properties."""
    scorer = RewardScorer("smiles")
    completions = [fill_completion(smiles, completion)]
    prompts = [""] * len(completions)

    rewards = scorer(prompts, completions)
    assert rewards.sum().item() == float("[SMILES]" in completion)


@pytest.mark.parametrize("completion, smiles", product(COMPLETIONS, SMILES))
def test_valid_smiles(completion, smiles, valid_smiles_scorer_filler):
    """Test the function molecular_properties."""
    scorer, filler = valid_smiles_scorer_filler
    completions = [filler(smiles, completion)]
    prompts = [""] * len(completions)

    rewards = scorer(prompts, completions)
    assert rewards.sum().item() == float(
        "[SMILES]" in completion and not ("FAKE" in smiles and len(smiles) == 1)
    )


@pytest.mark.parametrize(
    "completion, smiles, property1, property2",
    product(
        COMPLETIONS,
        SMILES,
        np.random.choice(KNOWN_PROPERTIES, 3),
        np.random.choice(KNOWN_PROPERTIES, 3),
    ),
)
def test_properties_single_prompt_reward(
    completion, smiles, property1, property2, property_scorer_filler
):
    """Test the function molecular_properties with 2 properties."""
    scorer, filler = property_scorer_filler
    completions = [filler(smiles, completion)]

    # 1- Test when optimizing 2 properties simultaneously
    prompts = [build_prompt(property1) + " --- " + build_prompt(property2)] * len(
        completions
    )
    rewards = scorer(prompts, completions)
    if "[SMILES]" in completion:
        is_reward_valid(rewards, smiles, [property1, property2])
    else:
        assert rewards.sum().item() == 0


@pytest.mark.parametrize(
    "completion, smiles, property1, property2",
    product(
        COMPLETIONS,
        SMILES,
        np.random.choice(KNOWN_PROPERTIES, 3),
        np.random.choice(KNOWN_PROPERTIES, 3),
    ),
)
def test_properties_multi_prompt_rewards(
    completion, smiles, property1, property2, property_scorer_filler
):
    """Test the function molecular_properties with 2 properties."""
    scorer, filler = property_scorer_filler
    completions = [filler(smiles, completion)] * 2

    # 2- Test when optimizing 2 properties separately
    prompts = [build_prompt(property1)] * (len(completions) // 2) + [
        build_prompt(property2)
    ] * (len(completions) // 2)
    rewards = scorer(prompts, completions)
    if "[SMILES]" in completion:
        is_reward_valid(rewards[: (len(completions) // 2)], smiles, [property1])
        is_reward_valid(rewards[(len(completions) // 2) :], smiles, [property2])
    else:
        assert rewards.sum().item() == 0


@pytest.mark.parametrize(
    "property1, property2",
    product(
        np.random.choice(KNOWN_PROPERTIES, 8), np.random.choice(KNOWN_PROPERTIES, 8)
    ),
)
def test_multip_prompt_multi_generation(
    property1,
    property2,
    property_scorer_filler,
    n_generations=4,
):
    """Test the function molecular_properties."""
    scorer, filler = property_scorer_filler
    completion = COMPLETIONS[0]
    prompts = [build_prompt(property1)] * n_generations + [
        build_prompt(property2)
    ] * n_generations
    smiles = [
        propeties_csv.sample(np.random.randint(1, 4))["smiles"].tolist()
        for k in range(n_generations * 2)
    ]
    completions = [filler(s, completion) for s in smiles]

    rewards = scorer(prompts, completions)

    for i in range(n_generations * 2):
        if "[SMILES]" in completion:
            if i < n_generations:
                is_reward_valid(rewards[i], smiles[i], [property1])
            else:
                is_reward_valid(rewards[i], smiles[i], [property2])
        else:
            assert rewards[i].sum().item() == 0
