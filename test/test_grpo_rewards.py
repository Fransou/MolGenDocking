import os

from typing import List, Callable
from itertools import product
import pytest
import torch
import numpy as np
from tdc.metadata import docking_target_info

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


def get_fill_completions(no_flags: bool = False) -> Callable[[List[str], str], str]:
    def fill_completion(smiles: List[str], completion: str) -> str:
        """Fill the completion with the smiles."""
        smiles_joined: str = "".join(
            [
                "{} ".format(s) if no_flags else "<SMILES>{}</SMILES> ".format(s)
                for s in smiles
            ]
        )
        print(smiles)
        print(smiles_joined)
        return completion.replace("[SMILES]", smiles_joined)

    return fill_completion


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
def valid_smiles_scorer(request):
    """Fixture to test the function molecular_properties."""
    return RewardScorer(
        "valid_smiles", parse_whole_completion=request.param, rescale=False
    )


@pytest.fixture(scope="module")
def valid_smiles_filler(valid_smiles_scorer):
    """Fixture to test the function molecular_properties."""
    return get_fill_completions(valid_smiles_scorer.parse_whole_completion)


@pytest.fixture(scope="module", params=[True, False])
def property_scorer(request):
    """Fixture to test the function molecular_properties."""
    return RewardScorer(
        "propertys", parse_whole_completion=request.param, rescale=False
    )


@pytest.fixture(scope="module", params=product(COMPLETIONS, SMILES))
def completion(request, property_filler):
    """Fixture to test the function molecular_properties."""
    completion, smiles = request.param
    return property_filler(smiles, completion)


@pytest.fixture(scope="module")
def property_filler(property_scorer):
    """Fixture to test the function molecular_properties."""
    return get_fill_completions(property_scorer.parse_whole_completion)


@pytest.fixture(scope="module", params=product(COMPLETIONS, SMILES))
def completions_smiles(request, property_filler):
    """Fixture to test the function molecular_properties."""
    completion, smiles = request.param
    if "[SMILES]" not in completion or smiles == ["FAKE"]:
        smiles = []
    if "FAKE" in smiles:
        smiles = [s for s in smiles if s != "FAKE"]
    return [property_filler(smiles, completion)], smiles


@pytest.mark.parametrize("completion, smiles", product(COMPLETIONS, SMILES))
def test_valid_smiles(completion, smiles, valid_smiles_scorer, valid_smiles_filler):
    """Test the function molecular_properties."""
    completions = [valid_smiles_filler(smiles, completion)]
    prompts = [""] * len(completions)
    rewards = valid_smiles_scorer(prompts, completions)
    assert rewards.sum().item() == float(
        "[SMILES]" in completion and not ("FAKE" in smiles and len(smiles) == 1)
    )


@pytest.mark.parametrize(
    "property1, property2",
    product(
        np.random.choice(KNOWN_PROPERTIES, 3),
        np.random.choice(KNOWN_PROPERTIES, 3),
    ),
)
def test_properties_single_prompt_reward(
    property1, property2, property_scorer, completions_smiles
):
    """Test the function molecular_properties with 2 properties."""
    completions, smiles = completions_smiles
    prompts = [build_prompt(property1) + " --- " + build_prompt(property2)] * len(
        completions
    )
    rewards = property_scorer(prompts, completions)
    if smiles != []:
        is_reward_valid(rewards, smiles, [property1, property2])
    else:
        assert rewards.sum().item() == 0


@pytest.mark.parametrize(
    "property1, property2",
    product(
        np.random.choice(KNOWN_PROPERTIES, 3),
        np.random.choice(KNOWN_PROPERTIES, 3),
    ),
)
def test_properties_multi_prompt_rewards(
    property1, property2, property_scorer, completions_smiles
):
    """Test the function molecular_properties with 2 properties."""
    completions, smiles = completions_smiles
    completions = completions * 2

    # 2- Test when optimizing 2 properties separately
    prompts = [build_prompt(property1)] * (len(completions) // 2) + [
        build_prompt(property2)
    ] * (len(completions) // 2)
    rewards = property_scorer(prompts, completions)
    if smiles != []:
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
    property_scorer,
    property_filler,
    n_generations=4,
):
    """Test the function molecular_properties."""
    completion = "Here is a molecule: [SMILES] what are its properties?"
    prompts = [build_prompt(property1)] * n_generations + [
        build_prompt(property2)
    ] * n_generations
    smiles = [
        propeties_csv.sample(np.random.randint(1, 4))["smiles"].tolist()
        for k in range(n_generations * 2)
    ]
    completions = [property_filler(s, completion) for s in smiles]

    rewards = property_scorer(prompts, completions)

    for i in range(n_generations * 2):
        if smiles != []:
            if i < n_generations:
                is_reward_valid(rewards[i], smiles[i], [property1])
            else:
                is_reward_valid(rewards[i], smiles[i], [property2])
        else:
            assert rewards[i].sum().item() == 0


@pytest.mark.skipif(os.system("vina --help") == 32512, reason="requires vina")
@pytest.mark.parametrize("target", docking_target_info.keys())
def test_properties_single_prompt_vina_reward(
    target, property_scorer, completions_smiles
):
    """Test the function molecular_properties with 2 properties."""
    completions, smiles = completions_smiles
    prompts = [build_prompt(target)] * len(completions)
    rewards = property_scorer(prompts, completions)
    if smiles != []:
        assert isinstance(rewards, np.ndarray) or isinstance(rewards, list)
    else:
        assert rewards.sum().item() == 0
