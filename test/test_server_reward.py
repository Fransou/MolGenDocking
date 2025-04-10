from typing import List
from itertools import product
import pytest
import torch
import numpy as np

from mol_gen_docking.reward.grpo_rewards import RewardScorer, RewardScorerServer
from mol_gen_docking.reward.oracles import propeties_csv, PROPERTIES_NAMES_SIMPLE

PROP_LIST = list(PROPERTIES_NAMES_SIMPLE.keys())

SMILES = (
    [["FAKE"]]
    + [propeties_csv.sample(k)["smiles"].tolist() for k in range(1, 3)]
    + [propeties_csv.sample(2)["smiles"].tolist() + ["FAKE"]]
)

COMPLETIONS = [
    "Here is a molecule:[SMILES] what are its properties?",
    "This is an empty completion.",
]


def fill_completion(smiles: List[str], completion: str) -> str:
    """Fill the completion with the smiles."""
    smiles_joined: str = "".join(["<SMILES>{}</SMILES>".format(s) for s in smiles])
    return completion.replace("[SMILES]", smiles_joined)


def build_prompt(property: str) -> str:
    return property + " (maximize)"


@pytest.mark.parametrize("completion, smiles", product(COMPLETIONS, SMILES))
def test_smiles_reward(completion, smiles):
    """Test the function molecular_properties."""
    scorer = RewardScorerServer("smiles")
    completions = [fill_completion(smiles, completion)]
    prompts = [""] * len(completions)

    rewards = scorer(prompts, completions)
    assert rewards.sum().item() == float("[SMILES]" in completion)


@pytest.mark.parametrize("completion, smiles", product(COMPLETIONS, SMILES))
def test_valid_smiles(completion, smiles):
    """Test the function molecular_properties."""
    scorer = RewardScorerServer("valid_smiles")
    completions = [fill_completion(smiles, completion)]
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
        np.random.choice(PROP_LIST, 2),
        np.random.choice(PROP_LIST, 2),
    ),
)
def test_properties_single_prompt_reward(completion, smiles, property1, property2):
    """Test the function molecular_properties with 2 properties."""
    scorer_gt = RewardScorer("properties")
    scorer = RewardScorerServer("properties")
    completions = [fill_completion(smiles, completion)]

    # 1- Test when optimizing 2 properties simultaneously
    prompts = [build_prompt(property1) + " --- " + build_prompt(property2)] * len(
        completions
    )
    scorer.pre_query_properties(prompts, completions)

    rewards_gt = scorer_gt(prompts, completions)
    rewards = scorer(prompts, completions)
    if "[SMILES]" in completion:
        assert torch.allclose(rewards, rewards_gt)
    else:
        assert rewards_gt.sum().item() == 0


@pytest.mark.parametrize(
    "completion, smiles, property1, property2",
    product(
        COMPLETIONS,
        SMILES,
        np.random.choice(PROP_LIST, 2),
        np.random.choice(PROP_LIST, 2),
    ),
)
def test_properties_multi_prompt_rewards(completion, smiles, property1, property2):
    """Test the function molecular_properties with 2 properties."""
    scorer_gt = RewardScorer("properties")
    scorer = RewardScorerServer("properties")
    completions = [fill_completion(smiles, completion)] * 2

    # 2- Test when optimizing 2 properties separately
    prompts = [build_prompt(property1)] * (len(completions) // 2) + [
        build_prompt(property2)
    ] * (len(completions) // 2)

    scorer.pre_query_properties(prompts, completions)
    rewards_gt = scorer_gt(prompts, completions)
    rewards = scorer(prompts, completions)
    if "[SMILES]" in completion:
        assert torch.allclose(rewards, rewards_gt)
    else:
        assert rewards.sum().item() == 0


@pytest.mark.parametrize(
    "property1, property2",
    product(
        np.random.choice(PROP_LIST, 2),
        np.random.choice(PROP_LIST, 2),
    ),
)
def test_multip_prompt_multi_generation(property1, property2, n_generations=4):
    """Test the function molecular_properties."""
    scorer_gt = RewardScorer("properties")
    scorer = RewardScorerServer("properties")
    completion = COMPLETIONS[0]
    prompts = [build_prompt(property1)] * n_generations + [
        build_prompt(property2)
    ] * n_generations
    smiles = [
        propeties_csv.sample(np.random.randint(1, 4))["smiles"].tolist()
        for k in range(n_generations * 2)
    ]
    completions = [fill_completion(s, completion) for s in smiles]

    scorer.pre_query_properties(prompts, completions)
    rewards_gt = scorer_gt(prompts, completions)
    rewards = scorer(prompts, completions)
    for i in range(n_generations * 2):
        if "[SMILES]" in completion:
            assert torch.allclose(rewards, rewards_gt)
        else:
            assert rewards[i].sum().item() == 0
