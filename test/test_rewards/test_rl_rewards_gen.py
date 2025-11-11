from itertools import product

import numpy as np
import pytest
import ray
import torch
from rdkit import Chem

from mol_gen_docking.data.gen_dataset import (
    DatasetConfig,
)
from mol_gen_docking.reward.rl_rewards import (
    RewardScorer,
    has_bridged_bond,
)
from mol_gen_docking.reward.verifiers.generation_reward.property_utils import (
    rescale_property_values,
)

from .utils import (
    COMPLETIONS,
    DATA_PATH,
    OBJECTIVES_TO_TEST,
    PROP_LIST,
    PROPERTIES_NAMES_SIMPLE,
    SMILES,
    fill_completion,
    get_unscaled_obj,
    propeties_csv,
)

if not ray.is_initialized():
    ray.init(num_cpus=16)


cfg = DatasetConfig(data_path=DATA_PATH)

valid_scorer = RewardScorer(
    DATA_PATH,
    "valid_smiles",
    parse_whole_completion=True,
    rescale=False,
)
property_scorer = RewardScorer(
    DATA_PATH,
    "property",
    parse_whole_completion=False,
    rescale=False,
)

property_scorer_rescale = RewardScorer(
    DATA_PATH,
    "property",
    parse_whole_completion=False,
    rescale=True,
)


def is_reward_valid(rewards, smiles, properties):
    """Check if the reward is valid."""
    # Remove "FAKE" from smiles
    smiles = [s for s in smiles if s != "FAKE"]
    # Get the bridged_bond mask
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    bridged_mask = torch.tensor(
        [not has_bridged_bond(mol) if mol is not None else False for mol in mols]
    )

    if len(smiles) > 0:
        property_names = [PROPERTIES_NAMES_SIMPLE.get(p, p) for p in properties]
        props = torch.tensor(
            propeties_csv.set_index("smiles").loc[smiles, property_names].values
        )
        if bridged_mask.sum() == 0:
            props = torch.tensor(0.0)
        else:
            props = (props[bridged_mask].float()).mean()

        rewards = torch.tensor(rewards).mean()
        assert torch.isclose(rewards, props, atol=1e-3).all()


@pytest.mark.parametrize("completion, smiles", product(COMPLETIONS, SMILES))
def test_valid_smiles(completion, smiles):
    """Test the function molecular_properties."""
    completions = ["<answer> " + " ".join(smiles) + " </answer>"]
    completions = [completion.format(SMILES=c) for c in completions]
    rewards = np.array(
        valid_scorer(
            completions,
            metadata=[{"objectives": ["maximize"]}],
        )
    )
    assert rewards.sum() == float(
        "SMILES" in completion and not ("FAKE" in smiles and len(smiles) == 1)
    )


@pytest.mark.parametrize(
    "property1, property2",
    product(
        PROP_LIST,
        np.random.choice(PROP_LIST, 8),
    ),
)
def test_multi_prompt_multi_generation(  # 16 - 1 : 20/7 // 192 - 1 :
    property1,
    property2,
):
    """Test the reward function for a set of 2 prompts and multiple generations."""
    completion = "Here is a molecule: [SMILES] what are its properties?"
    metadata = [
        {"properties": [property1], "objectives": ["maximize"], "target": [0]}
    ] + [{"properties": [property2], "objectives": ["maximize"], "target": [0]}]
    smiles = [
        propeties_csv.sample(np.random.randint(1, 8))["smiles"].tolist()
        for k in range(2)
    ]
    completions = [fill_completion(s, completion) for s in smiles]

    rewards = property_scorer(completions, metadata, use_pbar=False)

    if smiles != []:
        is_reward_valid(rewards[0], smiles[0], [property1])
        is_reward_valid(rewards[1], smiles[1], [property2])
    else:
        assert sum(rewards[0]) == 0


@pytest.mark.parametrize(
    "prop, obj, smiles",
    list(
        product(
            PROP_LIST,
            OBJECTIVES_TO_TEST[1:],  # Skip "maximize" for this test
            [propeties_csv.sample(4)["smiles"].tolist() for k in range(8)],
        )
    ),
)
def test_all_prompts(prop, obj, smiles):
    """
    Test the reward function with the optimization of one property.
    Assumes the value of the reward function when using maximise is correct.
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    mask = torch.tensor(
        [not has_bridged_bond(m) if m is not None else False for m in mols]
    ).float()

    obj, target = get_unscaled_obj(obj, prop)
    n_generations = len(smiles)
    metadata = [
        {"properties": [prop], "objectives": [obj], "target": [target]}
    ] * n_generations + [
        {"properties": [prop], "objectives": ["maximize"], "target": [target]}
    ] * n_generations
    smiles = smiles * 2
    completions = [
        fill_completion(
            [s], "Here is a molecule: [SMILES] does it have the right properties?"
        )
        for s in smiles
    ]
    rewards = property_scorer_rescale(completions, metadata, debug=True, use_pbar=False)

    assert isinstance(rewards, (np.ndarray, list, torch.Tensor))
    rewards = torch.Tensor(rewards)
    assert not rewards.isnan().any()
    rewards_prop = rewards[:n_generations]
    rewards_max = rewards[n_generations:]
    objective = obj.split()[0]
    if objective == "maximize":
        val = rewards_max
    elif objective == "minimize":
        val = 1 - rewards_max
    else:
        target = rescale_property_values(prop, target, False)
        if objective == "below":
            val = (rewards_max <= target).float()
        elif objective == "above":
            val = (rewards_max >= target).float()
        elif objective == "equal":
            val = np.clip(1 - 100 * (rewards_max - target) ** 2, 0, 1)
    val = val * mask
    assert torch.isclose(rewards_prop, val * mask, atol=1e-4).all()
    property_scorer.rescale = False
