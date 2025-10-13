from itertools import product
from typing import Callable, List

import numpy as np
import pytest
import ray
import torch

from mol_gen_docking.baselines.reward_fn import get_reward_fn
from mol_gen_docking.data.dataset import (
    DatasetConfig,
    MolGenerationInstructionsDatasetGenerator,
)
from mol_gen_docking.reward.property_utils import rescale_property_values
from mol_gen_docking.reward.rl_rewards import RewardScorer

from .utils import (
    COMPLETIONS,
    DATA_PATH,
    OBJECTIVES_TO_TEST,
    PROP_LIST,
    PROPERTIES_NAMES_SIMPLE,
    SMILES,
    get_fill_completions,
    get_unscaled_obj,
    propeties_csv,
)

try:
    ray.init(num_cpus=16)
except Exception:
    ray.init()


cfg = DatasetConfig(data_path=DATA_PATH)

scorers = {
    "valid_smiles": RewardScorer(
        DATA_PATH,
        "valid_smiles",
        parse_whole_completion=True,
        rescale=False,
    ),
    "MolFilters": RewardScorer(
        DATA_PATH,
        "MolFilters",
        parse_whole_completion=True,
        rescale=False,
    ),
    "property": RewardScorer(
        DATA_PATH,
        "property",
        parse_whole_completion=True,
        rescale=False,
    ),
    "property_whole": RewardScorer(
        DATA_PATH,
        "property",
        parse_whole_completion=False,
        rescale=False,
    ),
}


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


@pytest.fixture(scope="module", params=[True])
def build_prompt(request, build_metada_pocket):
    def build_prompt_from_dataset(
        property: str | List[str], obj: str = "maximize"
    ) -> str:
        if isinstance(property, str):
            properties = [property]
        else:
            properties = property
        dummy = MolGenerationInstructionsDatasetGenerator(cfg)
        prompt, _ = dummy.fill_prompt(properties, [obj] * len(properties))
        return prompt

    def build_prompt_from_string(
        property: str | List[str], obj: str = "maximize"
    ) -> str:
        prefix = """A conversation between User and Assistant. The User asks a question,
             and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning p
            rocess is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think
            > <answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>.\nThis is the problem:
            \n"""
        return prefix + " " + build_prompt_from_dataset(property, obj)

    if request.param:
        return build_prompt_from_dataset
    else:
        return build_prompt_from_string


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
        rewards = torch.tensor(rewards).mean()
        assert torch.isclose(rewards, props, atol=1e-3).all()


@pytest.fixture(scope="module")
def valid_smiles_scorer(request):
    """Fixture to test the function molecular_properties."""
    return scorers["valid_smiles"]


@pytest.fixture(scope="module")
def filter_smiles_scorer(request):
    """Fixture to test the function molecular_properties."""
    return scorers["MolFilters"]


@pytest.fixture(scope="module")
def valid_smiles_filler(valid_smiles_scorer):
    """Fixture to test the function molecular_properties."""
    return get_fill_completions(valid_smiles_scorer.parse_whole_completion)


@pytest.fixture(scope="module")
def filter_smiles_filler(filter_smiles_scorer):
    """Fixture to test the function molecular_properties."""
    return get_fill_completions(filter_smiles_scorer.parse_whole_completion)


@pytest.fixture(scope="module", params=[True])
def property_scorer(request):
    """Fixture to test the function molecular_properties."""
    return scorers["property"] if request.param else scorers["property_whole"]


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
    rewards = np.array(valid_smiles_scorer(prompts, completions))
    assert rewards.sum() == float(
        "[SMILES]" in completion and not ("FAKE" in smiles and len(smiles) == 1)
    )


# @pytest.mark.parametrize("completion, smiles", product(COMPLETIONS, SMILES))
# def test_filter_smiles(completion, smiles, filter_smiles_scorer, filter_smiles_filler):
#     """Test the function molecular_properties."""
#     completions = [filter_smiles_filler(smiles, completion)]
#     prompts = [""] * len(completions)
#     rewards = np.array(filter_smiles_scorer(prompts, completions))
#     assert not np.isnan(rewards.sum())


@pytest.mark.parametrize(
    "property1, property2",
    product(
        np.random.choice(PROP_LIST, 8),
        np.random.choice(PROP_LIST, 8),
    ),
)
def test_multi_prompt_multi_generation(  # 16 - 1 : 20/7 // 192 - 1 :
    property1,
    property2,
    property_scorer,
    property_filler,
    build_prompt,
    n_generations=4,
):
    """Test the reward function for a set of 2 prompts and multiple generations."""
    completion = "Here is a molecule: [SMILES] what are its properties?"
    prompts = [build_prompt(property1)] * n_generations + [
        build_prompt(property2)
    ] * n_generations
    smiles = [
        propeties_csv.sample(np.random.randint(1, 8))["smiles"].tolist()
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
            assert sum(rewards[i]) == 0


@pytest.mark.parametrize(
    "prop, obj, smiles",
    list(
        product(
            PROP_LIST,
            OBJECTIVES_TO_TEST[1:],  # Skip "maximize" for this test
            [propeties_csv.sample(8)["smiles"].tolist() for k in range(1)],
        )
    ),
)
def test_all_prompts(prop, obj, smiles, property_scorer, property_filler, build_prompt):
    """
    Test the reward function with the optimization of one property.
    Assumes the value of the reward function when using maximise is correct.
    """
    obj = get_unscaled_obj(obj, prop)
    n_generations = len(smiles)
    prompts = [build_prompt(prop, obj)] * n_generations + [
        build_prompt(prop, "maximize")
    ] * n_generations

    smiles = smiles * 2
    completions = [
        property_filler(
            [s], "Here is a molecule: [SMILES] does it have the right properties?"
        )
        for s in smiles
    ]
    property_scorer.rescale = True
    rewards = property_scorer(prompts, completions, debug=True)

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
        target = rescale_property_values(prop, float(obj.split()[1]), False)
        if objective == "below":
            val = (rewards_max <= target).float()
        elif objective == "above":
            val = (rewards_max >= target).float()
        elif objective == "equal":
            val = np.clip(1 - 100 * (rewards_max - target) ** 2, 0, 1)
    assert torch.isclose(rewards_prop, val, atol=1e-4).all()
    property_scorer.rescale = False


@pytest.mark.parametrize("smiles, property", product(SMILES, PROP_LIST))
def test_baseline_reward_fn(smiles: str, property: str, build_prompt: Callable):
    prompt = build_prompt([property])
    reward_fn = get_reward_fn(prompt, DATA_PATH)
    s = smiles[0]
    reward = reward_fn(s)
    assert isinstance(reward, float)
