import os
import time
from itertools import product
from typing import List

import numpy as np
import pytest
import ray
import torch

from mol_gen_docking.data.rl_dataset import (
    DatasetConfig,
    MolGenerationInstructionsDataset,
)
from mol_gen_docking.reward.rl_rewards import RewardScorer

from .utils import (
    COMPLETIONS,
    DOCKING_PROP_LIST,
    OBJECTIVES_TO_TEST,
    PROP_LIST,
    PROPERTIES_NAMES_SIMPLE,
    SMILES,
    get_fill_completions,
    propeties_csv,
)

try:
    ray.init(num_cpus=16)
except Exception:
    ray.init()

cfg = DatasetConfig(data_path="data/mol_orz")

scorers = {
    "valid_smiles": RewardScorer(
        PROPERTIES_NAMES_SIMPLE,
        DOCKING_PROP_LIST,
        "valid_smiles",
        parse_whole_completion=True,
        rescale=False,
        oracle_kwargs=dict(ncpu=1, exhaustiveness=1),
    ),
    "property": RewardScorer(
        PROPERTIES_NAMES_SIMPLE,
        DOCKING_PROP_LIST,
        "property",
        parse_whole_completion=True,
        rescale=False,
        oracle_kwargs=dict(ncpu=1, exhaustiveness=1),
    ),
    "property_whole": RewardScorer(
        PROPERTIES_NAMES_SIMPLE,
        DOCKING_PROP_LIST,
        "property",
        parse_whole_completion=False,
        rescale=False,
        oracle_kwargs=dict(ncpu=1, exhaustiveness=1),
    ),
}


@pytest.fixture(scope="module", params=[True, False])
def build_prompt(request):
    def build_prompt_from_dataset(
        property: str | List[str], obj: str = "maximize"
    ) -> str:
        if isinstance(property, str):
            properties = [property]
        else:
            properties = property
        dummy = MolGenerationInstructionsDataset(cfg)
        return dummy.fill_prompt(properties, [obj] * len(properties))

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


@pytest.fixture(scope="module", params=[True, False])
def valid_smiles_scorer(request):
    """Fixture to test the function molecular_properties."""
    return scorers["valid_smiles"]


@pytest.fixture(scope="module")
def valid_smiles_filler(valid_smiles_scorer):
    """Fixture to test the function molecular_properties."""
    return get_fill_completions(valid_smiles_scorer.parse_whole_completion)


@pytest.fixture(scope="module", params=[True, False])
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
    rewards = valid_smiles_scorer(prompts, completions)
    assert rewards.sum().item() == float(
        "[SMILES]" in completion and not ("FAKE" in smiles and len(smiles) == 1)
    )


@pytest.mark.parametrize(
    "property1, property2",
    product(
        np.random.choice(PROP_LIST, 3),
        np.random.choice(PROP_LIST, 3),
    ),
)
def test_properties_single_prompt_reward(
    property1, property2, property_scorer, completions_smiles, build_prompt
):
    """Test the function molecular_properties with 2 properties."""
    completions, smiles = completions_smiles
    prompts = [build_prompt([property1, property2])] * len(completions)
    rewards = property_scorer(prompts, completions)
    if smiles != []:
        is_reward_valid(rewards, smiles, [property1, property2])
    else:
        assert sum(rewards) == 0


@pytest.mark.parametrize(
    "property1, property2",
    product(
        np.random.choice(PROP_LIST, 2),
        np.random.choice(PROP_LIST, 2),
    ),
)
def test_properties_multi_prompt_rewards(
    property1, property2, property_scorer, completions_smiles, build_prompt
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
        assert sum(rewards) == 0


@pytest.mark.parametrize(
    "property1, property2",
    product(
        np.random.choice(PROP_LIST, 8),
        np.random.choice(PROP_LIST, 8),
    ),
)
def test_multip_prompt_multi_generation(
    property1,
    property2,
    property_scorer,
    property_filler,
    build_prompt,
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
            assert sum(rewards[i]) == 0


@pytest.mark.skipif(os.system("qvina --help") == 32512, reason="requires vina")
@pytest.mark.parametrize("target", np.random.choice(PROP_LIST, 3))
def test_properties_single_prompt_vina_reward(
    target, property_scorer, property_filler, build_prompt, n_generations=16
):
    """Test the function molecular_properties with 2 properties."""
    prompts = [build_prompt(PROPERTIES_NAMES_SIMPLE[target])] * n_generations
    smiles = [
        propeties_csv.sample(np.random.randint(1, 4))["smiles"].tolist()
        for k in range(n_generations)
    ]
    completions = [
        property_filler(s, "Here is a molecule: [SMILES] what are its properties?")
        for s in smiles
    ]
    rewards = property_scorer(prompts, completions)

    assert isinstance(rewards, (np.ndarray, list, torch.Tensor))
    rewards = torch.Tensor(rewards)
    assert not rewards.isnan().any()


@pytest.mark.parametrize(
    "prop, obj, smiles",
    list(
        product(
            PROP_LIST,
            OBJECTIVES_TO_TEST,
            [propeties_csv.sample(1)["smiles"].tolist() for k in range(3)],
        )
    ),
)
def test_all_prompts(prop, obj, smiles, property_scorer, property_filler, build_prompt):
    """Test the function molecular_properties with 2 properties."""

    n_generations = len(smiles)
    prompts = [build_prompt(prop, obj)] * n_generations + [
        build_prompt(prop, "maximize")
    ] * n_generations

    smiles = smiles * 2
    completions = [
        property_filler([s], "Here is a molecule: [SMILES] what are its properties?")
        for s in smiles
    ]
    property_scorer.rescale = True
    rewards = property_scorer(prompts, completions, debug=True)

    assert isinstance(rewards, (np.ndarray, list, torch.Tensor))
    rewards = torch.Tensor(rewards)
    assert not rewards.isnan().any()
    rewards_prop = rewards[:n_generations]
    rewards_max = rewards[n_generations:]
    if obj == "maximize":
        val = rewards_max
    elif obj == "minimize":
        val = 1 - rewards_max
    elif obj == "below 0.5":
        val = (rewards_max <= 0.5).float()
    elif obj == "above 0.5":
        val = (rewards_max >= 0.5).float()
    elif obj == "equal 0.5":
        val = 1 - (rewards_max - 0.5) ** 2
    assert torch.isclose(rewards_prop, val, atol=1e-4).all()
    property_scorer.rescale = False


@pytest.mark.skipif(os.system("qvina --help") == 32512, reason="requires vina")
@pytest.mark.parametrize(
    "prop, smiles",
    list(
        product(
            PROP_LIST[:2],
            [propeties_csv.sample(1)["smiles"].tolist() for k in range(3)],
        )
    ),
)
def test_ray(prop, smiles, build_prompt):
    prompts = [build_prompt(prop, "maximize")] * len(smiles)
    filler = get_fill_completions(True)
    completions = [filler([s], "Here is a molecule: [SMILES]") for s in smiles]

    worker = (
        ray.remote(RewardScorer)
        .options(num_cpus=1)
        .remote(
            parse_whole_completion=True,
            oracle_kwargs=dict(ncpu=1, exhaustiveness=1),
        )  # type: ignore
    )
    result = worker.get_score.remote(prompts, completions)
    _ = ray.get(result)


@pytest.mark.skipif(
    os.system("qvina --help") == 32512 and os.environ.get("DEBUG_MODE", 0) == "1",
    reason="requires vina and debug mode",
)
@pytest.mark.parametrize(
    "property1",
    DOCKING_PROP_LIST[:10],
)
def test_runtime(
    property1,
    n_generation=4,
    time_per_gen=5,
):
    print(f"Testing runtime for {property1}")
    property_scorer = scorers["property"]
    property_filler = get_fill_completions(property_scorer.parse_whole_completion)

    dataset_cls = MolGenerationInstructionsDataset()

    def build_prompt(property1):
        """Build a prompt for the given property."""
        return dataset_cls.fill_prompt([property1], ["maximize"])

    completion = "Here is a molecule: [SMILES] what are its properties?"
    prompts = [build_prompt(property1)] * n_generation
    smiles = [propeties_csv.sample(1)["smiles"].tolist() for _ in range(n_generation)]
    completions = [property_filler(s, completion) for s in smiles]

    worker = (
        ray.remote(RewardScorer)
        .options(num_cpus=1)
        .remote(
            parse_whole_completion=True,
            oracle_kwargs=dict(ncpu=1, exhaustiveness=1),
        )  # type: ignore
    )

    t0 = time.time()
    result = worker.get_score.remote(prompts, completions)
    r = ray.get(result)
    t1 = time.time()
    print(f"Runtime: {t1 - t0} seconds")

    # Max for 16 generations should be around 30 seconds
    assert t1 - t0 < time_per_gen * n_generation, (
        f"Runtime is too long: {t1 - t0} seconds"
    )
    assert (torch.tensor(r) > 0).all(), (
        "Some rewards are not positive, check the oracle."
    )
