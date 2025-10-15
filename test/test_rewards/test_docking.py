import os
import time
from itertools import product
from typing import List

import numpy as np
import pytest
import ray
import torch

from mol_gen_docking.data.dataset import (
    DatasetConfig,
    MolGenerationInstructionsDatasetGenerator,
)
from mol_gen_docking.reward.rl_rewards import RewardScorer

from .utils import (
    COMPLETIONS,
    DATA_PATH,
    DOCKING_PROP_LIST,
    SMILES,
    get_fill_completions,
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
    "property": RewardScorer(
        DATA_PATH,
        "property",
        parse_whole_completion=True,
        rescale=False,
        oracle_kwargs=dict(
            n_cpu=int(os.environ.get("N_CPUS_DOCKING", 1)),
            exhaustiveness=4,
            docking_oracle="soft_docking",
            vina_mode="autodock-gpu_256wi",
        ),
    ),
    "property_whole": RewardScorer(
        DATA_PATH,
        "property",
        parse_whole_completion=False,
        rescale=False,
        oracle_kwargs=dict(
            n_cpu=int(os.environ.get("N_CPUS_DOCKING", 1)),
            exhaustiveness=4,
            docking_oracle="soft_docking",
            vina_mode="AutoDock-Vina",
            qv_dir="external_repositories/Vina-GPU-2.1/AutoDock-Vina-GPU-2.1",  # Vina executable
        ),
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


@pytest.mark.parametrize("target", DOCKING_PROP_LIST[:3])
def test_properties_single_prompt_vina_reward(
    target, property_scorer, property_filler, build_prompt, n_generations=1
):
    """Test the reward function runs for vina targets."""
    prompts = [build_prompt(target)] * n_generations
    smiles = [propeties_csv.iloc[:128]["smiles"].tolist() for k in range(n_generations)]
    completions = [
        property_filler(s, "Here is a molecule: [SMILES] what are its properties?")
        for s in smiles
    ]
    rewards = property_scorer(prompts, completions)

    assert isinstance(rewards, (np.ndarray, list, torch.Tensor))
    rewards = torch.Tensor(rewards)
    assert not rewards.isnan().any()


@pytest.mark.parametrize("property1", np.random.choice(DOCKING_PROP_LIST, 4))
def test_timeout(
    property1,
    n_generation=1,
):
    property_scorer = scorers["property"]
    property_filler = get_fill_completions(property_scorer.parse_whole_completion)

    dataset_cls = MolGenerationInstructionsDatasetGenerator(cfg)

    def build_prompt(property1):
        """Build a prompt for the given property."""
        return dataset_cls.fill_prompt([property1], ["maximize"])[0]

    completion = "Here is a molecule: [SMILES] what are its properties?"
    prompts = [build_prompt(property1)] * n_generation
    smiles = [propeties_csv.sample(1)["smiles"].tolist() for _ in range(n_generation)]
    completions = [property_filler(s, completion) for s in smiles]
    t0 = time.time()
    scorer = RewardScorer(
        DATA_PATH,
        "property",
        parse_whole_completion=True,
        oracle_kwargs=dict(ncpu=1, exhaustiveness=1024),
    )

    result = scorer(prompts, completions)
    t1 = time.time()
    assert (torch.tensor(result) == 0).all() or (t1 - t0) < 60, (
        "Some rewards are not positive, check the oracle."
    )
