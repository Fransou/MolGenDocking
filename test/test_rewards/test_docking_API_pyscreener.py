import os
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
    DATA_PATH,
    DOCKING_PROP_LIST,
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
            docking_oracle="pyscreener",
        ),
    ),
}


def build_prompt(property: str | List[str], obj: str = "maximize") -> str:
    if isinstance(property, str):
        properties = [property]
    else:
        properties = property
    dummy = MolGenerationInstructionsDatasetGenerator(cfg)
    prompt, _ = dummy.fill_prompt(properties, [obj] * len(properties))
    return prompt


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


@pytest.mark.parametrize("target", DOCKING_PROP_LIST[:64])
def test_properties_single_prompt_vina_reward(target, n_generations=1):
    """Test the reward function runs for vina targets."""
    property_filler = get_fill_completions(scorers["property"].parse_whole_completion)
    prompts = [build_prompt(target)] * n_generations
    smiles = [propeties_csv.iloc[:128]["smiles"].tolist() for k in range(n_generations)]
    completions = [
        property_filler(s, "Here is a molecule: [SMILES] what are its properties?")
        for s in smiles
    ]
    rewards = scorers["property"](prompts, completions)

    assert isinstance(rewards, (np.ndarray, list, torch.Tensor))
    rewards = torch.Tensor(rewards)
    assert not rewards.isnan().any()
