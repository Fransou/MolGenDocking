import os
import time as time
from typing import List

import numpy as np
import pytest
import torch

from mol_gen_docking.data.dataset import (
    DatasetConfig,
    MolGenerationInstructionsDatasetGenerator,
)
from mol_gen_docking.data.meeko_process import ReceptorProcess
from mol_gen_docking.reward.rl_rewards import RewardScorer

from .utils import (
    DATA_PATH,
    DOCKING_PROP_LIST,
    fill_df_time,
    get_fill_completions,
    propeties_csv,
)

cfg = DatasetConfig(data_path=DATA_PATH)
props_to_eval = DOCKING_PROP_LIST[:100]


@pytest.fixture(scope="module", params=[1, 2, 4, 8, 16])
def exhaustiveness(request):
    return request.param


@pytest.fixture(scope="module")
def scorer(has_gpu, exhaustiveness):
    if not has_gpu:
        return RewardScorer(
            DATA_PATH,
            "property",
            parse_whole_completion=True,
            rescale=False,
            oracle_kwargs=dict(
                n_cpu=int(os.environ.get("N_CPUS_DOCKING", exhaustiveness)),
                exhaustiveness=exhaustiveness,
                docking_oracle="pyscreener",
            ),
        )
    else:
        return RewardScorer(
            DATA_PATH,
            "property",
            parse_whole_completion=True,
            rescale=False,
            oracle_kwargs=dict(
                n_cpu=int(os.environ.get("N_CPUS_DOCKING", 1)),
                exhaustiveness=exhaustiveness,
                docking_oracle="soft_docking",
                vina_mode="autodock_gpu_256wi",
            ),
        )


@pytest.fixture(scope="module")
def receptor_process(has_gpu):
    if not has_gpu:
        return lambda x: ([], [])
    else:
        rp = ReceptorProcess(
            data_path=DATA_PATH,
        )

        def wrapped_fn(r: str | List[str]):
            if isinstance(r, str):
                r = [r]
            return rp.process_receptors(r, True)

        return wrapped_fn


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


def test_receptor_process(receptor_process):
    """Test receptor processing."""
    _, missed_targets = receptor_process(props_to_eval)
    assert len(missed_targets) == 0, (
        f"Receptors {missed_targets} could not be processed."
    )


@pytest.mark.parametrize("target", props_to_eval)
def test_docking_props(target, scorer, receptor_process, n_generations=4):
    """Test the reward function runs for vina targets."""
    property_filler = get_fill_completions(scorer.parse_whole_completion)
    prompts = [build_prompt(target)] * n_generations
    smiles = [[propeties_csv.iloc[i]["smiles"]] for i in range(n_generations)]
    completions = [
        property_filler(s, "Here is a molecule: [SMILES] what are its properties?")
        for s in smiles
    ]
    t0 = time.time()
    rewards = scorer(prompts, completions)
    t1 = time.time()
    assert isinstance(rewards, (np.ndarray, list, torch.Tensor))
    rewards = torch.Tensor(rewards)
    assert not rewards.isnan().any()
    assert rewards.shape[0] == n_generations
    fill_df_time(
        target,
        n_generations,
        t0=t0,
        t1=t1,
        method=scorer.oracle_kwargs.get("docking_oracle", "unknown"),
        exhaustiveness=scorer.oracle_kwargs.get("exhaustiveness", -1),
        scores=rewards.mean().item(),
    )


@pytest.mark.parametrize(
    "targets", [props_to_eval[i * 5 : (i + 1) * 5] for i in range(4)]
)
def test_multi_docking_props(targets, receptor_process, scorer, n_generations=2):
    """Test the reward function runs for vina targets."""
    _, missed = receptor_process(targets)
    assert len(missed) == 0, f"Receptor {targets} could not be processed."
    property_filler = get_fill_completions(scorer.parse_whole_completion)
    prompts = [build_prompt(target) for target in targets] * n_generations
    smiles = [
        [propeties_csv.iloc[i]["smiles"]] for i in range(n_generations * len(targets))
    ]
    completions = [
        property_filler(s, "Here is a molecule: [SMILES] what are its properties?")
        for s in smiles
    ]
    rewards = scorer(prompts, completions)
    assert isinstance(rewards, (np.ndarray, list, torch.Tensor))
    rewards = torch.Tensor(rewards)
    assert not rewards.isnan().any()
    assert rewards.shape[0] == n_generations * len(targets)
