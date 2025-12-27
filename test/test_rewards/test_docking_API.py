import os
import time as time
from typing import Dict, List

import numpy as np
import pytest
import torch

from mol_gen_docking.data.gen_dataset import (
    DatasetConfig,
)
from mol_gen_docking.data.meeko_process import ReceptorProcess
from mol_gen_docking.reward.rl_rewards import RewardScorer

from .utils import (
    DATA_PATH,
    DOCKING_PROP_LIST,
    fill_completion,
    propeties_csv,
)

cfg = DatasetConfig(data_path=DATA_PATH)
props_to_eval = DOCKING_PROP_LIST[:64]


@pytest.fixture(scope="module", params=[4, 8, 16])
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
                n_cpu=int(os.environ.get("N_CPUS_DOCKING", exhaustiveness)),
                exhaustiveness=exhaustiveness,
                docking_oracle="autodock_gpu",
                vina_mode="autodock_gpu_128wi",
            ),
        )


@pytest.fixture(scope="module")
def receptor_process(has_gpu):
    if not has_gpu:
        return lambda x: ([], [])
    else:
        rp = ReceptorProcess(data_path=DATA_PATH, pre_process_receptors=True)

        def wrapped_fn(r: str | List[str]):
            if isinstance(r, str):
                r = [r]
            return rp.process_receptors(r, allow_bad_res=True)

        return wrapped_fn


def build_metadatas(
    property: str | List[str], obj: str = "maximize"
) -> Dict[str, List[str] | List[float]]:
    if isinstance(property, str):
        properties = [property]
    else:
        properties = property
    return {
        "properties": properties,
        "objectives": [obj] * len(properties),
        "target": [0.0] * len(properties),
    }


def test_receptor_process(receptor_process):
    """Test receptor processing."""
    _, missed_targets = receptor_process(props_to_eval)
    assert len(missed_targets) == 0, (
        f"Receptors {missed_targets} could not be processed."
    )


@pytest.mark.parametrize("target", props_to_eval)
def test_docking_props(target, scorer, receptor_process, n_generations=4):
    """Test the reward function runs for vina targets."""
    target = [target, "CalcPhi"]
    metadatas = [build_metadatas(target)] * n_generations
    smiles = [[propeties_csv.iloc[i]["smiles"]] for i in range(n_generations)]
    completions = [
        fill_completion(s, "Here is a molecule: [SMILES] what are its properties?")
        for s in smiles
    ]
    rewards = scorer(completions, metadatas)[0]
    assert isinstance(rewards, (np.ndarray, list, torch.Tensor))
    rewards = torch.Tensor(rewards)
    assert not rewards.isnan().any()
    assert rewards.shape[0] == n_generations


@pytest.mark.parametrize(
    "targets", [props_to_eval[i * 5 : (i + 1) * 5] for i in range(4)]
)
def test_multi_docking_props(targets, receptor_process, scorer, n_generations=2):
    """Test the reward function runs for vina targets."""
    _, missed = receptor_process(targets)
    assert len(missed) == 0, f"Receptor {targets} could not be processed."
    metadatas = [build_metadatas(targets)] * n_generations
    smiles = [[propeties_csv.iloc[i]["smiles"]] for i in range(n_generations)]
    completions = [
        fill_completion(s, "Here is a molecule: [SMILES] what are its properties?")
        for s in smiles
    ]
    rewards = scorer(completions, metadatas)[0]
    assert isinstance(rewards, (np.ndarray, list, torch.Tensor))
    rewards = torch.Tensor(rewards)
    assert not rewards.isnan().any()
    assert rewards.shape[0] == n_generations
