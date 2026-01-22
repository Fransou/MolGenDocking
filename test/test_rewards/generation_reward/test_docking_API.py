import os
from typing import Dict, List

import numpy as np
import pytest
import torch

from mol_gen_docking.data.meeko_process import ReceptorProcess
from mol_gen_docking.reward.molecular_verifier import MolecularVerifier


@pytest.fixture(scope="module", params=[4, 8, 16])
def exhaustiveness(request):
    """Fixture for testing different exhaustiveness levels."""
    return request.param


@pytest.fixture(scope="module")
def scorer(has_gpu, data_path, exhaustiveness):
    """Create a RewardScorer configured for docking tests."""
    if not has_gpu:
        return MolecularVerifier(
            data_path,
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
        return MolecularVerifier(
            data_path,
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
def receptor_process(has_gpu, data_path):
    """Create a ReceptorProcess for GPU or return a dummy function for CPU."""
    if not has_gpu:
        return lambda x: ([], [])
    else:
        rp = ReceptorProcess(data_path=data_path, pre_process_receptors=True)

        def wrapped_fn(r: str | List[str]):
            if isinstance(r, str):
                r = [r]
            return rp.process_receptors(r, allow_bad_res=True)

        return wrapped_fn


def build_metadatas(
    property: str | List[str], obj: str = "maximize"
) -> Dict[str, List[str] | List[float]]:
    """Build metadata dictionary for reward scoring."""
    if isinstance(property, str):
        properties = [property]
    else:
        properties = property
    return {
        "properties": properties,
        "objectives": [obj] * len(properties),
        "target": [0.0] * len(properties),
    }


# =============================================================================
# Receptor Processing Tests
# =============================================================================


class TestReceptorProcessing:
    """Tests for receptor processing."""

    def test_receptor_process(self, receptor_process, docking_targets):
        """Test receptor processing for all docking targets."""
        _, missed_targets = receptor_process(docking_targets)
        assert len(missed_targets) == 0, (
            f"Receptors {missed_targets} could not be processed."
        )


# =============================================================================
# Single Property Docking Tests
# =============================================================================


class TestSinglePropertyDocking:
    """Tests for docking with single properties."""

    def test_docking_props(
        self,
        docking_targets,
        scorer,
        receptor_process,
        properties_csv,
        n_generations=4,
    ):
        """Test the reward function runs for docking targets."""
        for target in docking_targets:
            target = [target, "CalcPhi"]
            metadatas = [build_metadatas(target)] * n_generations
            smiles = [[properties_csv.iloc[i]["smiles"]] for i in range(n_generations)]
            completions = [
                f"Here is a molecule: <answer> {s} </answer> what are its properties?"
                for s in smiles
            ]
            rewards = scorer(completions, metadatas)[0]
            assert isinstance(rewards, (np.ndarray, list, torch.Tensor))
            rewards = torch.Tensor(rewards)
            assert not rewards.isnan().any()
            assert rewards.shape[0] == n_generations


# =============================================================================
# Multiple Property Docking Tests
# =============================================================================


class TestMultiplePropertyDocking:
    """Tests for docking with multiple properties."""

    def test_multi_docking_props(
        self,
        docking_targets,
        receptor_process,
        scorer,
        properties_csv,
        n_generations=2,
    ):
        """Test the reward function runs for multiple docking targets."""
        for targets in docking_targets[::2]:
            _, missed = receptor_process(targets)
            assert len(missed) == 0, f"Receptor {targets} could not be processed."
            metadatas = [build_metadatas(targets)] * n_generations
            smiles = [[properties_csv.iloc[i]["smiles"]] for i in range(n_generations)]
            completions = [
                f"Here is a molecule: <answer> {s} </answer> what are its properties?"
                for s in smiles
            ]
            rewards = scorer(completions, metadatas)[0]
            assert isinstance(rewards, (np.ndarray, list, torch.Tensor))
            rewards = torch.Tensor(rewards)
            assert not rewards.isnan().any()
            assert rewards.shape[0] == n_generations
