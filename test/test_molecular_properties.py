import os
from typing import List

import numpy as np
import pytest
from rdkit.Chem import rdMolDescriptors
from tdc import single_pred

from mol_gen_docking.reward.oracle_wrapper import get_oracle
from mol_gen_docking.reward.oracles.rdkit_oracle import RDKITOracle

from .utils import DATA_PATH, DOCKING_PROP_LIST, PROP_LIST, PROPERTIES_NAMES_SIMPLE


def is_rdkit_use(name: str):
    return name in PROP_LIST or name in PROPERTIES_NAMES_SIMPLE.values()


@pytest.fixture(
    params=[
        "hERG*Tox",
        pytest.param(
            "BBB_Martins*ADME",
            marks=pytest.mark.skipif(
                os.environ.get("TEST_LONG", "False") == "False", reason="Fast Test"
            ),
        ),
        pytest.param(
            "Caco2_Wang*ADME",
            marks=pytest.mark.skipif(
                os.environ.get("TEST_LONG", "False") == "False", reason="Fast Test"
            ),
        ),
    ],
    scope="module",
)
def smiles_data(request) -> List[str]:
    name, task_or = request.param.split("*")
    task_mod = getattr(single_pred, task_or)
    return task_mod(name=name).get_data().sample(100)["Drug"].tolist()


@pytest.fixture(
    params=[
        prop
        for prop in dir(rdMolDescriptors)
        if ("Calc" in prop and (is_rdkit_use(prop)))
    ]
)
def rdkit_oracle(request) -> RDKITOracle:
    return RDKITOracle(name=request.param)


@pytest.fixture(params=PROP_LIST)
def oracle(request):
    return get_oracle(
        request.param,
        path_to_data=DATA_PATH,
        property_name_mapping=PROPERTIES_NAMES_SIMPLE,
        docking_target_list=DOCKING_PROP_LIST,
    )


def test_RDKITOracle(rdkit_oracle, smiles_data):
    """
    Test the RDKITOracle class
    """
    props = rdkit_oracle(smiles_data)
    assert isinstance(props, list)
    assert len(props) == len(smiles_data)
    assert isinstance(props[0], float)
    props = np.array(props)
    props_solo = np.array([rdkit_oracle(smi) for smi in smiles_data])
    assert np.isclose(props, props_solo).all()


def test_oracles(oracle, smiles_data):
    """
    Test the RDKITOracle class
    """
    props = oracle(smiles_data)
    assert isinstance(props, list) or isinstance(props, np.ndarray)
    assert len(props) == len(smiles_data)
    assert isinstance(props[0], float)


@pytest.mark.skipif(os.system("qvina --help") == 32512, reason="requires vina")
def test_vina(smiles_data):
    """
    Tests the oracle with vina
    """
    oracle = get_oracle(
        "3pbl_docking",
        path_to_data=DATA_PATH,
        property_name_mapping=PROPERTIES_NAMES_SIMPLE,
        docking_target_list=DOCKING_PROP_LIST,
        ncpu=1,
        exhaustiveness=1,
    )
    props = oracle(smiles_data)
    assert len(props) == len(smiles_data)
