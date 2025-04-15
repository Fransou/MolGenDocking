from typing import List
import os

import numpy as np
import pytest
from tdc import single_pred
from rdkit.Chem import rdMolDescriptors

from mol_gen_docking.reward.oracles import PROPERTIES_NAMES_SIMPLE, get_oracle
from mol_gen_docking.reward.oracles.rdkit_oracle import RDKITOracle
from mol_gen_docking.utils.logger import create_logger

logger = create_logger(__name__)

PROP_LIST = [
    k
    for k in PROPERTIES_NAMES_SIMPLE.keys()
    if "docking" not in PROPERTIES_NAMES_SIMPLE[k]
]


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
    return get_oracle(request.param)


def test_RDKITOracle(rdkit_oracle, smiles_data):
    """
    Test the RDKITOracle class
    """
    props = rdkit_oracle(smiles_data)
    assert isinstance(props, list)
    assert len(props) == len(smiles_data)
    assert isinstance(props[0], float)


def test_oracles(oracle, smiles_data):
    """
    Test the RDKITOracle class
    """
    props = oracle(smiles_data)
    assert isinstance(props, list) or isinstance(props, np.ndarray)
    assert len(props) == len(smiles_data)
    assert isinstance(props[0], float)


@pytest.mark.skipif(os.system("vina --help") == 32512, reason="requires vina")
def test_vina(smiles_data):
    """
    Tests the oracle with vina
    """
    oracle = get_oracle("3pbl_docking")
    props = oracle(smiles_data[:3])
    assert isinstance(props, list)
    assert len(props) == len(smiles_data)
