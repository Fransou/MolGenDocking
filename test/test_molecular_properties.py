from typing import List

import pytest
from tdc import single_pred
from rdkit.Chem import rdMolDescriptors

from mol_gen_docking.molecular_properties import (
    KNOWN_PROPERTIES,
    PROPERTIES_NAMES_SIMPLE,
    RDKITOracle,
    get_oracle,
)
from mol_gen_docking.logger import create_logger

logger = create_logger(__name__)

def is_rdkit_use(name:str):
    return name in KNOWN_PROPERTIES or name in PROPERTIES_NAMES_SIMPLE.values()


@pytest.fixture(
    params=[
        "hERG*Tox",
        pytest.param("BBB_Martins*ADME", marks=pytest.mark.skip),
        pytest.param("Caco2_Wang*ADME", marks=pytest.mark.skip)
    ],
    scope="module"
)
def smiles_data(request) -> List[str]:
    name, task_or = request.param.split("*")
    task_mod = getattr(single_pred, task_or)
    return task_mod(name=name).get_data()["Drug"].tolist()

@pytest.fixture(
    params= [
        prop for prop in dir(rdMolDescriptors) if ("Calc" in prop and (is_rdkit_use(prop)))
    ]
)
def rdkit_oracle(request) -> RDKITOracle:
    return RDKITOracle(name=request.param)

@pytest.fixture(
    params= KNOWN_PROPERTIES
)
def oracle(request: str):
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
    assert isinstance(props, list)
    assert len(props) == len(smiles_data)
    assert isinstance(props[0], float)


def test_vina(smiles_data):
    """
    Tests the oracle with vina
    """
    oracle = get_oracle("3pbl_docking")
    props = oracle(smiles_data)
    assert isinstance(props, list)
    assert len(props) == len(smiles_data)
