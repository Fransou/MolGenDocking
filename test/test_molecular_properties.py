from typing import List

import pytest
from tdc.single_pred import ADME
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
        "hERG",
        pytest.param("BBB_Martins", marks=pytest.mark.slow),
        pytest.param("Caco2_Wang", marks=pytest.mark.slow)
    ],
)
def smiles_data(param: str) -> List[str]:
    return ADME(name=param).get_data()["Drug"].tolist()

@pytest.fixture(
    params= [
        prop for prop in dir(rdMolDescriptors) if ("Calc" in prop and (is_rdkit_use(prop)))
    ]
)
def rdkit_oracle(param:str) -> RDKITOracle:
    return RDKITOracle(name=param)

@pytest.fixture(
    params= KNOWN_PROPERTIES
)
def oracle(param: str):
    return get_oracle(param)



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
