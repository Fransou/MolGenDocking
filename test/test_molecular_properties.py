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


@pytest.mark.parametrize(
    "rdkit_poss",
    [rdkit_poss for rdkit_poss in dir(rdMolDescriptors) if "Calc" in rdkit_poss],
)
def test_RDKITOracle(rdkit_poss: str):
    """
    Test the RDKITOracle class
    """
    data = ADME(name="BBB_Martins").get_data()
    smiles = data["Drug"].tolist()[0:10]

    is_prop_considered = (
        rdkit_poss in KNOWN_PROPERTIES or rdkit_poss in PROPERTIES_NAMES_SIMPLE.values()
    )
    oracle = RDKITOracle(rdkit_poss)
    try:
        print(type(smiles))
        props = oracle(smiles)
        assert isinstance(props, list)
        assert len(props) == len(smiles)
        assert isinstance(props[0], float)
    except Exception as e:
        if is_prop_considered:
            logger.error(f"Error for property {rdkit_poss}: {e}")
            raise e
        else:
            logger.info(
                f"Property {rdkit_poss} is not considered and failed to compute"
            )
    if not is_prop_considered:
        logger.warning(
            f"Property {rdkit_poss} is not considered but successfully computed"
        )


@pytest.mark.parametrize("property_name", KNOWN_PROPERTIES)
def test_oracles(property_name: str):
    """
    Test the RDKITOracle class
    Args:
        task_name: str: The name of the task to test
    """
    data = ADME(name="BBB_Martins").get_data()
    smiles = data["Drug"].tolist()[0:10]

    oracle = get_oracle(property_name)
    try:
        props = oracle(smiles)
        assert isinstance(props, list)
        assert len(props) == len(smiles)
        assert isinstance(props[0], float)
    except Exception as e:
        logger.error(f"Error for property {property_name}")
        raise e
