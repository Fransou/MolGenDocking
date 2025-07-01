import os
from typing import List

import numpy as np
import pytest

from mol_gen_docking.reward.oracle_wrapper import get_oracle

from .utils import DATA_PATH, DOCKING_PROP_LIST, PROP_LIST, PROPERTIES_NAMES_SIMPLE, propeties_csv


def is_rdkit_use(name: str):
    return name in PROP_LIST or name in PROPERTIES_NAMES_SIMPLE.values()


@pytest.fixture(
    params=[
        propeties_csv.sample(np.random.randint(1, 5)),
    ],
    scope="module",
)
def smiles_data(request) -> List[str]:
    df = request.param
    return df["smiles"].tolist(), df



@pytest.fixture(params=PROP_LIST)
def oracle(request):
    return get_oracle(
        request.param,
        path_to_data=DATA_PATH,
        property_name_mapping=PROPERTIES_NAMES_SIMPLE,
        docking_target_list=DOCKING_PROP_LIST,
    )




def test_oracles(oracle, smiles_data):
    """
    Test the RDKITOracle class
    """
    smiles, df = smiles_data
    props = oracle(smiles, rescale=False)

    oracle_name = oracle.name.split("/")[-1]

    assert isinstance(props, list) or isinstance(props, np.ndarray)
    assert len(props) == len(smiles)
    assert isinstance(props[0], float)
    assert (np.array(props) == df[oracle_name].values).all()




@pytest.mark.skipif(os.system("vina --help") == 32512, reason="requires vina")
def test_vina(smiles_data):
    """
    Tests the oracle with vina, only checks it runs correctly.
    """
    smiles, _ = smiles_data
    oracle = get_oracle(
        "3pbl_docking",
        path_to_data=DATA_PATH,
        property_name_mapping=PROPERTIES_NAMES_SIMPLE,
        docking_target_list=DOCKING_PROP_LIST,
        ncpu=1,
        exhaustiveness=1,
    )
    props = oracle(smiles)
    assert len(props) == len(smiles)
