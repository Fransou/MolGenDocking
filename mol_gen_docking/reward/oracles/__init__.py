from tdc.oracles import Oracle, oracle_names

from .utils import (
    propeties_csv,
    PROPERTIES_NAMES_SIMPLE,
)
from .oracle_wrapper import OracleWrapper


def get_oracle(oracle_name: str):
    """
    Get the Oracle object for the specified name.
    :param name: Name of the Oracle
    :return: OracleWrapper object
    """
    oracle_wrapper = OracleWrapper()
    oracle_name = PROPERTIES_NAMES_SIMPLE.get(oracle_name, oracle_name)
    if oracle_name.endswith("docking_vina") or oracle_name.endswith("docking"):
        from .docking_oracle import PyscreenerOracle

        oracle_wrapper.assign_evaluator(PyscreenerOracle(oracle_name, ncpus=16))
    elif oracle_name.lower() in oracle_names:
        oracle_wrapper.assign_evaluator(Oracle(name=oracle_name, ncpus=4), oracle_name)
    else:
        from .rdkit_oracle import RDKITOracle

        oracle_wrapper.assign_evaluator(RDKITOracle(oracle_name), oracle_name)
    return oracle_wrapper


# Add the imports to __all__
__all__ = [
    "propeties_csv",
    "PROPERTIES_NAMES_SIMPLE",
    "get_oracle",
]
