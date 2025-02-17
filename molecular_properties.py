"""Reward functions for molecular optimization."""

import argparse
import logging
from typing import List, Union

from tdc import Oracle
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem

KNOWN_PROPERTIES = [
    "JNK3",
    "DRD2",
    "GSK3B",
    "SA",
    "QED",
    "logP",
    "Molecular Weight",
    "Num Aromatic Rings",
    "Num H-bond acceptors",
    "Num H-bond donors",
    "Num Rotatable Bonds",
    "TPSA",
]

PROPERTIES_NAMES_SIMPLE = {
    "Molecular Weight": "CalcExactMolWt",
    "Num Aromatic Rings": "CalcNumAromaticRings",
    "Num H-bond acceptors": "CalcNumHBA",
    "Num H-bond donors": "CalcNumHBD",
    "Num Rotatable Bonds": "CalcNumRotatableBonds",
    "TPSA": "CalcTPSA",
}


def create_logger(name: str, level: str = "INFO"):
    """
    Create a logger object with the specified name and level.
    :param name: Name of the logger
    :param level: Level of the logger
    :return: Logger object
    """
    logger = logging.getLogger(name)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s | %(name)s | %(levelname)s] %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(level)
    return logger


class RDKITOracle:
    """
    Class implementing all Rdkit descriptors.
    """

    def __init__(self, name: str):
        self.name = name
        self.descriptor = self.get_descriptor()

    def get_descriptor(self):
        """Get the descriptor from Rdkit."""
        if hasattr(rdMolDescriptors, self.name):
            return getattr(rdMolDescriptors, self.name)
        else:
            raise ValueError(f"Descriptor {self.name} not found in Rdkit.")

    def __call__(self, smi_mol: Union[str, Chem.Mol]) -> float:
        """Get the descriptor value for the molecule."""
        if isinstance(smi_mol, str):
            mol = Chem.MolFromSmiles(smi_mol)
        elif isinstance(smi_mol, Chem.Mol):
            mol = smi_mol
        else:
            raise ValueError(
                "Input must be a SMILES string or a RDKit molecule object."
            )
        return self.descriptor(mol)


class OracleWrapper:
    """
    Code based on the Oracle class from: https://github.com/wenhao-gao/mol_opt/blob/main/main/optimizer.py#L50

    Wraps the Oracle class from TDC, enabling sample efficient optimization of molecular properties.

    Args:
        args: Namespace containing the arguments for the optimization process.
            - debug (: bool) : Debug mode.
        mol_buffer: Dictionary containing the molecules and their properties.
    """

    def __init__(
        self,
        debug: bool = False,
    ):
        self.logger = create_logger(
            __name__ + "/" + self.__class__.__name__,
            level="DEBUG" if debug else "WARNING",
        )
        self.name = None
        self.evaluator = None
        self.task_label = None

    def assign_evaluator(self, evaluator: Union[Oracle, RDKITOracle]):
        self.evaluator = evaluator

    def score_smi(self, smi: str) -> float:
        """
        Function to score one molecule

        Arguments:
            smi: One SMILES string represents a molecule.

        Return:
            score: a float represents the property of the molecule.
        """
        if smi is None:
            return 0
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            return 0
        smi = Chem.MolToSmiles(mol)
        return float(self.evaluator(smi))

    def __call__(self, smis: Union[str, List[str]]) -> Union[float, List[float]]:
        """
        Score
        """
        if isinstance(smis, list):
            score_list = []
            for smi in smis:
                score_list.append(self.score_smi(smi))

        elif isinstance(smis, str):
            score_list = self.score_smi(smis)

        else:
            raise ValueError(
                "Input must be a SMILES string or a list of SMILES strings."
            )
        return score_list


def get_oracle(oracle: str):
    """
    Get the Oracle object for the specified name.
    :param name: Name of the Oracle
    :return: OracleWrapper object
    """
    oracle_wrapper = OracleWrapper()
    oracle = PROPERTIES_NAMES_SIMPLE.get(oracle, oracle)

    if oracle.endswith("docking"):
        oracle_wrapper.assign_evaluator(
            Oracle(name=oracle, ncpus=1),
        )
    else:
        try:
            oracle_wrapper.assign_evaluator(
                Oracle(name=oracle),
            )
        except ValueError:
            oracle_wrapper.assign_evaluator(
                RDKITOracle(oracle),
            )
    return oracle_wrapper


if __name__ == "__main__":
    smis = [
        "O=C(NCCCc1ccccc1)NCCc1cccs1",
        "CCCCOc1ccccc1C[C@H]1COC(=O)[C@@H]1Cc1ccc(Cl)c(Cl)c1",
        "O=c1[nH]nc2n1-c1ccc(OCc3ccc(F)cc3)cc1CCC2",
        "CCN1CCN(c2ccc(C3=CC4(CCc5cc(O)ccc54)c4ccc(O)cc43)cc2)CC1",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", type=str, default=smis)
    parser.add_argument("--oracle", type=str, default="")
    parser.add_argument("--max-oracle-calls", type=int, default=100)
    parser.add_argument("--freq-log", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    arguments = parser.parse_args()

    oracle = OracleWrapper()
    if not arguments.oracle == "":
        oracle = get_oracle(arguments.oracle)
        rewards = oracle(smis)
        print(rewards)
    else:
        for oracle_name in KNOWN_PROPERTIES:
            print(oracle_name)
            oracle = get_oracle(oracle_name)
            rewards = oracle(smis)
            print(oracle_name, rewards)
