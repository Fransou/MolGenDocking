"""Reward functions for molecular optimization."""

import logging
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Union

import numpy as np
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles
from tdc.oracles import Oracle, oracle_names

from mol_gen_docking.reward.property_utils import rescale_property_values


class OracleWrapper:
    """
    Code based on the Oracle class from:
     https://github.com/wenhao-gao/mol_opt/blob/main/main/optimizer.py#L50

    Wraps the Oracle class from TDC, enabling sample efficient
    optimization of molecular properties.

    Args:
        args: Namespace containing the arguments for the
         optimization process.
        debug (: bool) : Debug mode.
    """

    def __init__(
        self,
        is_docking: bool = False,
        debug: bool = False,
        internal_memory: int = 1000000,
    ):
        self.logger = logging.getLogger(
            __name__ + "/" + self.__class__.__name__,
        )
        self.is_docking = is_docking
        self.name: str = ""
        self.evaluator: Callable[[Any], Any] = lambda x: None
        self.task_label = None
        self.memory: OrderedDict[str, float] = OrderedDict()
        self.internal_memory = internal_memory

    def assign_evaluator(self, evaluator: Callable[[Any], Any], name: str) -> None:
        """Assign the evaluator to the OracleWrapper."""
        self.evaluator = evaluator
        self.name = name

    def score(self, inp: Union[str, Mol]) -> float:
        """
        Function to score one molecule

        Arguments:
            inp: One SMILES string represents a molecule.

        Return:
            score: a float represents the property of the molecule.
        """
        if inp is None:
            return 0
        if isinstance(inp, Mol):
            inp = MolToSmiles(inp)
        elif not isinstance(inp, str):
            raise ValueError(f"{inp} cannot be transformed into mol")

        if inp in self.memory:
            return self.memory[inp]

        # Get the canonical smiles
        inp = MolToSmiles(MolFromSmiles(inp), canonical=True)
        out: float = self.evaluator(inp)

        self.memory[inp] = out
        if len(self.memory) > self.internal_memory:
            self.memory.popitem(last=False)

        if isinstance(out, float):
            return out
        raise ValueError(f"{out} is a {type(out)}, not a float")

    def score_smiles_list(self, inps: List[str]) -> List[float]:
        """
        Function to score a list of molecules

        Arguments:
            inps: A list of SMILES strings represents molecules.

        Return:
            score_list: a list of floats represents the properties of the molecules.
        """
        # Get the list of smiles strings that are not in memory
        inps = [MolToSmiles(MolFromSmiles(inp), canonical=True) for inp in inps]
        inps_to_score = [inp for inp in inps if inp not in self.memory]
        local_scores = {inp: self.memory[inp] for inp in inps if inp in self.memory}

        if len(inps_to_score) > 0:
            # Evaluate the new smiles
            res_not_in_mem = self.evaluator(inps_to_score)
            # Store the results in memory
            for inp, score in zip(inps_to_score, res_not_in_mem):
                self.memory[inp] = score
                local_scores[inp] = score
        out: List[float] = [local_scores[inp] for inp in inps]
        if len(out) != len(inps):
            raise ValueError(
                f"Output length {len(out)} does not match input length {len(inps)}"
            )
        if len(self.memory) > self.internal_memory:
            # Remove the oldest entries if memory exceeds the limit
            while len(self.memory) > self.internal_memory:
                self.memory.popitem(last=False)
        return out

    def __call__(
        self, smis: Union[str, List[str]], rescale: bool = False
    ) -> Union[float, np.ndarray]:
        """
        Score
        """
        if isinstance(smis, list):
            score_list = self.score_smiles_list(smis)
        elif isinstance(smis, str):
            score_list = [self.score(smis)]
        else:
            raise ValueError(
                "Input must be a SMILES string or a list of SMILES strings."
            )

        if rescale:
            score_list = [
                rescale_property_values(
                    self.name.split("/")[-1], score, self.is_docking
                )
                for score in score_list
            ]

        score_arr: np.ndarray = np.array(score_list)
        return score_arr


def get_oracle(
    oracle_name: str,
    path_to_data: str = "",
    property_name_mapping: Dict[str, str] = {},
    docking_target_list: List[str] = [],
    docking_oracle: str = "pyscreener",
    **kwargs: Any,
) -> OracleWrapper:
    """
    Get the Oracle object for the specified name.
    :param name: Name of the Oracle
    :param property_name_mapping: Mapping of property names to Oracle names
    :param docking_target_list: List of docking targets
    :return: OracleWrapper object
    """

    oracle_name = oracle_name.replace(".", "")
    oracle_name = property_name_mapping.get(oracle_name, oracle_name)
    if oracle_name in docking_target_list:
        if docking_oracle == "pyscreener":
            from mol_gen_docking.reward.oracles.pyscreener_oracle import (
                PyscreenerOracle,
            )

            oracle_wrapper = OracleWrapper(is_docking=True)
            oracle_wrapper.assign_evaluator(
                PyscreenerOracle(oracle_name, path_to_data=path_to_data, **kwargs),
                f"docking_prop/{oracle_name}",
            )
        elif docking_oracle == "soft_docking":
            from mol_gen_docking.reward.oracles.vinagpu_oracle import (
                DockingMoleculeGpuOracle,
            )

            oracle_wrapper = OracleWrapper(is_docking=True)
            oracle_wrapper.assign_evaluator(
                DockingMoleculeGpuOracle(
                    path_to_data=path_to_data, receptor_name=oracle_name, **kwargs
                ),
                f"docking_prop/{oracle_name}",
            )

    elif oracle_name.lower() in oracle_names:
        oracle_wrapper = OracleWrapper()
        oracle_wrapper.assign_evaluator(
            Oracle(name=oracle_name, **kwargs), f"tdc/{oracle_name}"
        )

    else:
        from mol_gen_docking.reward.oracles.rdkit_oracle import RDKITOracle

        oracle_wrapper = OracleWrapper()
        oracle_wrapper.assign_evaluator(
            RDKITOracle(oracle_name), f"rdkit/{oracle_name}"
        )

    return oracle_wrapper
