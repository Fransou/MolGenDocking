"""Reward functions for molecular optimization."""

from typing import List, Union, Optional, Callable, Any
import warnings
import numpy as np


from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import MolToSmiles


from mol_gen_docking.utils.logger import create_logger

from mol_gen_docking.reward.oracles.utils import propeties_csv, oracles_not_to_rescale


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
        debug: bool = False,
    ):
        self.logger = create_logger(
            __name__ + "/" + self.__class__.__name__,
            level="DEBUG" if debug else "WARNING",
        )
        self.name: None | str = None
        self.evaluator: Callable[[Any], Any] = lambda x: None
        self.task_label = None

    def assign_evaluator(
        self, evaluator: Callable[[Any], Any], name: Optional[str] = None
    ):
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
        out: float | List[float] | None = self.evaluator(inp)
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
        out = self.evaluator(inps)
        if isinstance(out, list):
            return out
        raise ValueError(f"{out} is a {type(out)}, not a list")

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

        score_arr: np.ndarray = np.array(score_list)
        print(self.name)
        if rescale:
            if (
                self.name not in oracles_not_to_rescale
                and self.name in propeties_csv.columns
            ):
                prop_typical_values = propeties_csv[self.name]
                # Rescale the values
                score_arr = (score_arr - prop_typical_values.quantile(0.01)) / (
                    prop_typical_values.quantile(0.99)
                    - prop_typical_values.quantile(0.01)
                )
            elif self.name is not None and "docking" in self.name:
                # Rescale the values by adding 11, dividing by 10
                # a docking score of -10 is therefore a 0.1 and -7 is 0.4
                score_arr = (score_arr + 11) / 10
            else:
                warnings.warn(
                    "Typical values not found for the property. Returning the raw values."
                )
        print(score_arr)
        return score_arr
