"""Reward functions for molecular optimization."""

import argparse
import os.path

from typing import List, Union, Optional, Callable, Any
from multiprocessing import Pool
import warnings

import pandas as pd
import numpy as np

from tdc.oracles import Oracle, oracle_names
from tdc.generation import MolGen

from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles

from tqdm import tqdm

from mol_gen_docking.utils.logger import create_logger

PROPERTIES_NAMES_SIMPLE = {
    "Inhibition against glycogen synthase kinase-3 beta": "GSK3B",
    "Inhibition against c-Jun N-terminal kinase-3": "JNK3",
    "Bioactivity against dopamine receptor D2": "DRD2",
    "Synthetic accessibility": "SA",
    "Quantitative estimate of drug-likeness": "QED",
    "Molecular Weight": "CalcExactMolWt",
    "Number of Aromatic Rings": "CalcNumAromaticRings",
    "Number of H-bond acceptors": "CalcNumHBA",
    "Number of H-bond donors": "CalcNumHBD",
    "Number of Rotatable Bonds": "CalcNumRotatableBonds",
    "Fraction of C atoms Sp3 hybridised": "CalcFractionCSP3",
    "Topological Polar Surface Area": "CalcTPSA",
    "Hall-Kier alpha": "CalcHallKierAlpha",
    "Hall-Kier kappa 1": "CalcKappa1",
    "Hall-Kier kappa 2": "CalcKappa2",
    "Hall-Kier kappa 3": "CalcKappa3",
    "Kier Phi": "CalcPhi",
}

KNOWN_PROPERTIES = [
    "logP",
] + list(PROPERTIES_NAMES_SIMPLE.keys())

property_csv_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "properties.csv"
)
if not os.path.exists(property_csv_path):
    # Raise a warning, the properties file is not found
    warnings.warn("The properties file is not found. Launch this file to generate it.")
    propeties_csv = pd.DataFrame(columns=["smiles"])
else:
    propeties_csv = pd.read_csv(property_csv_path)


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
        raise ValueError(f"Descriptor {self.name} not found in Rdkit.")

    def __call__(
        self, smi_mol: Union[str, Mol, List[str], List[Mol]]
    ) -> Union[float, List[float]]:
        """Get the descriptor value for the molecule."""
        if isinstance(smi_mol, list):
            if len(smi_mol) == 0:
                return []
            if isinstance(smi_mol[0], str):
                mols = [MolFromSmiles(smi) for smi in smi_mol]
            elif isinstance(smi_mol[0], Mol):
                mols = smi_mol
            else:
                raise ValueError(
                    "Input must be a list of SMILES strings or a list of RDKit molecule objects."
                )
            return [
                float(self.descriptor(mol)) if mol is not None else 0 for mol in mols
            ]

        if isinstance(smi_mol, str):
            mol = MolFromSmiles(smi_mol)
        elif isinstance(smi_mol, Mol):
            mol = smi_mol
        else:
            raise ValueError(
                "Input must be a SMILES string or a RDKit molecule object."
            )
        if mol is None:
            return 0
        return float(self.descriptor(mol))


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
        self.evaluator: Callable[[Any], Any] | Oracle | RDKITOracle = lambda x: None
        self.task_label = None

    def assign_evaluator(
        self, evaluator: Oracle | RDKITOracle, name: Optional[str] = None
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
    ) -> Union[float, List[float]]:
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

        score_list = np.array(score_list)
        if rescale:
            if self.name is not None and self.name in propeties_csv.columns:
                prop_typical_values = propeties_csv[self.name]
                # Rescale the values
                score_list = (score_list - prop_typical_values.quantile(0.01)) / (
                    prop_typical_values.quantile(0.99)
                    - prop_typical_values.quantile(0.01)
                )
            else:
                print(
                    "Typical values not found for the property. Returning the raw values."
                )
        return score_list


def get_oracle(oracle_name: str):
    """
    Get the Oracle object for the specified name.
    :param name: Name of the Oracle
    :return: OracleWrapper object
    """
    oracle_wrapper = OracleWrapper()
    oracle_name = PROPERTIES_NAMES_SIMPLE.get(oracle_name, oracle_name)
    if oracle_name.endswith("docking_vina") or oracle_name.lower() in oracle_names:
        print(oracle_name)
        oracle_wrapper.assign_evaluator(Oracle(name=oracle_name, ncpus=64), oracle_name)
    else:
        oracle_wrapper.assign_evaluator(RDKITOracle(oracle_name), oracle_name)
    return oracle_wrapper


if __name__ == "__main__":
    """Create a dataset with molecules and the property they could be optimizing."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default="ZINC",
        help="Name of the dataset to use for the generation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for the property calculation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers for the property calculation",
    )
    parser.add_argument("--sub-sample", type=int, default=None, help="Sub-sample size")

    args = parser.parse_args()

    molgen = MolGen(name=args.name).get_data()
    if args.sub_sample:
        molgen = molgen.sample(args.sub_sample)
    # Limits the dataframe to a multiple of the batch size
    molgen = molgen.iloc[: len(molgen) - (len(molgen) % args.batch_size)]

    smiles_batches = [
        molgen["smiles"].tolist()[i * args.batch_size : (i + 1) * args.batch_size]
        for i in range(len(molgen) // args.batch_size)
    ]

    for i_name, oracle_name in enumerate(KNOWN_PROPERTIES):
        oracle_name = PROPERTIES_NAMES_SIMPLE.get(oracle_name, oracle_name)
        oracle = get_oracle(oracle_name)

        p_bar = tqdm(
            total=len(molgen),
            desc=f"[{i_name}/{len(KNOWN_PROPERTIES)}] Calculating {oracle_name}",
        )

        def get_property(batch: List[str]) -> dict:
            """Get the property for a batch of SMILES strings."""
            props = oracle(batch)
            return {smi: prop for smi, prop in zip(batch, props)}

        pool = Pool(args.num_workers)
        props = tqdm(
            pool.imap_unordered(get_property, smiles_batches),
            total=len(smiles_batches),
            desc=f"[{i_name}/{len(KNOWN_PROPERTIES)}] | Calculating {oracle_name}",
        )

        props = {k: v for d in props for k, v in d.items()}
        molgen[oracle_name] = molgen["smiles"].map(props)

    print(molgen.sample(10))

    molgen.to_csv("properties.csv", index=False)
