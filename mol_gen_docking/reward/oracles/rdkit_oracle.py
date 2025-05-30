"""Reward functions for molecular optimization."""

from typing import List, Union

from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdmolfiles import MolFromSmiles


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

    def __call__(self, smi_mol: Union[str, List[str]]) -> Union[float, List[float]]:
        """Get the descriptor value for the molecule."""
        if isinstance(smi_mol, list):
            if len(smi_mol) == 0:
                return []
            if isinstance(smi_mol[0], str):
                mols = [MolFromSmiles(smi) for smi in smi_mol]
            else:
                raise ValueError("Input must be a list of SMILES strings.")
            return [
                float(self.descriptor(mol)) if mol is not None else 0 for mol in mols
            ]

        if isinstance(smi_mol, str):
            mol = MolFromSmiles(smi_mol)
        else:
            raise ValueError("Input must be a SMILES string.")
        if mol is None:
            return 0
        return float(self.descriptor(mol))
