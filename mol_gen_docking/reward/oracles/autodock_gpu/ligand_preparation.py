from typing import List

from rdkit import Chem

try:
    from scrubber import Scrub  # Old name util v0.1.1
except ImportError:
    from molscrub import Scrub  # New name
import os
from dataclasses import dataclass
from time import sleep

from meeko import MoleculePreparation, PDBQTWriterLegacy


@dataclass(frozen=True)
class Ligand:
    smiles: str
    name: str
    out_dir: str
    max_attempts: int = 4


def process_ligand(ligand: Ligand, scrub_instance: Scrub) -> List[str]:
    """
    Process a single ligand by scrubbing and preparing it for docking.

    Args:
       args (tuple):

    Returns:
       list: A list of output file paths generated during the process.

    ### Credits to:
    """
    output_paths = []
    # Convert SMILES string to molecule object
    mol = Chem.MolFromSmiles(ligand.smiles)
    if mol is None:
        print(f"[SMILES] Failed to parse: {ligand.smiles}")
        return []

    # Apply scrub with retry in case of failure, primarily with conformer generation
    for attempt in range(1, ligand.max_attempts + 1):
        try:
            mol_states = list(scrub_instance(mol))
            break
        except RuntimeError as e:
            print(
                f"[Scrub Run {attempt}/{ligand.max_attempts}] Failed on {ligand.name}, molecule Smiles {ligand.smiles}: {e}"
            )
            sleep(0.1)
    else:
        print(f"[Scrub Run] Gave up on {ligand.name}")
        return []

    # Initialize MoleculePreparation instance
    mk_prep = MoleculePreparation()
    counter = 0  # Counter for multiple outcomes from the same ligand SMILES

    # Prepare the molecules and generate the pdbqt files
    for mol_state in mol_states:
        molsetup_list = mk_prep.prepare(mol_state)
        for molsetup in molsetup_list:
            pdbqt_string, success, error_msg = PDBQTWriterLegacy.write_string(molsetup)
            if success:
                path = os.path.join(ligand.out_dir, f"{ligand.name}_{counter}.pdbqt")
                with open(path, "w") as f:
                    f.write(pdbqt_string)
                output_paths.append(path)
                counter += 1
            else:
                print(f"[PDBQT] Write failed for {ligand.name}: {error_msg}")

    return output_paths


def process_smiles_list(
    smiles: List[str], target_name: str, out_dir: str = "tmp", max_attempts: int = 4
) -> List[List[str]]:
    os.makedirs(out_dir, exist_ok=True)
    scrub = Scrub(ph_low=7.4, ph_high=7.4, skip_tautomers=True)
    ligands = [
        Ligand(
            smiles=s,
            name=target_name + f"_{i}",
            out_dir=out_dir,
            max_attempts=max_attempts,
        )
        for i, s in enumerate(smiles)
    ]
    return [process_ligand(ligand, scrub_instance=scrub) for ligand in ligands]
