from prody import fetchPDB, parsePDB, writePDB, writePDBStream
from meeko import MoleculePreparation, ResidueChemTemplates
from meeko import Polymer
from meeko import PDBQTWriterLegacy
import numpy as np
from meeko.gridbox import get_gpf_string
from io import StringIO

def get_clean_chainA(pdb_id: str, to_obj: bool = True, to_file: str = None) -> str:
   # Fetch PDB and parse structure
   pdb_path = fetchPDB(pdb_id, folder='.', compressed=False)
   structure = parsePDB(pdb_path, altloc='first')

   # Select protein atoms only in chain A, excluding hetatoms (e.g., ligands, water)
   # keeps only the first listed conformer for each atom
   chainA = structure.select('protein and chain A')

   if chainA is None:
      raise ValueError(f"No protein atoms found in chain A for PDB {pdb_id}")

   if to_obj:
      # Return the cleaned chain A object
      return chainA

   if to_file:
      writePDB(to_file, chainA)

   # Get PDB string as well
   pdb_io = StringIO()
   writePDBStream(pdb_io, chainA)
   return pdb_io.getvalue()

def get_center_of_residue(pdb_id: str, chain_id: str, resname: str):
   # Fetch and parse structure with only the first altloc
   pdb_path = fetchPDB(pdb_id, folder='.', compressed=False)
   structure = parsePDB(pdb_path, altloc='first')

   # Find the residue
   sel_str = f"chain {chain_id} and resname {resname}"
   residue = structure.select(sel_str)

   if residue is None:
      raise ValueError(f"Residue {resname} in chain {chain_id} not found in PDB {pdb_id}")

   # Compute center of mass of the residue
   center = residue.getCoords().mean(axis=0)

   return center


# Getting the PDB string for 4EY7 and preprocessing it with ProDy
pdb_str = {}
for pdb_id in ['4EY7']:
   pdb_str[pdb_id] = get_clean_chainA(pdb_id, to_file=f"{pdb_id}_chainA.pdb")

# Constructing the polymer object
mk_prep = MoleculePreparation()
chem_templates = ResidueChemTemplates.create_from_defaults()
mypol = Polymer.from_prody(pdb_str['4EY7'], chem_templates, mk_prep)

# Writing the polymer object to a PDBQT file
rigid_pdbqt_string, flex_pdbqt_string = PDBQTWriterLegacy.write_string_from_polymer(mypol)
with open("4EY7_receptor.pdbqt", "w") as f:
   f.write(rigid_pdbqt_string) # here, we only write the rigid part of the receptor

# Specifying the needed atom types for the grid map calculation
# the following are consistent with the default options in the command line script: mk_prepare_receptor.py
# including all possible atom types for the ligand and receptor
any_lig_base_types = [
   "HD",
   "C",
   "A",
   "N",
   "NA",
   "OA",
   "F",
   "P",
   "SA",
   "S",
   "Cl",
   "Br",
   "I",
   "Si",
   "B",
]
rec_types = [
   "HD",
   "C",
   "A",
   "N",
   "NA",
   "OA",
   "F",
   "P",
   "SA",
   "S",
   "Cl",
   "Br",
   "I",
   "Mg",
   "Ca",
   "Mn",
   "Fe",
   "Zn",
]

# Calculating the center of the ligand in the PDB structure
# E20 is the residue name for Donepezil in the PDB structure
center = get_center_of_residue("4EY7", "A", "E20")

# Writing out the grid parameter file (GPF) for AutoDock-GPU
with open("4EY7_gpf.gpf", "w") as f:
   gpf_string, (npts_x, npts_y, npts_z) = get_gpf_string(center, [20.] * 3, "4EY7_receptor.pdbqt", rec_types, any_lig_base_types)
   f.write(gpf_string)