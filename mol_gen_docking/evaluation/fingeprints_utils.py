from typing import Callable

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def fp_name_to_fn(
    fp_name: str,
) -> Callable[[Chem.Mol], DataStructs.cDataStructs.ExplicitBitVect]:
    """
    Names in:
        - ecfp{diam}-{Nbits}
        - maccs
        - rdkit
        - Gobbi2d
        -
        - Avalon
    """

    if fp_name.startswith("ecfp"):
        d = int(fp_name[4])
        n_bits = int(fp_name.split("-")[1])
        assert fp_name == f"ecfp{d}-{n_bits}", f"Invalid fingerprint name: {fp_name}"

        def fp_fn(mol: Chem.Mol) -> DataStructs.cDataStructs.ExplicitBitVect:
            return AllChem.GetMorganFingerprintAsBitVect(mol, d // 2, n_bits)

        return fp_fn
    elif fp_name == "maccs":

        def fp_fn(mol: Chem.Mol) -> DataStructs.cDataStructs.ExplicitBitVect:
            return Chem.rdMolDescriptors.GetMACCSKeysFingerprint(mol)

        return fp_fn
    elif fp_name == "rdkit":

        def fp_fn(mol: Chem.Mol) -> DataStructs.cDataStructs.ExplicitBitVect:
            return Chem.RDKFingerprint(mol)

        return fp_fn
    elif fp_name == "Gobbi2d":
        from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

        def fp_fn(mol: Chem.Mol) -> DataStructs.cDataStructs.ExplicitBitVect:
            return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)

        return fp_fn
    elif fp_name == "Avalon":
        from rdkit.Avalon import pyAvalonTools

        def fp_fn(mol: Chem.Mol) -> DataStructs.cDataStructs.ExplicitBitVect:
            return pyAvalonTools.GetAvalonFP(mol)

        return fp_fn
    else:
        raise ValueError(f"Unknown fingerprint name: {fp_name}")


def get_sim_matrix(
    mols: list[Chem.Mol],
    fingerprint_name: str = "ecfp4-1024",
) -> np.ndarray[float]:
    fp_fn = fp_name_to_fn(fingerprint_name)
    fps = [fp_fn(mol) for mol in mols]
    sim_mat = [
        np.array(DataStructs.BulkTanimotoSimilarity(fp, fps[i + 1 :]))
        for i, fp in enumerate(fps[:-1])
    ]
    matrix: np.ndarray[float] = np.concatenate(sim_mat)
    return matrix
