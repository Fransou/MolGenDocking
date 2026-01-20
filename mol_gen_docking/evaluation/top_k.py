from typing import List

from rdkit import Chem


def top_k(
    mols: List[str] | List[Chem.Mol],
    scores: List[float],
    k: int,
    canonicalize: bool = True,
) -> float:
    smi_list: List[str]
    if canonicalize or isinstance(mols[0], Chem.Mol):
        if isinstance(mols[0], str):
            mols_list = [Chem.MolFromSmiles(smi) for smi in mols]
        else:
            mols_list = mols
        smi_list = [Chem.MolToSmiles(mol, canonical=True) for mol in mols_list]
    else:
        smi_list = mols

    # Drop ducplicates and keep idxs
    seen = set()
    unique_idxs = []
    for idx, smi in enumerate(smi_list):
        if smi not in seen:
            seen.add(smi)
            unique_idxs.append(idx)
    unique_scores = [scores[idx] for idx in unique_idxs] + [
        0.0 for _ in range(len(unique_idxs), k)
    ]
    unique_scores = sorted(unique_scores, reverse=True)[:k]
    return sum(unique_scores) / k
