from typing import Callable

import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, rdMolDescriptors

PROMPT_TEMPLATES = {
    "final_product": [
        "Give me the final product of the following multi-step synthesis:\n{reaction}\nwhere the SMARTS reprentation of the reaction is:\n{smarts}\nIf the reaction is impossible, return 'impossible'."
    ],
    "reactant": [
        "What is the missing reactant of this synthesis step:\n{reaction}\nwhere the SMARTS reprentation of the reaction is:\n{smarts}\nIf the reaction is impossible, return 'impossible'."
    ],
    "all_reactants": [
        "Given the following reaction in the SMARTS format:\n{smarts}\nprovide one or multiple reactants to obtain the following product: {product}\nIf such a reaction does not exist, return 'impossible'."
    ],
    "all_reactants_bb_ref": [
        "Given the following reaction in the SMARTS format:\n{smarts}\nprovide one molecule and one or more building blocks from:\n{building_blocks}\nto obtain the following product: {product}\nIf such a reaction does not exist, return 'impossible'."
    ],
    "smarts": [
        "Provide the SMARTS representation of the following synthesis step:\n{reaction}\nIf the reaction is impossible, return 'impossible'."
    ],
    "full_path": [
        "Propose a synthesis pathway to generate {product}. Your answer must include at most {n_reaction} reactions. Provide your answer in the following format:\nA + B -> C\n C -> D"
    ],
    "full_path_bb_ref": [
        "Propose a synthesis pathway to generate {product} using building blocks among:\n{building_blocks}\nYour answer must include at most {n_reaction} reactions. Provide your answer in the following format:\nA + B -> C\nC -> D\nIf it is impossible, return 'impossible'."
    ],
    "full_path_smarts_ref": [
        "Propose a synthesis pathway to generate {product} using reactions among the following SMARTS:\n{smarts}\nYour answer must include at most {n_reaction} reactions. Provide your answer in the following format:\nA + B -> C\nC -> D\nIf it is impossible, return 'impossible'."
    ],
    "full_path_smarts_bb_ref": [
        "Propose a synthesis pathway to generate {product} using building blocks among:\n{building_blocks}\n and reactions among the following SMARTS:\n{smarts}\nYour answer must include at most {n_reaction} reactions. Provide your answer in the following format:\nA + B -> C\nC -> D\nIf it is impossible, return 'impossible'."
    ],
}


def get_prop(k: str, mol: Chem.rdchem.Mol) -> float:
    if k.lower() == "qed":
        return float(QED.qed(mol))
    if hasattr(rdMolDescriptors, k):
        return float(getattr(rdMolDescriptors, k)(mol))
    raise NotImplementedError(f"Unknown property {k}")


PROP_RANGE = {
    "qed": (1, 0.3),
    "CalcExactMolWt": (600, 0),
    "CalcTPSA": (160, 0),
    "CalcNumHBA": (10, 0),
    "CalcNumHBD": (10, 0),
    "CalcNumRotatableBonds": (10, 1),
    "CalcNumRings": (7, 0),
    "CalcNumAromaticRings": (6, 0),
}


def logbeta(
    a: float, b: float, min_x: float = 0.0, max_x: float = 1.0
) -> Callable[[float], float]:
    def logbeta_fn(x: float) -> float:
        x_norm = (x - min_x) / (max_x - min_x)
        x_norm = np.clip(x_norm, 1e-6, 1 - 1e-6)
        return np.log(x_norm) * (a - 1) + np.log(1 - x_norm) * (b - 1)  # type: ignore

    return logbeta_fn


PROP_TARGET_DISTRIB_FN: dict[str, Callable[[float], float]] = {
    "qed": logbeta(8, 2.5, 0, 1),
    "CalcExactMolWt": logbeta(14, 11, 0, 600),
    "CalcTPSA": logbeta(4, 6, 0, 160),
    "CalcNumHBA": logbeta(5.5, 8, -1, 11),
    # "CalcNumHBD": beta(4.5,11,-1,7),
    "CalcNumRotatableBonds": logbeta(5, 5, -1, 11),
    "CalcNumAromaticRings": logbeta(5, 9, -1, 7),
}
