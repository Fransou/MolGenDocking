from typing import Callable, List

from mol_gen_docking.reward.oracles import (
    propeties_csv,
    PROPERTIES_NAMES_SIMPLE,
    DOCKING_TARGETS,
)

PROP_LIST: List[str] = [
    k
    for k in PROPERTIES_NAMES_SIMPLE.keys()
    if "docking" not in PROPERTIES_NAMES_SIMPLE[k]
]

DOCKING_PROP_LIST: List[str] = [
    k
    for k in PROPERTIES_NAMES_SIMPLE
    if "docking" in PROPERTIES_NAMES_SIMPLE[k]
    or PROPERTIES_NAMES_SIMPLE[k] in DOCKING_TARGETS
]

SMILES: List[List[str]] = (
    [["FAKE"]]
    + [propeties_csv.sample(k)["smiles"].tolist() for k in range(1, 3)]
    + [propeties_csv.sample(1)["smiles"].tolist() + ["FAKE"]]
)

COMPLETIONS: List[str] = [
    "Here is a molecule: [SMILES] what are its properties?",
    "This is an empty completion.",
]

OBJECTIVES_TO_TEST: List[str] = [
    "maximize",
    "minimize",
    "above 0.5",
    "below 0.5",
    "equal 0.5",
]


def get_fill_completions(no_flags: bool = False) -> Callable[[List[str], str], str]:
    def fill_completion(smiles: List[str], completion: str) -> str:
        """Fill the completion with the smiles."""
        smiles_joined: str = "".join(
            [
                "{} ".format(s) if no_flags else "<SMILES>{}</SMILES> ".format(s)
                for s in smiles
            ]
        )
        return completion.replace("[SMILES]", smiles_joined)

    return fill_completion
