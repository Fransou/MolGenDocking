import json
from typing import Callable, List

import pandas as pd

with open("data/mol_orz/names_mapping.json") as f:
    PROPERTIES_NAMES_SIMPLE: dict = json.load(f)
with open("data/mol_orz/docking_targets.json") as f:
    DOCKING_PROP_LIST: List[str] = json.load(f)
print(DOCKING_PROP_LIST)

PROP_LIST: List[str] = [
    k for k in PROPERTIES_NAMES_SIMPLE.values() if k not in DOCKING_PROP_LIST
]


propeties_csv = pd.read_csv("data/properties.csv", index_col=0)
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
