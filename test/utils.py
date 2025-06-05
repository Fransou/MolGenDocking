import json
import os
from typing import Callable, List

import pandas as pd

DATA_PATH = os.environ.get("DATA_PATH", "data/mol_orz")

with open(f"{DATA_PATH}/names_mapping.json") as f:
    PROPERTIES_NAMES_SIMPLE: dict = json.load(f)
with open(f"{DATA_PATH}/docking_targets.json") as f:
    DOCKING_PROP_LIST: List[str] = json.load(f)

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
