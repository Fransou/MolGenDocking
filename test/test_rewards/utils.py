import json
import os
from typing import Callable, List, Optional

import pandas as pd

from mol_gen_docking.reward.property_utils import inverse_rescale_property_values

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


def get_unscaled_obj(obj: str, prop: str) -> str:
    if len(obj.split()) == 1:
        return obj
    func = obj.split()[0]
    val = float(obj.split()[1])

    val = inverse_rescale_property_values(prop, val, prop in DOCKING_PROP_LIST)
    return f"{func} {val:.2f}"


def get_fill_completions(no_flags: bool = False) -> Callable[[List[str], str], str]:
    def fill_completion(smiles: List[str], completion: str) -> str:
        """Fill the completion with the smiles."""
        if no_flags:
            smiles_joined = " ".join(smiles)

        smiles_joined: str = "<answer> " + " ".join(smiles) + " </answer>"
        return completion.replace("[SMILES]", smiles_joined)

    return fill_completion


def fill_df_time(
    target: str,
    n_generations: int,
    t0: float,
    t1: float,
    method: str,
    exhaustiveness: int,
    scores: float,
    server: bool = False,
    t_pre: Optional[float] = None,
) -> None:
    # get_current_commit_hash
    commit_hash = (
        os.popen("git rev-parse --short HEAD").read().strip()
        if os.path.exists(".git")
        else "N/A"
    )
    df = pd.DataFrame(
        {
            "target": [target],
            "n_generations": [n_generations],
            "method": [method],
            "server": [server],
            "time_preparation_s": [t0 - t_pre] if t_pre is not None else [0],
            "time_docking_s": [t1 - t0],
            "time_per_molecule_s": [(t1 - t0) / n_generations],
            "scores": [scores],
            "exhaustiveness": [exhaustiveness],
            "commit_hash": [commit_hash],
        }
    )
    path = os.path.join("test", "test_docking.csv")
    df.to_csv(
        path,
        mode="a",
        index=False,
        header=not os.path.exists(path),
    )
