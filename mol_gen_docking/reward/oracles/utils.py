import os
from typing import List, Dict
import warnings
import pandas as pd
import json

from tqdm import tqdm
from multiprocessing import Pool

from mol_gen_docking.reward.property_utils.docking import (
    DOCKING_TARGETS,
    get_pdb_description,
)


IS_CONNECTED = True

PROPERTIES_NAMES_SIMPLE: Dict[str, str] = {}
if not os.path.exists(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "properties_names_simple.json"
    )
):
    PROPERTIES_NAMES_SIMPLE = {
        "Inhibition against glycogen synthase kinase-3 beta": "GSK3B",
        "Inhibition against c-Jun N-terminal kinase-3": "JNK3",
        "Bioactivity against dopamine receptor D2": "DRD2",
        "Synthetic accessibility": "SA",
        "Quantitative estimate of drug-likeness": "QED",
        "Molecular Weight": "CalcExactMolWt",
        "Number of Aromatic Rings": "CalcNumAromaticRings",
        "Number of H-bond acceptors": "CalcNumHBA",
        "Number of H-bond donors": "CalcNumHBD",
        "Number of Rotatable Bonds": "CalcNumRotatableBonds",
        "Fraction of C atoms Sp3 hybridised": "CalcFractionCSP3",
        "Topological Polar Surface Area": "CalcTPSA",
        "Hall-Kier alpha": "CalcHallKierAlpha",
        "Hall-Kier kappa 1": "CalcKappa1",
        "Hall-Kier kappa 2": "CalcKappa2",
        "Hall-Kier kappa 3": "CalcKappa3",
        "Kier Phi": "CalcPhi",
        "logP": "logP",
    }
    pool = Pool(8)
    docking_desc = list(
        tqdm(
            pool.imap(get_pdb_description, DOCKING_TARGETS),
            total=len(DOCKING_TARGETS),
            desc="Adding descriptions to docking targets",
        )
    )
    for pdb_id, desc in zip(DOCKING_TARGETS, docking_desc):
        if desc is not None:
            PROPERTIES_NAMES_SIMPLE[f"Binding affinity against {desc} ({pdb_id})"] = (
                pdb_id
            )
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "properties_names_simple.json"
        ),
        "w",
    ) as f:
        json.dump(PROPERTIES_NAMES_SIMPLE, f, indent=4)
else:
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "properties_names_simple.json"
        )
    ) as f:
        PROPERTIES_NAMES_SIMPLE = json.load(f)


oracles_not_to_rescale = ["GSK3B", "JNK3", "DRD2"]

OBJECTIVES_TEMPLATES: Dict[str, List[str]] = {
    "maximize": [
        "maximize {prop}",
    ],
    "minimize": [
        "minimize {prop}",
    ],
    "above": [
        "ensure {prop} is above {val}",
        "keep {prop} greater than {val}",
        "target {prop} values higher than {val}",
    ],
    "below": [
        "ensure {prop} is below {val}",
        "keep {prop} less than {val}",
        "target {prop} values lower than {val}",
    ],
    "equal": [
        "ensure {prop} is equal to {val}",
        "set {prop} to exactly {val}",
        "target a {prop} of {val}",
    ],
}

PROMPT_TEMPLATE: List[str] = [
    "Design a molecule that satisfies the following objectives: {objectives}.",
    "Create a compound that meets these goals: {objectives}.",
    "Suggest a molecule with the following optimization targets: {objectives}.",
    "Please propose a structure that fulfills these requirements: {objectives}.",
    "Generate a candidate molecule optimized for: {objectives}.",
]


property_csv_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "properties.csv"
)

if not os.path.exists(property_csv_path):
    # Raise a warning, the properties file is not found
    warnings.warn(
        "The properties file is not found. Launch 'reward/oracle_wrapper.py' to generate it."
    )
    propeties_csv = pd.DataFrame(columns=["smiles"])
else:
    propeties_csv = pd.read_csv(property_csv_path)
