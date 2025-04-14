import os
from typing import List, Dict
import warnings
import pandas as pd
import requests


def get_pdb_description(pdb_id):
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        title = data.get("struct", {}).get("title", "No title found")
        return title
    else:
        return f"Failed to retrieve data for {pdb_id}"


DOCKING_TARGETS: List[str] = [
    "3pbl",
    "1iep",
    "2rgp",
    "3eml",
    "3ny8",
    "4rlu",
    "4unn",
    "5mo4",
    "7l11",
]

PROPERTIES_NAMES_SIMPLE: Dict[str, str] = {
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

for target in DOCKING_TARGETS:
    PROPERTIES_NAMES_SIMPLE[
        f"Binding affinity against {get_pdb_description(target)} ({target})"
    ] = target + "_docking_vina"


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
