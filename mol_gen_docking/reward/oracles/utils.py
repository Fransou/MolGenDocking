import os
import warnings
import pandas as pd


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
