from typing import Dict, List, Tuple

PROMPT_INTR: List[Tuple[str, str]] = [
    (
        "I am working on designing a drug-like compound meeting a specific objective to be a possible drug. Given the objective to optimize, propose a drug-like molecule as a SMILES string. Here is the objective to optimize: {objectives}.",
        "I am working on designing a drug-like compound meeting specific objectives to be a possible drug. Given the objectives to optimize, propose a drug-like molecule as a SMILES string. Here are the objectives to optimize: {objectives}.",
    ),
    (
        "You are tasked with designing a drug-like molecule that satisfies a specific optimization criterion. Based on the objective provided, propose a drug-like compound as a SMILES string. Optimization target: {objectives}.",
        "You are tasked with designing a drug-like molecule that satisfies specific optimization criteria. Based on the objectives provided, propose a drug-like compound as a SMILES string. Optimization targets: {objectives}.",
    ),
]


PROMPT_TEMPLATE: List[str] = [e[0] + "|" + e[1] for e in PROMPT_INTR]


OBJECTIVES_TEMPLATES: Dict[str, List[str]] = {
    "maximize": [
        "Maximize {prop}",
    ],
    "minimize": [
        "Minimize {prop}",
    ],
    "above": [
        "Ensure {prop} is above {val}",
        "Keep {prop} over {val}",
        "Target {prop} values higher than {val}",
    ],
    "below": [
        "Ensure {prop} is below {val}",
        "Keep {prop} under {val}",
        "Target {prop} values lower than {val}",
    ],
    "equal": [
        "Ensure {prop} is equal to {val}",
        "Set {prop} to exactly {val}",
        "Target a {prop} of {val}",
    ],
}

POSSIBLE_POCKET_INFO: List[str] = [
    "number of alpha spheres",
    "mean alpha-sphere radius",
    "mean alpha-sphere solvent acc.",
    "hydrophobicity score",
    "polarity score",
    "amino acid based volume score",
    "pocket volume (monte carlo)",
    "charge score",
    "local hydrophobic density score",
    "number of apolar alpha sphere",
    "proportion of apolar alpha sphere",
]
