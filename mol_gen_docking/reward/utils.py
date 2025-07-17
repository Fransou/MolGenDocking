from itertools import product
from typing import Dict, List, Tuple

PROMPT_INTR : List[Tuple[str, str]] = [
    (
        "I am working on designing a drug-like compound meeting a specific objective to be a possible drug. Given the objective to optimize, propose a candidate molecule as a SMILES string. In your reasoning process, you can consider multiple candidate molecules and choose the best one. Here is the objective to optimize: {objectives}.",
        "I am working on designing a drug-like compound meeting specific objectives to be a possible drug. Given the objectives to optimize, propose a candidate molecule as a SMILES string. In your reasoning process, you can consider multiple candidate molecules and choose the best one. Here are the objectives to optimize: {objectives}.",
    ),
    (
        "You are tasked with designing a drug-like molecule that satisfies a specific optimization criterion. Based on the objective provided, propose a candidate compound as a SMILES string. Feel free to consider and compare multiple candidates before selecting the most suitable one. Optimization target: {objectives}.",
        "You are tasked with designing a drug-like molecule that satisfies specific optimization criteria. Based on the objectives provided, propose a candidate compound as a SMILES string. Feel free to consider and compare multiple candidates before selecting the most suitable one. Optimization targets: {objectives}.",
    ),
    (
        "Design a molecule suitable for drug development, optimizing for the following criterion: {objectives}. Provide the structure as a SMILES string. As part of your reasoning, you may explore different candidates before settling on the most optimal molecule.",
        "Design a molecule suitable for drug development, optimizing for the following criteria: {objectives}. Provide the structure as a SMILES string. As part of your reasoning, you may explore different candidates before settling on the most optimal molecule.",
    ),
    (
        "Act as a molecular design assistant. Your task is to propose a drug-like compound, optimizing for this objective: {objectives}. Provide your answer as a SMILES string. You are encouraged to evaluate multiple candidates before choosing the best one.",
        "Act as a molecular design assistant. Your task is to propose a drug-like compound, optimizing for these objectives: {objectives}. Provide your answer as a SMILES string. You are encouraged to evaluate multiple candidates before choosing the best one.",
    ),
]


PROMPT_TEMPLATE: List[str] = [e[0] + "|" + e[1] for e in PROMPT_INTR]


OBJECTIVES_TEMPLATES: Dict[str, List[str]] = {
    "maximize": [
        "maximize {prop}",
    ],
    "minimize": [
        "minimize {prop}",
    ],
    "above": [
        "ensure {prop} is above {val}",
        "keep {prop} over {val}",
        "target {prop} values higher than {val}",
    ],
    "below": [
        "ensure {prop} is below {val}",
        "keep {prop} under {val}",
        "target {prop} values lower than {val}",
    ],
    "equal": [
        "ensure {prop} is equal to {val}",
        "set {prop} to exactly {val}",
        "target a {prop} of {val}",
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
