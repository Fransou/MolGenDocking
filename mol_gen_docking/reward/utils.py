from itertools import product
from typing import Dict, List

PROMPT_TEMPLATE: List[str] = [
    b + e[0] + "|" + b + e[1]
    for b, e in product(
        [
            "Design a molecule ",
            "Create a compound ",
            "Suggest a molecule ",
            "Propose a structure that ",
            "Generate a candidate molecule ",
        ],
        [
            (
                "that satisfies the following objective: {objectives}.",
                "that satisfies the following objectives: {objectives}.",
            ),
            (
                "that meets this goal: {objectives}.",
                "that meets these goals: {objectives}.",
            ),
            (
                "with the following optimization target: {objectives}.",
                "with the following optimization targets: {objectives}.",
            ),
            (
                "that fulfills this requirement: {objectives}.",
                "that fulfills these requirements: {objectives}.",
            ),
            ("optimized for: {objectives}.", "optimized for: {objectives}."),
        ],
    )
]


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
