from typing import Dict, List

PROMPT_TEMPLATE: List[str] = [
    "Design a molecule that satisfies the following objectives: {objectives}.",
    "Create a compound that meets these goals: {objectives}.",
    "Suggest a molecule with the following optimization targets: {objectives}.",
    "Please propose a structure that fulfills these requirements: {objectives}.",
    "Generate a candidate molecule optimized for: {objectives}.",
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
