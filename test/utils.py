from mol_gen_docking.reward.oracles import (
    propeties_csv,
    PROPERTIES_NAMES_SIMPLE,
)

PROP_LIST = [
    k
    for k in PROPERTIES_NAMES_SIMPLE.keys()
    if "docking" not in PROPERTIES_NAMES_SIMPLE[k]
]

DOCKING_PROP_LIST = [
    k for k in PROPERTIES_NAMES_SIMPLE if "docking" in PROPERTIES_NAMES_SIMPLE[k]
]

SMILES = (
    [["FAKE"]]
    + [propeties_csv.sample(k)["smiles"].tolist() for k in range(1, 3)]
    + [propeties_csv.sample(1)["smiles"].tolist() + ["FAKE"]]
)

COMPLETIONS = [
    "Here is a molecule: [SMILES] what are its properties?",
    "This is an empty completion.",
]

OBJECTIVES_TO_TEST = ["maximize", "minimize", "below 0.5", "above 0.5", "equal 0.5"]
