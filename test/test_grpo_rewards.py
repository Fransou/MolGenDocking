from typing import List

from mol_gen_docking.utils.grpo_rewards import (
    molecular_reward,
    KNOWN_PROPERTIES,
    get_mol_props_from_prompt,
    get_reward_molecular_property,
)
from mol_gen_docking.data.grpo_dataset import MolInstructionsDataset
import torch

SMILES = [
    [
        "O=C(NCc1ccc(Cl)cc1)c1ccc2c(c1)OCCO2",
    ],
    [
        "CC1CN(Cc2ccc(Nc3ncc4cc(C(=O)N(C)C)n(C5CCCC5)c4n3)nc2)CCN1",
        "COCC[C@@H](C(=O)Nc1ccc(C(=O)O)cc1)n1cc(OC)c(-c2cc(Cl)ccc2C#N)cc1=O",
    ],
]

COMPLETIONS = [
    "Here is a molecule:[SMILES] what are its properties?",
    "These are the smiles:[SMILES].",
]


def fill_completion(smiles: List[str], completion: str) -> str:
    """Fill the completion with the smiles."""
    smiles = "".join(["<SMILES>{}</SMILES>".format(s) for s in smiles])
    return completion.replace("[SMILES]", smiles)


def test_molecular_properties():
    """Test the function molecular_properties."""
    for prop_name in KNOWN_PROPERTIES:
        for smiles, completion in zip(SMILES, COMPLETIONS):
            completion = fill_completion(smiles, completion)
            properties = molecular_reward(completion, prop_name)
            assert isinstance(properties, torch.Tensor)
            assert properties.shape == (len(smiles),)


def test_prompts():
    """Test the prompts."""
    dataset = MolInstructionsDataset()
    prompts = []
    n_props = []
    for prompt, completion, n_prop in dataset.generate(100, return_n_props=True):
        prompts.append(prompt)
        n_props.append(n_prop)

    objs = get_mol_props_from_prompt(prompts)
    for n_p, obj in zip(n_props, objs):
        assert isinstance(obj, dict)
        assert len(obj) == n_p


def test_get_reward_molecular_property():
    """Test the function get_reward_molecular_property."""
    dataset = MolInstructionsDataset()
    prompts = []
    for prompt, _ in dataset.generate(100):
        prompts.append(prompt)
    completions = []
    for completion, smi in zip(prompts, SMILES):
        completions.append(fill_completion(smi, completion))

    prompts = prompts * len(completions)
    completions = completions * len(prompts)
    assert len(get_reward_molecular_property(prompts, completions)) == len(prompts)
