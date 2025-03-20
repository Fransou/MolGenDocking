"""Rewards for the GRPO task."""

from typing import List, Any, Tuple
import re
import torch

from rdkit import Chem

from mol_gen_docking.utils.molecular_properties import get_oracle, KNOWN_PROPERTIES

ALL_ORACLES = {oracle: get_oracle(oracle) for oracle in KNOWN_PROPERTIES}


def molecular_reward(
    completion: Any, oracle: str, **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Strip smiles located between the <SMILES> and </SMILES>
    tags or selfies located between the <selfies> and </selfies> tags.
    """
    if isinstance(completion, list):
        assert len(completion) == 1
        completion = completion[0]
    if isinstance(completion, dict):
        assert "content" in completion
        completion = completion["content"]

    def parse_smiles(s):
        s_spl = s.split("<SMILES>")[1:]
        s_spl = [x.split("</SMILES>")[0] for x in s_spl]
        return s_spl

    oracle_fn = ALL_ORACLES[oracle]

    smiles = parse_smiles(completion)
    able_to_generate_smiles_reward = torch.tensor(float(len(smiles) > 0))

    # Check validity of the smiles
    smiles = [smi for smi in smiles if Chem.MolFromSmiles(smi) is not None]
    able_to_generate_valid_smiles_reward = torch.tensor(float(len(smiles) > 0))

    property_reward = torch.tensor(oracle_fn(smiles, rescale=True).clip(0, 1.5))

    return (
        property_reward,
        able_to_generate_smiles_reward + able_to_generate_valid_smiles_reward,
    )


def get_mol_props_from_prompt(prompts: Any) -> List[dict]:
    """
    Get molecular properties from prompt
    Locates the properties, and find the following pattern:
        "$PROPERTY ($OBJECTIVE)"
    Where objective can be: maximize, minimize, below x or above x
    """
    objectives = []
    for prompt in prompts:
        if isinstance(prompt, list):
            if len(prompt) == 1:
                prompt = prompt[0]
            else:
                prompt = [p for p in prompt if "role" in p and p["role"] == "user"]
                if len(prompt) == 1:
                    prompt = prompt[0]["content"]
                else:
                    raise ValueError("Prompt not found correctly.")
        if isinstance(prompt, dict):
            assert "content" in prompt
            prompt = prompt["content"]
        props = {}
        for prop in KNOWN_PROPERTIES:
            pattern = r"{} \((below|above) ([0-9.]+)\)".format(prop)
            match = re.search(pattern, prompt)
            if match:
                props[prop] = (match.group(1), float(match.group(2)))

            pattern = r"{} \((maximize|minimize)\)".format(prop)
            match = re.search(pattern, prompt)
            if match:
                props[prop] = (match.group(1), 0)
        objectives.append(props)
    return objectives


def get_reward_molecular_property(
    prompts: List[Any], completions: List[Any]
) -> List[float]:
    """
    Get reward for molecular properties
    """
    objectives_prompts = get_mol_props_from_prompt(prompts)
    rewards = []
    for objective, completion in zip(objectives_prompts, completions):
        reward = 0
        for prop in objective:
            mol_prop, validity_reward = molecular_reward(completion, prop)
            if objective[prop][0] == "below":
                reward += (
                    torch.mean((mol_prop < objective[prop][1]).float())
                    + validity_reward
                )
            elif objective[prop][0] == "above":
                reward += (
                    torch.mean((mol_prop > objective[prop][1]).float())
                    + validity_reward
                )
            elif objective[prop][0] == "maximize":
                reward += mol_prop.mean() + validity_reward
            elif objective[prop][0] == "minimize":
                reward += 1 - mol_prop.mean() + validity_reward

        rewards.append(reward)
    rewards = torch.tensor(rewards)
    # Replace nan with 0
    rewards[torch.isnan(rewards)] = 0
    return rewards
