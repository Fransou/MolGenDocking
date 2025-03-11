"""Rewards for the GRPO task."""

from typing import List, Any
import re
import torch

from mol_gen_docking.molecular_properties import get_oracle, KNOWN_PROPERTIES


def molecular_properties(completion: Any, oracle: str, **kwargs) -> torch.Tensor:
    """
    Strip smiles located between the <SMILES> and </SMILES>
    tags or selfies located between the <selfies> and </selfies> tags.
    """
    print(completion)
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

    oracle_fn = get_oracle(oracle)

    smiles = parse_smiles(completion)
    return torch.tensor(oracle_fn(smiles))


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
            assert len(prompt) == 1
            prompt = prompt[0]
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
    print("objectives :", objectives_prompts)
    rewards = []
    for objective, completion in zip(objectives_prompts, completions):
        reward = 0
        print("objective :", objective)
        for prop in objective:
            mol_prop = molecular_properties(completion, prop)
            if objective[prop][0] == "below":
                reward += torch.mean((mol_prop < objective[prop][1]).float())
            elif objective[prop][0] == "above":
                reward += torch.mean((mol_prop > objective[prop][1]).float())
            elif objective[prop][0] == "maximize":
                reward += mol_prop.mean()
            elif objective[prop][0] == "minimize":
                reward += -mol_prop.mean()
            print(reward)
        rewards.append(reward)
    print(rewards)
    return rewards
