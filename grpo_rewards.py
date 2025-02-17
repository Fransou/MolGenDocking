from typing import List, Any
from molecular_properties import get_oracle, KNOWN_PROPERTIES
import re

import torch


def molecular_properties(completion: Any, oracle: str, **kwargs):
    """
    Strip smiles located between the <smiles> and </smiles> tags or selfies located between the <selfies> and </selfies> tags.
    """
    if isinstance(completion, list):
        assert len(completion) == 1
        completion = completion[0]
    if isinstance(completion, dict):
        assert "content" in completion
        completion = completion["content"]

    def parse_smiles(s):
        s_spl = s.split("<smiles>")[1:]
        s_spl = [x.split("</smiles>")[0] for x in s_spl]
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
    print(objectives_prompts)
    rewards = []
    for objective, completion in zip(objectives_prompts, completions):
        reward = 0
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
        rewards.append(reward)
    return rewards


def get_reward_n_generated(prompts: List[Any], completions: List[Any]) -> List[float]:
    """
    Get reward for number of generated molecules.
    The amount is located in the prompt with a pattern like:
    "$N molecules"
    while molecules are located in the completion between the <smiles> and </smiles> tags.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        if isinstance(prompt, list):
            assert len(prompt) == 1
            prompt = prompt[0]
        if isinstance(prompt, dict):
            assert "content" in prompt
            prompt = prompt["content"]
        if isinstance(completion, list):
            assert len(completion) == 1
            completion = completion[0]
        if isinstance(completion, dict):
            assert "content" in completion
            completion = completion["content"]
        n_to_generate = int(re.search(r"([0-9]+) molecules", prompt).group(1))
        smiles = completion.split("<smiles>")[1:]
        rewards.append(len(smiles) == n_to_generate)
    return rewards


if __name__ == "__main__":
    instructions = [
        "The objective is: logP (maximize), Molecular Weight (below 500)",
        "The objective is: DRD2 (maximize), QED (maximize)",
    ]
    comp = [
        "These are the smiles:<smiles>CCO</smiles>, <smiles>CCCO</smiles>",
        "These are the smiles:<smiles>CCCCCCCCCCCCCCCCO</smiles>, <smiles>CCCO</smiles>",
    ]

    props = get_reward_molecular_property(instructions, comp)

    print(props)
