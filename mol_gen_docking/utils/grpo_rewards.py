"""Rewards for the GRPO task."""

from typing import List, Any
import re
import torch

from rdkit import Chem

from mol_gen_docking.utils.molecular_properties import get_oracle, KNOWN_PROPERTIES

ALL_ORACLES = {oracle: get_oracle(oracle) for oracle in KNOWN_PROPERTIES}


class RewardScorer:
    def __init__(self, reward: str):
        self.oracles = {oracle: get_oracle(oracle) for oracle in KNOWN_PROPERTIES}
        self.reward = reward
        self.__name__ = f"RewardScorer/{reward}"

    def get_mol_props_from_prompt(self, prompts: Any) -> List[dict]:
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

    def get_smiles_from_completion(self, comp: str) -> List[str]:
        """
        Get smiles from completion
        """
        s_spl = comp.split("<SMILES>")[1:]
        s_spl = [x.split("</SMILES>")[0] for x in s_spl]
        return s_spl

    def get_property_reward(self, smiles: List[str], prop: str) -> torch.Tensor:
        """
        Get property reward
        """
        oracle_fn = self.oracles[prop]
        property_reward = torch.tensor(oracle_fn(smiles, rescale=True))
        return property_reward

    def get_reward(self, objectives: List[dict], completion: List[str]) -> List[float]:
        smiles = self.get_smiles_from_completion(completion)
        valid_smiles = [smi for smi in smiles if Chem.MolFromSmiles(smi) is not None]
        if self.reward == "property":
            reward = torch.tensor(0.0)
            if len(valid_smiles) > 0:
                for prop in objectives:
                    mol_prop = self.get_property_reward(valid_smiles, prop)
                    if objectives[prop][0] == "below":
                        reward += torch.mean((mol_prop < objectives[prop][1]).float())
                    elif objectives[prop][0] == "above":
                        reward += torch.mean((mol_prop > objectives[prop][1]).float())
                    elif objectives[prop][0] == "maximize":
                        reward += mol_prop.mean().clip(0, 2)
                    elif objectives[prop][0] == "minimize":
                        reward += (1 - mol_prop.mean()).clip(0, 2)
        elif self.reward == "smiles":
            # Add 1 if a smile has been generated
            reward = torch.tensor(float(len(smiles) > 0))
        elif self.reward == "valid_smiles":
            # Add 1 if a valid smile has been generated
            reward = torch.tensor(float(len(valid_smiles) > 0))
        return reward

    def __call__(self, prompts: List[Any], completions: List[Any]) -> List[float]:
        """
        Get reward for molecular properties
        """
        objectives_prompts = self.get_mol_props_from_prompt(prompts)
        rewards = []
        for objective, completion in zip(objectives_prompts, completions):
            if isinstance(completion, list):
                assert len(completion) == 1
                completion = completion[0]
            if isinstance(completion, dict):
                assert "content" in completion
                completion = completion["content"]

            reward = self.get_reward(objective, completion)
            rewards.append(reward)
        rewards = torch.tensor(rewards)
        # Replace nan with 0
        rewards[torch.isnan(rewards)] = 0
        return rewards
