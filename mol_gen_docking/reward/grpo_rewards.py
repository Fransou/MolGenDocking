"""Rewards for the GRPO task."""

import requests
from typing import List, Any, Tuple, Dict
import re
import torch
import pandas as pd
import numpy as np

from rdkit import Chem

from mol_gen_docking.reward.oracles import get_oracle, PROPERTIES_NAMES_SIMPLE


class RewardScorer:
    def __init__(
        self, reward: str, rescale: bool = True, parse_whole_completion: bool = False
    ):
        self.rescale = rescale
        self.reward = reward
        self.oracles = {
            oracle: get_oracle(oracle)
            for oracle in PROPERTIES_NAMES_SIMPLE
            if "docking" not in PROPERTIES_NAMES_SIMPLE[oracle]
        }
        self.parse_whole_completion = parse_whole_completion
        self.__name__ = f"RewardScorer/{reward}"

    def get_mol_props_from_prompt(
        self, prompts: Any
    ) -> List[Dict[str, Tuple[Any, Any]]]:
        """
        Get molecular properties from prompt
        Locates the properties, and find the following pattern:
            "$PROPERTY ($OBJECTIVE)"
        Where objective can be: maximize, minimize, below x or above x
        """
        objectives: List[Dict[str, Tuple[Any, Any]]] = []
        for prompt in prompts:
            if isinstance(prompt, list):
                if len(prompt) == 1:
                    prompt = prompt[0]
                else:
                    prompt = [
                        p
                        for p in prompt
                        if ("role" in p and p["role"] == "user")
                        or ("from" in p and p["from"] == "human")
                    ]
                    if len(prompt) == 1:
                        prompt = prompt[0]
                    else:
                        raise ValueError("Prompt not found correctly.")
            if isinstance(prompt, dict):
                if "content" in prompt:
                    prompt = prompt["content"]
                elif "value" in prompt:
                    prompt = prompt["value"]
                else:
                    raise ValueError("Prompt not found correctly.")
            props = {}
            all_props = set(PROPERTIES_NAMES_SIMPLE.keys())
            docking_props = set(re.findall(r"\b\w+_docking\b", prompt))
            all_props.update(docking_props)

            for prop in all_props:
                for pattern, parser in [
                    (
                        r"{} \((below|above|equal) ([0-9.]+)\)",
                        lambda m: (m.group(1), float(m.group(2))),
                    ),
                    (r"{} \((maximize|minimize)\)", lambda m: (m.group(1), 0)),
                ]:
                    match = re.search(pattern.format(re.escape(prop)), prompt)
                    if match:
                        props[prop] = parser(match)
                        break
            objectives.append(props)
        return objectives

    def get_smiles_from_completion(self, comp: str) -> List[str]:
        """
        Get smiles from completion
        """
        if not self.parse_whole_completion:
            s_spl = comp.split("<SMILES>")[1:]
            s_spl = [x.split("</SMILES>")[0] for x in s_spl]
            return s_spl
        else:
            # Parse the whole completion with no "<SMILES>" tag
            # First we split the completion by newlines and spaces
            re.split("\n| ", comp)
            # Then we filter by removing any string that does not contain "C"
            s_spl = [x for x in comp.split() if "C" in x or x.count("c") > 1]
            # Finally we remove any string that is not a valid SMILES
            s_spl = [x for x in s_spl if Chem.MolFromSmiles(x) is not None]
            return s_spl

    def get_all_completions_smiles(self, completions: Any) -> List[List[str]]:
        smiles = []
        for completion in completions:
            if isinstance(completion, list):
                assert len(completion) == 1
                completion = completion[0]
            if isinstance(completion, dict):
                assert "content" in completion
                completion = completion["content"]

            smiles.append(self.get_smiles_from_completion(completion))
        return smiles

    def _get_property(self, smiles: List[str], prop: str) -> torch.Tensor:
        """
        Get property reward
        """
        if prop in self.oracles:
            oracle_fn = self.oracles[prop]
        else:
            oracle_fn = get_oracle(prop)
            self.oracles[prop] = oracle_fn
        property_reward = oracle_fn(smiles, rescale=self.rescale)
        return property_reward

    def fill_df_properties(self, df_properties: pd.DataFrame):
        for p in df_properties["property"].unique():
            smiles = df_properties[df_properties["property"] == p]["smiles"].tolist()
            values = self._get_property(smiles, p)
            df_properties.loc[df_properties["property"] == p, "value"] = values

    def get_reward(self, row: pd.Series) -> float:
        reward: float = 0
        obj = row["obj"]
        mol_prop = row["value"]
        target_value = row["target_value"]
        if obj == "below":
            reward += (mol_prop <= target_value).float()
        elif obj == "above":
            reward += (mol_prop >= target_value).float()
        elif obj == "maximize":
            reward += mol_prop
        elif obj == "minimize":
            reward += 1 - mol_prop
        elif obj == "equal":
            reward += 1 - (mol_prop - target_value) ** 2
        return reward

    def _get_smiles_list(self, completions: List[Any]) -> List[List[str]]:
        smiles = self.get_all_completions_smiles(completions)
        if self.reward == "smiles":
            # No need to continue
            return smiles
        valid_smiles = [
            [s for s in smiles_c if Chem.MolFromSmiles(s) is not None]
            for smiles_c in smiles
        ]
        return valid_smiles

    def _get_prop_to_smiles_dataframe(
        self,
        smiles_list_per_completion: List[List[str]],
        objectives: List[dict],
    ) -> pd.DataFrame:
        df_properties = pd.DataFrame(
            columns=[
                "smiles",
                "property",
                "value",
                "obj",
                "target_value",
                "id_completion",
            ]
        )

        for id_completion, (props, smiles) in enumerate(
            zip(objectives, smiles_list_per_completion)
        ):
            for s in smiles:
                for p in props:
                    df_properties.loc[len(df_properties)] = [
                        s,
                        p,
                        None,
                        props[p][0],
                        props[p][1],
                        id_completion,
                    ]
        return df_properties

    def __call__(self, prompts: List[Any], completions: List[Any]) -> List[float]:
        """
        Get reward for molecular properties
        """
        smiles_list_per_completion = self._get_smiles_list(completions)
        if self.reward == "smiles" or self.reward == "valid_smiles":
            return torch.tensor(
                [
                    float(len(valid_smiles_c) > 0)
                    for valid_smiles_c in smiles_list_per_completion
                ]
            )
        objectives = self.get_mol_props_from_prompt(prompts)
        df_properties = self._get_prop_to_smiles_dataframe(
            smiles_list_per_completion, objectives
        )
        print(objectives)
        self.fill_df_properties(df_properties)
        df_properties["reward"] = df_properties.apply(
            lambda x: self.get_reward(x), axis=1
        )

        rewards = []
        for id_completion, smiles in enumerate(smiles_list_per_completion):
            if len(smiles) > 0:
                reward = df_properties[
                    (df_properties["id_completion"] == id_completion)
                    & (df_properties["smiles"].isin(smiles))
                ]["reward"].mean()
                if self.rescale:
                    reward = np.clip(reward, 0, 1)
            else:
                reward = 0
            rewards.append(reward)

        return torch.tensor(rewards).float()


class RewardScorerServer(RewardScorer):
    def __init__(
        self, reward: str, rescale: bool = True, address="http://0.0.0.0:8000/"
    ):
        super().__init__(reward, rescale=rescale)
        self.address = address

    def _query_properties(self, smiles: List[str], prop: str) -> None:
        """
        Query properties, the server wil then start computing them
        """
        data = {"smiles": smiles}
        requests.post(f"{self.address}/property/{prop}/", params=data)

    def _get_property(self, smiles: List[str], prop: str) -> torch.Tensor:
        """
        Get property reward
        """
        data = {"smiles": smiles}
        r = requests.get(f"{self.address}/property/{prop}/", params=data).json()
        assert r["smiles"] == smiles
        return r["property"]

    def pre_query_properties(self, prompts: List[str], completions: List[Any]):
        smiles_list_per_completion = self._get_smiles_list(completions)
        objectives = self.get_mol_props_from_prompt(prompts)
        df_properties = self._get_prop_to_smiles_dataframe(
            smiles_list_per_completion, objectives
        )

        for props in df_properties["property"].unique():
            smiles = df_properties[df_properties["property"] == props][
                "smiles"
            ].tolist()
            self._query_properties(smiles, props)
