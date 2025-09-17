import json
import os
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import ray
from ray.experimental import tqdm_ray
from rdkit import Chem, RDLogger
from tdc.chem_utils.oracle.filter import MolFilter

from mol_gen_docking.reward.oracle_wrapper import OracleWrapper, get_oracle
from mol_gen_docking.reward.property_utils import rescale_property_values
from mol_gen_docking.reward.utils import OBJECTIVES_TEMPLATES

RDLogger.DisableLog("rdApp.*")


def template_to_regex(template: str) -> str:
    """
    Convert a single template string into a regex pattern.
    """
    # Escape special regex characters that might appear in text
    pattern = re.escape(template)
    # Replace escaped {prop} and {val} with regex groups
    pattern = pattern.replace(r"\{prop\}", r"(?P<prop>.+?)")
    if r"\{val\}" in pattern:
        pattern = pattern.replace(r"\{val\}", r"(?P<val>[-+]?\d+\.?\d*)")
    else:
        pattern += r"(?:[.!?]+|$)"
    return pattern


def generate_regex_patterns(templates: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    """
    Converts OBJECTIVES_TEMPLATES to a list of regex patterns with associated objective type.
    Returns: List of (regex_pattern, objective_type)
    """
    pattern_list = []
    for obj_type, template_list in templates.items():
        for tmpl in template_list:
            regex = template_to_regex(tmpl)
            pattern_list.append((regex, obj_type))
    return pattern_list


class RewardScorer:
    def __init__(
        self,
        path_to_mappings: Optional[str] = None,
        reward: Literal["property", "valid_smiles", "MolFilters"] = "property",
        rescale: bool = True,
        parse_whole_completion: bool = False,
        oracle_kwargs: Dict[str, Any] = {},
    ):
        if path_to_mappings is not None:
            with open(os.path.join(path_to_mappings, "names_mapping.json")) as f:
                property_name_mapping = json.load(f)
            with open(os.path.join(path_to_mappings, "docking_targets.json")) as f:
                docking_target_list = json.load(f)

        self.property_name_mapping = property_name_mapping
        self.docking_target_list = docking_target_list
        self.path_to_mappings = path_to_mappings

        self.rescale = rescale
        self.reward = reward
        self.oracle_kwargs = oracle_kwargs
        self.parse_whole_completion = parse_whole_completion
        self.__name__ = f"RewardScorer/{reward}"
        self.remote_tqdm = ray.remote(tqdm_ray.tqdm)

        self.oracles: Dict[str, OracleWrapper] = {}

        self.search_patterns = generate_regex_patterns(OBJECTIVES_TEMPLATES)
        if not ray.is_initialized():
            ray.init()

    @staticmethod
    def get_mol_props_from_prompt(
        prompts: List[Any], search_templates: List[Tuple[str, str]]
    ) -> List[Dict[str, Tuple[str, float]]]:
        """
        Get molecular properties from prompt.

        Returns: List of Dict[property -> (objective, target_value)]
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
            assert isinstance(prompt, str), "Prompt not found correctly."

            prompt = prompt.split("user: ")[-1].split("assistant: ")[0]
            if prompt.endswith(".") or prompt.endswith("?"):
                prompt = prompt[:-1]

            props = {}
            clauses = re.split(r";", prompt)
            for clause in clauses:
                if clause == "":
                    continue
                clause = clause.strip()
                if not clause:
                    continue
                for pattern, obj_type in search_templates:
                    match = re.search(pattern, clause, re.IGNORECASE)
                    if match:
                        prop = match.group("prop").strip()
                        if obj_type in ["above", "below", "equal"]:
                            val = match.group("val").strip()
                            assert f"{float(val):.2f}" == val, "Value is not a number"
                            val = float(val)
                        else:
                            val = 0.0
                        props[prop] = (obj_type, val)
                        break

            objectives.append(props)
        return objectives

    def get_smiles_from_completion(self, comp: str) -> List[str]:
        """
        Get smiles from completion
        """
        if not self.parse_whole_completion:
            matches = re.findall(r"<answer>(.*?)</answer>", comp, flags=re.DOTALL)
            if len(matches) > 0:
                comp = " ".join(matches)
            else:
                comp = ""
        # Now we identify which elements are possibly SMILES
        # First we split the completion by newlines and spaces
        re.split("\n| |.|\t|:", comp)
        # Then we filter by removing any string that does not contain "C"
        valid_smiles_pattern = re.compile(r"^[A-Za-z0-9=#:\+\-\[\]\(\)/\\@.%]+$")

        def filter_smiles(x: str) -> bool:
            if "e" in x or len(x) < 3:
                return False
            if "C" in x or x.count("c") > 2:
                return valid_smiles_pattern.fullmatch(x) is not None
            return False

        s_poss = [x for x in comp.split() if filter_smiles(x)]

        # Finally we remove any string that is not a valid SMILES
        def test_is_valid(smi: str) -> bool:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return False
            try:
                _ = Chem.MolToMolBlock(mol)
                return True
            except Exception as e:
                print(f"Error in MolToMolBlock for {smi}: {e}")
                return False

        s_spl = [
            x for x in s_poss if test_is_valid(x)
        ]  ### TODO: Maybe do not return the mean if mutliple molecules
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

    def fill_df_properties(self, df_properties: pd.DataFrame) -> None:
        @ray.remote(num_cpus=1)
        def _get_property(
            smiles: List[str],
            prop: str,
            rescale: bool = True,
            kwargs: Dict[str, Any] = {},
            pbar: Optional[Any] = None,
        ) -> List[float]:
            """
            Get property reward
            """
            oracle_fn = self.oracles.get(
                prop,
                get_oracle(
                    prop,
                    path_to_data=self.path_to_mappings if self.path_to_mappings else "",
                    docking_target_list=self.docking_target_list,
                    property_name_mapping=self.property_name_mapping,
                    **kwargs,
                ),
            )
            if prop not in self.oracles:
                self.oracles[prop] = oracle_fn
            property_reward: np.ndarray | float = oracle_fn(smiles, rescale=rescale)
            assert isinstance(property_reward, np.ndarray)
            if pbar is not None:
                pbar.update.remote(len(property_reward))

            return [float(p) for p in property_reward]

        all_properties = df_properties["property"].unique().tolist()
        prop_smiles = {
            p: df_properties[df_properties["property"] == p]["smiles"].unique().tolist()
            for p in all_properties
        }

        pbar = self.remote_tqdm.remote(  # type: ignore
            total=df_properties[["property", "smiles"]].drop_duplicates().shape[0],
            desc="[Properties]",
        )
        values_job = []
        for p in all_properties:
            smiles = prop_smiles[p]
            values_job.append(
                _get_property.remote(
                    smiles,
                    p,
                    rescale=self.rescale,
                    kwargs=self.oracle_kwargs,
                    pbar=pbar,
                )
            )

        all_values = ray.get(values_job)
        for idx_p, p in enumerate(all_properties):
            values = all_values[idx_p]
            smiles = prop_smiles[p]
            for s, v in zip(smiles, values):
                df_properties.loc[
                    (df_properties["smiles"] == s) & (df_properties["property"] == p),
                    "value",
                ] = v

        pbar.close.remote()  # type: ignore

    def get_reward(self, row: pd.Series) -> float:
        reward: float = 0
        obj = row["obj"]
        mol_prop = row["value"]
        target_value = row["target_value"]
        prop = row["property"]
        if self.rescale:
            target_value = rescale_property_values(
                prop, target_value, docking=mol_prop in self.docking_target_list
            )
        if obj == "below":
            reward += float(mol_prop <= target_value)
        elif obj == "above":
            reward += float(mol_prop >= target_value)
        elif obj == "maximize":
            reward += mol_prop
        elif obj == "minimize":
            reward += 1 - mol_prop
        elif obj == "equal":
            reward += np.clip(1 - 100 * (mol_prop - target_value) ** 2, 0, 1)
        return float(reward)

    def _get_smiles_list(self, completions: List[Any]) -> List[List[str]]:
        smiles = self.get_all_completions_smiles(completions)
        return smiles

    def _get_prop_to_smiles_dataframe(
        self,
        smiles_list_per_completion: List[List[str]],
        objectives: List[dict[str, Tuple[str, float]]],
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

    def get_score(
        self,
        prompts: List[Any],
        completions: List[Any],
        debug: bool = False,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[float]:
        """
        Get reward for molecular properties
        """

        smiles_list_per_completion = self._get_smiles_list(completions)
        if (
            self.reward == "valid_smiles"
        ):  # TODO: Always return 1 if at least one valid smiles
            return [
                float(len(valid_smiles_c) > 0)
                for valid_smiles_c in smiles_list_per_completion
            ]
        elif self.reward == "MolFilters":
            filters = MolFilter(
                filters=["PAINS", "SureChEMBL", "Glaxo"], property_filters_flag=False
            )
            all_smiles = {}
            for i, valid_smiles_c in enumerate(smiles_list_per_completion):
                for s in valid_smiles_c:
                    if s not in all_smiles:
                        all_smiles[s] = [i]
                    else:
                        all_smiles[s].append(i)
            filter_flags = filters(list(all_smiles.keys()))
            outputs: List[float] = [0.0 for _ in range(len(smiles_list_per_completion))]

            for s in filter_flags:
                for idx in all_smiles[s]:
                    outputs[idx] += 1 / len(smiles_list_per_completion[idx])
            return outputs

        # objectives: List[Dict[str, Tuple[str, float]]], for each completion dict of the properties to evaluate and the objective ("above", "below", "equal", "maximize", "minimize") and target value
        if metadata is None or not (
            all(
                [
                    "properties" in m and "objectives" in m and "target" in m
                    for m in metadata
                ]
            )
        ):
            objectives = self.get_mol_props_from_prompt(prompts, self.search_patterns)
        else:
            objectives = []
            for m in metadata:
                props = {}
                for p, obj, target in zip(
                    m["properties"], m["objectives"], m["target"]
                ):
                    props[p] = (obj, float(target))
                objectives.append(props)

        df_properties = self._get_prop_to_smiles_dataframe(
            smiles_list_per_completion, objectives
        )
        self.fill_df_properties(df_properties)
        df_properties["reward"] = df_properties.apply(
            lambda x: self.get_reward(x), axis=1
        )
        df_properties.to_csv("debug_properties.csv", index=False)

        rewards = []
        for id_completion, smiles in enumerate(smiles_list_per_completion):
            if len(smiles) > 0:
                reward = df_properties[
                    (df_properties["id_completion"] == id_completion)
                    & (df_properties["smiles"].isin(smiles))
                ]["reward"].mean()
                if self.rescale and not debug:
                    reward = np.clip(reward, 0, 1)
            else:
                reward = 0

            if np.isnan(reward) or reward is None:
                sub_table = df_properties[
                    (df_properties["id_completion"] == id_completion)
                    & (df_properties["smiles"].isin(smiles))
                ]
                log_table = ";".join(
                    f"{col}: {sub_table[col].tolist()}\n" for col in sub_table.columns
                )
                print(
                    f"Warning: Reward is None or NaN for completion id {id_completion} with smiles {smiles}\n",
                    f"Associated table :\n {log_table}",
                )
                reward = 0
            rewards.append(float(reward))
        return rewards

    def __call__(
        self, prompts: List[Any], completions: List[Any], debug: bool = False
    ) -> List[float]:
        """
        Call the scorer to get the rewards.
        """
        return self.get_score(prompts, completions, debug=debug)
