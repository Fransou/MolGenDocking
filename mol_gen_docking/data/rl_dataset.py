"""Dataset for generating prompts for molecule generation"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Tuple, Union

import numpy as np
from numpy import random
from tqdm import tqdm

from mol_gen_docking.reward.property_utils.classical_properties import (
    CLASSICAL_PROPERTIES_NAMES,
)
from mol_gen_docking.reward.utils import (
    OBJECTIVES_TEMPLATES,
    PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


OBJECTIVES = ["maximize", "minimize", "below", "above", "equal"]
DOCKING_SOLO_OBJECTIVES = ["minimize", "below", "equal"]
TARGET_VALUE_OBJECTIVES = ["below", "above", "equal"]


@dataclass
class RuleSet:
    """A simple class to keep track of the rules for generating prompts"""

    prompt_ids: Dict[int, List[str]] = field(
        default_factory=dict
    )  # n_props -> list of prompt_ids
    n_occ_prop: Dict[int, Dict[str, int]] = field(
        default_factory=dict
    )  # n_props -> {prop: n_occurrences}
    probs_docking_targets: float = field(
        default=0.5
    )  # Probability to draw a docking property
    max_occ: int = field(
        default=10
    )  # Maximum number of occurrences of a property per n_props
    max_docking_per_prompt: int = field(default=2)
    prohibited_props_at_n: Dict[int, List[str]] = field(default_factory=dict)

    def verify_and_update(self, metadata: Dict[str, Any]) -> bool:
        """
        Verify if the metadata meets the rules and update the occurrences.
        Returns True if the metadata is valid, False otherwise.
        """
        n_props = metadata["n_props"]
        properties = metadata["properties"]

        if n_props not in self.n_occ_prop:
            self.n_occ_prop[n_props] = {}
            self.prompt_ids[n_props] = []

        if (
            metadata["prompt_id"] in self.prompt_ids[n_props]
        ):  # Already generated this prompt
            logger.info(
                "Prompt %s already generated for n_props=%d",
                metadata["prompt_id"],
                n_props,
            )
            return False
        for prop in properties:
            if prop not in self.n_occ_prop[n_props]:
                self.n_occ_prop[n_props][prop] = 0
            if (
                self.n_occ_prop[n_props][prop] >= self.max_occ
            ):  # Too many occurrences of this property
                logger.info(
                    "Property %s has too many occurrences: %d", prop, self.max_occ
                )
                return False
        # Update occurrences
        self.prompt_ids[n_props].append(metadata["prompt_id"])
        for prop in properties:
            # Only account for docking targets
            if prop in CLASSICAL_PROPERTIES_NAMES.values():
                continue
            self.n_occ_prop[n_props][prop] += 1
            if self.n_occ_prop[n_props][prop] >= self.max_docking_per_prompt:
                # If we have too many docking properties, we prohibit this property for this n_props
                if n_props not in self.prohibited_props_at_n:
                    self.prohibited_props_at_n[n_props] = []
                self.prohibited_props_at_n[n_props].append(prop)
                logger.info(
                    "Prohibiting docking property %s for n_props=%d",
                    prop,
                    n_props,
                )
        return True

    def partial_reset(self):
        """Reinitialize the rule set, keeping the prompt_ids"""
        self.n_occ_prop = {}
        self.prohibited_props_at_n = {}


@dataclass
class DatasetConfig:
    """Configuration for the MolGenerationInstructionsDataset"""

    data_path: str
    max_n_props: int = 5
    vina: bool = False
    split_docking: List[float] = field(default_factory=lambda: [1])
    probs_docking_targets: float = 0.5
    max_occ: int = 10
    max_docking_per_prompt: int = 2


class MolGenerationInstructionsDataset:
    """A simple Dataset generating rule-based prompts for molecule generation"""

    def __init__(
        self,
        config: DatasetConfig,
    ):
        """
        :param max_n_props: Maximal number of properties to optimize
        """
        self.docking_targets: List[str] = []
        self.prop_name_mapping: Dict[str, str] = {}
        self.pockets_info: Dict[str, Any] = {}

        self.load_props(config.data_path)
        self.max_n_props = config.max_n_props

        self.std_properties: List[str] = [
            k
            for k in self.prop_name_mapping
            if self.prop_name_mapping[k] not in self.docking_targets
        ]
        self.docking_properties: List[str] = []
        self.docking_properties_split: List[List[str]] = [[]] * len(
            config.split_docking
        )
        if config.vina:
            # shuffle the docking properties
            self.docking_properties = [
                k
                for k in self.prop_name_mapping
                if self.prop_name_mapping[k] in self.docking_targets
            ]
            np.random.shuffle(self.docking_properties)
            i0 = 0
            for idx, p in enumerate(config.split_docking):
                i1 = i0 + int(len(self.docking_properties) * p)
                i1 = min(i1, len(self.docking_properties))
                self.docking_properties_split[idx] = self.docking_properties[i0:i1]
                i0 = i1

        self.obj_templates: Dict[str, List[str]] = OBJECTIVES_TEMPLATES
        self.templates: List[str] = PROMPT_TEMPLATE
        self.prop_key_list = list(self.prop_name_mapping.keys())
        self.rule_set = RuleSet(
            probs_docking_targets=config.probs_docking_targets,
            max_occ=config.max_occ,
            max_docking_per_prompt=config.max_docking_per_prompt,
        )

    def load_props(self, path: str):
        assert os.path.exists(path)
        docking_target_list_path = os.path.join(path, "docking_targets.json")
        prop_name_mapping_path = os.path.join(path, "names_mapping.json")
        pocket_info_path = os.path.join(path, "pockets_info.json")

        assert os.path.exists(docking_target_list_path), (
            f"File {docking_target_list_path} does not exist. Please check the data path."
        )
        assert os.path.exists(prop_name_mapping_path), (
            f"File {prop_name_mapping_path} does not exist. Please check the data path."
        )
        assert os.path.exists(pocket_info_path), (
            f"File {pocket_info_path} does not exist. Please check the data path."
        )

        with open(docking_target_list_path) as f:
            self.docking_targets = json.load(f)
        with open(prop_name_mapping_path) as f:
            self.prop_name_mapping = json.load(f)
        with open(pocket_info_path) as f:
            self.pockets_info = json.load(f)

    def fill_prompt(self, props: List[str], objs: List[str]) -> str:
        """
        Takes a list of properties and corresponding objectives
        and returns a diverse natural language prompt for multi-objective optimization.
        """
        if len(props) != len(objs):
            raise ValueError("props and objs must have the same length.")

        # Phrase templates for each type of objective

        phrases = []
        for prop, obj in zip(props, objs):
            obj_l = obj.lower()
            phrase: str = random.choice(self.obj_templates[obj_l.split()[0]])
            if obj_l.startswith("maximize") or obj_l.startswith("minimize"):
                phrase = phrase.format(prop=prop)
            elif obj_l.startswith("above"):
                val = obj_l.split()[-1]
                phrase = phrase.format(prop=prop, val=val)
            elif obj_l.startswith("below"):
                val = obj_l.split()[-1]
                phrase = phrase.format(prop=prop, val=val)
            elif obj_l.startswith("equal"):
                val = obj_l.split()[-1]
                phrase = phrase.format(prop=prop, val=val)
            else:
                raise ValueError("Unknown objective.")

            phrases.append(phrase)

        # Top-level prompt templates
        prompt: str = random.choice(self.templates)
        full_prompt = prompt.format(objectives="; ".join(phrases))
        return full_prompt

    def generate(
        self,
        n: int,
        docking_properties_list: List[str],
    ) -> Iterator[Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]]:
        """
        Generates n prompts randomly to generate molecules
        :param n: number of prompts to generate
        :param return_n_props: if True, returns the number of properties to optimize
        :return:
        """

        for _ in range(n):
            # Sample properties and objectives
            metadata: Dict[str, Any] = {}

            n_props: int = int(random.randint(1, 1 + self.max_n_props))
            allowed_docking_props = [
                p
                for p in docking_properties_list
                if p not in self.rule_set.prohibited_props_at_n.get(n_props, [])
            ]
            allowed_std_props = [
                p
                for p in self.std_properties
                if p not in self.rule_set.prohibited_props_at_n.get(n_props, [])
            ]

            property_list = allowed_std_props + allowed_docking_props

            if len(allowed_docking_props) == 0:
                probas = None

            else:
                probas = [
                    (1 - self.rule_set.probs_docking_targets) / len(allowed_std_props)
                ] * len(allowed_std_props) + [
                    self.rule_set.probs_docking_targets / len(allowed_docking_props)
                ] * len(allowed_docking_props)

            properties = list(
                random.choice(property_list, n_props, replace=False, p=probas)
            )
            # If n_props>=2, we ensure that we have at least one docking property
            if n_props >= 2:
                new_p = list(random.choice(allowed_docking_props, 1, replace=False))
                properties[0] = new_p[0]

            # If we generated more that self.rule_set.max_docking_per_prompt docking properties,
            # we remove the last and replace it by a standard property
            if (
                len([p for p in properties if p in allowed_docking_props])
                > self.rule_set.max_docking_per_prompt
            ):
                new_p = list(random.choice(allowed_std_props, 1, replace=False))
                properties[-1] = new_p[0]

            objectives = []
            for prop in properties:
                if (
                    len(prop) == 1
                    and self.prop_name_mapping[prop] in self.docking_targets
                ):
                    obj = random.choice(DOCKING_SOLO_OBJECTIVES)
                else:
                    obj = random.choice(OBJECTIVES)
                if obj in TARGET_VALUE_OBJECTIVES:
                    if self.prop_name_mapping[prop] in self.docking_targets:
                        v = random.randint(
                            1, 5
                        )  # Only docking scores between -10 and -7
                    else:
                        v = random.randint(1, 9)
                    obj += f" {v / 10}"
                objectives.append(obj)

            metadata["properties"] = [self.prop_name_mapping[p] for p in properties]
            metadata["objectives"] = [obj.split(" ")[0] for obj in objectives]
            metadata["target"] = [
                0 if len(obj.split(" ")) == 1 else obj.split(" ")[1]
                for obj in objectives
            ]
            identifier = "".join(
                sorted(
                    [
                        str(self.prop_key_list.index(prop))
                        + str(OBJECTIVES.index(obj.split(" ")[0]))
                        + str(
                            int(float(obj.split(" ")[-1]) * 10)
                            if len(obj.split(" ")) > 1
                            else 0
                        )
                        for prop, obj in zip(properties, objectives)
                    ]
                )
            )
            metadata["prompt_id"] = identifier
            metadata["n_props"] = n_props

            prompt_text = self.fill_prompt(properties, objectives)
            metadata["docking_metadata"] = {}
            for p in properties:
                if self.prop_name_mapping[p] in self.docking_targets:
                    pdb_id = self.prop_name_mapping[p]
                    if pdb_id in self.pockets_info:
                        pocket_data = self.pockets_info[pdb_id]
                        if not isinstance(pocket_data, dict):
                            pocket_data = dict(pocket_data)
                        metadata["docking_metadata"][self.prop_name_mapping[p]] = (
                            pocket_data
                        )
                    else:
                        metadata["docking_metadata"][self.prop_name_mapping[p]] = {
                            "pdb_id": pdb_id.split("_")[0]
                        }

            prompt: List[Dict[str, Any]] = []
            completion: List[Dict[str, Any]] = []

            prompt = [
                {"from": "human", "value": prompt_text[:-1] + "."},
                {"from": "assistant", "ground_truth": {"value": ""}},
            ]
            completion = [{}]
            yield prompt, completion, metadata

    def generate_with_rule(
        self, n: int, eval_name="", docking_split: int = 0
    ) -> Iterator[
        Tuple[
            Union[List[Dict[str, Any]], Dict[str, Any]],
            List[Dict[str, Any]],
            Dict[str, Any],
        ]
    ]:
        """
        Generates prompts, with at most n tries to obtain a prompt that meets the rule.
        """
        docking_prop_list: List[str] = self.docking_properties_split[docking_split]
        for _ in range(n):
            found = False
            for prompt, completions, metadata in self.generate(
                4 * n, docking_properties_list=docking_prop_list
            ):
                allowed = self.rule_set.verify_and_update(metadata)
                if allowed:
                    found = True
                    break
            if not found:
                break

            if eval_name == "":  # Generate a prompt for training
                yield prompt, completions, metadata
            else:  # Generate a prompt for evaluation
                new_prompt: Dict[str, Any] = {}
                new_prompt["prompt"] = [prompt[0]]
                new_prompt["final_answer"] = (
                    prompt[1].get("ground_truth", {}).get("value", "")
                )
                new_prompt["file_name"] = eval_name
                yield new_prompt, completions, metadata

    def generate_prompt_json(
        self, n: int, eval_name: str = "", docking_split: int = 0
    ) -> List[Any]:
        """
        Generates n prompts randomly to generate molecules.
        The generation is controlled by self.rule_set
        """
        assert docking_split < len(self.docking_properties_split), (
            f"docking_split must be less than the number of docking splits, here:{len(self.docking_properties_split)}"
        )
        out_dictionary = []
        p_bar = tqdm(total=n)
        for prompt, _, metadata in self.generate_with_rule(
            n, eval_name=eval_name, docking_split=docking_split
        ):
            # prompt["metadata"] = metadata
            out_dictionary.append(prompt)
            tqdm.update(p_bar)

        self.rule_set.partial_reset()
        return out_dictionary
