"""Dataset for generating prompts for molecule generation"""

import json
import logging
import os
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from numpy import random
from tqdm import tqdm

from mol_gen_docking.reward.property_utils import (
    CLASSICAL_PROPERTIES_NAMES,
    PROPERTY_ALLOWED_OBJECTIVES,
    inverse_rescale_property_values,
)
from mol_gen_docking.reward.utils import (
    OBJECTIVES_TEMPLATES,
    POSSIBLE_POCKET_INFO,
    PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

OBJECTIVES = ["maximize", "minimize", "above", "below", "equal"]
DOCKING_SOLO_OBJECTIVES = ["minimize", "below"]
DOCKING_OBJECTIVES = ["minimize", "below", "above"]
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
                    "Prohibiting %s for n_props=%d",
                    prop,
                    n_props,
                )
        return True

    def partial_reset(self) -> None:
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
    min_n_pocket_infos: int = -1
    chat_template: Dict[str, str] = field(
        default_factory=lambda: {"user": "role", "content": "content"}
    )


class MolGenerationInstructionsDataset:
    """A simple Dataset generating rule-based prompts for molecule generation"""

    def __init__(
        self,
        config: DatasetConfig,
    ):
        """
        :param max_n_props: Maximal number of properties to optimize
        """
        self.config = config
        self.system_prompt = (
            "A conversation between User and Assistant. "
            "The User asks a question, and the Assistant solves it.\n"
            "The reasoning process is enclosed within <think> </think>"
            " and answer is enclosed within <answer> </answer> tags, "
            "respectively, i.e., <think> reasoning process here </think> "
            "<answer> SMILES here </answer>."
        )
        properties_csv = pd.read_csv(
            os.path.join(os.path.dirname(config.data_path), "properties.csv")
        )
        self.possible_smiles = properties_csv[
            properties_csv.smiles.apply(len) < 32
        ].smiles
        self.chat_temp = config.chat_template

        self.docking_targets: List[str] = []
        self.prop_name_mapping: Dict[str, str] = {}
        self.pockets_info: Dict[str, Any] = {}

        self.add_pocket_info = config.min_n_pocket_infos > 0
        self.min_n_pocket_infos = config.min_n_pocket_infos
        self._load_props(config.data_path)
        self.max_n_props = config.max_n_props

        self.std_properties: List[str] = [
            k
            for k in self.prop_name_mapping
            if self.prop_name_mapping[k] not in self.docking_targets
            and k in CLASSICAL_PROPERTIES_NAMES
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
            self._extract_splits(config.split_docking)  # Train, Val, Test (no leakage)

        self.obj_templates: Dict[str, List[str]] = OBJECTIVES_TEMPLATES
        self.templates: List[str] = PROMPT_TEMPLATE
        self.prop_key_list = list(self.prop_name_mapping.keys())
        self.rule_set = RuleSet(
            probs_docking_targets=config.probs_docking_targets,
            max_occ=config.max_occ,
            max_docking_per_prompt=config.max_docking_per_prompt,
        )

    def save_sim_matrices(self) -> None:
        # Get similarity matrix per split
        with open(
            os.path.join(self.config.data_path, "val_dist_to_train.json"), "w"
        ) as f:
            json.dump(self._get_similarity_matrix(0, 1, self.config.data_path), f)
        with open(
            os.path.join(self.config.data_path, "val_dist_to_train.json"), "w"
        ) as f:
            json.dump(self._get_similarity_matrix(0, 2, self.config.data_path), f)
        with open(
            os.path.join(self.config.data_path, "val_dist_to_val.json"), "w"
        ) as f:
            json.dump(self._get_similarity_matrix(1, 1, self.config.data_path), f)
        with open(
            os.path.join(self.config.data_path, "test_dist_to_test.json"), "w"
        ) as f:
            json.dump(self._get_similarity_matrix(2, 2, self.config.data_path), f)

    @staticmethod
    def _get_allowed_props(
        original_prop_list: List[str], rule_set: RuleSet, n_props: int
    ) -> List[str]:
        return [
            p
            for p in original_prop_list
            if p not in rule_set.prohibited_props_at_n.get(n_props, [])
        ]

    def _get_similarity_matrix(
        self, i0: int, i1: int, path: str
    ) -> Dict[str, Dict[str, float]]:
        pdb_ids_list0 = [
            self.prop_name_mapping[p]
            for p in self.docking_properties_split[i0]
            if not self.prop_name_mapping[p].endswith("_docking")
        ]
        pdb_ids_list1 = [
            self.prop_name_mapping[p]
            for p in self.docking_properties_split[i1]
            if not self.prop_name_mapping[p].endswith("_docking")
        ]

        similarities: Dict[str, Dict[str, float]] = {p: {} for p in pdb_ids_list0}
        args = [(p0, p1, path) for p0 in pdb_ids_list0 for p1 in pdb_ids_list1]
        pool = Pool(4)
        results = list(
            tqdm(
                pool.imap(self.get_similarity, args),
                total=len(args),
                desc=f"Similarity computing between {i0} and {i1}",
            )
        )
        for r, ar in zip(results, args):
            similarities[ar[0]][ar[1]] = r

        return similarities

    def get_similarity(self, inp: Tuple[str, str, str]) -> float:
        from tmtools import tm_align
        from tmtools.io import get_residue_data, get_structure

        pdb0, pdb1, path = inp

        pocket_id0 = self.pockets_info[pdb0]["metadata"]["pocket_id"]
        pocket_id1 = self.pockets_info[pdb1]["metadata"]["pocket_id"]

        pocket0 = get_structure(
            os.path.join(
                path,
                "pdb_files",
                pdb0 + "_processed_out",
                "pockets",
                f"pocket{pocket_id0}_atm.pdb",
            )
        )
        pocket1 = get_structure(
            os.path.join(
                path,
                "pdb_files",
                pdb1 + "_processed_out",
                "pockets",
                f"pocket{pocket_id1}_atm.pdb",
            )
        )

        s0 = get_structure(os.path.join(path, "pdb_files", pdb0 + "_processed.pdb"))
        s1 = get_structure(os.path.join(path, "pdb_files", pdb1 + "_processed.pdb"))

        def get_closest_chain(structure: Any, pocket: Any) -> Any:
            pocket_atoms = [atom.get_coord() for atom in pocket.get_atoms()]
            pocket_center = np.mean(pocket_atoms, axis=0)

            min_dist = float("inf")
            closest_chain = None
            for chain in structure.get_chains():
                try:
                    chain_coords = np.array(
                        [atom.get_coord() for atom in chain.get_atoms()]
                    )
                    if chain_coords.shape[0] < 3:
                        continue
                    chain_center = np.mean(chain_coords, axis=0)
                    dist = float(np.linalg.norm(pocket_center - chain_center))
                    if dist < min_dist:
                        min_dist = dist
                        closest_chain = chain
                except Exception:
                    continue
            return closest_chain

        chain0 = get_closest_chain(s0, pocket0)
        chain1 = get_closest_chain(s1, pocket1)

        def normalize_resnames(chain: Any) -> None:
            rename_map = {
                "HIE": "HIS",
                "HIP": "HIS",
                "HID": "HIS",
                "ASH": "ASP",
                "GLH": "GLU",
                "CYX": "CYS",
                "CSS": "CYS",
                # Add more if needed
            }
            for res in chain:
                if res.resname in rename_map:
                    res.resname = rename_map[res.resname]

        normalize_resnames(chain0)
        normalize_resnames(chain1)

        try:
            coords0, seq0 = get_residue_data(chain0)
            coords1, seq1 = get_residue_data(chain1)
            if len(seq0) < 3 or len(seq1) < 3:
                out = 10.0
            out = tm_align(coords0, coords1, seq0, seq1).rmsd
        except Exception:
            out = 10.0
        return out

    def _load_props(self, path: str) -> None:
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

    def _extract_splits(self, split_docking: List[float]) -> None:
        np.random.shuffle(self.docking_properties)
        i0 = 0
        for idx, p in enumerate(split_docking):
            i1 = i0 + int(len(self.docking_properties) * p)
            i1 = min(i1, len(self.docking_properties))
            self.docking_properties_split[idx] = self.docking_properties[i0:i1]
            i0 = i1

    def fill_prompt(
        self, props: List[str], objs: List[str]
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Takes a list of properties and corresponding objectives
        and returns a diverse natural language prompt for multi-objective optimization.
        """
        if len(props) != len(objs):
            raise ValueError("props and objs must have the same length.")

        # Phrase templates for each type of objective

        phrases = []
        phrases_mm = []
        path_to_mm_object = []
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
            # Check if it is a docking target
            if self.prop_name_mapping.get(prop, prop) in self.docking_targets:
                phrases_mm.append("<|image_pad|> " + phrase)
                path_to_mm_object.append(
                    {
                        "type": "image",  # HACK
                        "path": os.path.join(
                            "pockets_embeddings",
                            self.prop_name_mapping.get(prop, prop) + "_embeddings.pt",
                        ),
                    }
                )
            else:
                phrases_mm.append(phrase)

        # Top-level prompt templates
        prompt: str = random.choice(self.templates).split("|")[int(len(props) > 1)]
        full_prompt = prompt.format(objectives="; ".join(phrases))
        prompt_mm = prompt.format(objectives="; ".join(phrases_mm))

        full_prompt_mm: List[Dict[str, str]] = [
            {"type": "text", "text": prompt_mm}
        ] + path_to_mm_object

        return full_prompt, full_prompt_mm

    def _generate_pocket_additional_data(self, properties: List[str]) -> Dict[str, Any]:
        pocket_datas: Dict[str, Any] = {}
        if self.add_pocket_info:
            for p in properties:
                pdb_id = self.prop_name_mapping[p]
                if pdb_id in self.docking_targets and pdb_id in self.pockets_info:
                    pocket_metadata = self.pockets_info[pdb_id].get("metadata", {})
                    if not isinstance(pocket_metadata, dict):
                        pocket_metadata = dict(pocket_metadata)
                    if self.min_n_pocket_infos < len(POSSIBLE_POCKET_INFO):
                        n_props = np.random.randint(
                            self.min_n_pocket_infos, len(POSSIBLE_POCKET_INFO)
                        )
                    else:
                        n_props = len(POSSIBLE_POCKET_INFO)
                    dict_keys = np.random.choice(
                        POSSIBLE_POCKET_INFO, n_props, replace=False
                    )
                    pocket_data = {
                        k: pocket_metadata[k] for k in dict_keys if k in pocket_metadata
                    }
                    pocket_datas[pdb_id] = pocket_data
        return pocket_datas

    def _get_prompt_metadata(
        self,
        properties: List[str],
        objectives: List[str],
        identifier: str,
        n_props: int,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        metadata["properties"] = [self.prop_name_mapping[p] for p in properties]
        metadata["objectives"] = [obj.split(" ")[0] for obj in objectives]
        metadata["target"] = [
            0.0 if len(obj.split(" ")) == 1 else float(obj.split(" ")[1])
            for obj in objectives
        ]
        metadata["prompt_id"] = identifier
        metadata["n_props"] = n_props
        metadata["docking_metadata"] = []
        for p in properties:
            if self.prop_name_mapping[p] in self.docking_targets:
                pdb_id = self.prop_name_mapping[p]
                if pdb_id in self.pockets_info:
                    infos = {}

                    pocket_data = self.pockets_info[pdb_id]
                    if not isinstance(pocket_data, dict):
                        pocket_data = dict(pocket_data)
                    infos = pocket_data
                    if "pdb_id" not in infos:
                        infos["pdb_id"] = pdb_id
                else:
                    infos = {"pdb_id": pdb_id.split("_")[0]}
                metadata["docking_metadata"].append(infos)
        return metadata

    def _sample_properties(
        self, n_props: int, docking_properties_list: List[str]
    ) -> List[str]:
        allowed_docking_props = self._get_allowed_props(
            docking_properties_list, self.rule_set, n_props=n_props
        )
        allowed_std_props = self._get_allowed_props(
            self.std_properties, self.rule_set, n_props=n_props
        )

        if len(allowed_docking_props) == 0:
            probas = None
        else:
            allowed_docking_props = np.random.choice(
                allowed_docking_props,
                min(self.rule_set.max_docking_per_prompt, len(allowed_docking_props)),
                replace=False,
            ).tolist()
            probas = [
                (1 - self.rule_set.probs_docking_targets) / len(allowed_std_props)
            ] * len(allowed_std_props) + [
                self.rule_set.probs_docking_targets / len(allowed_docking_props)
            ] * len(allowed_docking_props)

        property_list = allowed_std_props + allowed_docking_props

        properties = list(
            random.choice(property_list, n_props, replace=False, p=probas)
        )
        # If n_props>=2, we ensure that we have at least one docking property
        if n_props >= 2 and len(np.intersect1d(allowed_docking_props, properties)) == 0:
            properties[0] = allowed_docking_props[0]
        return properties

    def add_pocket_info_to_prompt(
        self, full_prompt: str, pocket_data: Dict[str, Any]
    ) -> str:
        if not pocket_data == {}:
            new_sentence = (
                " Here are some descriptors of the pocket"
                + "s" * int(len(pocket_data) > 1)
                + " we want the compound to bind to: \n"
            )
            for p in pocket_data:
                new_sentence += p + ":\n"
                for k in pocket_data[p]:
                    new_sentence += "    " + k + ":" + str(pocket_data[p][k]) + "\n"

            full_prompt += new_sentence
        return full_prompt

    def get_obj_from_prop(self, properties: List[str]) -> List[str]:
        objectives: List[str] = []
        n_dock_props = len(
            [p for p in properties if self.prop_name_mapping[p] in self.docking_targets]
        )
        for prop in properties:
            short_prop = self.prop_name_mapping[prop]
            if n_dock_props == 1 and short_prop in self.docking_targets:
                obj = random.choice(DOCKING_SOLO_OBJECTIVES)
            elif short_prop in self.docking_targets:
                obj = random.choice(DOCKING_OBJECTIVES)
            else:
                obj = random.choice(PROPERTY_ALLOWED_OBJECTIVES[short_prop])

            if obj in TARGET_VALUE_OBJECTIVES:
                # Find the value to target
                if short_prop in self.docking_targets:
                    v = random.random() * 0.4 + 0.1
                    # Only docking scores between -10 and -7
                else:
                    v = random.random() * 0.6 + 0.2

                obj += f" {inverse_rescale_property_values(short_prop, v / 10, short_prop in self.docking_targets):.2f}"
            objectives.append(obj)
        return objectives

    def generate_text_prompts(
        self, properties: List[str], objectives: List[str], pocket_datas: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        prompt_text, prompt_multimodal = self.fill_prompt(properties, objectives)
        prompt_text_with_pocket = self.add_pocket_info_to_prompt(
            prompt_text, pocket_datas
        )

        sys_prompt = self.system_prompt

        prompt: Dict[str, List[Dict[str, Any]]] = {
            "standard": [
                {
                    self.chat_temp["user"]: "system",
                    self.chat_temp["content"]: sys_prompt,
                },
                {
                    self.chat_temp["user"]: "user",
                    self.chat_temp["content"]: prompt_text,
                },
            ],
            "with_pocket_descriptors": [
                {self.chat_temp["user"]: "system", "content": sys_prompt},
                {
                    self.chat_temp["user"]: "user",
                    self.chat_temp["content"]: prompt_text_with_pocket,
                },
            ],
            "multimodal": [
                {
                    self.chat_temp["user"]: "system",
                    self.chat_temp["content"]: [{"type": "text", "text": sys_prompt}],
                },
                {
                    self.chat_temp["user"]: "user",
                    self.chat_temp["content"]: prompt_multimodal,
                },
            ],
        }
        return prompt

    def generate(
        self,
        n: int,
        docking_properties_list: List[str],
    ) -> Iterator[Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]]:
        """
        Generates n prompts randomly to generate molecules
        :param n: number of prompts to generate
        :param return_n_props: if True, returns the number of properties to optimize
        :return:
        """
        for _ in range(n):
            possible_n = [
                i
                for i in range(1, self.max_n_props + 1)
                if len(self.rule_set.prohibited_props_at_n.get(i, []))
                < len(docking_properties_list)
            ]
            n_props: int = int(np.random.choice(possible_n))
            properties = self._sample_properties(n_props, docking_properties_list)
            objectives = self.get_obj_from_prop(properties)

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

            pocket_datas = self._generate_pocket_additional_data(properties)
            prompt = self.generate_text_prompts(properties, objectives, pocket_datas)

            metadata = self._get_prompt_metadata(
                properties, objectives, identifier, n_props
            )
            yield prompt, metadata

    def generate_with_rule(
        self, n: int, docking_split: int = 0
    ) -> Iterator[
        Tuple[
            Dict[str, List[Dict[str, Any]]],
            Dict[str, Any],
        ]
    ]:
        """
        Generates prompts, with at most n tries to obtain a prompt that meets the rule.
        """
        docking_prop_list: List[str] = self.docking_properties_split[docking_split]
        for _ in range(n):
            found = False
            for prompt, metadata in self.generate(
                4 * n, docking_properties_list=docking_prop_list
            ):
                allowed = self.rule_set.verify_and_update(metadata)
                if allowed:
                    found = True
                    break
            if not found:
                break
            yield prompt, metadata

    def generate_hf_dataset(self, n: int, docking_split: int) -> Dataset:
        assert docking_split < len(self.docking_properties_split), (
            f"docking_split must be less than the number of docking splits, here:{len(self.docking_properties_split)}"
        )
        data_dict: Dict[str, Any] = {
            "prompt": [],
            "prompt_pocket_descriptors": [],
            "prompt_multimodal": [],
            "properties": [],
            "objectives": [],
            "target": [],
            "prompt_id": [],
            "n_props": [],
            "docking_metadata": [],
        }
        p_bar = tqdm(total=n)
        for prompt, metadata in self.generate_with_rule(n, docking_split=docking_split):
            jsonize_dict(metadata)
            data_dict["prompt"].append(prompt["standard"])
            data_dict["prompt_pocket_descriptors"].append(
                prompt["with_pocket_descriptors"]
            )
            data_dict["prompt_multimodal"].append(prompt["multimodal"])
            for k in metadata:
                if k in data_dict:
                    data_dict[k].append(metadata[k])
            tqdm.update(p_bar)
        print(data_dict["prompt_multimodal"])
        self.rule_set.partial_reset()
        return Dataset.from_dict(data_dict)


def jsonize_dict(d: Dict[Any, Any]) -> None:
    for k, v in d.items():
        if isinstance(v, dict):
            jsonize_dict(v)
        elif isinstance(v, np.ndarray):
            d[k] = v.tolist()
