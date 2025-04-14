"""Dataset for generating prompts for molecule generation"""

from typing import Iterator, Tuple, Dict, List, Literal, Any
from numpy import random
from datasets import Dataset
from tqdm import tqdm
from itertools import chain

from mol_gen_docking.reward.oracles import PROPERTIES_NAMES_SIMPLE

OBJECTIVES = ["maximize", "minimize", "below", "above", "equal"]
DOCKING_SOLO_OBJECTIVES = ["minimize", "below", "equal"]
TARGET_VALUE_OBJECTIVES = ["below", "above", "equal"]


class MolGenerationInstructionsDataset:
    """A simple Dataset generating rule-based prompts for molecule generation"""

    def __init__(self, max_n_props: int = 5, vina: bool = False):
        """
        :param max_n_props: Maximal number of properties to optimize
        """
        self.max_n_props = max_n_props
        if not vina:
            self.known_properties = [
                k
                for k in PROPERTIES_NAMES_SIMPLE
                if "docking" not in PROPERTIES_NAMES_SIMPLE[k]
            ]
        else:
            self.known_properties = list(PROPERTIES_NAMES_SIMPLE.keys())
        self.template = "Can you generate a molecule in the SMILES format optimizing the following properties:"
        self.system_prompt = (
            "You are a helpful assistant. You can generate drug-like molecules"
            + " in the SMILES format between <SMILES> and </SMILES> tags."
        )

    def fill_prompt(self, prompt: str, property: str, objective: str) -> str:
        """Fills a prompt with a property and objective"""
        return prompt + f" {property} ({objective}),"

    def generate(
        self, n: int, format: Literal["chat_format", "orz"] = "chat_format"
    ) -> Iterator[
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, Dict[str, Any]]
    ]:
        """
        Generates n prompts randomly to generate molecules
        :param n: number of prompts to generate
        :param return_n_props: if True, returns the number of properties to optimize
        :return:
        """
        for _ in range(n):
            metadata: Dict[str, Any] = {}

            n_props: int = int(random.randint(1, 1 + self.max_n_props))
            properties = random.choice(self.known_properties, n_props, replace=False)
            objectives = []
            for prop in properties:
                if len(prop) == 1 and "docking" in PROPERTIES_NAMES_SIMPLE[prop]:
                    obj = random.choice(DOCKING_SOLO_OBJECTIVES)
                else:
                    obj = random.choice(OBJECTIVES)
                if obj in TARGET_VALUE_OBJECTIVES:
                    v = random.randint(1, 4)
                    obj += f" {v / 10}"
                objectives.append(obj)

            metadata["properties"] = list(properties)
            metadata["objectives"] = [obj.split(" ")[0] for obj in objectives]
            metadata["target"] = [
                0 if len(obj.split(" ")) == 1 else obj.split(" ")[1]
                for obj in objectives
            ]
            metadata["n_props"] = n_props

            prompt_text = self.template
            for prop, obj in zip(properties, objectives):
                prompt_text = self.fill_prompt(prompt_text, prop, obj)
            prompt: List[Dict[str, Any]] = []
            completion: List[Dict[str, Any]] = []
            if format == "chat_format":
                prompt = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": prompt_text[:-1] + ".",  # Remove the last comma
                    },
                ]
                completion = [
                    {
                        "role": "assistant",
                        "content": r"<think>",
                    }
                ]
                yield prompt, completion, n_props, metadata
            elif format == "orz":
                prompt = [
                    {"from": "human", "value": prompt_text[:-1] + "."},
                    {"from": "assistant", "ground_truth": {"value": ""}},
                ]
                completion = [{}]
                yield prompt, completion, n_props, metadata

    def generate_prompt_json(
        self, n: int, format: Literal["chat_format", "orz"] = "chat_format"
    ) -> List[List[Dict[str, Any]]]:
        """
        Generates n prompts randomly to generate molecules.
        The same optimization problem cannot be generated multiple times,
        evwn with different target values.
        """
        out_dictionary = []
        already_seen: List[str] = []
        p_bar = tqdm(total=n)
        for prompt, _, _, metadata in self.generate(n, format=format):
            hash = "".join(
                list(chain(*[metadata["properties"], metadata["objectives"]]))
            )
            if hash in already_seen:
                for p, _, _, met in self.generate(n, format=format):
                    found = False
                    hash = "".join(list(chain(*[met["properties"], met["objectives"]])))
                    if hash not in already_seen:
                        prompt = p
                        metadata = met
                        found = True
                        break
                if not found:
                    break
            out_dictionary.append(prompt)
            already_seen.append(hash)
            tqdm.update(p_bar)
        return out_dictionary

    def __call__(self, n: int) -> Dataset:
        out_dictionary: Dict[str, List[List[Dict[str, str]]]] = {
            "prompt": [],
            "completion": [],
        }
        for prompt, completion, *_ in self.generate(n):
            out_dictionary["prompt"].append(prompt)
            out_dictionary["completion"].append(completion)
        del out_dictionary["completion"]

        dataset = Dataset.from_dict(out_dictionary)
        return dataset
