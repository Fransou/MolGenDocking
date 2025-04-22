"""Dataset for generating prompts for molecule generation"""

from typing import Iterator, Tuple, Dict, List, Literal, Any, Union
from numpy import random
from datasets import Dataset
from tqdm import tqdm

from mol_gen_docking.reward.oracles import (
    PROPERTIES_NAMES_SIMPLE,
    OBJECTIVES_TEMPLATES,
    PROMPT_TEMPLATE,
)

OBJECTIVES = ["maximize", "minimize", "below", "above", "equal"]
DOCKING_SOLO_OBJECTIVES = ["minimize", "below", "equal"]
TARGET_VALUE_OBJECTIVES = ["below", "above", "equal"]


class MolGenerationInstructionsDataset:
    """A simple Dataset generating rule-based prompts for molecule generation"""

    def __init__(
        self, max_n_props: int = 5, vina: bool = False, n_max_occurence: int = 10
    ):
        """
        :param max_n_props: Maximal number of properties to optimize
        """
        self.max_n_props = max_n_props
        self.n_max_occurence = n_max_occurence
        if not vina:
            self.known_properties = [
                k
                for k in PROPERTIES_NAMES_SIMPLE
                if "docking" not in PROPERTIES_NAMES_SIMPLE[k]
            ]
        else:
            self.known_properties = list(PROPERTIES_NAMES_SIMPLE.keys())
        self.obj_templates: Dict[str, List[str]] = OBJECTIVES_TEMPLATES
        self.templates: List[str] = PROMPT_TEMPLATE

        self.system_prompt = (
            "You are a helpful assistant. You can generate drug-like molecules"
            + " in the SMILES format between <SMILES> and </SMILES> tags."
        )
        self.rule_set: Dict[
            int, Dict[str, int]
        ] = {}  # n_times a given propery was used for each
        # mulit-obj prompt
        self.generated: Dict[int, List[str]] = {}  # List of generated prompts

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
        self, n: int, format: Literal["chat_format", "orz"] = "chat_format"
    ) -> Iterator[Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]]:
        """
        Generates n prompts randomly to generate molecules
        :param n: number of prompts to generate
        :param return_n_props: if True, returns the number of properties to optimize
        :return:
        """
        for _ in range(n):
            metadata: Dict[str, Any] = {}

            n_props: int = int(random.randint(1, 1 + self.max_n_props))
            properties = list(
                random.choice(self.known_properties, n_props, replace=False)
            )
            objectives = []
            for prop in properties:
                if len(prop) == 1 and "docking" in PROPERTIES_NAMES_SIMPLE[prop]:
                    obj = random.choice(DOCKING_SOLO_OBJECTIVES)
                else:
                    obj = random.choice(OBJECTIVES)
                if obj in TARGET_VALUE_OBJECTIVES:
                    if "docking" in PROPERTIES_NAMES_SIMPLE[prop]:
                        v = random.randint(
                            1, 5
                        )  # Only docking scores between -10 and -7
                    else:
                        v = random.randint(0, 10)
                    obj += f" {v / 10}"
                objectives.append(obj)

            metadata["properties"] = properties
            metadata["objectives"] = [obj.split(" ")[0] for obj in objectives]
            metadata["target"] = [
                0 if len(obj.split(" ")) == 1 else obj.split(" ")[1]
                for obj in objectives
            ]
            metadata["prompt_id"] = "".join(
                sorted(
                    [
                        prop + " " + obj + " | "
                        for prop, obj in zip(properties, objectives)
                    ]
                )
            )[:-3]
            metadata["n_props"] = n_props

            prompt_text = self.fill_prompt(properties, objectives)

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
                yield prompt, completion, metadata
            elif format == "orz":
                prompt = [
                    {"from": "human", "value": prompt_text[:-1] + "."},
                    {"from": "assistant", "ground_truth": {"value": ""}},
                ]
                completion = [{}]
                yield prompt, completion, metadata

    def generate_with_rule(
        self, n: int, format: Literal["chat_format", "orz"], eval_name=""
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
        for _ in range(n):
            found = False
            for prompt, completions, metadata in self.generate(4 * n, format=format):
                n_props = metadata["n_props"]
                if n_props not in self.rule_set:
                    self.rule_set[n_props] = {prop: 0 for prop in self.known_properties}
                    self.generated[n_props] = []
                allowed = metadata["prompt_id"] not in self.generated[n_props]

                for prop in metadata["properties"]:
                    allowed = allowed and (
                        self.rule_set[metadata["n_props"]][prop] < self.n_max_occurence
                    )

                if allowed:
                    found = True
                    break
            if not found:
                break
            for prop in metadata["properties"]:
                self.rule_set[metadata["n_props"]][prop] += 1
            self.generated[n_props].append(metadata["prompt_id"])
            if eval_name != "":
                yield prompt, completions, metadata
            else:
                new_prompt: Dict[str, Any] = {}
                new_prompt["prompt"] = [prompt[0]]
                new_prompt["final_answer"] = prompt[1]["ground_truth"]["value"]
                new_prompt["file_name"] = eval_name
                yield new_prompt, completions, metadata

    def generate_prompt_json(
        self,
        n: int,
        format: Literal["chat_format", "orz"] = "chat_format",
        eval_name: str = "",
    ) -> List[Any]:
        """
        Generates n prompts randomly to generate molecules.
        The generation is controlled by the self.rule_set dictionary
        """
        out_dictionary = []
        p_bar = tqdm(total=n)
        for prompt, _, metadata in self.generate_with_rule(
            n, format=format, eval_name=eval_name
        ):
            out_dictionary.append(prompt)
            tqdm.update(p_bar)
        return out_dictionary

    def __call__(
        self, n: int, format: Literal["chat_format", "orz"] = "chat_format"
    ) -> Dataset:
        out_dictionary: Dict[str, List[List[Dict[str, Any]]]] = {
            "prompt": [],
            "completion": [],
        }
        for prompt, completion, *_ in self.generate_with_rule(n, format=format):
            assert isinstance(prompt, list)
            out_dictionary["prompt"].append(prompt)
            out_dictionary["completion"].append(completion)
        del out_dictionary["completion"]

        dataset = Dataset.from_dict(out_dictionary)
        return dataset
