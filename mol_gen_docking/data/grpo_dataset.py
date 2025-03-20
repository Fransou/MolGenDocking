"""Dataset for generating prompts for molecule generation"""

from typing import Iterator
from numpy import random
from datasets import Dataset
from tokenizers import Tokenizer

from mol_gen_docking.utils.grpo_rewards import KNOWN_PROPERTIES

OBJECTIVES = ["maximize", "minimize"]


class MolInstructionsDataset:
    """A simple Dataset generating rule-based prompts for molecule generation"""

    def __init__(self, max_n_props: int = 2, vina: bool = False):
        """
        :param max_n_props: Maximal number of properties to optimize
        """
        self.max_n_props = max_n_props
        if not vina:
            self.known_properties = [k for k in KNOWN_PROPERTIES if "docking" not in k]
        else:
            self.known_properties = KNOWN_PROPERTIES
        self.template = (
            "Can you generate a molecule optimizing the following properties:"
        )
        self.system_prompt = (
            "You are a helpful assistant. You can generate drug-like molecules"
            + " in the SMILES format between <SMILES> and </SMILES> tags."
        )

    def fill_prompt(self, prompt: str, property: str, objective: str) -> str:
        """Fills a prompt with a property and objective"""
        return prompt + f" {property} ({objective}),"

    def generate(self, n: int, return_n_props: bool = False) -> Iterator[str]:
        """
        Generates n prompts randomly to generate molecules
        :param n: number of prompts to generate
        :param return_n_props: if True, returns the number of properties to optimize
        :return:
        """
        for _ in range(n):
            n_props = random.randint(1, self.max_n_props)
            properties = random.choice(self.known_properties, n_props, replace=False)
            objectives = random.choice(OBJECTIVES, n_props)
            prompt = self.template
            for prop, obj in zip(properties, objectives):
                prompt = self.fill_prompt(prompt, prop, obj)

            prompt = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": prompt[:-1] + ".",  # Remove the last comma
                },
            ]
            completion = [
                {
                    "role": "assistant",
                    "content": r"<SMILES>O=C(NCc1ccc(Cl)cc1)c1ccc2c(c1)OCCO2</SMILES>",
                }
            ]
            if not return_n_props:
                yield prompt, completion
            else:
                yield prompt, completion, n_props

    def __call__(self, n: int, tokenizer: Tokenizer):
        out_dictionary = {"prompt": [], "completion": []}
        for prompt, completion in self.generate(n):
            out_dictionary["prompt"].append(prompt)
            out_dictionary["completion"].append(completion)
        del out_dictionary["completion"]

        dataset = Dataset.from_dict(out_dictionary)
        return dataset
