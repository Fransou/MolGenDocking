"""Dataset for generating prompts for molecule generation"""

from typing import Iterator
from numpy import random

from mol_gen_docking.grpo_rewards import KNOWN_PROPERTIES

OBJECTIVES = ["maximize", "minimize"]


class MolInstructionsDataset:
    """A simple Dataset generating rule-based prompts for molecule generation"""

    def __init__(self, max_n_props: int = 3, vina: bool = False):
        """
        :param max_n_props: Maximal number of properties to optimize
        """
        self.max_n_props = max_n_props
        if not vina:
            self.known_properties = [k for k in KNOWN_PROPERTIES if "docking" not in k]
        else:
            self.known_properties = KNOWN_PROPERTIES
        self.template = "I am a chemist working in drug discovery. Can you generate the SMILES representation of a molecule optimizing the following properties:"

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
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",  # Remove the last comma
                },
                {
                    "role": "user",
                    "content": prompt[:-1] + ".",  # Remove the last comma
                },
            ]
            if not return_n_props:
                yield prompt
            else:
                yield prompt, n_props
