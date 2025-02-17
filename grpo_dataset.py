from IPython.core.debugger import prompt

from grpo_rewards import KNOWN_PROPERTIES
from numpy import random

OBJECTIVES = ["maximize", "minimize"]


class MolInstructionsDataset:
    def __init__(self, max_n_props: int = 3, max_n_generated: int = 10):
        self.max_n_props = max_n_props
        self.max_n_generated = max_n_generated
        self.known_properties = KNOWN_PROPERTIES
        self.template = "Generate a molecule optimizing the following properties:"

    def fill_prompt(self, prompt: str, property: str, objective: str):
        return prompt + f" {property} ({objective}),"

    def generate(self, n: int):
        for _ in range(n):
            n_props = random.randint(1, self.max_n_props)
            properties = random.choice(self.known_properties, n_props, replace=False)
            objectives = random.choice(OBJECTIVES, n_props)
            prompt = self.template
            for prop, obj in zip(properties, objectives):
                prompt = self.fill_prompt(prompt, prop, obj)
            yield prompt[:-1] + "."


if __name__ == "__main__":
    dataset = MolInstructionsDataset()
    for prompt in dataset.generate(10):
        print(prompt)
