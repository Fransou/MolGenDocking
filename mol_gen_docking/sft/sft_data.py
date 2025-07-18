"""Preprocess the instruction dataset for the model training."""

from typing import Any, Dict, List, Tuple

from datasets import Dataset, concatenate_datasets, load_dataset
from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.RDLogger import RDLogger

RDLogger.DisableLog("rdApp.*")

SMolInstruct_tasks = [
    "forward_synthesis",
    "retrosynthesis",
    "molecule_captioning",
    "molecule_generation",
    # "property_prediction-esol",
    # "property_prediction-lipo",
    # "property_prediction-bbbp",
    # "property_prediction-clintox",
    # "property_prediction-hiv",
    # "property_prediction-sider",
]


class InstructionDatasetProcessor:
    """
    Preprocess the instruction dataset for the model training.
    """

    def __init__(self, name: str, n_proc: int = 8):
        """
        :param name: Name of the dataset
        """
        self.name = name
        self.n_proc = n_proc
        self.processed = False

        if name == "SMolInstruct":
            self.dataset = load_dataset(
                "osunlp/SMolInstruct",
                tasks=SMolInstruct_tasks,
                insert_core_tags=False
            )
        elif name == "Mol-Instructions":
            self.dataset = load_dataset(
                "zjunlp/Mol-Instructions",
                "Molecule-oriented Instructions",
                trust_remote_code=True,
            )

        else:
            raise ValueError("Unknown dataset")

    def process_line(self, line: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process a line of the dataset.
        :param line: Line of the dataset
        :return: An instruction and a completion
        """

        instruction = line.get("instruction", "")
        inp = line.get("input", "")
        out = line["output"]

        raw_input = line.get("raw_input", "")
        if ("C" in raw_input or "c" in raw_input) and not "e" in raw_input: # Can be a smiles
            if MolFromSmiles(raw_input) is not None:
                new_raw_input = "; ".join(raw_input.split("."))
                inp = inp.replace(raw_input, new_raw_input)
        raw_output = line.get("raw_output", "")
        if ("C" in raw_output or "c" in raw_output) and not "e" in raw_output: # Can be a smiles
            if MolFromSmiles(raw_output) is not None:
                new_raw_output = "; ".join(raw_output.split("."))
                out = out.replace(raw_output, new_raw_output)

        prompt = [
            {
                "role": "user",
                "content": instruction + " " + inp,
            },
        ]
        completion = [
            {"role": "assistant", "content": out},
        ]

        return {
            "prompt": prompt,
            "completion": completion,
        }

    def get_training_corpus(
        self,
        train_size: int = -1,
    ) -> Tuple[Dataset, Dataset]:
        """
        Get the training corpus.
        :param train_size: Amount of training data
        :param test_size: Amount of testing data
        :return: Training and testing datasets
        """
        if self.processed:
            return self.dataset["train"], self.dataset["test"]

        cols_to_remove = [
            col
            for col in ["input", "instruction", "output"]
            if col in self.dataset.column_names
        ]
        for k in self.dataset.keys():
            self.dataset[k] = self.dataset[k].map(
                self.process_line,
                num_proc=self.n_proc,
                remove_columns=cols_to_remove,
                # load_from_cache_file=False,
            )
        # If train and test are not specified, flatten the dataset and split it
        if not ("train" in self.dataset.keys() and "test" in self.dataset.keys()):
            self.dataset = concatenate_datasets(
                [self.dataset[k] for k in self.dataset.keys()]
            )
            if train_size == -1:
                train_size = int(0.9 * len(self.dataset))
                test_size = len(self.dataset) - train_size
            else:
                train_size = min(train_size, int(0.9 * len(self.dataset)))
                test_size = int(0.1 * train_size)

            self.dataset = self.dataset.train_test_split(
                train_size=train_size, test_size=test_size, seed=42
            )
        elif train_size < 0.9 * len(self.dataset["train"]):
            self.dataset = self.dataset["train"].train_test_split(
                train_size=train_size, test_size=int(0.1 * train_size), seed=42
            )
        self.processed = True

        self.dataset["train"].shuffle()
        self.dataset["train"].flatten_indices()

        return self.dataset["train"], self.dataset["test"]
