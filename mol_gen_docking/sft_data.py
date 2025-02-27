"""Preprocess the instruction dataset for the model training."""

from typing import Tuple, Dict

from datasets import load_dataset, Dataset, concatenate_datasets

import selfies as sf
from rdkit import Chem


special_tok = {
    "smiles": "<SMILES>",
    "smiles_end": "</SMILES>",
    "selfies": "<SELFIES>",
    "selfies_end": "</SELFIES>",
    "molformula": "<MOLFORMULA>",
    "molformula_end": "</MOLFORMULA>",
    "iupac": "<IUPAC>",
    "iupac_end": "</IUPAC>",
    "NUMBER": "<NUMBER>",
    "NUMBER_end": "</NUMBER>",
}


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
            self.dataset = load_dataset("osunlp/SMolInstruct")
        elif name == "Mol-Instructions":
            self.dataset = load_dataset(
                "zjunlp/Mol-Instructions", "Molecule-oriented Instructions"
            )

        else:
            raise ValueError("Unknown dataset")

    def is_selfies(self, string) -> bool:
        """
        Check if the string is a valid SELFIES.
        :param string: Input string
        :return: True if the string is a valid SELFIES, False otherwise
        """
        for spe_tok in special_tok.values():
            if spe_tok in string:
                return False
        try:
            if sf.decoder(string) == "":
                return False
            return True
        except Exception:
            return False

    def is_smiles(self, string) -> bool:
        """
        Check if the string is a valid SMILES.
        :param string: Input string
        :return: True if the string is a valid SMILES, False otherwise
        """
        for spe_tok in special_tok.values():
            if spe_tok in string:
                return False
        try:
            if Chem.MolFromSmiles(string) is None:
                return False
            return True
        except Exception:
            return False

    def process_str(self, line: Dict[str, str]) -> Dict[str, str]:
        """
        Process a line of the dataset.
        :param line: Line of the dataset
        :return: An instruction and a completion
        """

        instruction = line.get("instruction", "")
        inp = line.get("input", "")
        out = line["output"]
        if self.is_selfies(inp):
            inp = special_tok["selfies"] + inp + special_tok["selfies_end"]
        elif self.is_selfies(out):
            out = special_tok["selfies"] + out + special_tok["selfies_end"]

        return {"prompt": instruction + inp, "completion": out}

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
            for col in self.dataset.column_names
        ]
        self.dataset = self.dataset.map(
            self.process_str, num_proc=self.n_proc, remove_columns=cols_to_remove
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
        print(self.dataset)
        return self.dataset["train"], self.dataset["test"]
