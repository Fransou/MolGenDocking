"""Preprocess the instruction dataset for the model training."""

from typing import Tuple, Dict

from datasets import load_dataset, Dataset, concatenate_datasets

import selfies as sf
from rdkit import Chem


special_tok = {"smiles": "<SMILES>", "smiles_end": "</SMILES>"}

SMolInstruct_tasks = [
    "forward_synthesis",
    "retrosynthesis",
    "molecule_captioning",
    "molecule_generation",
    "property_prediction-esol",
    "property_prediction-lipo",
    "property_prediction-bbbp",
    "property_prediction-clintox",
    "property_prediction-hiv",
    "property_prediction-sider",
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
            )
        elif name == "Mol-Instructions":
            self.dataset = load_dataset(
                "zjunlp/Mol-Instructions",
                "Molecule-oriented Instructions",
                trust_remote_code=True,
            )

        else:
            raise ValueError("Unknown dataset")

        self.system_prompt = (
            "You are a helpful assistant. You can describe molecules"
            + " in the SMILES format between the <SMILES> and </SMILES> tags."
        )

    def process_str(self, string) -> str:
        """
        Check if the string is a valid SELFIES, or if it contains special tokens.
        Avoids useless tasks, and processes the string if needed.
        :param string: Input string
        :return: The input string processed
        """
        if string == "":
            return ""
        found_a_special_tok = False
        for spe_tok in special_tok.values():
            if spe_tok in string:
                found_a_special_tok = True
        if not found_a_special_tok:
            try:
                mol = sf.decoder(string)
                if mol != "":
                    # Get the molecule
                    mol = Chem.MolFromSmiles(mol)
                    # Remove stereochemistry
                    Chem.RemoveStereochemistry(mol)
                    # Get the SMILES
                    string = Chem.MolToSmiles(mol)
                    return special_tok["smiles"] + string + special_tok["smiles_end"]
            except Exception as e:
                del e

        return string

    def process_line(self, line: Dict[str, str]) -> Dict[str, str]:
        """
        Process a line of the dataset.
        :param line: Line of the dataset
        :return: An instruction and a completion
        """

        instruction = line.get("instruction", "")
        inp = line.get("input", "")
        out = line["output"]

        inp = self.process_str(inp)
        instruction = self.process_str(instruction)
        out = self.process_str(out)

        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": instruction + inp,
            },
            {"role": "assistant", "content": out},
        ]
        return {"messages": message}

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
                load_from_cache_file=False,
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

        for k in self.dataset.keys():
            self.dataset[k] = self.dataset[k].select_columns(["messages"])

        self.processed = True

        self.dataset["train"].shuffle()
        self.dataset["train"].flatten_indices()
        return self.dataset["train"], self.dataset["test"]
