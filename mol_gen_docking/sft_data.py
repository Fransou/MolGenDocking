from datasets import load_dataset, Dataset

import selfies as sf
from rdkit import Chem

from tqdm import trange


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
    def __init__(self, name: str):
        if name == "SMolInstruct":
            self.dataset = load_dataset(
                "zjunlp/Mol-Instructions", "Molecule-oriented Instructions"
            )
        elif name == "Mol-Instructions":
            self.dataset = load_dataset("osunlp/SMolInstruct")
        else:
            raise ValueError("Unknown dataset")

    def is_selfies(self, string):
        for spe_tok in special_tok.keys():
            if spe_tok in string:
                return False
        try:
            if sf.decoder(string) == "":
                return False
            return True
        except Exception:
            return False

    def is_smiles(self, string):
        for spe_tok in special_tok.keys():
            if spe_tok in string:
                return False
        try:
            if Chem.MolFromSmiles(string) is None:
                return False
            return True
        except Exception:
            return False

    def get_training_corpus(self, train_size, test_size):
        corpus = []
        for k in self.dataset.keys():
            print(self.dataset[k])
            for i in trange(len(self.dataset[k])):
                instruction = self.dataset[k][i].get("instruction", "")
                inp = self.dataset[k][i].get("input", "")
                out = self.dataset[k][i]["output"]
                if self.is_selfies(inp):
                    inp = special_tok["selfies"] + inp + special_tok["selfies_end"]
                elif self.is_selfies(out):
                    out = special_tok["selfies"] + out + special_tok["selfies_end"]
                corpus.append({"prompt": instruction + inp, "completion": out})
        dataset = Dataset.from_list(corpus)
        dataset = dataset.train_test_split(
            train_size=train_size, test_size=test_size, seed=42
        )
        return dataset["train"], dataset["test"]