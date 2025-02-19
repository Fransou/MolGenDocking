import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


class MolTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model, self.tokenizer = self.get_model()
        self.dataset, self.eval_dataset = self.get_dataset()

    def get_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=self.args.local_files_only
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            local_files_only=self.args.local_files_only
        )
        return model, tokenizer

    def get_dataset(self):
       raise NotImplementedError

    def get_trainer(self):
        raise NotImplementedError

    def __call__(self):
        os.environ["WANDB_MODE"]="offline"
        trainer = self.get_trainer()
        print("LAUNCHING TRAINING")
        trainer.train()
