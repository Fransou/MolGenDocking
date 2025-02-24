"""Base class for the trainer."""

import os
import argparse
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from datasets import Dataset
import submitit


class MolTrainer(submitit.helpers.Checkpointable):
    """Base class for the trainer."""

    def __init__(
        self, args: argparse.Namespace, datasets: Optional[Tuple[Dataset, Dataset]] = None
    ):
        """
        :param args: Parameters for the training
        :param datasets: training and evaluation datasets (if None, will be loaded)
        """
        super().__init__()

        self.args = args
        self.checkpoint_path = ""
        self.model = None
        self.tokenizer = None

        if datasets is None:
            self.dataset, self.eval_dataset = self.get_dataset()
        else:
            self.dataset, self.eval_dataset = datasets

    def retrieve_checkpoint_step(self) -> str:
        """
        Retrieve the last checkpoint step
        :return: path of the last checkpoint
        """
        checkpoints_step = sorted(
            [int(d.split("-")[-1]) for d in os.listdir(self.args.output_dir) if d.startswith("checkpoint-")],
            reverse=True,
        )

        for step in checkpoints_step:
            path_ckpt = os.path.join(self.args.output_dir, "checkpoint-" + str(step))
            files = list(os.listdir(path_ckpt))
            if len(files) >= 3 and "trainer_state.json" in files:
                print("Recovering Checkpoint :" + path_ckpt)
                return path_ckpt
        return ""

    def get_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load the model and tokenizer."""
        model = AutoModelForCausalLM.from_pretrained(
            (
                self.args.model_name
                if self.checkpoint_path == ""
                else self.checkpoint_path
            ),
            torch_dtype="auto",
            device_map="auto",
            local_files_only=self.args.local_files_only,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            (
                self.args.model_name
                if self.checkpoint_path == ""
                else self.checkpoint_path
            ),
            local_files_only=self.args.local_files_only,
        )
        return model, tokenizer

    def get_dataset(self) -> Tuple[Dataset, Dataset]:
        """Loads the dataset."""
        raise NotImplementedError

    def get_trainer(self) -> Trainer:
        """Get the trainer."""
        raise NotImplementedError

    def checkpoint(self) -> submitit.helpers.DelayedSubmission:
        """Checkpoint the training."""
        training_callable = type(self)(self.args, (self.dataset, self.eval_dataset))
        print("RESUMING TRAINING")
        return submitit.helpers.DelayedSubmission(training_callable)

    def __call__(self):
        """
        Launch the training
        """
        os.environ["WANDB_MODE"] = "offline"

        self.checkpoint_path = self.retrieve_checkpoint_step()
        self.model, self.tokenizer = self.get_model()
        trainer = self.get_trainer()

        print("LAUNCHING TRAINING")
        trainer.train(
            resume_from_checkpoint=(
                False if self.checkpoint_path == "" else self.checkpoint_path
            )
        )
