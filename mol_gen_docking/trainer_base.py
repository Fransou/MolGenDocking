import os
import argparse
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import submitit


class MolTrainer(submitit.helpers.Checkpointable):
    def __init__(self, args: argparse.Namespace, datasets:Optional[Tuple[Dataset]]=None):
        super().__init__()

        self.args = args
        self.checkpoint_path = self.retrieve_checkpoint_step()

        self.model, self.tokenizer = self.get_model()
        if datasets is None:
            self.dataset, self.eval_dataset = self.get_dataset()
        else:
            self.dataset, self.eval_dataset = datasets

    def retrieve_checkpoint_step(self):
        checkpoints_step = sorted(
            [int(d.split("-")[-1]) for d in os.listdir(self.args.output_dir)],
            reverse=True,
        )

        for step in checkpoints_step:
            path_ckpt = os.path.join(self.args.output_dir, "checkpoint-" + str(step))
            files = list(os.listdir(path_ckpt))
            if len(files) >= 3 and "trainer_state.json" in files:
                print("Recovering Checkpoint :" + path_ckpt)
                return path_ckpt
        return ""

    def get_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name if self.checkpoint_path == "" else self.checkpoint_path,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=self.args.local_files_only,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name if self.checkpoint_path == "" else self.checkpoint_path,
            local_files_only=self.args.local_files_only,
        )
        return model, tokenizer

    def get_dataset(self) -> Tuple[Dataset]:
        raise NotImplementedError

    def get_trainer(self):
        raise NotImplementedError

    def checkpoint(self):
        training_callable = type(self)(self.args, (self.dataset, self.eval_dataset))
        return submitit.helpers.DelayedSubmission(training_callable)

    def __call__(self):
        os.environ["WANDB_MODE"] = "offline"
        trainer = self.get_trainer()
        print("LAUNCHING TRAINING")
        trainer.train(
            resume_from_checkpoint=False if self.checkpoint_path == "" else self.checkpoint_path
        )
