import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import submitit

class MolTrainer(submitit.helpers.Checkpointable):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
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

    def checkpoint(self, path:str):
        training_callable = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(training_callable)

    def __call__(self):
        os.environ["WANDB_MODE"]="offline"
        trainer = self.get_trainer()
        print(
            "LAUNCHING TRAINING"
        )
        ckpt_to_resume =False
        possible_ckpts_step = sorted([
            int(d.split('-')[-1]) for d in os.listdir(self.args.checkpoint_dir)
        ], reverse=True)

        for step in possible_ckpts_step:
            path_ckpt = os.path.join(self.args.checkpoint_dir, "checkpoint-"+str(step))
            if len(os.listdir(path_ckpt)) >= 3:
                return trainer.train(resume_from_checkpoint = path_ckpt)
        return trainer.train()
