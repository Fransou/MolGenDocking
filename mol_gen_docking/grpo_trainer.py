import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from mol_gen_docking.grpo_dataset import MolInstructionsDataset
from mol_gen_docking.grpo_rewards import get_reward_molecular_property



class GRPOMolTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args


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
        # Load the dataset
        dataset = Dataset.from_dict(
            {"prompt": list(MolInstructionsDataset().generate(self.args.n_prompts))}
        )
        eval_dataset = Dataset.from_dict(
            {"prompt": list(MolInstructionsDataset().generate(10))}
        )
        return dataset, eval_dataset

    def __call__(self):
        model, tokenizer = self.get_model()

        training_args = GRPOConfig(
            output_dir=self.args.output_dir,
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            num_generations=2,
            push_to_hub=False,
        )

        dataset, eval_dataset = self.get_dataset()
        print("FINISHED")
        return None

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[get_reward_molecular_property],
            args=training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            reward_processing_classes=tokenizer,
        )
        trainer.train()

def launch_grpo_training(args: argparse.Namespace):
    trainer = GRPOMolTrainer(args)
    return trainer()