"""Trainer callable for GRPO."""

import argparse
from typing import Tuple, Optional

from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

from mol_gen_docking.data.grpo_dataset import MolInstructionsDataset
from mol_gen_docking.utils.grpo_rewards import RewardScorer, RewardScorerServer
from mol_gen_docking.trainer.trainer_base import MolTrainer
from mol_gen_docking.utils.grpo_reward_tokenizer import wrap_tokenizer


class GRPOMolTrainer(MolTrainer):
    """
    Trainer for GRPO for molecular generation.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        datasets: Optional[Tuple[Dataset, Dataset]] = None,
    ):
        """
        :param args: Parameters for the training
        :param datasets: training and evaluation datasets (if None, will be loaded)
        """
        super().__init__(args, datasets)

    def get_dataset(self) -> Tuple[Dataset, Dataset]:
        """Loads the dataset."""
        dataset = MolInstructionsDataset(vina=self.args.vina)(self.args.n_prompts)
        eval_dataset = MolInstructionsDataset(vina=self.args.vina)(
            self.args.n_prompts // 10
        )
        return dataset, eval_dataset

    def get_trainer(self) -> GRPOTrainer:
        """:return: Trainer for GRPO."""

        peft_config = self.get_peft_config(False)

        training_args = GRPOConfig(
            beta=0,
            output_dir=self.args.output_dir,
            run_name=self.args.output_dir,
            num_train_epochs=self.args.num_train_epochs,
            eval_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            save_steps=100,
            eval_steps=100,
            logging_steps=1,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            per_device_train_batch_size=self.args.batch_size
            * self.args.num_generations,
            per_device_eval_batch_size=self.args.batch_size * self.args.num_generations,
            dataloader_num_workers=self.args.dataloader_num_workers,
            num_generations=self.args.num_generations,
            push_to_hub=False,
            bf16=True,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            save_total_limit=3,
            use_vllm=self.args.vllm,
            report_to="wandb",
            log_completions=True,
        )

        trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=[
                RewardScorerServer("property"),
                RewardScorer("smiles"),
                RewardScorer("valid_smiles"),
            ],
            args=training_args,
            train_dataset=self.dataset,
            eval_dataset=self.eval_dataset,
            processing_class=wrap_tokenizer(self.tokenizer),
            peft_config=peft_config,
        )

        return trainer
