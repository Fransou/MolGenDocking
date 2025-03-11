"""Trainer callable for GRPO."""

import argparse
from typing import Tuple, Optional

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from mol_gen_docking.grpo_dataset import MolInstructionsDataset
from mol_gen_docking.grpo_rewards import get_reward_molecular_property
from mol_gen_docking.trainer.trainer_base import MolTrainer


class GRPOMolTrainer(MolTrainer):
    """
    Trainer for GRPO for molecular generation.
    """

    def __init__(
        self, args: argparse.Namespace, datasets: Optional[Tuple[Dataset]] = None
    ):
        """
        :param args: Parameters for the training
        :param datasets: training and evaluation datasets (if None, will be loaded)
        """
        super().__init__(args, datasets)

    def get_dataset(self) -> Tuple[Dataset]:
        """Loads the dataset."""
        dataset = Dataset.from_dict(
            {"prompt": list(MolInstructionsDataset(vina=self.args.vina).generate(self.args.n_prompts))}
        )
        eval_dataset = Dataset.from_dict(
            {"prompt": list(MolInstructionsDataset(vina=self.args.vina).generate(10))}
        )
        return dataset, eval_dataset

    def get_trainer(self) -> GRPOTrainer:
        """:return: Trainer for GRPO."""
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

        return GRPOTrainer(
            model=self.model,
            reward_funcs=[get_reward_molecular_property],
            args=training_args,
            train_dataset=self.dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            reward_processing_classes=self.tokenizer,
        )
