"""Trainer callable for SFT."""

import argparse
from typing import Tuple, Optional

from trl import SFTTrainer, SFTConfig, setup_chat_format
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset

from mol_gen_docking.sft_data import InstructionDatasetProcessor
from mol_gen_docking.trainer.trainer_base import MolTrainer


class SFTMolTrainer(MolTrainer):
    """
    Trainer for SFT for molecular instructions.
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
        return InstructionDatasetProcessor(self.args.dataset).get_training_corpus(
            self.args.train_size
        )

    def get_trainer(self) -> SFTTrainer:
        """:return: Trainer for SFT."""
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.args.lora_config.get("r", 8),
            lora_alpha=self.args.lora_config.get("lora_alpha", 32),
            lora_dropout=self.args.lora_config.get("lora_dropout", 0.1),
        )
        self.model = get_peft_model(self.model, peft_config)
        try:
            self.model, self.tokenizer = setup_chat_format(self.model, self.tokenizer)
        except ValueError:
            pass

        training_args = SFTConfig(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.num_train_epochs,
            overwrite_output_dir=True,
            eval_strategy="epoch",
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            push_to_hub=False,
            logging_steps=len(self.dataset) // self.args.batch_size,
            save_strategy="epoch",
            run_name=self.args.output_dir,
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
        )
        return trainer
