"""Trainer callable for SFT."""

import argparse
from typing import Tuple, Optional

from trl import SFTTrainer, SFTConfig, setup_chat_format
from peft import get_peft_model
from datasets import Dataset, concatenate_datasets

from mol_gen_docking.data.sft_data import InstructionDatasetProcessor
from mol_gen_docking.trainer.trainer_base import MolTrainer


class SFTMolTrainer(MolTrainer):
    """
    Trainer for SFT for molecular instructions.
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
        tuple_datasets = tuple(
            InstructionDatasetProcessor(d).get_training_corpus(self.args.train_size)
            for d in self.args.dataset
        )
        # Concatenate the datasets
        return (
            concatenate_datasets([d[0] for d in tuple_datasets]),
            concatenate_datasets([d[1] for d in tuple_datasets]),
        )

    def get_trainer(self) -> SFTTrainer:
        """:return: Trainer for SFT."""
        peft_config = self.get_peft_config(True)
        self.model = get_peft_model(self.model, peft_config)
        try:
            self.model, self.tokenizer = setup_chat_format(self.model, self.tokenizer)
        except ValueError:
            pass
        self.model.print_trainable_parameters()

        training_args = SFTConfig(
            output_dir=self.args.output_dir,
            run_name=self.args.output_dir,
            num_train_epochs=self.args.num_train_epochs,
            eval_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            save_steps=1000,
            eval_steps=1000,
            logging_steps=10,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            dataloader_num_workers=self.args.dataloader_num_workers,
            max_seq_length=1024,
            dataset_num_proc=8,
            packing=True,
            bf16=True,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            save_total_limit=3,
            dataloader_prefetch_factor=2,
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
        )

        return trainer
