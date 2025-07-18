"""Trainer callable for SFT."""

import argparse
import json
import os
from typing import Optional, Tuple

import torch
from datasets import Dataset, concatenate_datasets
from peft import AutoPeftModelForCausalLM, LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, setup_chat_format

from mol_gen_docking.sft.sft_data import InstructionDatasetProcessor


class SFTMolTrainer:
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
        self.args = args
        self.checkpoint_path = ""
        self.last_epoch = 0
        self.model: AutoModelForCausalLM | None = None
        self.tokenizer: AutoTokenizer | None = None
        if datasets is None:
            datasets = None, None

        self.dataset: None | Dataset = datasets[0]
        self.eval_dataset = datasets[1]

        self.checkpoint_path = self.retrieve_checkpoint_step()

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

    def retrieve_checkpoint_step(self) -> str:
        """
        Retrieve the last checkpoint step
        :return: path of the last checkpoint
        """
        checkpoints_step = sorted(
            [
                int(d.split("-")[-1])
                for d in os.listdir(self.args.output_dir)
                if d.startswith("checkpoint-")
            ],
            reverse=True,
        )

        for step in checkpoints_step:
            path_ckpt = os.path.join(self.args.output_dir, "checkpoint-" + str(step))
            files = list(os.listdir(path_ckpt))
            if len(files) >= 3 and "trainer_state.json" in files:
                trainer_state = json.load(
                    open(os.path.join(path_ckpt, "trainer_state.json"))
                )
                self.last_epoch = trainer_state["epoch"]
                return path_ckpt
        return ""

    def get_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load the model and tokenizer."""
        args = dict(
            torch_dtype=torch.bfloat16,
            local_files_only=self.args.local_files_only,
            use_cache=False,
            attn_implementation=(
                self.args.attention if not self.args.attention == "vanilla" else None
            ),
        )
        # if not hasattr(self.args, "vllm") or not self.args.vllm:
        #     args["device_map"] = "auto"
        ckpt = (
            self.args.model_name if self.checkpoint_path == "" else self.checkpoint_path
        )
        if self.checkpoint_path != "" and os.path.exists(
            os.path.join(ckpt, "adapter_config.json")
        ):
            print("============= Loading PEFT model =================")
            model = AutoPeftModelForCausalLM.from_pretrained(ckpt, **args)
        else:
            model = AutoModelForCausalLM.from_pretrained(ckpt, **args)

        tokenizer = AutoTokenizer.from_pretrained(
            ckpt,
            local_files_only=self.args.local_files_only,
            padding_side="left",
            use_cache=False,
        )
        return model, tokenizer

    def get_peft_config(self) -> LoraConfig:
        assert self.model is not None and self.tokenizer is not None, (
            "Model and tokenizer must be initialized before calling get_peft_config"
        )
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.args.lora_config.get("r", 8),
            lora_alpha=self.args.lora_config.get("lora_alpha", 32),
            lora_dropout=self.args.lora_config.get("lora_dropout", 0.1),
            target_modules="*_proj",
        )

    def get_trainer(self) -> SFTTrainer:
        """:return: Trainer for SFT."""
        print(self.model)
        peft_config = self.get_peft_config()
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
            save_steps=500,
            eval_steps=500,
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
            completion_only_loss=True,
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
        )

        return trainer

    def __call__(self) -> None:
        """
        Launch the training
        """
        if self.args.slurm:
            os.environ["WANDB_MODE"] = "offline"

        self.checkpoint_path = self.retrieve_checkpoint_step()
        self.model, self.tokenizer = self.get_model()

        if self.dataset is None:
            self.dataset, self.eval_dataset = self.get_dataset()
        print(self.dataset[0])
        trainer = self.get_trainer()

        print(
            "LAUNCHING TRAINING with checkpoint: ",
            self.checkpoint_path if self.checkpoint_path != "" else "None",
        )
        self.tokenizer.padding_side = "left"

        trainer.train(
            resume_from_checkpoint=(
                False if self.checkpoint_path == "" else self.checkpoint_path
            )
        )
