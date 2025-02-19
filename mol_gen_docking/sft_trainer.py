import os
import argparse

from trl import SFTTrainer, SFTConfig, setup_chat_format
from peft import LoraConfig, TaskType, get_peft_model

from mol_gen_docking.sft_data import InstructionDatasetProcessor, special_tok
from mol_gen_docking.trainer_base import MolTrainer



class SFTMolTrainer(MolTrainer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def get_dataset(self):
        # Load the dataset
        return InstructionDatasetProcessor(self.args.dataset).get_training_corpus(
            self.args.train_size,int(0.1*self.args.train_size)
        )

    def get_trainer(self):
        # Expand model vocab
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(special_tok.values())}
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=self.args.lora_config.get("r", 8),
            lora_alpha=self.args.lora_config.get("lora_alpha", 32),
            lora_dropout=self.args.lora_config.get("lora_dropout", 0.1),
        )
        self.model = get_peft_model(self.model, peft_config)
        try:
            self.model, self.tokenizer = setup_chat_format(
                self.model, self.tokenizer
            )
        except ValueError:
            pass

        training_args = SFTConfig(
            output_dir=self.args.output_dir,
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            push_to_hub=False,
            logging_steps=len(self.dataset) // self.args.batch_size,
            save_strategy="epoch"
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
        )
        return trainer
