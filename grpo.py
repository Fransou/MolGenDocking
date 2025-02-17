import argparse
import json

from datasets import Dataset
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from mol_gen_docking.grpo_rewards import get_reward_molecular_property
from mol_gen_docking.grpo_dataset import MolInstructionsDataset


def get_model(args):
    # peft_config = LoraConfig(
    #    task_type=TaskType.SEQ_2_SEQ_LM,
    #    inference_mode=False,
    #    r=args.lora_config.get("r", 8),
    #    lora_alpha=args.lora_config.get("lora_alpha", 32),
    #    lora_dropout=args.lora_config.get("lora_dropout", 0.1)
    # )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype="auto", device_map="auto"
    )

    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    return model


def get_dataset(args):
    # Load the dataset
    dataset = Dataset.from_dict(
        {"prompt": list(MolInstructionsDataset().generate(args.n_prompts))}
    )
    eval_dataset = Dataset.from_dict(
        {"prompt": list(MolInstructionsDataset().generate(10))}
    )
    return dataset, eval_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model on the Mol-Instructions dataset"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="The model name to train",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=100,
        help="The number of training examples to use",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="The batch size to use"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="The learning rate to use"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="The weight decay to use"
    )
    parser.add_argument(
        "--output_dir", type=str, default="test", help="The output directory to use"
    )
    parser.add_argument(
        "--lora-config",
        type=json.loads,
        default={"r": 8, "lora_alpha": 32, "lora_dropout": 0.1},
        help="The LoRA configuration to use",
    )
    parser.add_argument(
        "--n_prompts", type=int, default=16, help="The number of prompts to generate"
    )
    args = parser.parse_args()
    model = get_model(args)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_generations=2,
        push_to_hub=False,
    )

    dataset, eval_dataset = get_dataset(args)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[get_reward_molecular_property],
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
