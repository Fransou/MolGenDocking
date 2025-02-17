import math
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
import argparse
from trl import SFTTrainer, SFTConfig, setup_chat_format

from mol_gen_docking.sft_data import InstructionDatasetProcessor


def get_model_and_dataset(args: argparse.Namespace):
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=args.lora_config.get("r", 8),
        lora_alpha=args.lora_config.get("lora_alpha", 32),
        lora_dropout=args.lora_config.get("lora_dropout", 0.1),
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype="auto", device_map="auto"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    try:
        model, tokenizer = setup_chat_format(model, tokenizer)
    except ValueError:
        pass

    return (
        model,
        tokenizer,
        InstructionDatasetProcessor(args.dataset).get_training_corpus(),
    )


def main(args: argparse.Namespace):
    model, tokenizer, dataset = get_model_and_dataset(args)

    train_size = args.train_size
    test_size = int(0.1 * train_size)

    downsampled_dataset = dataset.train_test_split(
        train_size=train_size, test_size=test_size, seed=42
    )

    # Show the training loss with every epoch
    logging_steps = len(downsampled_dataset["train"]) // args.batch_size

    training_args = SFTConfig(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        push_to_hub=False,
        logging_steps=logging_steps,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=downsampled_dataset["train"],
        eval_dataset=downsampled_dataset["test"],
        tokenizer=tokenizer,
    )

    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    trainer.train()

    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model on the Mol-Instructions dataset"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
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
        "--dataset", type=str, default="Mol-Instructions", help="The dataset to use"
    )
    parser.add_argument(
        "--lora-config",
        type=json.loads,
        default={"r": 8, "lora_alpha": 32, "lora_dropout": 0.1},
        help="The LoRA configuration to use",
    )
    args = parser.parse_args()

    main(args)
