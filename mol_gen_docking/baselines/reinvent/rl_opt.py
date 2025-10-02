import argparse
import json
import os
from typing import Any, Callable, Dict, List

from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

from mol_gen_docking.baselines.reinvent.trainers import VanillaReinventTrainer
from mol_gen_docking.reward.rl_rewards import RewardScorer


def get_reward_fn(
    metadata: Dict[str, Any], datasets_path: str
) -> Callable[[str | List[str]], float | List[float]]:
    SCORER = RewardScorer(datasets_path, "property", parse_whole_completion=True)

    def reward_fn(completions: str | List[str]) -> float | List[float]:
        if isinstance(completions, str):
            smiles = [completions]
            return SCORER([""], [completions], metadata=[metadata])[0]
        return SCORER([""], completions, metadata=[metadata] * len(smiles))

    return reward_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/sair_rl/eval_data/eval_prompts",
        help="Dataset name",
    )
    parser.add_argument("--datasets-path", type=str, default="data/sair_rl")

    parser.add_argument(
        "--model_name",
        type=str,
        default="Franso/reinvent_301K",
        help="Name of the model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )

    args = parser.parse_args()
    dataset = load_from_disk(args.data_path)
    with open(os.path.join(args.datasets_path, "docking_targets.json")) as f:
        docking_targets = json.load(f)

    for row in dataset:
        metadata = {k: row[k] for k in ["properties", "objectives", "target"]}
        print(metadata)
        if any([prop in docking_targets for prop in metadata["properties"]]):
            continue

        reward_fn = get_reward_fn(metadata, args.datasets_path)

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        training_args = GRPOConfig(
            output_dir=args.output_dir,
            run_name=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            eval_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            save_steps=500,
            eval_steps=500,
            logging_steps=10,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            dataloader_num_workers=args.dataloader_num_workers,
            max_seq_length=1024,
            dataset_num_proc=8,
            packing=True,
            bf16=True,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_total_limit=3,
            dataloader_prefetch_factor=2,
            completion_only_loss=True,
        )

        train_dataset = Dataset.from_dict({"prompt": ""})

        trainer = VanillaReinventTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )

        trainer.train()
