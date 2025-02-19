import argparse
import json

def add_trainer_args(parser):
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
        "--num-train-epochs", type=int, default=100, help="The number of epochs to use"
    )


def add_model_data_args(parser):
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="The model name to train",
    )
    parser.add_argument(
        "--lora-config",
        type=json.loads,
        default={"r": 8, "lora_alpha": 32, "lora_dropout": 0.1},
        help="The LoRA configuration to use",
    )
    parser.add_argument("--local-files-only", action="store_true")

def add_slurm_args(parser):
    parser.add_argument("--timeout-min", type=int, default=15)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--mem-gb", type=int, default=200)
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--tasks-per-node", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--slurm-account", type=str, default="def-ibenayed")
    parser.add_argument("--slurm-job-name", type=str, default="MolGenDocking")
