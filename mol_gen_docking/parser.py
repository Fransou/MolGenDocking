"""Common Parser modifications for CLI"""
import argparse
import json
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class SlurmArgs:
    timeout_min: int = 15
    nodes: int = 1
    mem_gb: int = 200
    cpus_per_task: int = 8
    tasks_per_node: int = 1
    gpus_per_node: int = 1
    slurm_account: str = "def-ibenayed"
    slurm_job_name: str = "MolGenDocking"


class MolTrainerParser:
    """
    Parser for the MolGenDocking training scripts
    """
    def __init__(self, description: str = "Train a model"):
        self.parser = argparse.ArgumentParser(
            description=description
        )
        self.add_trainer_args()
        self.add_model_data_args()
        self.add_slurm_args()


    def add_trainer_args(self):
        """
        Add the arguments for the trainer
        :param self.parser: self.parser to modify
        :return:
        """
        self.add_argument(
            "--train_size",
            type=int,
            default=10000000,
            help="The number of training examples to use",
        )
        self.add_argument(
            "--batch_size", type=int, default=2, help="The batch size to use"
        )
        self.add_argument(
            "--learning_rate", type=float, default=2e-5, help="The learning rate to use"
        )
        self.add_argument(
            "--weight_decay", type=float, default=0.01, help="The weight decay to use"
        )
        self.add_argument(
            "--output_dir", type=str, default="test", help="The output directory to use"
        )
        self.add_argument(
            "--num-train-epochs", type=int, default=100, help="The number of epochs to use"
        )
        self.add_argument(
        "--dataloader-num-workers", type=int, default=4, help="The number of workers to use for the dataloader"
        )

    def add_model_data_args(self):
        """
        Add the arguments for the model and data
        :param self.parser: self.parser to modify
        :return:
        """
        self.add_argument(
            "--model_name",
            type=str,
            default="Qwen/Qwen2-0.5B-Instruct",
            help="The model name to train",
        )
        self.add_argument(
            "--lora-config",
            type=json.loads,
            default={"r": 8, "lora_alpha": 32, "lora_dropout": 0.1},
            help="The LoRA configuration to use",
        )
        self.add_argument(
            "--attention", type=str, default="vanilla", help="The attention to use"
        )
        self.add_argument("--local-files-only", action="store_true")


    def add_slurm_args(self):
        """
        Add the arguments for the SLURM configuration
        :param self.parser: self.parser to modify
        :return:
        """
        self.add_argument("--timeout-min", type=int, default=15)
        self.add_argument("--nodes", type=int, default=1)
        self.add_argument("--mem-gb", type=int, default=200)
        self.add_argument("--cpus-per-task", type=int, default=8)
        self.add_argument("--tasks-per-node", type=int, default=1)
        self.add_argument("--gpus-per-node", type=int, default=1)
        self.add_argument("--slurm-account", type=str, default="def-ibenayed")
        self.add_argument("--slurm-job-name", type=str, default="MolGenDocking")
        self.add_argument("--max-slurm-runs", type=int, default=5)

    def add_argument(self, *args, **kwargs):
        """
        Add an argument to the parser
        :param args: The arguments to add
        :param kwargs: The keyword arguments to add
        :return:
        """
        self.parser.add_argument(*args, **kwargs)

    def parse_args(self) -> Tuple[argparse.Namespace, SlurmArgs]:
        """
        Parse the arguments
        :return: The parsed arguments
        """
        args= self.parser.parse_args()
        slurm_args = SlurmArgs(
            timeout_min=args.timeout_min,
            nodes=args.nodes,
            mem_gb=args.mem_gb,
            cpus_per_task=args.cpus_per_task,
            tasks_per_node=args.tasks_per_node,
            gpus_per_node=args.gpus_per_node,
            slurm_account=args.slurm_account,
            slurm_job_name=args.slurm_job_name,
        )
        return args, slurm_args



