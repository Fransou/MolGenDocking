"""Launches SFT training"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Tuple

import submitit

from mol_gen_docking.sft.sft_trainer import SFTMolTrainer


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
        self.parser = argparse.ArgumentParser(description=description)
        self.add_trainer_args()
        self.add_model_data_args()
        self.add_slurm_args()

        self.add_argument(
            "--slurm", dest="slurm", action="store_true", help="Use SLURM for training"
        )
        self.add_argument(
            "--no-slurm",
            dest="slurm",
            action="store_false",
            help="Do not use SLURM for training",
        )
        self.parser.set_defaults(slurm=False)

    def add_trainer_args(self) -> None:
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
            "--batch_size", type=int, default=8, help="The batch size to use"
        )
        self.add_argument(
            "--learning_rate", type=float, default=1e-4, help="The learning rate to use"
        )
        self.add_argument(
            "--weight_decay", type=float, default=0.01, help="The weight decay to use"
        )
        self.add_argument(
            "--output_dir", type=str, default="test", help="The output directory to use"
        )
        self.add_argument(
            "--num-train-epochs",
            type=int,
            default=5,
            help="The number of epochs to use",
        )
        self.add_argument(
            "--dataloader-num-workers",
            type=int,
            default=1,
            help="The number of workers to use for the dataloader",
        )
        self.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=2,
            help="The number of gradient accumulation steps",
        )

    def add_model_data_args(self) -> None:
        """
        Add the arguments for the model and data
        :param self.parser: self.parser to modify
        :return:
        """
        self.add_argument(
            "--model_name",
            type=str,
            default="Qwen/Qwen2.5-0.5B",
            help="The model name to train",
        )
        self.add_argument(
            "--lora-config",
            type=json.loads,
            default={"r": 4, "lora_alpha": 8, "lora_dropout": 0.0},
            help="The LoRA configuration to use",
        )
        self.add_argument(
            "--attention",
            type=str,
            default="flash_attention_2",
            help="The attention to use",
        )
        self.add_argument("--local-files-only", action="store_true")

    def add_slurm_args(self) -> None:
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

    def add_argument(self, *args: Any, **kwargs: Any) -> None:
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
        args = self.parser.parse_args()
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


if __name__ == "__main__":
    mol_parser = MolTrainerParser(
        description="Train a model on the Mol-Instructions dataset"
    )

    mol_parser.add_argument(
        "--dataset",
        nargs="+",
        default=["SMolInstruct"],
        help="Dataset to use",
    )

    args, slurm_args = mol_parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    trainer = SFTMolTrainer(args)
    if args.slurm:
        current_epoch = trainer.last_epoch
        n_runs = 0
        while current_epoch < args.num_train_epochs and n_runs < args.max_slurm_runs:
            executor = submitit.AutoExecutor(
                folder="log_test",
            )
            executor.update_parameters(**slurm_args.__dict__)
            try:
                job = executor.submit(trainer)
                job.result()
                break
            except submitit.core.utils.UncompletedJobError:
                current_epoch = trainer.last_epoch
                n_runs += 1
            except Exception as e:
                raise e
    else:
        trainer()

    model, tokenizer = trainer.model, trainer.tokenizer
    if args.output_dir == "debug":
        import shutil

        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    if model is not None and tokenizer is not None:
        model = model.merge_and_unload()
        model.save_pretrained(os.path.join(args.output_dir, "model"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "model"))
