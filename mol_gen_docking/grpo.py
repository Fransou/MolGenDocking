"""GRPO training script"""

import os
import submitit
from mol_gen_docking.utils.parser import MolTrainerParser
from mol_gen_docking.trainer.grpo_trainer import GRPOMolTrainer


if __name__ == "__main__":
    mol_parser = MolTrainerParser(
        description="Train a model on the Mol-Instructions dataset"
    )
    mol_parser.add_argument(
        "--n-prompts", type=int, default=2048, help="The number of prompts to generate"
    )
    mol_parser.add_argument(
        "--num-generations",
        type=int,
        default=8,
        help="The number of generations to use for the training",
    )

    mol_parser.add_argument("--vina", action="store_true", dest="vina")
    mol_parser.add_argument("--no-vina", action="store_false", dest="vina")

    mol_parser.add_argument("--vllm", action="store_true", dest="vllm")
    mol_parser.add_argument("--no-vllm", action="store_false", dest="vllm")

    args, slurm_args = mol_parser.parse_args()

    if args.output_dir == "debug" and os.path.exists(args.output_dir):
        import shutil

        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    executor = submitit.AutoExecutor(folder="log_test")
    executor.update_parameters(**slurm_args.__dict__)

    trainer = GRPOMolTrainer(args)
    if args.slurm:
        job = executor.submit(trainer)
        job.result()
    else:
        trainer(True)
