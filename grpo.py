"""GRPO training script"""
import submitit
from mol_gen_docking.parser import MolTrainerParser
from mol_gen_docking.trainer.grpo_trainer import GRPOMolTrainer


if __name__ == "__main__":
    mol_parser = MolTrainerParser(
        description="Train a model on the Mol-Instructions dataset"
    )
    mol_parser.add_argument(
        "--n_prompts", type=int, default=16, help="The number of prompts to generate"
    )
    args, slurm_args = mol_parser.parse_args()

    executor = submitit.AutoExecutor(folder="log_test")
    executor.update_parameters(**slurm_args.__dict__)

    trainer = GRPOMolTrainer(args)
    job = executor.submit(trainer)
    job.result()
