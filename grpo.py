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
    mol_parser.add_argument("--vina", action="store_true", dest="vina")
    mol_parser.add_argument("--no-vina", action="store_false", dest="vina")

    args, slurm_args = mol_parser.parse_args()

    executor = submitit.AutoExecutor(folder="log_test")
    executor.update_parameters(**slurm_args.__dict__)

    trainer = GRPOMolTrainer(args)
    if args.slurm:
        job = executor.submit(trainer)
        job.result()
    else:
        trainer()
