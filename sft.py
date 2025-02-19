import argparse
import submitit

from mol_gen_docking.parser import add_trainer_args, add_model_data_args,add_slurm_args
from mol_gen_docking.sft_trainer import SFTMolTrainer




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model on the Mol-Instructions dataset"
    )
    add_trainer_args(parser)
    add_model_data_args(parser)
    add_slurm_args(parser)
    parser.add_argument(
        "--dataset", type=str, default="Mol-Instructions", help="The dataset to use"
    )


    args = parser.parse_args()
    trainer = SFTMolTrainer(args)
    executor = submitit.AutoExecutor(folder="log_test")
    executor.update_parameters(
        timeout_min=args.timeout_min,
        nodes=args.nodes,
        mem_gb=args.mem_gb,
        cpus_per_task=args.cpus_per_task,
        tasks_per_node=args.tasks_per_node,
        gpus_per_node=args.gpus_per_node,
        slurm_account=args.slurm_account,
        slurm_job_name=args.slurm_job_name,
    )

    job=executor.submit(trainer)
    job.result()
