import os
import argparse
import submitit

from mol_gen_docking.parser import add_trainer_args, add_model_data_args
from mol_gen_docking.sft_trainer import SFTMolTrainer




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model on the Mol-Instructions dataset"
    )
    add_trainer_args(parser)
    add_model_data_args(parser)
    parser.add_argument(
        "--dataset", type=str, default="Mol-Instructions", help="The dataset to use"
    )


    args = parser.parse_args()
    trainer = SFTMolTrainer(args)
    executor = submitit.AutoExecutor(folder="log_test")
    executor.update_parameters(
        timeout_min=5,
        nodes=1,
        mem_gb=200,
        cpus_per_task=8,
        tasks_per_node=1,
        gpus_per_node=1,
        slurm_account="def-ibenayed",
        slurm_job_name="SFT"
    )

    job=executor.submit(trainer)
    job.result()
