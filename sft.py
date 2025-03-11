"""Launches SFT training"""

import os

import submitit
from mol_gen_docking.parser import MolTrainerParser
from mol_gen_docking.trainer.sft_trainer import SFTMolTrainer


if __name__ == "__main__":
    mol_parser = MolTrainerParser(
        description="Train a model on the Mol-Instructions dataset"
    )

    mol_parser.add_argument(
        "--dataset", type=str, default="Mol-Instructions", help="The dataset to use"
    )

    args, slurm_args = mol_parser.parse_args()
    trainer = SFTMolTrainer(args)

    if args.slurm:
        current_epoch = trainer.last_epoch
        n_runs = 0
        while current_epoch < args.num_train_epochs and n_runs < args.max_slurm_runs:
            print(f"Starting run {n_runs} at epoch {current_epoch}")
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
        print("Finished training")
    else:
        trainer()

    model, tokenizer = trainer.get_model()
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(os.path.join(args.output_dir, "model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "model"))
