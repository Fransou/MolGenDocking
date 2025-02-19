import argparse
import json
import submitit

from mol_gen_docking.sft_trainer import launch_sft_training




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model on the Mol-Instructions dataset"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="The model name to train",
    )
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
        "--dataset", type=str, default="Mol-Instructions", help="The dataset to use"
    )
    parser.add_argument(
        "--lora-config",
        type=json.loads,
        default={"r": 8, "lora_alpha": 32, "lora_dropout": 0.1},
        help="The LoRA configuration to use",
    )
    args = parser.parse_args()

    executor = submitit.AutoExecutor(folder="log_test")
    executor.update_parameters(
        timeout_min=5,
        nodes=1,
        mem_gb=200,
        cpus_per_task=8,
        tasks_per_node=1,
        gpus_per_node=1,
        account="def-ibenayed",
        job_name="GRPO"
    )

    job=executor.submit(launch_sft_training, args)
    job.result()
