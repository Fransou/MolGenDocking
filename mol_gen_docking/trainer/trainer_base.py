"""Base class for the trainer."""

import os
import json
import argparse
import functools
from typing import Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from datasets import Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, TaskType

from mol_gen_docking.data import special_tok
from torch.profiler import profile, record_function, ProfilerActivity


def torch_profiler_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
        ) as prof:
            with record_function("model_inference"):
                result = func(*args, **kwargs)
        # Save chrome trace
        prof.export_chrome_trace("profile.json")
        return result

    return wrapper


def record_function_decorator_builder(name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with record_function(name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class MolTrainer:
    """Base class for the trainer."""

    def __init__(
        self,
        args: argparse.Namespace,
        datasets: Optional[Tuple[Dataset, Dataset]] = None,
    ):
        """
        :param args: Parameters for the training
        :param datasets: training and evaluation datasets (if None, will be loaded)
        """
        super().__init__()

        self.args = args
        self.checkpoint_path = ""
        self.last_epoch = 0
        self.model: AutoModelForCausalLM | None = None
        self.tokenizer: AutoTokenizer | None = None
        if datasets is None:
            datasets = None, None

        self.dataset: None | Dataset = datasets[0]
        self.eval_dataset = datasets[1]

        self.checkpoint_path = self.retrieve_checkpoint_step()

    def retrieve_checkpoint_step(self) -> str:
        """
        Retrieve the last checkpoint step
        :return: path of the last checkpoint
        """
        checkpoints_step = sorted(
            [
                int(d.split("-")[-1])
                for d in os.listdir(self.args.output_dir)
                if d.startswith("checkpoint-")
            ],
            reverse=True,
        )

        for step in checkpoints_step:
            path_ckpt = os.path.join(self.args.output_dir, "checkpoint-" + str(step))
            files = list(os.listdir(path_ckpt))
            if len(files) >= 3 and "trainer_state.json" in files:
                trainer_state = json.load(
                    open(os.path.join(path_ckpt, "trainer_state.json"))
                )
                self.last_epoch = trainer_state["epoch"]
                return path_ckpt
        return ""

    def get_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load the model and tokenizer."""
        args = dict(
            torch_dtype=torch.bfloat16,
            local_files_only=self.args.local_files_only,
            use_cache=False,
            attn_implementation=(
                self.args.attention if not self.args.attention == "vanilla" else None
            ),
        )
        if not hasattr(self.args, "vllm") or not self.args.vllm:
            args["device_map"] = "auto"
        ckpt = (
            self.args.model_name if self.checkpoint_path == "" else self.checkpoint_path
        )
        if self.checkpoint_path != "" and os.path.exists(
            os.path.join(ckpt, "adapter_config.json")
        ):
            print("============= Loading PEFT model =================")
            model = AutoPeftModelForCausalLM.from_pretrained(ckpt, **args)
        else:
            model = AutoModelForCausalLM.from_pretrained(ckpt, **args)

        tokenizer = AutoTokenizer.from_pretrained(
            ckpt,
            local_files_only=self.args.local_files_only,
            padding_side="left",
            use_cache=False,
        )
        return model, tokenizer

    def get_dataset(self) -> Tuple[Dataset, Dataset]:
        """Loads the dataset."""
        raise NotImplementedError

    def get_trainer(self) -> Trainer:
        """Get the trainer."""
        raise NotImplementedError

    def get_peft_config(self, train_tokens: bool = False) -> LoraConfig:
        assert self.model is not None or self.tokenizer is not None, (
            "Model and tokenizer must be initialized before calling get_peft_config"
        )
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.args.lora_config.get("r", 8),
            lora_alpha=self.args.lora_config.get("lora_alpha", 32),
            lora_dropout=self.args.lora_config.get("lora_dropout", 0.1),
            target_modules=["q_proj", "v_proj"],
            trainable_token_indices=(
                {
                    "embed_tokens": [
                        self.tokenizer.convert_tokens_to_ids(t)
                        for t in special_tok.values()
                    ],
                }
                if train_tokens and self.tokenizer is not None
                else {}
            ),
        )

    def __call__(self, profile: bool = False):
        """
        Launch the training
        """
        if self.args.slurm:
            os.environ["WANDB_MODE"] = "offline"

        # wandb.require("legacy-service")

        self.checkpoint_path = self.retrieve_checkpoint_step()
        self.model, self.tokenizer = self.get_model()

        if self.dataset is None:
            self.dataset, self.eval_dataset = self.get_dataset()

        trainer = self.get_trainer()

        print(
            "LAUNCHING TRAINING with checkpoint: ",
            self.checkpoint_path if self.checkpoint_path != "" else "None",
        )
        self.tokenizer.padding_side = "left"

        if profile:
            trainer.train = torch_profiler_decorator(trainer.train)
            trainer._prepare_inputs = record_function_decorator_builder(
                "prepare_inputs"
            )(trainer._prepare_inputs)
            if hasattr(trainer, "_generate_and_score_completions"):
                trainer._generate_and_score_completions = (
                    record_function_decorator_builder("generate_and_score_completions")(
                        trainer._generate_and_score_completions
                    )
                )
            if hasattr(trainer, "reward_funcs"):
                for i in range(len(trainer.reward_funcs)):
                    trainer.reward_funcs[i] = record_function_decorator_builder(
                        f"reward_func_{i}"
                    )(trainer.reward_funcs[i])

        trainer.train(
            resume_from_checkpoint=(
                False if self.checkpoint_path == "" else self.checkpoint_path
            )
        )
