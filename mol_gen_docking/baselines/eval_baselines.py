import argparse
import os
from dataclasses import dataclass
from typing import Callable

from datasets import load_from_disk
from molopt.base import BaseOptimizer

from mol_gen_docking.reward.rl_rewards import RewardScorer


@dataclass(frozen=False)
class BaselineConfig:
    baseline_name: str = "SmilesGA"
    n_jobs: int = -1
    max_oracle_calls: int = 1000
    freq_log: int = 100
    output_dir: str = "baseline_results"
    log_results: bool = True
    patience: int = 5

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "BaselineConfig":
        return cls(
            baseline_name=args.BASELINE_NAME,
            n_jobs=args.n_jobs,
            max_oracle_calls=args.max_oracle_calls,
            freq_log=args.freq_log,
            output_dir=args.output_dir,
            log_results=args.log_results,
            patience=args.patience,
        )

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument("BASELINE_NAME", type=str)
        parser.add_argument(
            "--n_jobs",
            type=int,
            default=-1,
            help="Number of jobs to run in parallel. -1 means using all available cores.",
        )
        parser.add_argument(
            "--max_oracle_calls",
            type=int,
            default=1000,
            help="Maximum number of oracle calls.",
        )
        parser.add_argument(
            "--freq_log",
            type=int,
            default=100,
            help="Frequency of logging results.",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="baseline_results",
            help="Directory to save the results.",
        )
        parser.add_argument(
            "--log_results",
            action="store_true",
            help="Whether to log the results.",
        )
        parser.add_argument(
            "--patience",
            type=int,
            default=5,
            help="Patience for early stopping.",
        )

    def __post_init__(self):
        self.output_dir = os.path.join(self.output_dir, self.baseline_name)
        os.makedirs(self.output_dir, exist_ok=True)


def get_mol_opt_cls(baseline_name: str) -> BaseOptimizer:
    if baseline_name == "GPBO":
        from molopt.gpbo import GPBO

        return GPBO
    elif baseline_name == "GraphGA":
        from molopt.graph_ga import GraphGA

        return GraphGA
    elif baseline_name == "Graph_MCTS":
        from molopt.graph_mcts import Graph_MCTS

        return Graph_MCTS
    elif baseline_name == "JTVAE":
        from molopt.jt_vae import JTVAE

        return JTVAE
    elif baseline_name == "MIMOSA":
        from molopt.mimosa import MIMOSA

        return MIMOSA
    elif baseline_name == "MolDQN":
        from molopt.moldqn import MolDQN

        return MolDQN
    elif baseline_name == "REINVENT":
        from molopt.reinvent import REINVENT

        return REINVENT
    elif baseline_name == "Screening":
        from molopt.screening import Screening

        return Screening
    elif baseline_name == "SmilesGA":
        from molopt.smiles_ga import SmilesGA

        return SmilesGA
    elif baseline_name == "Stoned":
        from molopt.stoned import Stoned

        return Stoned
    raise ValueError(f"Baseline {baseline_name} is not supported. ")


def get_reward_fn(prompt: str, datasets_path: str) -> Callable[[str], float]:
    SCORER = RewardScorer(datasets_path, "property", parse_whole_completion=False)

    def reward_fn(smiles: str) -> float:
        if smiles is None:
            return 0.0
        reward = SCORER([prompt], [smiles])[0]
        return reward

    return reward_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate prompts for molecular generation tasks."
    )
    BaselineConfig.add_arguments(parser)
    parser.add_argument(
        "--data-path", type=str, default="data/mol_orz/eval_data/eval_prompts"
    )
    parser.add_argument("--datasets-path", type=str, default="data/mol_orz")
    args = parser.parse_args()

    config = BaselineConfig.from_args(args)
    dataset = load_from_disk(args.data_path)

    for prompt in dataset["prompt"]:
        reward_fn = get_reward_fn(prompt, args.datasets_path)
        optimizer = get_mol_opt_cls(config.baseline_name)(
            smi_file=None,
            n_jobs=config.n_jobs,
            max_oracle_calls=config.max_oracle_calls,
            freq_log=config.freq_log,
            output_dir=config.output_dir,
            log_results=config.log_results,
        )
        optimizer.optimize(
            oracle=get_reward_fn(prompt, args.datasets_path),
            patience=config.patience,
            seed=0,
        )

    print(dataset)
