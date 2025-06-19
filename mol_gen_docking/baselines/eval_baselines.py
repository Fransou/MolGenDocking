import argparse
import os
from dataclasses import dataclass
from typing import List

from datasets import load_from_disk
from molopt.base import BaseOptimizer, Oracle
from rdkit import Chem

from mol_gen_docking.baselines.reward_fn import get_reward_fn


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


class BatchedOracle(Oracle):
    def __init__(
        self, max_oracle_calls=10000, freq_log=100, output_dir="results", mol_buffer={}
    ) -> None:
        super().__init__(
            max_oracle_calls=10000, freq_log=100, output_dir="results", mol_buffer={}
        )

    def batch_score_smi(self, smi_lst: List[str]) -> List[float]:
        if len(self.mol_buffer) > self.max_oracle_calls:
            return [0.0 for _ in smi_lst]
        scores = [0.0 for _ in smi_lst]
        to_score_idx = []
        for i, smi in enumerate(smi_lst):
            if smi is None:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None or len(smi) == 0:
                continue
            to_score_idx.append(i)

        smis_to_score = [
            smi_lst[i] for i in to_score_idx if smi_lst[i] not in self.mol_buffer
        ]
        res_scores: List[float] = self.evaluator(smis_to_score)
        for smi, score in zip(smi_lst, res_scores):
            self.mol_buffer[smi] = [score, len(self.mol_buffer) + 1]

        for i in range(len(scores)):
            if i not in to_score_idx:
                scores[i] = self.mol_buffer[smi][0]
        return scores

    def __call__(self, smiles_lst):
        """
        Score
        """
        if isinstance(smiles_lst, list):
            score_list = self.batch_score_smi(smiles_lst)
            for smi in smiles_lst:
                if (
                    len(self.mol_buffer) % self.freq_log == 0
                    and len(self.mol_buffer) > self.last_log
                ):
                    self.sort_buffer()
                    self.log_intermediate()
                    self.last_log = len(self.mol_buffer)
                    self.save_result(self.task_label)
        else:  ### a string of SMILES
            score_list = self.score_smi(smiles_lst)
            if (
                len(self.mol_buffer) % self.freq_log == 0
                and len(self.mol_buffer) > self.last_log
            ):
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
                self.save_result(self.task_label)
        return score_list

    @classmethod
    def from_oracle(cls, oracle: Oracle):
        return cls(
            oracle.max_oracle_calls,
            freq_log=oracle.freq_log,
            output_dir=oracle.output_dir,
            mol_buffer=oracle.mol_buffer,
        )


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
        optimizer.oracle = BatchedOracle.from_oracle(optimizer.oracle)
        optimizer.optimize(
            oracle=get_reward_fn(prompt, args.datasets_path),
            patience=config.patience,
            seed=0,
        )

    print(dataset)
