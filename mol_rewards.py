"""Reward functions for molecular optimization."""
import os
from argparse import Namespace
import logging
from typing import List, Union, Optional, Dict
import yaml

from tdc import Oracle
import tdc
import numpy as np
from rdkit import Chem

def create_logger(name:str, level:str="INFO"):
    """
    Create a logger object with the specified name and level.
    :param name: Name of the logger
    :param level: Level of the logger
    :return: Logger object
    """
    logger = logging.getLogger(name)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s | %(name)s | %(levelname)s] %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(level)
    return logger

def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls


class OracleWrapper:
    """
    Code based on the Oracle class from: https://github.com/wenhao-gao/mol_opt/blob/main/main/optimizer.py#L50

    Wraps the Oracle class from TDC, enabling sample efficient optimization of molecular properties.

    Args:
        args: Namespace containing the arguments for the optimization process.
            - max_oracle_calls (: int) : Maximum number of oracle calls.
            - freq_log (: int) : Frequency of logging the results.
            - output_dir (: str) : Directory to save the results
            - debug (: bool) : Debug mode.
        mol_buffer: Dictionary containing the molecules and their properties.
    """
    def __init__(
            self,
            args: Optional[Namespace] = None,
            mol_buffer: Dict[str, List[Union[float, int]]] = {}
    ):
        self.logger = create_logger(__name__ + "/" + self.__class__.__name__, level="DEBUG" if args.debug else "WARNING")
        self.name = None
        self.evaluator = None
        self.task_label = None
        if args is None:
            self.max_oracle_calls = 10000
            self.freq_log = 100
        else:
            self.args = args
            self.max_oracle_calls = args.max_oracle_calls
            self.freq_log = args.freq_log
        self.mol_buffer = mol_buffer
        self.sa_scorer = tdc.Oracle(name='SA')
        self.diversity_evaluator = tdc.Evaluator(name='Diversity')
        self.last_log = 0

    @property
    def budget(self) -> int:
        return self.max_oracle_calls

    def assign_evaluator(self, evaluator: Oracle):
        self.evaluator = evaluator

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def save_result(self, suffix: Optional[str]=None):

        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)

    def log_intermediate(self, mols: Optional[List[Chem.Mol]] = None, scores: Optional[List[float]] = None, finish: bool = False):

        if finish:
            temp_top100 = list(self.mol_buffer.items())[:100]
            smis = [item[0] for item in temp_top100]
            scores = [item[1][0] for item in temp_top100]
            n_calls = self.max_oracle_calls
        else:
            if mols is None and scores is None:
                if len(self.mol_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 mols in buffer
                    temp_top100 = list(self.mol_buffer.items())[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = len(self.mol_buffer)
                else:
                    results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))[
                              :self.max_oracle_calls]
                    temp_top100 = sorted(results, key=lambda kv: kv[1][0], reverse=True)[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = self.max_oracle_calls
            else:
                # Otherwise, log the input moleucles
                smis = [Chem.MolToSmiles(m) for m in mols]
                n_calls = len(self.mol_buffer)

        # Uncomment this line if want to log top-10 moelucles figures, so as the best_mol key values.
        # temp_top10 = list(self.mol_buffer.items())[:10]

        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)
        avg_sa = np.mean(self.sa_scorer(smis))
        diversity_top100 = self.diversity_evaluator(smis)

        print(f'{n_calls}/{self.max_oracle_calls} | '
              f'avg_top1: {avg_top1:.3f} | '
              f'avg_top10: {avg_top10:.3f} | '
              f'avg_top100: {avg_top100:.3f} | '
              f'avg_sa: {avg_sa:.3f} | '
              f'div: {diversity_top100:.3f}')

        # try:
        print({
            "avg_top1": avg_top1,
            "avg_top10": avg_top10,
            "avg_top100": avg_top100,
            "auc_top1": top_auc(self.mol_buffer, 1, finish, self.freq_log, self.max_oracle_calls),
            "auc_top10": top_auc(self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls),
            "auc_top100": top_auc(self.mol_buffer, 100, finish, self.freq_log, self.max_oracle_calls),
            "avg_sa": avg_sa,
            "diversity_top100": diversity_top100,
            "n_oracle": n_calls,
        })

    def __len__(self) -> int:
        return len(self.mol_buffer)

    def score_smi(self, smi:str) -> float:
        """
        Function to score one molecule

        Argguments:
            smi: One SMILES string represents a molecule.

        Return:
            score: a float represents the property of the molecule.
        """
        if len(self.mol_buffer) > self.max_oracle_calls:
            return 0
        if smi is None:
            return 0
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            return 0
        smi = Chem.MolToSmiles(mol)
        if smi in self.mol_buffer:
            pass
        else:
            self.mol_buffer[smi] = [float(self.evaluator(smi)), len(self.mol_buffer) + 1]
        return self.mol_buffer[smi][0]

    def __call__(self, smis: Union[str, List[str]]) -> Union[float, List[float]]:
        """
        Score
        """
        if isinstance(smis, list):
            score_list = []
            for smi in smis:
                score_list.append(self.score_smi(smi))
                if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                    self.sort_buffer()
                    self.log_intermediate()
                    self.last_log = len(self.mol_buffer)
                    self.save_result(self.task_label)
        elif isinstance(smis, str):
            score_list = self.score_smi(smis)
            if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
                self.save_result(self.task_label)
        else:
            raise ValueError("Input must be a SMILES string or a list of SMILES strings.")
        return score_list

    @property
    def finish(self):
        return len(self.mol_buffer) >= self.max_oracle_calls



