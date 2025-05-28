import ray
import numpy as np
from typing import List

from mol_gen_docking.reward.grpo_rewards import RewardScorer


@ray.remote(num_cpus=32)
class RewardWorker:
    def __init__(self):
        self._reward_fn = RewardScorer("property", parse_whole_completion=True)

    def get_score(self, prompts: List[str], outputs: List[str]) -> List[float]:
        rewards = self._reward_fn(prompts, outputs).tolist()
        # Check if there ar Nans or None values in the results
        for i, (p, c, r) in enumerate(zip(prompts, outputs, rewards)):
            if r is None or np.isnan(r):
                print(
                    f"Warning: Reward is None or NaN for prompt: {p}, completion: {c}"
                )
                rewards[i] = 0
        return rewards


@ray.remote(num_cpus=2)
class RewardWorkerValid:
    def __init__(self):
        self._reward_fn = RewardScorer("valid_smiles", parse_whole_completion=True)

    def get_score(self, prompts: List[str], outputs: List[str]) -> List[float]:
        rewards = self._reward_fn(prompts, outputs).tolist()
        return rewards
