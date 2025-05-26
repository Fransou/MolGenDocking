import ray
from typing import List

from mol_gen_docking.reward.grpo_rewards import RewardScorer


@ray.remote(num_cpus=16)
class RewardWorker:
    def __init__(self):
        self._reward_fn = RewardScorer("property", parse_whole_completion=True)

    def get_score(self, prompts: List[str], outputs: List[str]) -> List[float]:
        r = self._reward_fn(prompts, outputs).tolist()
        return r


@ray.remote(num_cpus=2)
class RewardWorkerValid:
    def __init__(self):
        self._reward_fn = RewardScorer("valid_smiles", parse_whole_completion=True)

    def get_score(self, prompts: List[str], outputs: List[str]) -> List[float]:
        r = self._reward_fn(prompts, outputs).tolist()
        return r
