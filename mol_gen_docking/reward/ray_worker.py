import ray
from typing import List

from mol_gen_docking.reward.grpo_rewards import RewardScorer


@ray.remote(num_cpus=64)
class RewardWorker:
    def __init__(self):
        self._reward_valid_molecules = RewardScorer(
            "valid_smiles", parse_whole_completion=True
        )
        self._reward_properties = RewardScorer("property", parse_whole_completion=True)

    def get_score(self, prompts: List[str], outputs: List[str]) -> List[float]:
        return self._reward_properties(prompts, outputs)
