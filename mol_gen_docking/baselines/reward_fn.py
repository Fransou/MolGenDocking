from typing import Callable

from mol_gen_docking.reward.rl_rewards import RewardScorer


def get_reward_fn(prompt: str, datasets_path: str) -> Callable[[str], float]:
    SCORER = RewardScorer(datasets_path, "property", parse_whole_completion=True)

    def reward_fn(smiles: str) -> float:
        if smiles is None:
            print(smiles)
            return 0.0
        reward = SCORER([prompt], [smiles])[0]
        return reward

    return reward_fn
