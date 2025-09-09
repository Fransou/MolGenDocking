from typing import Any, Callable, Dict, List

from mol_gen_docking.reward.rl_rewards import RewardScorer


def get_reward_fn(
    metadata: Dict[str, Any], datasets_path: str
) -> Callable[[str | List[str]], float | List[float]]:
    SCORER = RewardScorer(datasets_path, "property", parse_whole_completion=True)

    def reward_fn(smiles: str | List[str]) -> float | List[float]:
        if isinstance(smiles, str):
            smiles = [smiles]
            return SCORER([""], [smiles], metadata=[metadata])[0]
        return SCORER([""], smiles, metadata=[metadata] * len(smiles))

    return reward_fn
