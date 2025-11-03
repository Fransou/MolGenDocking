import ray

from mol_gen_docking.reward.rl_rewards import RewardScorer

from .utils import (
    DATA_PATH,
)

if not ray.is_initialized():
    ray.init(num_cpus=16)


property_scorer = RewardScorer(DATA_PATH, "property", rescale=False)


def test_reaction():
    target = "Nc1ccccc1C=O + O=C1CC1N=C"
    completions = [
        "<answer> {} </answer>".format(v)
        for v in [
            target,
            "c1(C=O)c(N)cccc1, C=NC1CC1=O",
            "c1(C=O)c(N)cccc1 + C=NC1CC1=O",
            "Nc1ccccc1C=O, CCC",
            "ksdjgf",
        ]
    ] * 4

    metadata = [
        {"objectives": [obj], "properties": [""], "target": [target]}
        for obj in ["product", "product_full", "reactant", "reactant_full"]
        for _ in range(5)
    ]
    rewards = property_scorer(completions, metadata, use_pbar=False)
    assert rewards == [1.0, 1.0, 1.0, 0.1, 0] * 4
