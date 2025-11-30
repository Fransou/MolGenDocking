import numpy as np

from mol_gen_docking.reward.rl_rewards import RewardScorer

from .utils import (
    DATA_PATH,
)

property_scorer = RewardScorer(DATA_PATH, "property", rescale=False)


def test_regression():
    target = np.random.random()
    completions = [
        "<answer> {} </answer>".format(v * np.sign(np.random.random() - 0.5) + target)
        for v in [0, 0.01, 0.5, 1]
    ] + ["ksdjgf"]
    metadata = [
        {"objectives": ["regression"], "properties": [""], "target": [target]}
    ] * 5
    rewards = property_scorer(completions, metadata)[0]
    assert sorted(rewards)[::-1] == rewards


def test_classification():
    completions = [
        "<answer> {} </answer>".format(v) for v in [1, 1, 0, 0, "bbfhdsbfsj"]
    ]
    metadata = [
        {"objectives": ["classification"], "properties": [""], "target": [1]}
    ] * 5
    rewards = property_scorer(completions, metadata)[0]
    assert rewards == [1.0, 1.0, 0.0, 0.0, 0.0]

    metadata = [
        {"objectives": ["classification"], "properties": [""], "target": [0]}
    ] * 5
    rewards = property_scorer(completions, metadata)[0]
    assert rewards == [0.0, 0.0, 1.0, 1.0, 0.0]


def test_with_generation():
    completions = ["<answer> {} </answer>".format(v) for v in [1, 1, 0, 0, 0]] + [
        "<answer> CCC{} </answer>".format(smi) for smi in ["C" * i for i in range(5)]
    ]
    metadata = [
        {"objectives": ["classification"], "properties": [""], "target": [1]}
    ] * 5 + [
        {
            "objectives": ["maximize"],
            "properties": ["CalcNumRotatableBonds"],
            "target": [1],
        }
    ] * 5
    rewards = property_scorer(completions, metadata, debug=True)[0]
    assert rewards == [1.0, 1.0, 0.0, 0.0, 0.0] + [0, 1, 2, 3, 4]

    metadata = [
        {"objectives": ["classification"], "properties": [""], "target": [0]}
    ] * 5 + [
        {
            "objectives": ["maximize"],
            "properties": ["CalcNumRotatableBonds"],
            "target": [1],
        }
    ] * 5
    rewards = property_scorer(completions, metadata)[0]
    assert rewards == [0.0, 0.0, 1.0, 1.0, 1.0] + [0, 1, 2, 3, 4]
