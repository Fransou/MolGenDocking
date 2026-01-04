import logging
import re
from typing import Any, Dict, List, Tuple

import numpy as np

from mol_gen_docking.reward.verifiers.abstract_verifier import Verifier


class MolPropVerifier(Verifier):
    def __init__(self, reward: str) -> None:
        super().__init__()
        self.reward = reward
        self.logger = logging.getLogger("MolPropVerifier")

    def get_score(
        self, completions: List[Any], metadata: List[Dict[str, Any]]
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        parsed_answer = []
        for answer in completions:
            matches = re.findall(r"<answer>(.*?)</answer>", answer, flags=re.DOTALL)
            self.logger.info(f"Matches: {matches}")
            if len(matches) == 1:
                try:
                    y = matches[0]
                    if y.lower() in ["true", "yes"]:
                        y = 1
                    elif y.lower() in ["false", "no"]:
                        y = 0
                    else:
                        y = float(y)
                    parsed_answer.append(y)
                except ValueError:
                    parsed_answer.append(None)
            else:
                parsed_answer.append(None)
        if self.reward == "valid_smiles":
            return [float(isinstance(y, float)) for y in parsed_answer], [
                {} for _ in parsed_answer
            ]
        rewards = []
        for meta, y in zip(metadata, parsed_answer):
            if y is None:
                rewards.append(0.0)
            else:
                if meta["objectives"][0] == "regression":
                    std = meta.get("norm_var", 1.0)
                    rewards.append(
                        np.clip(
                            1 - ((y - meta["target"][0]) / std) ** 2,
                            a_min=0.0,
                            a_max=1.0,
                        )
                    )
                elif meta["objectives"][0] == "classification":
                    rewards.append(float(y == meta["target"][0]))
                else:
                    self.logger.error(f"Not a valid objective: {meta['objectives'][0]}")
                    raise NotImplementedError
        self.logger.info(f"Rewards: {rewards}")
        return rewards, [{} for _ in parsed_answer]
