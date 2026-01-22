import logging
import re
from typing import Any, Dict, List

import numpy as np

from mol_gen_docking.reward.verifiers.abstract_verifier import Verifier
from mol_gen_docking.reward.verifiers.mol_prop_reward.mol_prop_verifier_pydantic_model import (
    MolPropVerifierConfigModel,
    MolPropVerifierMetadataModel,
    MolPropVerifierOutputModel,
)


class MolPropVerifier(Verifier):
    def __init__(self, verifier_config: MolPropVerifierConfigModel) -> None:
        super().__init__()
        self.verifier_config = verifier_config
        self.logger = logging.getLogger("MolPropVerifier")

    def get_score(
        self, completions: List[Any], metadata: List[Dict[str, Any]]
    ) -> List[MolPropVerifierOutputModel]:
        verifier_metadatas: List[MolPropVerifierMetadataModel] = []
        for answer in completions:
            matches = re.findall(r"<answer>(.*?)</answer>", answer, flags=re.DOTALL)
            self.logger.info(f"Matches: {matches}")
            if len(matches) == 1:
                try:
                    split_answer = re.split("\n| |\t|:|`|'", matches[0])
                    ys: List[float | int | None] = []
                    for spl in split_answer:
                        if spl.lower() in ["true", "yes"]:
                            ys.append(1)
                        elif spl.lower() in ["false", "no"]:
                            ys.append(0)
                        else:
                            if re.match(r"(\+|-)?\d+(\.\d+)?", spl):
                                ys.append(float(spl))
                    if len(ys) == 0:
                        verifier_metadatas.append(MolPropVerifierMetadataModel(None))
                        continue
                    if len(ys) > 1:
                        self.logger.info(f"Too many values found in answer: {matches}")
                        verifier_metadatas.append(MolPropVerifierMetadataModel(None))
                        continue
                    verifier_metadatas.append(MolPropVerifierMetadataModel(ys[0]))
                except ValueError:
                    verifier_metadatas.append(MolPropVerifierMetadataModel(None))
            else:
                verifier_metadatas.append(MolPropVerifierMetadataModel(None))
        if self.verifier_config.reward == "valid_smiles":
            return [
                MolPropVerifierOutputModel(
                    rewards=float(isinstance(verifier_meta.y, (float, int))),
                    verifier_metadata=verifier_meta,
                )
                for verifier_meta in verifier_metadatas
            ]

        rewards = []
        for meta, verifier_meta in zip(metadata, verifier_metadatas):
            if verifier_meta.y is None:
                rewards.append(0.0)
            else:
                if meta["objectives"][0] == "regression":
                    std = meta.get("norm_var", 1.0)
                    rewards.append(
                        np.clip(
                            1 - ((verifier_meta.y - meta["target"][0]) / std) ** 2,
                            a_min=0.0,
                            a_max=1.0,
                        )
                    )
                elif meta["objectives"][0] == "classification":
                    rewards.append(float(verifier_meta.y == meta["target"][0]))
                else:
                    self.logger.error(f"Not a valid objective: {meta['objectives'][0]}")
                    raise NotImplementedError
        self.logger.info(f"Rewards: {rewards}")
        return [
            MolPropVerifierOutputModel(
                rewards=reward, verifier_metadatas=verifier_metadata
            )
            for reward, verifier_metadata in zip(rewards, verifier_metadatas)
        ]
