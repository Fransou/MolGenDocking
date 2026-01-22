import logging
import re
from typing import List

import numpy as np

from mol_gen_docking.reward.verifiers.abstract_verifier import (
    Verifier,
    VerifierInputBatchModel,
)
from mol_gen_docking.reward.verifiers.mol_prop_reward.input_metadata import (
    MolPropVerifierInputMetadataModel,
)
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
        self, inputs: VerifierInputBatchModel
    ) -> List[MolPropVerifierOutputModel]:
        completions = inputs.completions
        assert all(
            isinstance(meta, MolPropVerifierInputMetadataModel)
            for meta in inputs.metadatas
        )
        metadatas: List[MolPropVerifierInputMetadataModel] = inputs.metadatas

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
                        verifier_metadatas.append(
                            MolPropVerifierMetadataModel(extracted_answer=None)
                        )
                        continue
                    if len(ys) > 1:
                        self.logger.info(f"Too many values found in answer: {matches}")
                        verifier_metadatas.append(
                            MolPropVerifierMetadataModel(extracted_answer=None)
                        )
                        continue
                    verifier_metadatas.append(
                        MolPropVerifierMetadataModel(extracted_answer=ys[0])
                    )
                except ValueError:
                    verifier_metadatas.append(
                        MolPropVerifierMetadataModel(extracted_answer=None)
                    )
            else:
                verifier_metadatas.append(
                    MolPropVerifierMetadataModel(extracted_answer=None)
                )

        if self.verifier_config.reward == "valid_smiles":
            return [
                MolPropVerifierOutputModel(
                    reward=float(
                        isinstance(verifier_meta.extracted_answer, (float, int))
                    ),
                    verifier_metadata=verifier_meta,
                )
                for verifier_meta in verifier_metadatas
            ]

        rewards = []
        for meta, verifier_meta in zip(metadatas, verifier_metadatas):
            if verifier_meta.extracted_answer is None:
                rewards.append(0.0)
            else:
                if meta.objectives[0] == "regression":
                    std = meta.norm_var if meta.norm_var is not None else 1.0
                    rewards.append(
                        np.clip(
                            1
                            - ((verifier_meta.extracted_answer - meta.target[0]) / std)
                            ** 2,
                            a_min=0.0,
                            a_max=1.0,
                        )
                    )
                elif meta.objectives[0] == "classification":
                    rewards.append(
                        float(verifier_meta.extracted_answer == meta.target[0])
                    )
                else:
                    self.logger.error(f"Not a valid objective: {meta.objectives[0]}")
                    raise NotImplementedError

        self.logger.info(f"Rewards: {rewards}")
        return [
            MolPropVerifierOutputModel(
                reward=reward, verifier_metadata=verifier_metadata
            )
            for reward, verifier_metadata in zip(rewards, verifier_metadatas)
        ]
