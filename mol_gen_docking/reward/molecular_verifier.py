import logging
from typing import Any, Callable, Dict, List

import ray
from ray.experimental import tqdm_ray
from rdkit import RDLogger

from mol_gen_docking.reward.molecular_verifier_pydantic_model import (
    BatchMolecularVerifierOutputModel,
    MolecularVerifierConfigModel,
)
from mol_gen_docking.reward.verifiers import (
    GenerationVerifier,
    GenerationVerifierOutputModel,
    MolPropVerifier,
    MolPropVerifierOutputModel,
    ReactionVerifier,
    ReactionVerifierOutputModel,
    VerifierInputBatchModel,
    VerifierOutputModel,
    assign_to_inputs,
)

RDLogger.DisableLog("rdApp.*")


class MolecularVerifier:
    def __init__(
        self,
        verifier_config: MolecularVerifierConfigModel,
    ):
        self.verifier_config = verifier_config
        self.__name__ = "RewardScorer/MolecularVerifier"
        self.remote_tqdm = ray.remote(tqdm_ray.tqdm)

        self._generation_verifier: None | GenerationVerifier = None
        self._mol_prop_verifier: None | MolPropVerifier = None
        self._reaction_verifier: None | ReactionVerifier = None
        self.logger = logging.getLogger("RewardScorer")

        if not ray.is_initialized():
            ray.init()

    @property
    def generation_verifier(self) -> GenerationVerifier:
        if self._generation_verifier is not None:
            return self._generation_verifier
        assert self.verifier_config.generation_verifier_config is not None
        self._generation_verifier = GenerationVerifier(
            verifier_config=self.verifier_config.generation_verifier_config
        )
        return self._generation_verifier

    @property
    def mol_prop_verifier(self) -> MolPropVerifier:
        if self._mol_prop_verifier is not None:
            return self._mol_prop_verifier
        assert self.verifier_config.mol_prop_verifier_config is not None
        self._mol_prop_verifier = MolPropVerifier(
            verifier_config=self.verifier_config.mol_prop_verifier_config
        )
        return self._mol_prop_verifier

    @property
    def reaction_verifier(self) -> ReactionVerifier:
        if self._reaction_verifier is not None:
            return self._reaction_verifier
        assert self.verifier_config.reaction_verifier_config is not None
        self._reaction_verifier = ReactionVerifier(
            verifier_config=self.verifier_config.reaction_verifier_config
        )

        return self._reaction_verifier

    def _get_generation_score(
        self,
        inputs: VerifierInputBatchModel,
        debug: bool = False,
        use_pbar: bool = False,
    ) -> List[GenerationVerifierOutputModel]:
        """
        Get reward for molecular properties
        """
        if debug:  # Testing purposes
            self.generation_verifier.debug = True
        elif self._generation_verifier is not None:
            self.generation_verifier.debug = False
        return self.generation_verifier.get_score(inputs)

    def _get_prop_pred_score(
        self,
        inputs: VerifierInputBatchModel,
        debug: bool = False,
        use_pbar: bool = False,
    ) -> List[MolPropVerifierOutputModel]:
        return self.mol_prop_verifier.get_score(inputs)

    def _get_reaction_score(
        self,
        inputs: VerifierInputBatchModel,
        debug: bool = False,
        use_pbar: bool = False,
    ) -> List[ReactionVerifierOutputModel]:
        return self.reaction_verifier.get_score(inputs)

    def get_score(
        self,
        completions: List[Any],
        metadata: List[Dict[str, Any]],
        debug: bool = False,
        use_pbar: bool = False,
    ) -> BatchMolecularVerifierOutputModel:
        assert len(completions) == len(metadata)
        obj_to_fn: Dict[
            str,
            Callable[
                [
                    VerifierInputBatchModel,
                    bool,
                    bool,
                ],
                List[VerifierOutputModel],
            ],
        ] = {
            "generation": self._get_generation_score,
            "mol_prop": self._get_prop_pred_score,
            "reaction": self._get_reaction_score,
        }
        idxs: Dict[str, List[int]] = {"generation": [], "mol_prop": [], "reaction": []}
        completions_per_obj: Dict[str, List[str]] = {
            "generation": [],
            "mol_prop": [],
            "reaction": [],
        }
        metadata_per_obj: Dict[str, List[Dict[str, Any]]] = {
            "generation": [],
            "mol_prop": [],
            "reaction": [],
        }
        for i, (completion, meta) in enumerate(zip(completions, metadata)):
            assigned = assign_to_inputs(completion, meta)
            idxs[assigned].append(i)
            completions_per_obj[assigned].append(completion)
            metadata_per_obj[assigned].append(meta)

        rewards = [0.0 for _ in range(len(metadata))]
        metadata = [{} for _ in range(len(metadata))]
        for key, fn in obj_to_fn.items():
            if len(completions_per_obj[key]) > 0:
                outputs_obj = fn(
                    VerifierInputBatchModel(
                        completions=completions_per_obj[key],
                        metadatas=metadata_per_obj[key],
                    ),
                    debug,
                    use_pbar,
                )
                for i, output in zip(idxs[key], outputs_obj):
                    rewards[i] = output.reward
                    metadata[i] = output.verifier_metadata
        return BatchMolecularVerifierOutputModel(
            rewards=rewards, verifier_metadatas=metadata
        )

    def __call__(
        self,
        completions: List[Any],
        metadata: List[Dict[str, Any]],
        debug: bool = False,
        use_pbar: bool = False,
    ) -> BatchMolecularVerifierOutputModel:
        """
        Call the scorer to get the rewards.
        """
        return self.get_score(
            completions=completions, metadata=metadata, debug=debug, use_pbar=use_pbar
        )
