from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings

from mol_gen_docking.reward import (
    GenerationVerifierConfigModel,
    MolecularVerifierConfigModel,
    MolPropVerifierConfigModel,
    ReactionVerifierConfigModel,
)


class MolecularVerifierServerSettings(BaseSettings):
    scorer_exhaustiveness: int = 8
    scorer_ncpus: int = 8
    docking_concurrency_per_gpu: int = 2
    max_concurrent_requests: int = 128
    reaction_matrix_path: str = "data/rxn_matrix.pkl"
    docking_oracle: Literal["pyscreener", "autodock_gpu"] = "autodock_gpu"
    vina_mode: str = "autodock_gpu_256wi"
    data_path: str = "data/molgendata"
    buffer_time: int = 20
    parse_whole_completion: bool = False

    def __post_init__(self) -> None:
        assert self.scorer_exhaustiveness > 0, "Exhaustiveness must be greater than 0"
        assert self.scorer_ncpus > 0, "Number of CPUs must be greater than 0"
        assert self.max_concurrent_requests > 0, (
            "Max concurrent requests must be greater than 0"
        )
        assert (
            self.scorer_ncpus
            == self.scorer_exhaustiveness * self.max_concurrent_requests
        ), "Number of CPUs must be equal to exhaustiveness"
        assert self.docking_concurrency_per_gpu > 0, (
            "GPU utilization per docking run must be > 0"
        )

        assert Path(self.reaction_matrix_path).exists(), (
            f"Reaction matrix file {self.reaction_matrix_path} does not exist"
        )

    def to_molecular_verifier_config(
        self, reward: Literal["property", "valid_smiles"] = "property"
    ) -> MolecularVerifierConfigModel:
        """Convert server settings to a verifier configuration model.

        Creates a complete MolecularVerifierConfigModel with all sub-verifier
        configurations (generation, reaction, and property) initialized from
        the server settings.

        Args:
            reward: Type of reward to compute. Options are "property" for
                property-based rewards or "valid_smiles" for validity-based
                rewards. Default is "property".

        Returns:
            MolecularVerifierConfigModel: A fully configured verifier model with:
                - GenerationVerifierConfigModel for molecular generation tasks
                - ReactionVerifierConfigModel for reaction tasks
                - MolPropVerifierConfigModel for property prediction
                All sub-configs automatically synced to the specified reward type.

        Example:
            ```python
            settings = MolecularVerifierServerSettings()
            config = settings.to_molecular_verifier_config(reward="property")
            verifier = MolecularVerifier(verifier_config=config)
            ```
        """
        # Create oracle kwargs from server settings
        oracle_kwargs = {
            "exhaustiveness": self.scorer_exhaustiveness,
            "n_cpu": self.scorer_ncpus,
            "docking_oracle": self.docking_oracle,
            "vina_mode": self.vina_mode,
        }

        # Create GenerationVerifierConfigModel
        generation_config = GenerationVerifierConfigModel(
            path_to_mappings=self.data_path,
            reward=reward,
            rescale=True,
            oracle_kwargs=oracle_kwargs,
            docking_concurrency_per_gpu=self.docking_concurrency_per_gpu,
        )

        # Create ReactionVerifierConfigModel
        reaction_config = ReactionVerifierConfigModel(
            reaction_matrix_path=self.reaction_matrix_path,
            reward=reward,
        )

        # Create MolPropVerifierConfigModel
        molprop_config = MolPropVerifierConfigModel(reward=reward)

        # Create and return MolecularVerifierConfigModel
        return MolecularVerifierConfigModel(
            parse_whole_completion=self.parse_whole_completion,
            reward=reward,
            generation_verifier_config=generation_config,
            reaction_verifier_config=reaction_config,
            mol_prop_verifier_config=molprop_config,
        )
