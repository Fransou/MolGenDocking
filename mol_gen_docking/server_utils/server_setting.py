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
    """Configuration settings for the Molecular Verifier server.

    This class manages all configuration parameters for running the Molecular Verifier
    server, including docking settings, oracle configuration, and verifier behavior.
    Settings can be loaded from environment variables, .env files, or passed directly.

    The settings are used to initialize verifier configurations for molecular property
    prediction, reaction verification, and molecular generation tasks. It provides
    validation of settings and conversion to verifier config models.

    Attributes:
        scorer_exhaustiveness (int): Exhaustiveness parameter for docking scoring.
            Controls the thoroughness of the docking search. Higher values increase
            accuracy but require more computation time.
            Default: 8

        scorer_ncpus (int): Number of CPU cores to allocate for scoring operations.
            Must be compatible with exhaustiveness and max_concurrent_requests.
            Default: 8

        docking_concurrency_per_gpu (int): Number of concurrent docking jobs per GPU.
            Controls GPU utilization.
            Default: 2

        reaction_matrix_path (str): Path to the pickled reaction matrix file.
            Must exist and be accessible for reaction verification tasks.
            Default: "data/rxn_matrix.pkl"

        docking_oracle (Literal["pyscreener", "autodock_gpu"]): The docking software
            to use for molecular docking. "pyscreener" for PyScreener docking or
            "autodock_gpu" for GPU-accelerated AutoDock.
            Default: "autodock_gpu"

        vina_mode (str): Command mode for AutoDock GPU execution.
            Example: "autodock_gpu_256wi" for 256 work items mode.
            Only used when docking_oracle is "autodock_gpu".
            Default: "autodock_gpu_256wi"

        data_path (str): Path to the molecular data directory containing:
            - names_mapping.json: Property name mappings
            - docking_targets.json: Target definitions for docking
            Used by generation tasks, and must be accessible.
            Default: "data/molgendata"

        buffer_time (int): Time in seconds to buffer requests before processing.
            Allows batch processing of concurrent requests to improve efficiency.
            Default: 20

        parse_whole_completion (bool): Whether to parse entire completion output
            or only extract answers from tagged regions (<answer>...</answer>).
            Default: False

    Example:
        ```python
        from mol_gen_docking.server_utils.server_setting import MolecularVerifierServerSettings
        from mol_gen_docking.reward import MolecularVerifier

        # Load settings from environment variables or .env file
        settings = MolecularVerifierServerSettings()

        # Convert to verifier configuration
        config = settings.to_molecular_verifier_config(reward="property")

        # Create verifier instance
        verifier = MolecularVerifier(verifier_config=config)
        ```

    Environment Variables:
        Settings can be configured via environment variables (non-case sensitive) names:

        - SCORER_EXHAUSTIVENESS
        - SCORER_NCPUS
        - DOCKING_CONCURRENCY_PER_GPU
        - REACTION_MATRIX_PATH
        - DOCKING_ORACLE
        - VINA_MODE
        - DATA_PATH
        - BUFFER_TIME
        - PARSE_WHOLE_COMPLETION
    """

    scorer_exhaustiveness: int = 8
    scorer_ncpus: int = 8
    docking_concurrency_per_gpu: int = 2
    reaction_matrix_path: str = "data/rxn_matrix.pkl"
    docking_oracle: Literal["pyscreener", "autodock_gpu"] = "autodock_gpu"
    vina_mode: str = "autodock_gpu_256wi"
    data_path: str = "data/molgendata"
    buffer_time: int = 20
    parse_whole_completion: bool = False

    def __post_init__(self) -> None:
        """Validate all settings after initialization.

        Performs critical validation checks to ensure settings are consistent
        and valid before the server starts. This method is called automatically
        after the pydantic BaseSettings initializes the instance.

        Raises:
            AssertionError: If any of the following conditions are not met:

                - scorer_exhaustiveness > 0
                - scorer_ncpus > 0
                - max_concurrent_requests > 0
                - scorer_ncpus == scorer_exhaustiveness * max_concurrent_requests
                - docking_concurrency_per_gpu > 0
                - reaction_matrix_path file exists

        Note:
            The constraint scorer_ncpus == scorer_exhaustiveness * max_concurrent_requests
            ensures that CPU allocation matches the docking configuration.
            For example, with exhaustiveness=8 and max_concurrent=16, you need 128 CPUs.
        """
        assert self.scorer_exhaustiveness > 0, "Exhaustiveness must be greater than 0"
        assert self.scorer_ncpus > 0, "Number of CPUs must be greater than 0"
        assert self.max_concurrent_requests > 0, (
            "Max concurrent requests must be greater than 0"
        )

        assert Path(self.reaction_matrix_path).exists(), (
            f"Reaction matrix file {self.reaction_matrix_path} does not exist"
        )

    def to_molecular_verifier_config(
        self, reward: Literal["property", "valid_smiles"] = "property"
    ) -> MolecularVerifierConfigModel:
        """Convert server settings to a complete verifier configuration model.

        Creates a MolecularVerifierConfigModel with all sub-verifier configurations
        (generation, reaction, and property) initialized from the server settings.
        This method bridges the gap between server-level settings and verifier-level
        configurations, ensuring consistent parameter propagation.

        Args:
            reward (Literal["property", "valid_smiles"]): Type of reward to compute.
                - "property": Use property-based rewards for molecular optimization.
                  Evaluates predicted property values against target objectives.
                - "valid_smiles": Use validity-based rewards. Returns 1.0 for valid
                  molecules and 0.0 for invalid ones.
                Default: "property"

        Returns:
            MolecularVerifierConfigModel: A fully configured verifier model containing:

                - GenerationVerifierConfigModel: For molecular generation tasks with:
                  - path_to_mappings set to data_path
                  - reward type synchronized
                  - rescale enabled
                  - oracle_kwargs with docking settings
                  - docking_concurrency_per_gpu configured

                - ReactionVerifierConfigModel: For reaction verification with:
                  - reaction_matrix_path configured
                  - reward type synchronized
                  - reaction_reward_type set to "tanimoto" (default)

                - MolPropVerifierConfigModel: For property prediction with:
                  - reward type synchronized

                All sub-configs are automatically synced to the specified reward type.

        Raises:
            ValueError: If any of the referenced configuration paths don't exist.
                This can happen if data_path or reaction_matrix_path are invalid.
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
