from typing import Literal

from pydantic import BaseModel, Field


class MolPropVerifierConfigModel(BaseModel):
    """Pydantic model for molecular verifier configuration.

    This model defines the configuration parameters for the MolecularVerifier class,
    providing validation and documentation for all configuration options.

    Attributes:
        path_to_mappings: Optional path to property mappings and docking targets configuration directory.
        reward: Type of reward to compute. Either "property" for property-based rewards or "valid_smiles"
                for validity-based rewards.
        rescale: Whether to rescale the rewards to a normalized range.
        parse_whole_completion: Whether to parse the entire completion output or only the content
                                between answer tags.
        reaction_matrix_path: Path to the reaction matrix pickle file used for reaction verification.
        oracle_kwargs: Dictionary of keyword arguments to pass to the docking oracle. Can include:
                       - exhaustiveness: Docking exhaustiveness parameter
                       - n_cpu: Number of CPUs for docking
                       - docking_oracle: Type of docking oracle ("pyscreener" or "autodock_gpu")
                       - vina_mode: Command mode for AutoDock GPU
        docking_concurrency_per_gpu: Number of concurrent docking runs to allow per GPU.
                                     Default is 2 (uses ~1GB per run on 80GB GPU).
    """

    reward: Literal["property", "valid_smiles"] = Field(
        default="property",
        description='Reward type: "property" for property-based or "valid_smiles" for validity-based rewards',
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        validate_assignment = True
        json_schema_extra = {
            "example": {
                "reward": "property",
            }
        }
