import os
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from mol_gen_docking.reward.verifiers.abstract_verifier import (
    VerifierOutputModel,
)

ReactionObjT = Literal[
    "final_product",
    "reactant",
    "all_reactants",
    "all_reactants_bb_ref",
    "smarts",
    "full_path",
    "full_path_bb_ref",
    "full_path_smarts_ref",
    "full_path_smarts_bb_ref",
    "analog_gen",
]


class ReactionVerifierConfigModel(BaseModel):
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

    reaction_matrix_path: str = Field(
        default="data/rxn_matrix.pkl",
        description="Path to the reaction matrix pickle file for reaction verification",
    )

    reaction_reward_type: Literal["binary", "tanimoto"] = Field(
        default="tanimoto",
        description="For retro-synthesis, assign reward based on the exact match (binary) or Tanimoto similarity of the last product",
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "reward": "property",
                "reaction_matrix_path": "data/rxn_matrix.pkl",
                "reaction_reward_type": "tanimoto",
            }
        }

    @model_validator(mode="after")  # type: ignore
    def check_reaction_matrix_path(self) -> "ReactionVerifierConfigModel":
        """Validate that the reaction matrix path exists."""
        if not os.path.exists(self.reaction_matrix_path):
            raise ValueError(
                f"Reaction matrix path {self.reaction_matrix_path} does not exist."
            )
        return self


class ReactionVerifierMetadataModel(BaseModel):
    """Metadata model for reaction verifier results.

    Attributes:
        prop_valid: Proportion of valid reaction steps (0.0 to 1.0).
        correct_last_product: Whether the final product is correct.
        correct_bb: Whether all building blocks used are valid.
    """

    valid: float = Field(
        default=0.0,
        description="Is the answer valid. If the task is to propose a synthesis route, this is the proportion of valid reaction steps (0.0 to 1.0).",
    )
    correct_product: float = Field(
        default=0.0,
        description="Whether the product is correct. For synthesis tasks, if we use tanimoto similarity, similarity to the target molecule, for SMARTS prediction, do both of the chemical reactions lead to the correct product.",
    )
    correct_reactant: bool = Field(
        default=False,
        description="Whether all reactants are correct.",
    )


class ReactionVerifierOutputModel(VerifierOutputModel):
    """Output model for reaction verifier results.

    Attributes:
        reward: The computed reward for the reaction verification.
        verifier_metadata: Metadata related to the reaction verification process.
    """

    reward: float = Field(
        ...,
        description="The computed reward for the reaction verification.",
    )
    verifier_metadata: ReactionVerifierMetadataModel = Field(
        ...,
        description="Metadata related to the reaction verification process.",
    )
