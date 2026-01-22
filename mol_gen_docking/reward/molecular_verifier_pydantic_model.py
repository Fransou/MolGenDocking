from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from mol_gen_docking.reward.verifiers import (
    GenerationVerifierConfigModel,
    GenerationVerifierMetadataModel,
    MolPropVerifierConfigModel,
    MolPropVerifierMetadataModel,
    ReactionVerifierConfigModel,
    ReactionVerifierMetadataModel,
)


class MolecularVerifierConfigModel(BaseModel):
    """Pydantic model for molecular verifier configuration.

    This model defines the configuration parameters for the MolecularVerifier class,
    providing validation and documentation for all configuration options.

    The reward field is automatically propagated to all sub-verifier configurations.
    """

    parse_whole_completion: bool = Field(
        default=False,
        description="Whether to parse the whole completion from the model or just the extracted answer between special tokens.",
    )
    reward: Literal["valid_smiles", "property"] = Field(
        default="property",
        description="Type of reward to use for molecular verification.",
    )
    generation_verifier_config: Optional[GenerationVerifierConfigModel] = Field(
        None,
        description="Configuration for generation verifier, required if reward is 'valid_smiles'.",
    )
    mol_prop_verifier_config: Optional[MolPropVerifierConfigModel] = Field(
        None,
        description="Configuration for molecular property verifier, required if reward is 'property'.",
    )
    reaction_verifier_config: Optional[ReactionVerifierConfigModel] = Field(
        None,
        description="Configuration for reaction verifier, required if reward is 'reaction'.",
    )

    @model_validator(mode="after")  # type: ignore
    def propagate_reward_to_subconfigs(self) -> "MolecularVerifierConfigModel":
        """Propagate the reward field to all sub-verifier configurations."""
        if self.generation_verifier_config is not None:
            self.generation_verifier_config.reward = self.reward
            self.generation_verifier_config.parse_whole_completion = (
                self.parse_whole_completion
            )
        if self.mol_prop_verifier_config is not None:
            self.mol_prop_verifier_config.reward = self.reward
        if self.reaction_verifier_config is not None:
            self.reaction_verifier_config.reward = self.reward
        return self

    class Config:
        """Pydantic configuration for the MolecularVerifierConfigModel."""

        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "parse_whole_completion": False,
                "reward": "property",
                "generation_verifier_config": {
                    "path_to_mappings": "data/molgendata",
                    "rescale": True,
                    "oracle_kwargs": {
                        "exhaustiveness": 8,
                        "n_cpu": 8,
                        "docking_oracle": "autodock_gpu",
                        "vina_mode": "autodock_gpu_256wi",
                    },
                    "docking_concurrency_per_gpu": 2,
                },
                "mol_prop_verifier_config": {},
                "reaction_verifier_config": {
                    "reaction_matrix_path": "data/rxn_matrix.pkl",
                    "reaction_reward_type": "tanimoto",
                },
            }
        }


class BatchMolecularVerifierOutputModel(BaseModel):
    """Output model for molecular verifier results.

    Attributes:
        reward: The computed reward for the molecular verification.
    """

    rewards: list[float] = Field(
        ...,
        description="List of computed rewards for the molecular verification.",
    )
    verifier_metadatas: list[
        GenerationVerifierMetadataModel
        | ReactionVerifierMetadataModel
        | MolPropVerifierMetadataModel
    ] = Field(
        ...,
        description="List of metadata from each verifier used in the molecular verification.",
    )
