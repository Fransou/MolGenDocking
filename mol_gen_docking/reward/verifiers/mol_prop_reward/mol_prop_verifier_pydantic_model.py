from typing import Literal

from pydantic import BaseModel, Field

from mol_gen_docking.reward.verifiers.abstract_verifier_pydantic_model import (
    VerifierOutputModel,
)


class MolPropVerifierConfigModel(BaseModel):
    """Pydantic model for molecular verifier configuration.

    This model defines the configuration parameters for the MolecularVerifier class,
    providing validation and documentation for all configuration options.
    """

    reward: Literal["property", "valid_smiles"] = Field(
        default="property",
        description='Reward type: "property" for property-based or "valid_smiles" for validity-based rewards',
    )
    parsing_method: Literal["none", "answer_tags", "boxed"] = Field(
        default="answer_tags",
        description="Method to parse model completions for SMILES or property values.",
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        validate_assignment = True
        json_schema_extra = {
            "example": {
                "reward": "property",
                "parsing_method": "answer_tags",
            }
        }


class MolPropVerifierMetadataModel(BaseModel):
    """Metadata model for molecular property verifier results.

    Attributes:
    property_verif_extracted_answer (float): The extracted answer string from the model completion.
    property_verif_extraction_success (bool): Indicates whether the answer extraction was successful.
    """

    property_verif_extracted_answer: float = Field(
        ...,
        description="The extracted answer string from the model completion.",
    )
    property_verif_extraction_success: bool = Field(
        ...,
        description="Indicates whether the answer extraction was successful.",
    )


class MolPropVerifierOutputModel(VerifierOutputModel):
    """Output model for molecular property verifier results.

    Attributes:

    """

    reward: float = Field(
        ...,
        description="The computed reward for the molecular property verification.",
    )
    verifier_metadata: MolPropVerifierMetadataModel = Field(
        ...,
        description="Metadata related to the molecular property verification process.",
    )
