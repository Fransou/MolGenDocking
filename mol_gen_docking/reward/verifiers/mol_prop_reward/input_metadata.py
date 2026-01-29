from typing import List, Literal

from pydantic import BaseModel, Field

MolPropObjT = Literal["regression", "classification"]


class MolPropVerifierInputMetadataModel(BaseModel):
    """Input metadata model for molecular property verifier.

    Attributes:
        prompt: The input prompt string used for verification.
    """

    objectives: List[MolPropObjT] = Field(
        ...,
        description="The type of objective for the property: regression or classification.",
    )
    properties: List[str] = Field(
        default_factory=list,
        description="The molecular properties to be verified.",
    )
    target: List[float | int] = Field(
        ...,
        description="The target value for the molecular property to verify against.",
    )
    norm_var: float | int | None = Field(
        default=None,
        description="Normalization variance for regression objectives.",
    )
