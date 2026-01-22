from typing import List, Literal

from pydantic import BaseModel, Field, model_validator

GenerationObjT = Literal["maximize", "minimize", "above", "below"]


class GenerationVerifierInputMetadataModel(BaseModel):
    """Input metadata model for generation verifier.

    Attributes:
        properties: List of property names to verify.
    """

    properties: List[str] = Field(
        ...,
        description="List of property names to verify.",
    )
    objectives: List[GenerationObjT] = Field(
        ...,
        description="List of objectives for each property: maximize, minimize, above, or below.",
    )
    target: List[float] = Field(
        ...,
        description="List of target values for each property.",
    )

    @model_validator(mode="after")  # type: ignore
    def validate_properties(self) -> "GenerationVerifierInputMetadataModel":
        """Validate that properties, objectives, and target have the same length."""
        if not (len(self.properties) == len(self.objectives) == len(self.target)):
            raise ValueError(
                "Length of properties, objectives, and target must be the same."
            )
        return self
