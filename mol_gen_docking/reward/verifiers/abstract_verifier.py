from typing import Any, Dict, List

from pydantic import BaseModel, Field, model_validator

from .generation_reward.input_metadata import GenerationVerifierInputMetadataModel
from .mol_prop_reward.input_metadata import MolPropVerifierInputMetadataModel
from .reaction_reward.input_metadata import ReactionVerifierInputMetadataModel


class VerifierOutputModel(BaseModel):
    reward: float = Field(..., description="Reward score assigned by the verifier.")
    verifier_metadata: Dict[str, Any] = Field(
        ..., description="Additional metadata from the verification process."
    )


class VerifierInputBatchModel(BaseModel):
    completions: List[str] = Field(
        ..., description="List of model completions to be verified."
    )
    metadatas: (
        List[GenerationVerifierInputMetadataModel]
        | List[MolPropVerifierInputMetadataModel]
        | List[ReactionVerifierInputMetadataModel]
    ) = Field(
        ...,
        description="List of metadata corresponding to each completion.",
    )

    @model_validator(mode="after")  # type: ignore
    def check_lengths(self) -> "VerifierInputBatchModel":
        if len(self.completions) != len(self.metadatas):
            raise ValueError("`completions` and `metadatas` must have the same length")
        return self


class Verifier:
    def __init__(self) -> None:
        pass

    def get_score(self, inputs: VerifierInputBatchModel) -> List[VerifierOutputModel]:
        raise NotImplementedError
