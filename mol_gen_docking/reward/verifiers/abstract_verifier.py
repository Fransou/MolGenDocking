from typing import Any, Dict, List

from pydantic import BaseModel, Field


class VerifierOutputModel(BaseModel):
    reward: float = Field(..., description="Reward score assigned by the verifier.")
    verifier_metadata: Dict[str, Any] = Field(
        ..., description="Additional metadata from the verification process."
    )


class Verifier:
    def __init__(self) -> None:
        pass

    def get_score(
        self, completions: List[Any], metadata: List[Dict[str, Any]]
    ) -> List[VerifierOutputModel]:
        raise NotImplementedError
