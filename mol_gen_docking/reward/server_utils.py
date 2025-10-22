from typing import Any, Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class MolecularVerifierSettings(BaseSettings):
    """
    Protocol for molecular docking.
    """

    scorer_exhaustiveness: int = 2
    scorer_cpus: int = 2
    max_concurrent_requests: int = 128

    data_path: str = "data"

    def __post_init__(self) -> None:
        assert self.scorer_exhaustiveness > 0, "Exhaustiveness must be greater than 0"
        assert self.scorer_cpus > 0, "Number of CPUs must be greater than 0"
        assert self.max_concurrent_requests > 0, (
            "Max concurrent requests must be greater than 0"
        )
        assert (
            self.scorer_cpus
            == self.scorer_exhaustiveness * self.max_concurrent_requests
        ), "Number of CPUs must be equal to exhaustiveness"


class MolecularVerifierQuery(BaseModel):
    """
    Query model for the MolecularVerifier.
    """

    query: list[str]
    prompts: list[str]
    metadata: Optional[list[dict[str, Any]]] = None


class MolecularVerifierResponse(BaseModel):
    """
    Response model for the VerifierServer.
    """

    reward: float
    error: str | None = None
    meta: dict | None = None
    next_turn_feedback: str | None = None
