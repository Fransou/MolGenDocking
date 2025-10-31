from typing import Any, Literal, Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class MolecularVerifierSettings(BaseSettings):
    """
    Protocol for molecular docking.
    """

    scorer_exhaustiveness: int = 8
    scorer_ncpus: int = 8
    gpu_utilization_gpu_docking: float = 0.05
    max_concurrent_requests: int = 128
    docking_oracle: Literal["pyscreener", "autodock_gpu"] = "pyscreener"
    vina_mode: str = "autodock_gpu_256wi"  # Command used to run autodock gpu
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
        assert self.gpu_utilization_gpu_docking > 0.0, (
            "GPU utilization per docking run must be > 0"
        )


class MolecularVerifierQuery(BaseModel):
    """
    Query model for the MolecularVerifier.
    """

    query: Optional[list[str]] = None
    prompts: Optional[list[str]] = None
    metadata: Optional[list[dict[str, Any]]] = None


class MolecularVerifierResponse(BaseModel):
    """
    Response model for the VerifierServer.
    """

    reward: float
    reward_list: list[float]
    error: str | None = None
    meta: dict[str, list[Any]] | None = None
    next_turn_feedback: str | None = None
