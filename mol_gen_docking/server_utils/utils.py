from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class MolecularVerifierSettings(BaseSettings):
    """
    Protocol for molecular docking.
    """

    scorer_exhaustiveness: int = 8
    scorer_ncpus: int = 8
    docking_concurrency_per_gpu: int = 2
    max_concurrent_requests: int = 128
    reaction_matrix_path: str = "data/rxn_matrix.pkl"
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
        assert self.docking_concurrency_per_gpu > 0, (
            "GPU utilization per docking run must be > 0"
        )

        assert Path(self.reaction_matrix_path).exists(), (
            f"Reaction matrix file {self.reaction_matrix_path} does not exist"
        )


class MolecularVerifierQuery(BaseModel):
    """
    Query model for the MolecularVerifier.
    """

    metadata: list[dict[str, Any]]
    query: list[str]
    prompts: Optional[list[str]] = None


class MolecularVerifierResponse(BaseModel):
    """
    Response model for the VerifierServer.
    """

    reward: float
    reward_list: list[float]
    error: str | None = None
    meta: dict[str, list[Any]] | None = None
    next_turn_feedback: str | None = None
