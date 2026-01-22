from pathlib import Path
from typing import Any, List, Literal, Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class MolecularVerifierSettings(BaseSettings):
    """
    Protocol for molecular docking.
    Args:
        scorer_exhaustiveness (int): Exhaustiveness parameter for the docking scorer.
        scorer_ncpus (int): Number of CPUs to use for the docking scorer.
        docking_concurrency_per_gpu (int): Number of concurrent docking runs per GPU.
        max_concurrent_requests (int): Maximum number of concurrent requests to handle.
        reaction_matrix_path (str): Path to the reaction matrix file.
        docking_oracle (Literal["pyscreener", "autodock_gpu"]): Docking oracle to use.
        vina_mode (str): Command used to run autodock gpu.
        data_path (str): Path to the data directory.
        buffer_time (int): Buffer time in seconds used to gather concurrent requests before computation.
        parse_whole_completion (bool): Whether to parse the whole completion output.
    """

    scorer_exhaustiveness: int = 8
    scorer_ncpus: int = 8
    docking_concurrency_per_gpu: int = 2
    max_concurrent_requests: int = 128
    reaction_matrix_path: str = "data/rxn_matrix.pkl"
    docking_oracle: Literal["pyscreener", "autodock_gpu"] = "pyscreener"
    vina_mode: str = "autodock_gpu_256wi"  # Command used to run autodock gpu
    data_path: str = "data"
    buffer_time: int = 20
    parse_whole_completion: bool = False

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


class MolecularVerifierServerQuery(BaseModel):
    """
    Query model for the MolecularVerifier.
    Args:
        metadata (list[dict[str, Any]]): List of metadata dictionaries for prompt.
        query (list[str]): List of generated completions.
        prompts (Optional[list[str]]): Optional list of prompts for each molecule.
    """

    metadata: List[dict[str, Any]]
    query: List[str]
    prompts: Optional[List[str]] = None


class MolecularVerifierServerMetadata(BaseModel):
    """
    Metadata model for the MolecularVerifier.
    Args:
        smiles_extraction_failure (Optional[str]): Error message for SMILES extraction failure.
        all_smi_rewards (Optional[list[float]]): List of rewards for all SMILES.
        all_smi (Optional[list[str]]): List of all SMILES strings.
        individual_rewards (Optional[list[float]]): List of individual rewards.
        properties (Optional[list[str]]): List of properties evaluated.
        extracted_answer (Optional[str]): Extracted answer from molecule property prediction.
        prop_valid (Optional[float]): Validity score of the property prediction.
        correct_last_product (Optional[bool]): Whether the last product is correct.
        correct_bb (Optional[bool]): Whether the building block is correct.
        Reactants_contained (Optional[bool]): Whether reactants are contained in the prediction.
        Products_contained (Optional[bool]): Whether products are contained in the prediction.
    """

    # MOL GENERATION ARGS
    smiles_extraction_failure: Optional[str] = None
    all_smi_rewards: Optional[List[float]] = None
    all_smi: Optional[List[str]] = None
    individual_rewards: Optional[List[float]] = None
    properties: Optional[List[str]] = None

    # MOL PROP PRED ARGS
    extracted_answer: Optional[str] = None

    # CHEMICAL REACTION ARGS
    prop_valid: Optional[float] = None
    correct_last_product: Optional[bool] = None
    correct_bb: Optional[bool] = None

    Reactants_contained: Optional[bool] = None
    Products_contained: Optional[bool] = None


class MolecularVerifierServerResponse(BaseModel):
    """
    Response model for the VerifierServer.
    Args:
        reward (float): Overall reward score.
        reward_list (list[float]): List of individual reward scores.
        error (Optional[str]): Optional error message.
        meta (Optional[dict[str, list[Any]]]): Optional metadata dictionary.
        next_turn_feedback (Optional[str]): Optional feedback for the next turn.
    """

    reward: float
    reward_list: List[float]
    error: Optional[str] = None
    meta: Optional[List[MolecularVerifierServerMetadata]] = None
    next_turn_feedback: Optional[str] = None
