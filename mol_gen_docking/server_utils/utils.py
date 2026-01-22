from pathlib import Path
from typing import Any, List, Literal, Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class MolecularVerifierServerSettings(BaseSettings):
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
    docking_oracle: Literal["pyscreener", "autodock_gpu"] = "autodock_gpu"
    vina_mode: str = "autodock_gpu_256wi"
    data_path: str = "data/molgendata"
    buffer_time: int = 20
    parse_whole_completion: bool = False

    def __post_init__(self) -> None:
        assert self.scorer_exhaustiveness > 0, "Exhaustiveness must be greater than 0"
        assert self.scorer_ncpus > 0, "Number of CPUs must be greater than 0"
        assert self.max_concurrent_requests > 0, (
            "Max concurrent requests must be greater than 0"
        )
        assert (
            self.scorer_ncpus
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
    """Response from the Molecular Verifier server.

    Contains the computed reward scores and detailed metadata for each
    evaluated completion.

    Attributes:
        reward: Overall reward score combining all evaluated properties.
            Typically normalized to [0.0, 1.0] range when rescaling is enabled.

        reward_list: List of individual reward scores, one per evaluated
            completion in the request.

        error: Error message if scoring failed. None if successful.

        meta: List of metadata dictionaries with detailed scoring information.
            Contains extraction failures, property rewards, and verification
            results for each completion. One metadata object per query item.

        next_turn_feedback: Optional feedback for multi-turn conversations.
            Can be used to guide subsequent model generations.

    Example:
        ```json
        {
          "reward": 0.75,
          "reward_list": [0.75],
          "error": null,
          "meta": [
            {
              "smiles_extraction_failure": null,
              "all_smi": ["CC(C)Cc1ccc(cc1)"],
              "all_smi_rewards": [0.75],
              "properties": ["GSK3B", "CalcLogP"],
              "individual_rewards": [1.0, 0.5]
            }
          ],
          "next_turn_feedback": null
        }
        ```
    """

    reward: float
    reward_list: List[float]
    error: Optional[str] = None
    meta: List[MolecularVerifierServerMetadata] = []
    next_turn_feedback: Optional[str] = None
