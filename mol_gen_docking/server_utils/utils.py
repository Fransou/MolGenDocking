from typing import Any, List, Optional

from pydantic import BaseModel


class MolecularVerifierServerQuery(BaseModel):
    """Input query model for the Molecular Verifier server.

    Represents a complete request to the molecular verifier service,
    containing metadata for scoring and completions to evaluate.

    Attributes:
        metadata: List of metadata dictionaries, one per query item.
            Each dictionary will be converted to a MolecularVerifierMetadata
            object for scoring.

        query: List of completion strings from the language model.
            Each completion should contain the answer wrapped in tags:
            `<answer>SMILES</answer>` or `<|answer_start|>...</|answer_end|>`

            The parser extracts content between these tags unless
            parse_whole_completion is enabled.

        prompts: Optional. Original prompts used to generate the completions.
            Useful for tracking and debugging. If provided, should have
            same length as query list.


    Example:
    ```json
    {
      "query": "Here is a molecules: <answer>CC(C)Cc1ccc(cc1)C(C)C(=O)O</answer>",
      "prompt": "Generate a molecule that binds to my target protein with high affinity and has more than 3 rotatable bonds.",
      "metadata": [
        {
          "properties": ["CalcNumRotatableBonds", "sample_228234_model_0"],
          "objectives": ["above", "minimize"],
          "target": [3.0, 0.0]
        }
      ]
    }
    ```
    """

    metadata: List[dict[str, Any]]
    query: List[str]
    prompts: Optional[List[str]] = None


class MolecularVerifierServerMetadata(BaseModel):
    """Metadata returned with each scored molecule.

    Contains detailed information about the scoring result, including
    extracted SMILES, rewards for individual properties, and task-specific
    verification results.

    Attributes:
        smiles_extraction_failure: Error message if SMILES extraction failed.
            None if extraction was successful.

        all_smi: List of all valid SMILES strings extracted from the completion.

        all_smi_rewards: List of reward values corresponding to each SMILES.

        individual_rewards: List of individual reward values for each property
            evaluated on the first SMILES.

        properties: List of property names that were evaluated.

        extracted_answer: Extracted answer text from property prediction tasks.

        valid: Validity score for reaction predictions.
            Range: [0.0, 1.0]

        correct_product: Whether the product matches expected output
            in reaction tasks. For synthesis tasks with tanimoto similarity,
            this is the similarity score to the target molecule.

        correct_reactant: Whether all reactants are from the allowed
            building blocks in reaction synthesis tasks.
    """

    # Generation Verifier Fields
    smiles_extraction_failure: Optional[str] = None
    all_smi_rewards: Optional[List[float]] = None
    all_smi: Optional[List[str]] = None
    individual_rewards: Optional[List[float]] = None
    properties: Optional[List[str]] = None

    # Molecular Property Verifier Fields
    extracted_answer: Optional[float | int] = None

    # Reaction Verifier Fields
    valid: Optional[float] = None
    correct_product: Optional[float] = None
    correct_reactant: Optional[bool] = None


class MolecularVerifierServerResponse(BaseModel):
    """Response from the Molecular Verifier server.

    Contains the computed reward scores and detailed metadata for each
    evaluated completion.

    Attributes:
        reward: Overall reward score combining all evaluated properties.
            Typically normalized to [0.0, 1.0] range when rescaling is enabled.

        reward_list: List of individual reward scores, one per evaluated
            completion in the request (for batch processing).

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
