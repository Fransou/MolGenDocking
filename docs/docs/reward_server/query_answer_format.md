# Format of the Queries and Answers for the Molecular Verifier Server

All queries to the Molecular Verifier server and its responses follow specific formats defined using pydantic models. Below we outline the expected structure for both requests and responses.

## Request Format

We provide in [server_utils.py](https://github.com/Fransou/MolGenDocking/blob/main/mol_gen_docking/server_utils/utils.py) a pydantic model for request verifier queries and response:
```python
class MolecularVerifierQuery(BaseModel):
    """
    Query model for the MolecularVerifier.
    Args:
        metadata (list[dict[str, Any]]): List of metadata dictionaries for prompt.
        query (list[str]): List of generated completions.
        prompts (Optional[list[str]]): Optional list of prompts for each molecule.
    """

    metadata: list[dict[str, Any]]
    query: list[str]
    prompts: Optional[list[str]] = None
```
Example:
```json
{
  "query": "Here is a molecules: <answer>CC(C)Cc1ccc(cc1)C(C)C(=O)O</answer>",
  "prompt": "Generate a molecule that binds to GSK3B with high affinity and is drug-like",
  "metadata": [
    {
      "properties": ["CalcNumRotatableBonds", "sample_228234_model_0"],
      "objectives": ["above", "minimize"],
      "target": [3.0, 0.0]
    }
  ]
}
```

### Field Descriptions

- **query** (str): The completion containing generated molecule(s)
  - Should include `<answer>SMILES</answer>` tags
  - Parser extracts content between these tags if the rewarder is not set to parse the whole completion.

- **prompt** (str): The original prompt used to generate the completion (optional).

- **metadata** (list): List of evaluation contexts
  - Each dictionary can contain:
    - **properties** (list): List of property names to compute
    - **objectives** (list): List of objectives for each property (e.g., "above", "below", "minimize", "maximize")
    - **target** (list): Target values for each property (above, below).
    - other optional information as needed.

## Response Format
Similarly, we provide a pydantic model for the response:
```python

class MolecularVerifierResponse(BaseModel):
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
    reward_list: list[float]
    error: str | None = None
    meta: dict[str, list[Any]] | None = None
    next_turn_feedback: str | None = None
```

Example Response:
```json
{
  "reward": 0.0,
  "reward_list": [1.0, 0.0],
  "error": null,
  "meta": {
    ...
  },
  "next_turn_feedback": 0.0
}
```

### Field Descriptions

- **reward** (float): Overall reward score combining all properties
- **reward_list** (list): List of individual reward scores for each property
- **error** (str, optional): Error message if any issues occurred
- **meta** (dict, optional): Additional metadata about the scoring
- **next_turn_feedback** (str, optional): Feedback for the next turn in a conversation (for multi-turn scenarios)
