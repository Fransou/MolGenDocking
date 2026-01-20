# Getting Started with the Reward Server

The Reward Server is a FastAPI application that evaluates molecular structures based on various scoring functions including docking, drug-likeness, and bioactivity predictions.

## Quick Start

### Installation

For GPU-accelerated docking, ensure AutoDock-GPU is installed:

```bash
# Install AutoDock-GPU (see https://github.com/ccsb-scripps/AutoDock-GPU)
# The installation process depends on your system
```

### Starting the Server

Set required environment variables:

```bash
export DOCKING_ORACLE=autodock_gpu
export DATA_PATH=data
export MAX_CONCURRENT_REQUESTS=128
```

Start the server:

```bash
uvicorn --host 0.0.0.0 --port 8000 mol_gen_docking.server:app
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

## Basic Usage

### Python Client

```python
import requests

response = requests.post(
    "http://localhost:8000/get_reward",
    json={
        "query": "<answer>CC(C)Cc1ccc(cc1)C(C)C(=O)O</answer>",
        "prompt": "[Textual prompt used to generate the molecule]",
        "metadata": [
             {
                 "properties": ["sample_654138_model_0", "CalcExactMolWt"],
                 "objectives": ["below", "below"],
                 "target": [-10.86, 197.27]
             }
        ]
    }
)
```

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

- **prompt** (str): The original prompt used to generate the completion

- **metadata** (list): List of evaluation contexts
  - Each dictionary can contain:
    - **properties** (list): List of property names to compute
    - **objectives** (list): List of objectives for each property (e.g., "above", "below", "minimize", "maximize")
    - **target** (list): Target values for each property (above, below).

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

## Server Configuration

The server can be configured using environment variables, which are then stored in a pydantic settings model ([source](https://github.com/Fransou/MolGenDocking/blob/main/mol_gen_docking/server_utils/utils.py)):

```python

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
```


## Next Steps

- [Configuration Guide](configuration.md) - Advanced settings
- [API Reference](api.md) - Complete API documentation
- [Datasets](../datasets/overview.md) - Available docking targets
