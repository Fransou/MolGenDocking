# Configuration of the Molecular Verifier Server

## Using a Pydantic BaseSettings

We use Pydantic's `BaseSettings` to manage the configuration of our application through environment variables. It automatically reads and validates environment variables, converting them to Python types defined in your settings class.


**How it works:**
When you define a class that inherits from `BaseSettings`, Pydantic:

1. Looks for environment variables matching the field names (case-insensitive)
2. Converts them to the appropriate Python types
3. Validates them against the field constraints
4. Falls back to default values if environment variables are not provided

**Setting configuration:**
Simply export environment variables before starting the server:
```bash
export DOCKING_ORACLE="autodock_gpu"
export DATA_PATH="/path/to/data"
export MAX_CONCURRENT_REQUESTS="256"
uvicorn ...
```

Or set them inline:
```bash
DOCKING_ORACLE=autodock_gpu DATA_PATH=./data uvicorn ...
```

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

### Environment Variables
- **DOCKING_ORACLE**: Docking oracle to use (`autodock_gpu` or `pyscreener`)
- **DATA_PATH**: Path to the data directory for generation tasks (not necessarily where the prompts are stored, but must contain the `names_mapping.json` files and the `pdb_files` directory).
- **MAX_CONCURRENT_REQUESTS**: Maximum number of concurrent requests the server can handle.
- **SCORER_EXHAUSTIVENESS**: Exhaustiveness parameter for the docking scorer (default: 8).
- **SCORER_NCPUS**: Number of CPUs to use for the docking scorer
- **DOCKING_CONCURRENCY_PER_GPU**: Number of concurrent docking runs per GPU (default: 2).
- **BUFFER_TIME**: Buffer time in seconds used to gather concurrent requests before computation (default: 20).
- **PARSE_WHOLE_COMPLETION**: Whether to parse the whole completion output (default: False). Use sparingly as it may lead to invalid SMILES extraction.
- **REACTION_MATRIX_PATH**: Path to the reaction matrix file (default: `data/rxn_matrix.pkl`) for molecular reaction based tasks.
- **VINA_MODE**: Command used to run autodock gpu (default: `autodock_gpu_256wi`).

A pydantic settings model will extract all environement variables with the same name as the fields above.
To configure the server, simply set the corresponding environment variables before starting the server.
