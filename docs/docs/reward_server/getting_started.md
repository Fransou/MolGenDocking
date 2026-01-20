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

Set required environment variables ([see here for all options](#server-configuration))

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


## Next Steps

- Explore how to configure the server in [Server Configuration](server_configuration.md)
- Understand the query and response formats in [Query and Answer Format](query_answer_format.md)
- Learn about the full API in [API Documentation](api.md)
