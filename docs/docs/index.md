# MolGenDocking: Molecular Generation and Docking Benchmarks

Welcome to MolGenDocking, a comprehensive framework for molecular generation tasks with integrated protein-ligand docking evaluation. This project provides datasets, benchmarks, and a reward server for training and evaluating models that generate drug-like molecules optimized for specific biological targets.

## Overview

MolGenDocking addresses the challenge of *de novo* molecular generation, with a benchmark designed for Large Language Models (LLMs) and other generative architectures.
Our dataset currently supports 3 downstream tasks:

| Dataset                           | Size             | Source                                                                                                       | Purpose                                                            |
|-----------------------------------|------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| ***De Novo* Molecular Generation**          | ~4,500 complexes | [SAIR](https://huggingface.co/datasets/SandboxAQ/SAIR) and [SIU](https://huggingface.co/datasets/bgao95/SIU) | Generate molecules optimizing a set of up to three properties      |
| **Molecular Property prediction** | ~100,000 tasks   | [Polaris](https://polarishub.io/)                                                                            | Predict the properties of a molecule (regression + classification) |
| **RetroSynthesis Tasks**                     | ~200,000 reactions | [Enamine](https://enamine.net/building-blocks/building-blocks-catalog)         | Retro-synthesis planning, reactants, products, SMARTS prediction   |


⚙️ **Reward Server API**
We use AutoDock-GPU for fast GPU-accelerated docking calculations, and we also support Vina for CPU-based docking.
The Molecular Verifier server is built using FastAPI, ad supports concurrent requests, ensuring efficient handling of multiple docking evaluations.

## Quick Start

### Installation

```bash
git clone https://github.com/Fransou/MolGenDocking.git
cd MolGenDocking
pip install -e .
```

### Running the Reward Server

```bash
export DOCKING_ORACLE=autodock_gpu
... # Set other environment variables as needed
export DATA_PATH=... # Path to your data directory
uvicorn --host 0.0.0.0 --port 8000 mol_gen_docking.server:app
```

### Using the API

```python
import requests

response = requests.post(
    "http://localhost:8000/get_reward",
    json={
        "query": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "prompt": "Generate a drug-like molecule...",
        "metadata": [
            {
                "properties": ["QED", "protein_1"],
                "objectives": ["above", "minimize"],
                "target": [0.7, 0.0]
            }
        ]
    }
)
```

## Citation

If you use MolGenDocking in your research, please cite:

```bibtex
@software{molgendocking2025,
  title={MolGenDocking: Molecular Generation and Docking Benchmarks},
  author={Your Name},
  year={2025},
  url={https://github.com/Fransou/MolGenDocking}
}
```

## License

[Your License Here]

## Support

For issues, questions, or contributions, please visit our [GitHub repository](https://github.com/Fransou/MolGenDocking).
