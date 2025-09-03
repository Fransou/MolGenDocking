[![Tests-PIP](https://github.com/Fransou/MolGenDocking/actions/workflows/test_pip_installation.yml/badge.svg)](https://github.com/Fransou/MolGenDocking/actions/workflows/test_pip_installation.yml)
![coverage](./coverage.svg)


# Reward Server for molecular generation

This repository contains a reward server for molecular generation tasks. It provides an API to evaluate molecular structures based on various scoring functions, including docking scores, drug-likeness, and synthetic accessibility.

## Deploying the reward server

The reward server can be deployed usng the "uvicorn" python library:

```bash
python -m mol_gen_docking.fast_api_reward_server --data-path ...
```
--data-path: path to a folder where the necessary data files are stored (e.g. docking receptor files, etc.)

--port: port number for the server (default: 8000)

--host: IP of the server

--scorer-exhaustivness: Number of runs for the docking scoring function (default: 2). Higher values yield more accurate results but take longer to compute.

--scorer-ncpus: Number of CPUs to used for each docking simulation. Default should be equal to the exhaustivness value.