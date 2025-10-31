[![Tests-PIP](https://github.com/Fransou/MolGenDocking/actions/workflows/test_pip_installation.yml/badge.svg)](https://github.com/Fransou/MolGenDocking/actions/workflows/test_pip_installation.yml)



#  Reward Server for molecular generation

This repository contains a reward server for molecular generation tasks. It provides an API to evaluate molecular structures based on various scoring functions, including docking scores, drug-likeness, and synthetic accessibility.

# :nut_and_bolt: Installation

To install the required packages to run the molecular reward server, clone the repository and install the package using pip:

```bash
git clone https://github.com/Fransou/MolGenDocking.git
cd MolGenDocking
pip install -e .
```

This project requires Python >= 3.10. For molecular docking objectives, we rely mainly on AutoDock-GPU. Ensure AutoDock-GPU is installed and available in your PATH (see [AutoDock-GPU documentation](https://github.com/ccsb-scripps/AutoDock-GPU) for installation instructions). Additionally, Vina support (see [VINA documentation](https://github.com/ccsb-scripps/AutoDock-Vina) for installation details) is included through the pyscreener library, which is automatically installed as a dependency.

We provide docke images to run the server with all dependencies installed:
```bash
docker pull fransou/molgendata:latest
```

# :bookmark_tabs: Deploying the reward server

The server configuration is managed through environment variables using Pydantic Settings. Set the following environment variables before starting the server:

```bash
export DOCKING_ORACLE=autodock_gpu
export SCORER_EXHAUSTIVENESS=8
export SCORER_NCPUS=8
export GPU_UTILIZATION_GPU_DOCKING=0.05
export MAX_CONCURRENT_REQUESTS=128
export VINA_MODE=autodock_gpu_256wi
export DATA_PATH=data
```

Then start the server with uvicorn:

```bash
uvicorn --host 0.0.0.0 --port 8000 mol_gen_docking.server:app
```

## Configuration Parameters

- `DOCKING_ORACLE`: Docking method to use (`pyscreener` or `autodock_gpu`, default: `pyscreener`)
- `SCORER_EXHAUSTIVENESS`: Number of runs for the docking scoring function (default: 8). Higher values yield more accurate results but take longer to compute.
- `SCORER_NCPUS`: Number of CPUs to use for each docking simulation (default: 8). Should be equal to exhaustiveness for pyscreener.
- `GPU_UTILIZATION_GPU_DOCKING`: GPU utilization per docking run (default: 0.05), controls the maximum amount of parrallel docking runs that can be performed (only based on memmory).
- `MAX_CONCURRENT_REQUESTS`: Maximum concurrent requests (default: 128)
- `VINA_MODE`: Command used to run AutoDock GPU (default: `autodock_gpu_256wi`)
- `DATA_PATH`: Path to folder containing necessary data files (e.g., docking receptor files, default: `data`)

---
The server can then be requested, and will treat requests at the "get_reward" endpoint:
- query: A completion (str) that will be parsed. In the completion, the scorer will parse all the content between the <answer> and </answer> tags. The parsed content will be interpreted as a SMILES string if possible and evaluated by the scoring function.
- prompt: The prompt (str) that was used to generate the completion.
- metadata: List of dictionaries providing additional context for each query. Each dictionary should include keys such as 'properties' (list of property names), 'objectives' (list of objectives), and 'target' (target protein or property).


# :microscope: Molecular Generation


This repository provides tools and baselines for molecular generation tasks in the training process, where the objective is to design novel molecules that optimize specific properties or objectives.

## Data creation

The data creation process involves downloading and processing datasets for molecular generation tasks. This includes the SAIR and SIU datasets, where structures were extracted from protein-ligand complexes.

### Docking Properties

#### SAIR Dataset
The SAIR (SandboxAQ/SAIR) dataset is sourced from Hugging Face and contains protein-ligand interaction data.

To download and process the SAIR dataset:
1. Download the parquet file using the [SAIR_download](mol_gen_docking/data/scripts/SAIR_download.py) script.
2. Extract structure files and identify pockets using [SAIR_identify_pockets](mol_gen_docking/data/scripts/SAIR_identify_pockets.py).

#### SIU Dataset
The SIU dataset provides protein pocket information extracted from PDB structures. The processing involves:
- Loading pickled data with pocket coordinates and labels from the huggingface repo, and extracting the corresponding PDB files.
- Computing pocket metadata (size, center, average pIC50/pKd) with [SIU_download](mol_gen_docking/data/scripts/SIU_download.py)

Both datasets are used to create docking targets for molecular generation tasks, providing diverse protein structures for evaluating generated molecules.

For both datasets, we use [write_post_processed_data](mol_gen_docking/data/scripts/write_post_processed_data.py) to find textual descriptions of the proteins.

### Other properties

### Supported Properties

The system includes classical molecular properties computed using RDKit and other libraries:

- **Bioactivity Properties**: GSK3B (Glycogen Synthase Kinase-3 Beta), JNK3 (c-Jun N-terminal Kinase-3), DRD2 (Dopamine Receptor D2)
- **Drug-likeness Metrics**: SA (Synthetic Accessibility), QED (Quantitative Estimate of Drug-likeness)
- **Physicochemical Properties**:
  - Molecular Weight (CalcExactMolWt)
  - Aromatic Rings (CalcNumAromaticRings)
  - H-bond Acceptors/Donors (CalcNumHBA/CalcNumHBD)
  - Rotatable Bonds (CalcNumRotatableBonds)
  - Fraction of sp3 Carbon Atoms (CalcFractionCSP3)
  - Topological Polar Surface Area (CalcTPSA)
  - Kier-Hall Molecular Shape Indices (CalcHallKierAlpha, CalcPhi)
  - LogP (partition coefficient)

### Prompt generation

Finally, we use [generate_gen_dataset](mol_gen_docking/data/scripts/generate_gen_dataset.py) to generate a dataset of properties to optimize :

```bash
python mol_gen_docking/data/scripts/generate_gen_dataset.py --n-prompts 512 --data-path data/mol_orz
```

#### Arguments

- `--n-prompts`: The number of prompts to generate (default: 512).
- `--max-dock-prop-per-prompt`: The maximum number of docking properties per prompt (default: 2).
- `--n-props-probs`: Probabilities for the number of properties per prompt (default: [0.5, 0.3, 0.2], corresponding to 1, 2, and 3 properties respectively).
- `--data-path`: Path to the dataset directory (default: "data/mol_orz").
- `--split-docking`: Split ratios for docking targets across train, validation, and test sets (default: [0.8, 0.1, 0.1]).

#### RuleSet for Prompt Generation

The `RuleSet` class enforces diversity in the generated prompts by tracking and limiting the occurrences of properties across different numbers of properties per prompt. It includes the following attributes:

- `prompt_ids`: A dictionary mapping the number of properties (`n_props`) to a list of generated prompt IDs to avoid duplicates.
- `n_occ_prop`: A dictionary mapping `n_props` to another dictionary of property names and their occurrence counts.
- `probs_docking_targets`: Probability of selecting a docking property (default: 0.5).
- `max_occ`: Maximum number of occurrences allowed for a property per `n_props` (default: 10 for `n_props > 0`, half for `n_props = 0`).
- `max_docking_per_prompt`: Maximum number of docking properties allowed per prompt (default: 2).
- `prohibited_props_at_n`: A dictionary mapping `n_props` to lists of properties that have reached the maximum occurrences and are thus prohibited.

The `verify_and_update` method checks if a new prompt's metadata complies with the rules (e.g., not exceeding occurrence limits) and updates the occurrence counts if valid. The `partial_reset` method reinitializes occurrence tracking while preserving prompt IDs for continuity.

### Metadatas

The metadata generated for each prompt includes the following fields:

- `properties`: A list of property names (e.g., docking targets or classical properties like "logP", "TPSA") that the generated molecule should optimize.
- `objectives`: A list of optimization objectives corresponding to each property, such as "maximize", "minimize", "above", "below", or "equal".
- `target`: A list of target values (floats) for objectives that require a specific threshold (e.g., for "above 5.0" or "below -8.0"). For objectives without targets, the value is 0.0.
- `prompt_id`: A unique string identifier for the prompt, generated based on the properties, objectives, and target values to ensure diversity and avoid duplicates.
- `n_props`: The number of properties being optimized in the prompt (integer).
- `docking_metadata`: A list of dictionaries containing additional information for docking targets, including:
  - `pdb_id`: The PDB identifier of the protein target.
  - Other pocket-related metadata such as size, center coordinates, and average binding affinities (pIC50/pKd) if available.

These metadata fields are used to track the generation rules, ensure diversity in prompts, and provide context for evaluating the generated molecules against the specified objectives.

# :mag: Property Prediction

Property prediction tasks involve predicting molecular properties such as bioactivity, drug-likeness, and physicochemical characteristics. The repository supports both regression and classification objectives for various properties.

### Objectives

Property prediction supports two main objective types:
- **Regression**: Predict continuous numerical values (e.g., molecular weight, logP)
- **Classification**: Predict binary outcomes (e.g., active/inactive against a target)

### Datasets

Property prediction tasks utilize datasets from Polaris, including:
- **MolecularML**: ChEMBL datasets for various targets (e.g., moleculeace-chembl204-ki, moleculeace-chembl234-ki)
- **TDCommons**: Toxicology and drug discovery datasets (e.g., ames, bbb-martins, caco2-wang, herg)
- **ADME**: Absorption, Distribution, Metabolism, Excretion properties (e.g., fang-hclint, fang-perm, fang-rppb)

The reward function evaluates predictions by comparing them to ground truth values, providing scores based on accuracy for classification or mean squared error for regression tasks.

## :pill: Chemical Reactions

Chemical reaction tasks focus on predicting reaction outcomes, including forward and retrosynthetic predictions. The system supports various reaction prediction objectives using the USPTO dataset.

### Task Types

The repository implements several reaction prediction tasks:

- **Product Prediction**: Given reactants and conditions, predict the reaction products
- **Reactant Prediction**: Given products, predict the missing reactants
- **Full Reaction Prediction**: Predict complete reaction schemes including all reactants and products
- **Retrosynthetic Analysis**: Work backwards from products to identify possible synthetic routes

### Objectives

Reaction tasks use specific objective types:
- **product**: Predict reaction products from given reactants
- **reactant**: Identify missing reactants in a reaction
- **reactant_full**: Predict all reactants needed for a given set of products
- **product_full**: Predict all products from a given set of reactants

### Dataset

Reaction tasks utilize the USPTO (United States Patent and Trademark Office) dataset, processed from the Hugging Face chenxran/uspto_full dataset. The dataset contains chemical reactions with SMILES representations of reactants and products.

### Evaluation

The reaction reward function evaluates predictions by comparing canonical SMILES strings of predicted and actual molecules. It provides partial credit for correct predictions and handles multiple molecule comparisons within reactions. The scoring accounts for molecular equivalence regardless of SMILES representation order.
