# Data Access

## Downloading Datasets
Our dataset is available on [huggingface.](https://huggingface.co/datasets/Franso/MolGenData)
The three tasks are separated into three compressed folders: `molgendata.tar.gz`, `polaris.tar.gz`, and `synthesis.tar.gz`.

```bash
# Extract tarball datasets
tar -xzf molgendata.tar.gz
tar -xzf analog_gen.tar.gz
tar -xzf synthesis.tar.gz
```

To perform *de novo* molecular generation with docking constraints, the path to the `molgendata` folder should be provided to the reward server via the `DATA_PATH` environment variable. (See [Reward Server Configuration](../reward_server/getting_started.md) for more details.)

## Data Organization

```
molgendata/              # Docking targets
├── docking_targets.json
├── names_mapping.json
├── pockets_info.json
├── pdb_files/
├── train_prompts.jsonl
├── eval_data/
│   └── eval_prompts.jsonl
│   └── eval_prompts_ood.jsonl
└── test_data/
    └── test_prompts_ood.jsonl

synthesis/               # Reaction data
├── train_prompts.jsonl
├── chembl_test.jsonl
└── enamine_test.jsonl

polaris/                 # Multi-task benchmarks
├── eval_concatenated.jsonl
└── [dataset]
    ├── train.jsonl
    ├── eval.jsonl
    └── test.jsonl
```


### Data Format

Our data are stored in JSONL format, represented by a pydantic base model ([source](https://github.com/Fransou/MolGenDocking/blob/main/mol_gen_docking/data/pydantic_dataset.py)):

**JSONL Format**

Each line in a JSONL file represents a `Sample` object with the following structure:

```json
{
  "identifier": "sample_001",
  "source": "dataset_name",
  "conversations": [
    {
      "identifier": "conversation_001",
      "meta": {
        "properties": ["QED", ...],
        "objectives": ["above", ...],
        "target": [0.5, ...],
        ... # Additional metadata such as pocket box, target information, etc.
      },
      "messages": [
        {
          "role": "system",
          "content": "You are a molecular generation assistant...",
          "meta": {}
        },
        {
          "role": "user",
          "content": "Generate a molecule that binds to GSK3B with high affinity",
          "meta": {},
          "identifier": "msg_001"
        }
      ]
    }
  ]
}
```
