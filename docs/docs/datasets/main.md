# Data Access

## Overview

MolGenDocking addresses the challenge of *de novo* molecular generation, with a benchmark designed for Large Language Models (LLMs) and other generative architectures.
Our dataset currently supports 3 downstream tasks:

| Dataset                            | Size           | Source                                                                                                       | Purpose                                                            |
|------------------------------------|----------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| ***De Novo* Molecular Generation** | ~50k prompts   | [SAIR](https://huggingface.co/datasets/SandboxAQ/SAIR) and [SIU](https://huggingface.co/datasets/bgao95/SIU) | Generate molecules optimizing a set of up to three properties      |
| **Molecular Property prediction**  | ~50k prompts   | [Polaris](https://polarishub.io/)                                                                            | Predict the properties of a molecule (regression + classification) |
| **RetroSynthesis Tasks**           | ~50k reactions | [Enamine](https://enamine.net/building-blocks/building-blocks-catalog)                                       | Retro-synthesis planning, reactants, products, SMARTS prediction   |


## Downloading Datasets
Our dataset is available on [huggingface.](https://huggingface.co/datasets/Franso/MolGenData)
The three tasks are separated into three compressed folders: `molgendata.tar.gz`, `polaris.tar.gz`, and `synthesis.tar.gz`.

```bash
# Extract tarball datasets
tar -xzf molgendata.tar.gz
tar -xzf analog_gen.tar.gz
tar -xzf synthesis.tar.gz
```
!!! warning
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

```python
class Message(BaseModel):
    """Represents a single message in a conversation"""

    role: Literal["system", "user", "assistant"]
    content: str
    meta: Dict[str, Any] = Field(default_factory=dict)
    identifier: Optional[str] = None
    multimodal_document: Optional[Dict[str, Any]] = None


class Conversation(BaseModel):
    """Complete conversation structure"""

    meta: Dict[
        str, Any
    ]
    messages: List[Message]
    system_prompt: Optional[str] = None
    available_tools: Optional[List[str]] = None
    truncate_at_max_tokens: Optional[int] = None
    truncate_at_max_image_tokens: Optional[int] = None
    output_modalities: Optional[List[str]] = None
    identifier: str
    references: List[Any] = Field(default_factory=list)
    rating: Optional[float] = None
    source: Optional[str] = None
    training_masks_strategy: str
    custom_training_masks: Optional[Dict[str, Any]] = None


class Sample(BaseModel):
    """Root model containing all conversations"""

    identifier: str
    conversations: List[Conversation]
    trajectories: List[Any] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = None
```

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
