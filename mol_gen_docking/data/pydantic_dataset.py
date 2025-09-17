import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


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
    ]  # This should contain everything necessary to compute the reward, ground truth whatever
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


def write_jsonl(output_file: Path, samples: list[Sample]) -> None:
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(output_file, "w") as fout:
        for sample in samples:
            fout.write(json.dumps(sample.model_dump(), ensure_ascii=False) + "\n")


def read_jsonl(input_file: Path) -> list[Sample]:
    assert input_file.exists(), input_file

    samples: list[Sample] = []
    with open(input_file) as fin:
        for line in fin:
            samples.append(Sample(**json.loads(line)))

    return samples
