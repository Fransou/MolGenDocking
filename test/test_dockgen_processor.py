import json
from typing import Any, Dict, List, cast

import pytest
import torch
from transformers import AutoProcessor, AutoTokenizer

from mol_gen_docking.models.hf_model import DockGenModel, DockGenProcessor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("test/data/test_mm_messages.json") as f:
    MESSAGES = json.load(f)


@pytest.fixture(scope="module", params=["local", "remote"])
def tokenizer_processor(request):
    if request.param == "local":
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        processor = DockGenProcessor(tokenizer=tokenizer)
    else:
        processor = AutoProcessor.from_pretrained(
            "Franso/DockGen-Qwen3-0.6B", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    return tokenizer, processor


def get_outs_tok_processor(
    tokenizer: DockGenModel,
    processor: List[Dict[str, Any]],
    messages: List[Dict[str, Any]] | List[List[Dict[str, Any]]],
) -> Any:
    chat = processor.apply_chat_template(messages, tokenize=True)
    chat_text = processor.apply_chat_template(messages, tokenize=False)
    if isinstance(messages[0], list):
        cast(messages, List[List[Dict[str, Any]]])
        messages_tok = [
            [
                {
                    "role": m["role"],
                    "content": "".join(
                        [c["text"] for c in m["content"] if c["type"] == "text"]
                    ),
                }
                for m in message
            ]
            for message in messages
        ]
    else:
        cast(messages, List[Dict[str, Any]])
        messages_tok = [
            {
                "role": m["role"],
                "content": "".join(
                    [c["text"] for c in m["content"] if c["type"] == "text"]
                ),
            }
            for m in messages
        ]

    chat_tok = tokenizer.apply_chat_template(
        messages_tok, tokenize=True, return_tensors="pt", padding="max_length"
    )
    chat_tok_text = tokenizer.apply_chat_template(
        messages_tok, tokenize=False, return_tensors="pt", padding="max_length"
    )
    return chat, chat_text, chat_tok, chat_tok_text


@pytest.mark.parametrize("message", MESSAGES)
def test_processor_only_text(tokenizer_processor, message: List[Dict[str, Any]]):
    tokenizer, processor = tokenizer_processor

    chat, chat_text, chat_tok, chat_tok_text = get_outs_tok_processor(
        tokenizer, processor, message
    )

    assert chat_text == chat_tok_text, "Tokenized chat text does not match"
    assert (chat == chat_tok).all(), "Tokenized chat does not match"


def test_processor_only_text_batched(tokenizer_processor):
    tokenizer, processor = tokenizer_processor
    messages = MESSAGES
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    processor = DockGenProcessor(tokenizer=tokenizer)

    chat, chat_text, chat_tok, chat_tok_text = get_outs_tok_processor(
        tokenizer, processor, messages
    )

    assert chat_text == chat_tok_text, "Tokenized chat text does not match"
    assert (chat == chat_tok).all(), "Tokenized chat does not match"


@pytest.mark.parametrize("message", MESSAGES)
def test_processor_mm(tokenizer_processor, message: List[Dict[str, Any]]):
    tokenizer, processor = tokenizer_processor
    for m in message:
        if m["role"] == "user":
            mm_info = [
                {"type": "image", "path": "test/data/test_embed_0.pt"},
                {"type": "text", "text": "This is a test object."},
            ]
            m["content"] = m["content"] + mm_info

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    processor = DockGenProcessor(tokenizer=tokenizer)

    chat, chat_text, chat_tok, chat_tok_text = get_outs_tok_processor(
        tokenizer, processor, message
    )

    assert chat_text == chat_tok_text, "Tokenized chat text does not match"
    assert (chat == chat_tok).all(), "Tokenized chat does not match"


def test_processor_mm_batched(tokenizer_processor):
    tokenizer, processor = tokenizer_processor
    messages = MESSAGES
    for i in range(len(messages)):
        for m in messages[i]:
            if m["role"] == "user":
                mm_info = [
                    {"type": "image", "path": "test/data/test_embed_0.pt"},
                    {"type": "text", "text": "This is a test object."},
                ]
                m["content"] = m["content"] + mm_info

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    processor = DockGenProcessor(tokenizer=tokenizer)

    chat, chat_text, chat_tok, chat_tok_text = get_outs_tok_processor(
        tokenizer, processor, messages
    )

    assert chat_text == chat_tok_text, "Tokenized chat text does not match"
    assert (chat == chat_tok).all(), "Tokenized chat does not match"
