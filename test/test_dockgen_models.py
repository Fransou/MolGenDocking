import json

import pytest
import torch
import vllm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mm_messages():
    with open("test/data/test_mm_messages.json") as f:
        MESSAGES = json.load(f)
    mm_messages = []
    for i, message in enumerate(MESSAGES):
        new_message = []
        for m in message:
            new_message.append(m)
            if m["role"] == "user":
                new_message[-1]["content"][0]["text"] = (
                    new_message[-1]["content"][0]["text"][:-1] + " <|image_pad|>."
                )
                mm_info = [
                    {"type": "image", "path": "test/data/test_embed_0.pt"},
                ]
                new_message[-1]["content"] = new_message[-1]["content"] + mm_info
                break
            mm_messages.append(new_message)
    return mm_messages


READY_FOR_TEST = DEVICE == "cuda"

HF_MODEL_NAME = "Franso/DockGen-Qwen3-0.6B"
mm_messages = get_mm_messages()


@pytest.fixture(scope="module")
def model_hf():
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(HF_MODEL_NAME, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    return model, processor, tokenizer


@pytest.fixture(scope="module")
def vllm_model(model_name):
    return vllm.LLM(
        model_name, task="generate", trust_remote_code=True, gpu_memory_utilization=0.8
    )


def test_generation(hf_model, vllm_model, tokenizer_processor):
    tokenizer, processor = tokenizer_processor

    chat_templated = processor.apply_chat_template(
        mm_messages, tokenize=True, return_dict=True
    )

    gen = hf_model.generate(chat_templated)
    print(gen)
