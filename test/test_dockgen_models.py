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
    hf_model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(HF_MODEL_NAME, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    return hf_model, processor, tokenizer


@pytest.fixture(scope="module")
def vllm_model():
    return vllm.LLM(
        HF_MODEL_NAME,
        task="generate",
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
    )


def test_generation(hf_model, vllm_model):
    hf_model, processor, tokenizer = hf_model
    chat_templated = processor.apply_chat_template(
        mm_messages[:2], tokenize=True, return_dict=True
    )

    hf_generation = hf_model.generate(**chat_templated)
    # Remove prompt text
    hf_generation = [
        g[inp_ids.shape[0] :]
        for g, inp_ids in zip(hf_generation, chat_templated["input_ids"])
    ]
    hf_text = tokenizer.batch_decode(hf_generation, skip_special_tokens=True)

    requests = [
        vllm.inputs.TokensPrompt(
            prompt_token_ids=chat_templated["input_ids"][i],
            multi_modal_data={"image": chat_templated["pixel_values"][i]},
        )
        for i in range(chat_templated["input_ids"].shape[0])
    ]
    vllm_generations = vllm_model.generate(
        prompts=requests, sampling_params=vllm.SamplingParams(temperature=0.0)
    )
    vllm_text = [g.outputs[0].text for g in vllm_generations]

    for hf_t, vllm_t in zip(hf_text, vllm_text):
        assert hf_t == vllm_t
