"""A script to resize the embedding layer of a model to include special tokens"""

import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import tokenizers

from mol_gen_docking.data import special_tok


def resize_model(model_path: str, output_path: str, add_spe_tokens: bool = False):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if add_spe_tokens:
        special_tokens = [
            tokenizers.AddedToken(t, special=True, lstrip=True, rstrip=True)
            for _, t in special_tok.items()
        ]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        model.resize_token_embeddings(len(tokenizer))
        model.tie_weights()
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resize the model to include special tokens"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="The path to the model to resize",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="The path to save the new model",
        default="qwen_2_0.5B_resized",
    )

    parser.add_argument(
        "--add-spe-tokens",
        action="store_true",
        help="Add special tokens to the model",
    )

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    model, tokenizer = resize_model(
        args.model_path, args.output_path, args.add_spe_tokens
    )

    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
