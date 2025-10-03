import argparse
from typing import Generator, List

from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)


def get_training_corpus(dataset: Dataset) -> Generator[List[str], None, None]:
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["smiles"]


def create_reinvent_model(
    checkpoint: str = "BEE-spoke-data/smol_llama-81M-tied",
    N_voc: int = 60,
    num_hidden_layers: int = 6,
    num_attention_heads: int = 24,
    head_dim: int = 32,
) -> None:
    old_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)

    ds = load_dataset("jarod0411/zinc10M")["train"]
    tokenizer = old_tokenizer.train_new_from_iterator(get_training_corpus(ds), N_voc)

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer.backend_tokenizer,
        model_input_names=["input_ids", "attention_mask"],
    )
    fast_tok.add_special_tokens({"pad_token": "<pad>"})
    fast_tok.add_special_tokens({"bos_token": "<s>"})
    fast_tok.add_special_tokens({"eos_token": "</s>"})
    fast_tok.add_special_tokens({"unk_token": "<unk>"})

    print("#-#" * 20)
    print(f"Tokenizer vocab :\n {fast_tok.vocab}")
    print(fast_tok("CCC"))
    print("#-#" * 20)

    config = model.config.__dict__
    config["vocab_size"] = len(fast_tok)
    config["num_hidden_layers"] = num_hidden_layers  # 6
    config["num_attention_heads"] = num_attention_heads  # 24
    config["head_dim"] = head_dim  # 32

    config["num_key_value_heads"] = config["num_attention_heads"]
    config["hidden_size"] = config["num_attention_heads"] * config["head_dim"]
    config["intermediate_size"] = int(config["hidden_size"] * 2.7)

    cfg = LlamaConfig(**config)
    model = LlamaForCausalLM(cfg)
    n_params = model.num_parameters(only_trainable=True)

    if n_params // 1e6 > 0:
        n_params = f"{int(n_params // 1e6)}M"
    else:
        n_params = f"{int(n_params // 1e3)}K"

    NAME = f"reinvent_{n_params}"
    print("#-#" * 20)
    print(f"Model name : {NAME}")
    print("#-#" * 20)

    assert model.forward(**tokenizer("CCC", return_tensors="pt")) is not None

    fast_tok.save_pretrained(f"./{NAME}")
    fast_tok.push_to_hub(f"Franso/{NAME}")
    model.push_to_hub(f"Franso/{NAME}")

    print(AutoTokenizer.from_pretrained(f"Franso/{NAME}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a REINVENT model with custom architecture"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="BEE-spoke-data/smol_llama-81M-tied",
        help="Base model checkpoint to start from",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=60, help="Vocabulary size for the tokenizer"
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=6,
        help="Number of hidden layers in the model",
    )
    parser.add_argument(
        "--num_attention_heads", type=int, default=24, help="Number of attention heads"
    )
    parser.add_argument(
        "--head_dim", type=int, default=32, help="Dimension of each attention head"
    )

    args = parser.parse_args()

    create_reinvent_model(
        checkpoint=args.checkpoint,
        N_voc=args.vocab_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        head_dim=args.head_dim,
    )
