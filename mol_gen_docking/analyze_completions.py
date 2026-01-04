import argparse
from pathlib import Path

import jsonlines
import numpy as np
from transformers import AutoTokenizer


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score molecular completions.")
    parser.add_argument(
        "--input_files",
        type=str,
        required=True,
        help="Path to the input file containing molecular completions (can be a directory).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model used for generation.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    completions = []
    if Path(args.input_files).is_file():
        input_files = [args.input_files]
    else:
        directory = Path(args.input_files)
        input_files = [str(f) for f in directory.glob("*_scored.jsonl")]

    for input_file in input_files:
        with jsonlines.open(input_file) as reader:
            for item in reader:
                completions.append(item)
    # Compute min, max, mean, median, and quantiles of the length of the outputs
    tokenizer = (
        AutoTokenizer.from_pretrained(args.model_name) if args.model_name else None
    )
    if tokenizer:
        n_tokens_output = [
            len(tokenizer.encode(item["output"])) for item in completions
        ]
    else:
        n_tokens_output = [len(item["output"]) // 4 for item in completions]
    print("Output length stats (in number of tokens):")
    min_toks = min(n_tokens_output)
    max_toks = max(n_tokens_output)
    mean_toks = sum(n_tokens_output) / len(n_tokens_output)
    diff_min = int((mean_toks - min_toks) / (max_toks - min_toks) * 100)
    diff_max = int((max_toks - mean_toks) / (max_toks - min_toks) * 100)

    print(
        f"Distribution: {min_toks}|{'-' * diff_min}|{mean_toks}|{'-' * diff_max}|{max_toks}"
    )
    print(
        "===".join(
            [f"|{int(np.quantile(n_tokens_output, q))}|" for q in np.linspace(0, 1, 11)]
        )
    )

    show = input("Show generations?")
    if show == "y":
        for item in completions:
            if float(item["reward"]) == 0 and item["reward_meta"][
                "smiles_extraction_failure"
            ] in ["multiple", "no_smiles", "no_valid_smiles", "no_answer"]:
                print(f"Output: {item['output']}")
                print(f"Reason: {item['reward_meta']['smiles_extraction_failure']}")
                print("-----")
                print("-----")
                input("Press Enter to continue...")
