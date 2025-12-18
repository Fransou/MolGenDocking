import argparse

import numpy as np

from mol_gen_docking.reward.rl_rewards import (
    RewardScorer,
)
from mol_gen_docking.server_utils.utils import (
    MolecularVerifierSettings,
)
from tqdm import tqdm
import jsonlines
from mol_gen_docking.data.meeko_process import ReceptorProcess
from transformers import AutoTokenizer
from pathlib import Path

verifier_settings = MolecularVerifierSettings()
reward_scorer = RewardScorer(
    reward="valid_smiles",
    path_to_mappings=verifier_settings.data_path,
    parse_whole_completion=False,
    reaction_matrix_path=verifier_settings.reaction_matrix_path,
)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score molecular completions."
    )
    parser.add_argument(
        "--input_files",
        type=str,
        required=True,
        help="Path to the input file containing molecular completions (can be a regex in a directory).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model used for generation.",
    )
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()
    completions = []
    if Path(args.input_files).is_file():
        input_files = [args.input_files]
    else:
        directory = Path("/".join(args.input_files.split("/")[:-1]))
        pattern = args.input_files.split("/")[-1]
        input_files = sorted([str(p) for p in directory.glob(pattern)])

    for input_file in input_files:
        with jsonlines.open(input_file) as reader:
            for item in reader:
                completions.append(item)
    all_responses, rew_meta = reward_scorer.get_score(
            completions=[item["output"] for item in completions],
            metadata=[item.get("metadata", {}) for item in completions],
        )
    # Compute min, max, mean, median, and quantiles of the length of the outputs
    tokenizer = AutoTokenizer.from_pretrained(args.model_name) if args.model_name else None
    if tokenizer:
        n_tokens_output = [len(tokenizer.encode(item["output"])) for item in completions]
    else:
        n_tokens_output = [len(item["output"])//4 for item in completions]
    print(f"Output length stats (in number of tokens):")
    min_toks = min(n_tokens_output)
    max_toks = max(n_tokens_output)
    mean_toks = sum(n_tokens_output)/len(n_tokens_output)
    diff_min = int((mean_toks-min_toks) / (max_toks-min_toks) * 100)
    diff_max = int((max_toks-mean_toks) / (max_toks-min_toks) * 100)

    print(f"Distribution: {min_toks}|{'-'*diff_min}|{mean_toks}|{'-'*diff_max}|{max_toks}")
    print("===".join([f"|{int(np.quantile(n_tokens_output, q))}|" for q in np.linspace(0,1,11)]))

    results = []
    print(f"Generated {sum(all_responses)} valid completions for {len(completions)} inputs ({int(100*sum(all_responses) / len(completions))}%).")
    show = input("Show generations?")
    if show == "y":
        for item, response, fail in zip(completions, all_responses, rew_meta):
            if response == 0:
                print(f"Output: {item['output']}")
                print(f"Reason: {fail}")
            print("-----")
            print("-----")
            input("Press Enter to continue...")
