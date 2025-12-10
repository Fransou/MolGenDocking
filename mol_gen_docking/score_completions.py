import argparse
from mol_gen_docking.reward.rl_rewards import (
    RewardScorer,
)
from mol_gen_docking.server_utils.utils import (
    MolecularVerifierSettings,
)
from tqdm import tqdm
import jsonlines
from mol_gen_docking.data.meeko_process import ReceptorProcess

verifier_settings = MolecularVerifierSettings()
reward_scorer = RewardScorer(
    path_to_mappings=verifier_settings.data_path,
    parse_whole_completion=False,
    docking_concurrency_per_gpu=verifier_settings.docking_concurrency_per_gpu,
    reaction_matrix_path=verifier_settings.reaction_matrix_path,
    oracle_kwargs=dict(
        exhaustiveness=verifier_settings.scorer_exhaustiveness,
        n_cpu=verifier_settings.scorer_ncpus,
        docking_oracle=verifier_settings.docking_oracle,
        vina_mode=verifier_settings.vina_mode,
    ),
)
receptor_process = ReceptorProcess(
    data_path=verifier_settings.data_path, pre_process_receptors=True
)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score molecular completions."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input file containing molecular completions.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()
    completions = {}
    with jsonlines.open(args.input_file) as reader:
        for item in reader:
            props = item["metadata"]["properties"]
            found=False
            for p in props:
                if p in receptor_process.pockets:
                    if p not in completions:
                        completions[p] = []
                    completions[p].append(item)
                    found = True
                    break
            if not found:
                if "no_docking" not in completions:
                    completions["no_docking"] = []
                completions["no_docking"].append(item)
    completions = [item for sublist in completions.values() for item in sublist]

    all_responses = []
    for idx in tqdm(range(0,len(completions), args.batch_size), desc="Scoring completions"):
        batch = completions[idx: min(idx + args.batch_size, len(completions))]
        # 1 pre-process
        all_targets = []
        for item in batch:
            all_targets.extend(item["metadata"]["properties"])
        all_targets = list(set(all_targets))
        all_targets = [r for r in all_targets if r in receptor_process.pockets]
        receptor_process.process_receptors(all_targets, allow_bad_res=True, use_pbar=True)
        # 2 get reward
        response,_ = reward_scorer.get_score(
            completions=[item["output"] for item in batch],
            metadata=[item.get("metadata", {}) for item in batch],
        )
        all_responses.extend(response)
    results = []
    for item, response in zip(completions, all_responses):
        results.append({
            "output": item["output"],
            "metadata": item.get("metadata", {}),
            "reward": response,
        })
    with jsonlines.open(args.input_file.replace(".jsonl", "_scored.jsonl"), mode="w") as writer:
        writer.write_all(results)

