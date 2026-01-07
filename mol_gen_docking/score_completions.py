import argparse
from pathlib import Path
from typing import Any

import jsonlines
from tqdm import tqdm

from mol_gen_docking.data.meeko_process import ReceptorProcess
from mol_gen_docking.reward.rl_rewards import (
    RewardScorer,
)
from mol_gen_docking.server_utils.utils import (
    MolecularVerifierSettings,
)

verifier_settings = MolecularVerifierSettings()
reward_scorer = RewardScorer(
    path_to_mappings=verifier_settings.data_path,
    parse_whole_completion=verifier_settings.parse_whole_completion,
    docking_concurrency_per_gpu=verifier_settings.docking_concurrency_per_gpu,
    reaction_matrix_path=verifier_settings.reaction_matrix_path,
    oracle_kwargs=dict(
        exhaustiveness=verifier_settings.scorer_exhaustiveness,
        n_cpu=verifier_settings.scorer_ncpus,
        docking_oracle=verifier_settings.docking_oracle,
        vina_mode=verifier_settings.vina_mode,
    ),
)
receptor_process: None | ReceptorProcess = None


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score molecular completions.")
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
    parser.add_argument(
        "--mol-generation",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mol_generation:
        receptor_process = ReceptorProcess(
            data_path=verifier_settings.data_path, pre_process_receptors=True
        )

    if Path(args.input_file).is_file():
        input_files = [args.input_file]
    else:
        directory = Path(args.input_file)
        input_files = [
            str(f)
            for f in directory.glob("*.jsonl")
            if not str(f).endswith("_scored.jsonl")
            and not Path(str(f).replace(".jsonl", "_scored.jsonl")).exists()
        ]

    for input_file in input_files:
        output_path = input_file.replace(".jsonl", "_scored.jsonl")
        print("=== Computing Properties")
        completions: dict[str, list[dict[str, Any]]] = {}

        # Group completions by target property
        with jsonlines.open(input_file) as reader:
            for item in reader:
                if not args.mol_generation:
                    id = item["metadata"]["prompt_id"]
                    if id not in completions:
                        completions[id] = []
                    completions[id].append(item)
                else:
                    props = item["metadata"]["properties"]
                    found = False
                    assert receptor_process is not None
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
        completions_ordered: list[dict[str, Any]] = [
            item for sublist in completions.values() for item in sublist
        ]

        # Computing rewards
        all_responses: list[float] = []
        all_metas: list[dict[str, Any]] = []
        for idx in tqdm(
            range(0, len(completions_ordered), args.batch_size),
            desc="Scoring completions",
        ):
            batch = completions_ordered[
                idx : min(idx + args.batch_size, len(completions_ordered))
            ]
            # 1 pre-process
            if args.mol_generation:
                all_targets = []
                for item in batch:
                    all_targets.extend(item["metadata"]["properties"])
                all_targets = list(set(all_targets))

                assert receptor_process is not None
                all_targets = [r for r in all_targets if r in receptor_process.pockets]
                print(f"Processing targets:\n{all_targets}")
                receptor_process.process_receptors(
                    all_targets, allow_bad_res=True, use_pbar=False
                )
            # 2 get reward
            response, meta = reward_scorer.get_score(
                completions=[item["output"] for item in batch],
                metadata=[item.get("metadata", {}) for item in batch],
            )
            all_responses.extend(response)
            all_metas.extend(meta)

        # Save results
        results = []
        for item, r, r_metadata in zip(completions_ordered, all_responses, all_metas):
            results.append(
                {
                    "output": item["output"],
                    "metadata": item.get("metadata", {}),
                    "reward": r,
                    "reward_meta": r_metadata,
                }
            )

        print("=== Saving")
        with jsonlines.open(output_path, mode="w") as writer:
            writer.write_all(tqdm(results, desc=f"Saving {output_path} |"))
