import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm.auto import tqdm

from mol_gen_docking.data.pydantic_dataset import (
    Conversation,
    Message,
    Sample,
    write_jsonl,
)
from mol_gen_docking.data.reactions.reaction_matrix import ReactantReactionMatrix
from mol_gen_docking.data.reactions.utils import PROMPT_TEMPLATES
from mol_gen_docking.dataset.scripts.reaction_task.utils import ReactionTaskSampler

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. "
    "The User asks a question, and the Assistant solves it.\n"
    "The reasoning process is enclosed within <think> </think>"
    " and answer is enclosed within <answer> </answer> tags, "
    "respectively, i.e., <think> reasoning process here </think> "
    "<answer> answer here </answer>."
)


def data_dict_to_pydantic(data_dict: dict, key: str = "prompt") -> List[Sample]:
    sample_list: List[Sample] = []

    for i in range(len(data_dict[key])):
        messages: List[Message] = [Message(**msg) for msg in data_dict[key][i]]
        conv = Conversation(
            messages=messages,
            system_prompt=None,
            available_tools=None,
            truncate_at_max_tokens=None,
            truncate_at_max_image_tokens=None,
            output_modalities=None,
            identifier=data_dict["prompt_id"][i],
            references=[],
            rating=None,
            source=None,
            training_masks_strategy="none",
            custom_training_masks=None,
            meta={
                "properties": data_dict["properties"][i],
                "objectives": data_dict["objectives"][i],
                "target": data_dict["target"][i],
                "prompt_id": data_dict["prompt_id"][i],
                "full_reaction": data_dict["full_reaction"][i],
                "or_smarts": data_dict["or_smarts"][i],
                "smarts": np.unique(data_dict["smarts"][i]).tolist(),
                "reactants": data_dict["reactants"][i],
                "products": data_dict["products"][i],
                "building_blocks": np.unique(data_dict["building_blocks"][i]).tolist(),
                "idx_chosen": data_dict["idx_chosen"][i],
                "n_building_blocks": len(data_dict["building_blocks"][i]),
            },
        )
        sample_list.append(
            Sample(
                identifier=data_dict["prompt_id"][i],
                conversations=[conv],
                trajectories=[],
                meta={},
                source=None,
            )
        )
    return sample_list


def get_matrix(args: argparse.Namespace) -> ReactantReactionMatrix:
    rxn_matrix: ReactantReactionMatrix = pickle.load(
        open(os.path.join(args.data_path, "rxn_matrix.pkl"), "rb")
    )
    return rxn_matrix


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_path", type=str, required=True)
    parser.add_argument("-o", "--out_path", type=str, default="")

    parser.add_argument(
        "--n_bb_max",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--n_smarts_max",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--data_json",
        type=str,
        default="mol_gen_docking/data/utils/enamine_1k.json",
    )

    args = parser.parse_args()
    if args.out_path == "":
        args.out_path = os.path.join(
            args.data_path,
            "synthesis",
            f"train_{args.n_reaction_retry}_{args.n_bb_retry}.jsonl",
        )
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    return args


def update_data_dict(
    data_dict: Dict[str, Any],
    args: argparse.Namespace,
    i: int,
    smi: str,
    prop: str,
    idx_chosen: int,
    label: List[str],
    bb: List[str],
) -> None:
    prompt_text = np.random.choice(PROMPT_TEMPLATES[prop]).format(
        product=smi, building_blocks=bb, n_reaction=5
    )
    data_dict["prompt"].append(
        [
            {
                "role": "user",
                "content": prompt_text,
            },
        ]
    )
    data_dict["properties"].append([prop])
    data_dict["objectives"].append([prop])
    data_dict["target"].append([smi])
    data_dict["prompt_id"].append(
        f"{args.data_json.split('/')[-1].replace('.json', '')}:{i}"
    )
    data_dict["full_reaction"].append("")
    data_dict["smarts"].append([])
    data_dict["or_smarts"].append([])
    data_dict["reactants"].append([])
    data_dict["products"].append([])
    data_dict["building_blocks"].append(bb)
    data_dict["original_building_blocks"].append([])
    data_dict["idx_chosen"].append(0)


def main(args: argparse.Namespace) -> None:
    data_dict: Dict[str, Any] = {
        "prompt": [],
        "properties": [],
        "objectives": [],
        "target": [],
        "prompt_id": [],
        "full_reaction": [],
        "smarts": [],
        "reactants": [],
        "or_smarts": [],
        "products": [],
        "building_blocks": [],
        "original_building_blocks": [],
        "idx_chosen": [],
    }
    rxn_matrix = get_matrix(args)

    task_sampler = ReactionTaskSampler(args, rxn_matrix)
    i = 0
    with open(args.data_json) as f:
        smi_list = json.load(f)

    for smi in tqdm(smi_list, "Processing prompts", total=len(smi_list)):
        prop, idx_chosen, label, bb = task_sampler.sample_eval(smi)
        update_data_dict(data_dict, args, i, smi, prop, idx_chosen, label, bb)
        i += 1

    dataset = data_dict_to_pydantic(data_dict)
    out_path = args.out_path
    write_jsonl(
        Path(os.path.join(out_path)),
        dataset,
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
