import argparse
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
from mol_gen_docking.data.reactions.projection_dataset import TextualProjectionDataset
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


def get_proj_dataset(args: argparse.Namespace) -> TextualProjectionDataset:
    rxn_matrix: ReactantReactionMatrix = pickle.load(
        open(os.path.join(args.data_path, "rxn_matrix.pkl"), "rb")
    )
    return TextualProjectionDataset(
        rxn_matrix,
        max_num_atoms=60,
        max_smiles_len=192,
        max_num_reactions=5,
        init_stack_weighted_ratio=0.0,
        virtual_length=args.n_prompts,
        n_retry=args.n_reaction_retry,
        n_attempts_per_reaction=args.n_bb_retry,
    )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_path", type=str, required=True)
    parser.add_argument("-o", "--out_path", type=str, default="")
    parser.add_argument("-n", "--n_prompts", type=int, default=50000)
    parser.add_argument(
        "--proba-obj",
        nargs="+",
        type=float,
        default=[0.05, 0.05, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15],
    )

    ### Prior information args
    parser.add_argument(
        "--n_bb_max",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--n_smarts_max",
        type=int,
        default=10,
    )

    ### Generation sampling args
    parser.add_argument(
        "--n_reaction_retry",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--n_bb_retry",
        type=int,
        default=5,
    )

    args = parser.parse_args()
    if args.out_path == "":
        args.out_path = os.path.join(
            args.data_path,
            "synthesis",
            f"train_{args.n_reaction_retry}_{args.n_bb_retry}.jsonl",
        )
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    assert len(args.proba_obj) == len(PROMPT_TEMPLATES)
    args.proba_obj = [p / sum(args.proba_obj) for p in args.proba_obj]
    return args


def update_data_dict(
    data_dict: Dict[str, Any],
    args: argparse.Namespace,
    i: int,
    reactants: List[List[str]],
    products: List[str],
    or_smarts: List[str],
    prop: str,
    idx_chosen: int,
    label: List[str],
    building_blocks: List[str],
    original_building_blocks: List[str],
    smarts: List[str],
) -> None:
    reaction_str_list = [
        " + ".join(r) + " -> " + p for r, p in zip(reactants, products)
    ]
    reaction_str = "\n".join(reaction_str_list)
    for lab in label:
        reaction_str = reaction_str.replace(f"{lab}", "?")

    # 4 - Create_prompt
    prompt_text = np.random.choice(PROMPT_TEMPLATES[prop]).format(
        reaction=reaction_str,
        smarts="\n".join(smarts),
        product=products[-1],
        building_blocks=building_blocks,
        n_reaction=len(reactants),
    )
    if len(reactants) == 1:
        prompt_text = prompt_text.replace("multi-step synthesis", "synthesis step")

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
    data_dict["target"].append(label)
    data_dict["prompt_id"].append(f"synth:prop::{i}")
    data_dict["full_reaction"].append("\n".join(reaction_str_list))
    data_dict["smarts"].append(smarts)
    data_dict["or_smarts"].append(or_smarts)
    data_dict["reactants"].append(reactants)
    data_dict["products"].append(products)
    data_dict["building_blocks"].append(building_blocks)
    data_dict["original_building_blocks"].append(
        original_building_blocks
    )  # Not used here
    data_dict["idx_chosen"].append(idx_chosen)


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
    proj_dataset = get_proj_dataset(args)

    task_sampler = ReactionTaskSampler(args, proj_dataset._reaction_matrix)
    i = 0

    for reactants, products, or_smarts, pass_filters in tqdm(
        proj_dataset, "Post-processing prompts", total=len(proj_dataset)
    ):
        if len(reactants) == 0 or len(products) == 0 or len(or_smarts) == 0:
            continue

        prop, idx_chosen, label, bb, smarts, or_bb = task_sampler.sample(
            reactants=reactants, products=products, or_smarts=or_smarts
        )

        update_data_dict(
            data_dict,
            args,
            i,
            reactants,
            products,
            or_smarts=or_smarts,
            prop=prop,
            idx_chosen=idx_chosen,
            label=label,
            building_blocks=bb,
            original_building_blocks=or_bb,
            smarts=smarts,
        )
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
