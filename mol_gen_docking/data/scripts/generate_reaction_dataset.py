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
                "impossible": data_dict["impossible"][i],
                "smarts": np.unique(data_dict["smarts"][i]).tolist(),
                "reactants": data_dict["reactants"][i],
                "products": data_dict["products"][i],
                "building_blocks": np.unique(data_dict["building_blocks"][i]).tolist(),
                "idx_chosen": data_dict["idx_chosen"][i],
                "n_building_blocks": len(data_dict["building_blocks"][i]),
                "pass_filters": data_dict["pass_filters"][i],
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


def get_bb_blocks(
    original_building_blocks: list[str],
    all_reactants: list[str],
    args: argparse.Namespace,
) -> list[str]:
    n_to_choose = np.random.randint(2 * len(original_building_blocks), args.n_bb_max)
    building_blocks = sample_l(n_to_choose, all_reactants)
    for smi in original_building_blocks:
        building_blocks.append(smi)
    building_blocks = list(set(building_blocks))
    np.random.shuffle(building_blocks)
    return list(building_blocks)


def get_objective_label(
    args: argparse.Namespace,
    reactants: list[list[str]],
    products: list[str],
    or_smarts: list[str],
    pass_filters: list[bool],
) -> tuple[str, int, List[str]]:
    prop = np.random.choice(list(PROMPT_TEMPLATES.keys()), p=args.proba_obj)
    idx_chosen: int = 0
    if "full_path" not in prop and prop != "final_product":
        idx_chosen = int(np.random.choice(list(range(len(reactants)))))
        reactants = [reactants[idx_chosen]]
        products = [products[idx_chosen]]
        or_smarts = [or_smarts[idx_chosen]]
        pass_filters = [pass_filters[idx_chosen]]
    # 2 - Get a label
    label: List[str] = ["n/a"]
    if prop == "final_product" or prop.startswith("full_path"):
        label = [products[-1]]
    elif prop == "reactant":
        label = [np.random.choice(reactants[0])]
    elif prop == "smarts":
        label = [or_smarts[0]]
    elif prop in ["all_reactants", "all_reactants_bb_ref"]:
        label = reactants[0]
    return prop, idx_chosen, label


def sample_l(n: int, list_to_sample: List[str]) -> List[str]:
    idxs = list(range(len(list_to_sample)))
    np.random.shuffle(idxs)
    idxs = idxs[:n]
    return [list_to_sample[i] for i in idxs]


def get_smarts_bb(
    args: argparse.Namespace,
    prop: str,
    reactants: list[list[str]],
    or_smarts: list[str],
    all_reactants: list[str],
    all_reactions: list[str],
) -> tuple[list[str], list[str], list[str]]:
    original_building_blocks = []
    for l_reactants in reactants:
        for smi in l_reactants:
            if smi in all_reactants:
                original_building_blocks.append(smi)
    if prop not in [
        "all_reactants_bb_ref",
        "full_path_bb_ref",
        "full_path_smarts_bb_ref",
    ]:
        building_blocks = []
    else:
        building_blocks = get_bb_blocks(original_building_blocks, all_reactants, args)
    if prop in ["full_path_smarts_ref", "full_path_smarts_bb_ref"]:
        n_smarts_max = np.random.randint(len(or_smarts), args.n_smarts_max)
        smarts = list(
            set(or_smarts + np.random.choice(all_reactions, n_smarts_max).tolist())
        )
        np.random.shuffle(smarts)
    else:
        smarts = or_smarts

    return original_building_blocks, building_blocks, smarts


def make_impossible_decision(
    impossible_proba: float,
    prop: str,
    smarts: list[str],
    reactants: list[list[str]],
    building_blocks: list[str],
    all_reactants: list[str],
    all_reactions: list[str],
    original_building_blocks: list[str],
    or_smarts: list[str],
) -> tuple[bool, list[str], list[list[str]], list[str]]:
    impossible = False
    if np.random.random() <= args.impossible_proba:
        if prop in ["reactant", "final_product", "all_reactants"]:
            for i_smart in range(len(smarts)):
                new_smart = smarts[i_smart]
                while new_smart == smarts[i_smart]:
                    id_new_smart = np.random.randint(0, len(all_reactions))
                    new_smart = all_reactions[id_new_smart]
                smarts[i_smart] = new_smart
            smarts = np.unique(smarts).tolist()
            impossible = True
        elif prop == "smarts":
            for i_reactants in range(len(reactants[0])):
                new_reactant = reactants[0][i_reactants]
                while new_reactant == reactants[0][i_reactants]:
                    new_reactant = np.random.choice(all_reactants)
                reactants[0][i_reactants] = new_reactant
                impossible = True
        elif prop in ["full_path_smarts_ref", "full_path_smarts_bb_ref"]:
            n = len(smarts)
            smarts = np.random.choice(all_reactions, n + len(or_smarts)).tolist()
            smarts = [s for s in smarts if s not in or_smarts][:n]
            impossible = True
        elif prop in ["full_path_bb_ref", "all_reactants_bb_ref"]:
            n = len(building_blocks)
            building_blocks = sample_l(n, all_reactants)
            building_blocks = [
                s for s in building_blocks if s not in original_building_blocks
            ][:n]
            impossible = True
    return impossible, smarts, reactants, building_blocks


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
    parser.add_argument(
        "--impossible_proba",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--n_bb_max",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--n_smarts_max",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--n_reaction_retry",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--n_bb_retry",
        type=int,
        default=10,
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
    return args


def update_data_dict(
    data_dict: Dict[str, Any],
    args: argparse.Namespace,
    i: int,
    reactants: list[list[str]],
    products: list[str],
    or_smarts: list[str],
    pass_filters: list[bool],
    all_reactants: list[str],
    all_reactions: list[str],
) -> None:
    if len(reactants) == 0 or len(products) == 0 or len(or_smarts) == 0:
        return None
    prop, idx_chosen, label = get_objective_label(
        args, reactants, products, or_smarts, pass_filters
    )
    original_building_blocks, building_blocks, smarts = get_smarts_bb(
        args,
        prop,
        reactants,
        or_smarts,
        all_reactants,
        all_reactions,
    )
    reaction_str_list = [
        " + ".join(r) + " -> " + p for r, p in zip(reactants, products)
    ]
    reaction_str = "\n".join(reaction_str_list)
    for lab in label:
        reaction_str = reaction_str.replace(f"{lab}", "?")

    impossible, smarts, reactants, building_blocks = make_impossible_decision(
        args.impossible_proba,
        prop,
        smarts,
        reactants,
        building_blocks,
        all_reactants,
        all_reactions,
        original_building_blocks,
        or_smarts,
    )

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
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt_text,
            },
        ]
    )
    data_dict["properties"].append([prop])
    data_dict["objectives"].append([prop])
    data_dict["target"].append(label)
    data_dict["prompt_id"].append(f"synth{i}")
    data_dict["full_reaction"].append("\n".join(reaction_str_list))
    data_dict["smarts"].append(smarts)
    data_dict["or_smarts"].append(or_smarts)
    data_dict["impossible"].append(impossible)
    data_dict["reactants"].append(reactants)
    data_dict["products"].append(products)
    data_dict["building_blocks"].append(building_blocks)
    data_dict["idx_chosen"].append(idx_chosen)
    data_dict["pass_filters"].append(pass_filters)


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
        "impossible": [],
        "building_blocks": [],
        "idx_chosen": [],
        "pass_filters": [],
    }
    proj_dataset = get_proj_dataset(args)

    all_reactants = [r.smiles for r in proj_dataset._reaction_matrix._reactants]
    all_reactions = [r.smarts for r in proj_dataset._reaction_matrix.reactions]
    i = 0

    for reactants, products, or_smarts, pass_filters in tqdm(
        proj_dataset, "Post-processing prompts", total=len(proj_dataset)
    ):
        update_data_dict(
            data_dict,
            args,
            i,
            reactants,
            products,
            or_smarts,
            pass_filters,
            all_reactants,
            all_reactions,
        )
        i += 1

        # if impossible:
        #     print(r" \\\ IMPOSSIBLE ///")
        # print(prompt_text)
        # print("---")

    dataset = data_dict_to_pydantic(data_dict)
    out_path = args.out_path
    write_jsonl(
        Path(os.path.join(out_path)),
        dataset,
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
