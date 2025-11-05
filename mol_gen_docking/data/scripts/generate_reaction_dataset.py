import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from mol_gen_docking.data.pydantic_dataset import (
    Conversation,
    Message,
    Sample,
    write_jsonl,
)

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
                "solvent": data_dict["solvent"][i],
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


PROMPT_TEMPLATES = {
    "product_no_solvent": [
        "Give me the missing product of the following chemical reaction: {input}."
    ],
    "reactant_no_solvent": [
        "What is the missing reactant of the following chemical reaction: {input} ?"
    ],
    "solvent": ["In which solvent can the following reaction take place: {input} ?"],
    "product": [
        "Give me the missing product of the following chemical reaction{SOLVENT_INFO}: {input}."
    ],
    "reactant": [
        "What is the missing reactant of the following chemical reaction{SOLVENT_INFO}: {input} ?"
    ],
    "reactant_full": [
        "Generate a possible chemical reaction to obtain the following products{SOLVENT_INFO}: {input}."
    ],
    "product_full": [
        "Find all the products of the following chemical reaction{SOLVENT_INFO}: {input}."
    ],
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_path", type=str, required=True)
    parser.add_argument("-s", "--subsample", type=float, default=1.0)
    parser.add_argument(
        "-p",
        "--proba-obj",
        nargs="+",
        type=float,
        default=[0.1, 0.1, 0.1, 0.2, 0.2, 0.15, 0.15],
    )

    args = parser.parse_args()
    assert len(args.proba_obj) == len(PROMPT_TEMPLATES)
    uspto_llm = pd.read_csv(os.path.join(args.data_path, "uspto_llm.csv"))
    train_ids = uspto_llm["id"].sample(frac=0.7)
    valid_ids = uspto_llm[~uspto_llm["id"].isin(train_ids)]["id"].sample(frac=0.5)
    test_ids = uspto_llm[
        ~uspto_llm["id"].isin(train_ids) & ~uspto_llm["id"].isin(valid_ids)
    ]["id"]

    data = {
        "train": uspto_llm[uspto_llm["id"].isin(train_ids)],
        "valid": uspto_llm[uspto_llm["id"].isin(valid_ids)],
        "test": uspto_llm[uspto_llm["id"].isin(test_ids)],
    }

    for split in ["train", "valid", "test"]:
        data_split = data[split]
        data_dict: Dict[str, Any] = {
            "prompt": [],
            "properties": [],
            "objectives": [],
            "target": [],
            "prompt_id": [],
            "full_reaction": [],
            "solvent": [],
        }
        for i, (_, row) in tqdm(
            enumerate(data_split.iterrows()), desc=split, total=len(data_split)
        ):
            if args.subsample < 1:
                if np.random.random() > args.subsample:
                    continue

            reactants = row["rs>>ps"].split(">>")[0].split(".")
            products = row["rs>>ps"].split(">>")[1].split(".")
            solvent = ".".join(json.loads(row["solvents"].replace("'", '"')))

            probas = np.array(args.proba_obj)
            if solvent == "":
                probas[2] = 0
                probas = probas / probas.sum()
            obj = np.random.choice(list(PROMPT_TEMPLATES.keys()), p=args.proba_obj)

            label: str
            reaction = " + ".join(reactants) + " -> " + " + ".join(products)
            if obj == "reactant_no_solvent" or obj == "reactant":
                idx_label = np.random.choice(len(reactants))
                label = reactants[idx_label]
            elif obj == "product_no_solvent" or obj == "product":
                idx_label = np.random.choice(len(products))
                label = products[idx_label]
            elif obj == "solvent":
                label = solvent
            elif obj == "reactant_full":
                label = " + ".join(reactants)
            elif obj == "product_full":
                label = " + ".join(products)
            else:
                raise ValueError(f"Unknown reaction type: {obj}")

            if obj != solvent:
                reaction = reaction.replace(label, "?")
            template = np.random.choice(PROMPT_TEMPLATES[obj])

            if "no_solvent" in obj:
                prompt_text = template.format(input=reaction)
            else:
                solvent_info = "performed in the solvent: {}".format(solvent)
                if solvent == "":
                    prompt_text = template.format(input=reaction, SOLVENT_INFO="")
                else:
                    prompt_text = template.format(
                        input=reaction, SOLVENT_INFO=solvent_info
                    )
            prompt = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt_text,
                },
            ]
            data_dict["prompt"].append(prompt)
            data_dict["properties"].append([""])
            data_dict["objectives"].append([obj])
            data_dict["target"].append([label])
            data_dict["prompt_id"].append(row["id"])
            data_dict["full_reaction"].append(
                " + ".join(reactants) + " -> " + " + ".join(products)
            )
            data_dict["solvent"].append(solvent)
        pydantic_dataset = data_dict_to_pydantic(data_dict)
        print(f"Generated {split} with {len(data_dict['prompt'])} reactions")
        os.makedirs(os.path.join(args.data_path, "uspto"), exist_ok=True)
        data_path = os.path.join(args.data_path, "uspto")
        if split == "valid":
            split = "eval"
        write_jsonl(
            Path(os.path.join(data_path, f"{split}.jsonl")),
            pydantic_dataset,
        )
