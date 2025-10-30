import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
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
                "target": 0,
                "prompt_id": data_dict["prompt_id"][i],
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
    "product": ["Give me the product of the following chemical reaction: {input}."],
    "reactant": [
        "What is the missing reactant of the following chemical reaction: {input}"
    ],
    "reactant_full": [
        "Generate a possible chemical reaction to obtain the following products: {input}"
    ],
    "product_full": [
        "Find all the products of the following chemical reaction: {input}"
    ],
}


if __name__ == "__main__":
    import argparse

    import datasets

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_path", type=str, required=True)
    parser.add_argument("-s", "--subsample", type=float, default=1.0)

    args = parser.parse_args()

    data = datasets.load_dataset("chenxran/uspto_full")

    for split in ["train", "valid", "test"]:
        data_split = data[split]
        data_dict: Dict[str, Any] = {
            "prompt": [],
            "properties": [],
            "objectives": [],
            "target": [],
            "prompt_id": [],
        }
        for i, row in tqdm(
            enumerate(data_split), desc=split, total=len(data_split["reaction"])
        ):
            if split == "train" and args.subsample < 1:
                if np.random.random() > args.subsample:
                    continue
            reactants = row["reaction"].split(">")[0].split(".")
            products = row["reaction"].split(">")[1].split(".")
            obj = np.random.choice(list(PROMPT_TEMPLATES.keys()))

            label: str
            reaction = " + ".join(reactants) + " -> " + " + ".join(products)
            if obj == "reactant":
                idx_label = np.random.choice(len(reactants))
                label = reactants[idx_label]
            elif obj == "product":
                idx_label = np.random.choice(len(products))
                label = products[idx_label]
            elif obj == "reactant_full":
                label = " + ".join(reactants)
            elif obj == "product_full":
                label = " + ".join(products)
            else:
                raise ValueError(f"Unknown reaction type: {obj}")

            reaction = reaction.replace(label, "?")
            template = np.random.choice(PROMPT_TEMPLATES[obj])
            prompt_text = template.format(input=reaction)
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
            data_dict["prompt_id"].append(f"uspto_{split}_{i}")
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
