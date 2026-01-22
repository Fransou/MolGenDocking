import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm.auto import tqdm

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

PROMPT = "Generate an analog of {SMILES} with its corresponding synthetic pathway. Provide your answer in the following format:\nA + B -> C\n C -> D\n with at most 5 steps, where the last product is the analog of {SMILES}."


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
                "building_blocks": data_dict["building_blocks"][i],
                "smarts": data_dict["smarts"][i],
                "n_steps_max": data_dict["n_steps_max"][i],
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


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_path", type=str, required=True)
    parser.add_argument("-o", "--out_path", type=str, required=True)
    parser.add_argument("-n", "--n_prompts", type=int, default=10000)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    return args


def update_data_dict(
    data_dict: Dict[str, Any], args: argparse.Namespace, smiles: str, i: int
) -> None:
    prompt_text = PROMPT.format(SMILES=smiles)

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
    data_dict["properties"].append(["analog_gen"])
    data_dict["objectives"].append(["analog_gen"])
    data_dict["target"].append([smiles])
    data_dict["prompt_id"].append(f"analog_gen{i}")
    data_dict["building_blocks"].append([])
    data_dict["smarts"].append([])
    data_dict["n_steps_max"].append(5)


def main(args: argparse.Namespace) -> None:
    data_dict: Dict[str, Any] = {
        "prompt": [],
        "properties": [],
        "objectives": [],
        "target": [],
        "prompt_id": [],
        "building_blocks": [],
        "smarts": [],
        "n_steps_max": [],
    }
    ds = load_dataset("jarod0411/zinc10M")["train"].shuffle()[: args.n_prompts]
    for i, smiles in enumerate(tqdm(ds["smiles"], total=len(ds))):
        update_data_dict(data_dict, args, smiles=smiles, i=i)

    dataset = data_dict_to_pydantic(data_dict)
    out_path = args.out_path
    write_jsonl(
        Path(os.path.join(out_path)),
        dataset,
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
