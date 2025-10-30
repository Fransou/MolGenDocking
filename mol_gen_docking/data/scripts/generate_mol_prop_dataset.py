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
                "target": data_dict["target"][i],
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


POLARIS_DATASET_PROP_NAME = {
    "polaris/adme-fang-solu-1": "log-solubility",
    "polaris/adme-fang-rppb-1": "rat plasma protein binding",
    "polaris/adme-fang-perm-1": "permeability",
    "polaris/adme-fang-hclint-1": "human liver microsomal stability",
    "polaris/adme-fang-rclint-1": "rat liver microsomal stability",
    "tdcommons/pgp-broccatelli": "an inhibitor of the P-glycoprotein",
    "tdcommons/vdss-lombardo": "volume of distribution at steady state",
    "tdcommons/bbb-martins": "able to penetrate the blood-brain barrier",
    "tdcommons/caco2-wang": "rate of compounds passing through the Caco-2 cells",
    "tdcommons/dili": "inducing liver injuries",
    "tdcommons/herg": "a blocker of hERG",
    "tdcommons/ames": "mutagenic",
    "tdcommons/half-life-obach": "duration for the concentration of the molecule in the body to be reduced by half",
    "tdcommons/lipophilicity-astrazeneca": "lipophilicity",
    "tdcommons/ppbr-az": "human plasma protein binding rate",
    "tdcommons/clearance-hepatocyte-az": "drug clearance (hepatocyte)",
    "tdcommons/clearance-microsome-az": "drug clearance (microsome)",
    "tdcommons/ld50-zhu": "acute toxicity LD50",
    "tdcommons/cyp3a4-veith": "inhibiting CYP3A4",
    "tdcommons/cyp2d6-veith": "inhibiting CYP2D6",
    "tdcommons/cyp2c9-veith": "inhibiting CYP2C9",
    "tdcommons/cyp2c9-substrate-carbonmangels": "a substrate of CYP2C9",
    "tdcommons/cyp2d6-substrate-carbonmangels": "a substrate of CYP2D6",
    "tdcommons/cyp3a4-substrate-carbonmangels": "a substrate of CYP3A4",
    "tdcommons/solubility-aqsoldb": "solubility",
    "molecularml/moleculeace-chembl2034-ki": "pKi against the Nuclear hormone receptor subfamily 3 group C member 1",
    "molecularml/moleculeace-chembl1871-ki": "pKi against the Nuclear hormone receptor subfamily 3 group C member 4",
    "molecularml/moleculeace-chembl239-ec50": "pEC50 against the Nuclear hormone receptor subfamily 1 group C member 1",
    "molecularml/moleculeace-chembl3979-ec50": "pEC50 against the Nuclear hormone receptor subfamily 1 group C member 2",
    "molecularml/moleculeace-chembl235-ec50": "pEC50 against the Nuclear hormone receptor subfamily 1 group C member 3",
    "molecularml/moleculeace-chembl2047-ec50": "pEC50 against the Nuclear hormone receptor subfamily 1 group H member 4",
    "molecularml/moleculeace-chembl2147-ki": "pKi against the CAMK protein kinase PIM family",
    "molecularml/moleculeace-chembl4792-ki": "pKi against the Orexin receptor family",
    "molecularml/moleculeace-chembl236-ki": "pKi against the Opioid receptor family",
    "molecularml/moleculeace-chembl234-ki": "pKi against the Dopamine receptor family",
    "molecularml/moleculeace-chembl1862-ki": "pKi against the Tyrosine protein kinase Abl family",
    "molecularml/moleculeace-chembl2835-ki": "pKi against the Tyrosine protein kinase JakA family",
    "molecularml/moleculeace-chembl238-ki": "pKi against the SLC06 neurotransmitter transporter family",
    "molecularml/moleculeace-chembl204-ki": "pKi against the Serine protease S1A subfamily",
    "molecularml/moleculeace-chembl214-ki": "pKi against the Serotonin receptor",
    "molecularml/moleculeace-chembl264-ki": "pKi against the Histamine receptor",
    "molecularml/moleculeace-chembl4203-ki": "pKi against the CMGC protein kinase CLK family",
    "molecularml/moleculeace-chembl262-ki": "pKi against the protein kinase GSK family",
    "molecularml/moleculeace-chembl4616-ec50": "pEC50 against the GRP-related receptor ",
}

PROMPT_TEMPLATES = {
    "regression": ["Give me the {property} of the following molecule: {mol}."],
    "classification": ["Tell me if {mol} is {property}."],
}

REGRESSION_TRANSFORMER = {"tdcommons/vdss-lombardo": "log"}


if __name__ == "__main__":
    import argparse

    import polaris as po
    from polaris.utils.types import TargetType

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_path", type=str, required=True)
    parser.add_argument(
        "-p",
        "--property_dataset",
        nargs="+",
        type=str,
        default=POLARIS_DATASET_PROP_NAME.keys(),
    )

    args = parser.parse_args()
    cumulative_length = 0
    for property_dataset in args.property_dataset:
        benchmark = po.load_benchmark(property_dataset)

        train, test = benchmark.get_train_test_split()
        obj: str

        targettype = list(benchmark.target_types.values())[0]
        if targettype == TargetType.REGRESSION:
            obj = "regression"
        elif targettype == TargetType.CLASSIFICATION:
            obj = "classification"
        else:
            raise NotImplementedError(f"{targettype} not covered")

        data_dict: Dict[str, Any] = {
            "prompt": [],
            "properties": [],
            "objectives": [],
            "target": [],
            "prompt_id": [],
        }

        final_ys = []
        for i, (smiles, y) in tqdm(enumerate(train)):
            prop_name = POLARIS_DATASET_PROP_NAME[property_dataset]

            y = float(y)
            if REGRESSION_TRANSFORMER.get(property_dataset, None) == "log":
                y = np.log(y)
                prop_name = "log-" + prop_name
            elif property_dataset in REGRESSION_TRANSFORMER:
                raise NotImplementedError(
                    f"{REGRESSION_TRANSFORMER[property_dataset]} not covered"
                )

            template = np.random.choice(PROMPT_TEMPLATES[obj])
            prompt_text = template.format(mol=smiles, property=prop_name)
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
            data_dict["properties"].append(property_dataset)
            data_dict["objectives"].append(obj)
            data_dict["target"].append(y)
            data_dict["prompt_id"].append(
                f"{property_dataset.replace('/', ':')}_train_{i}"
            )
            final_ys.append(y)

        pydantic_dataset = data_dict_to_pydantic(data_dict)

        n_train = int(len(pydantic_dataset) * 0.8)
        idx_train = np.random.choice(len(train), n_train, replace=False)
        idx_val = [idx for idx in range(len(train)) if idx not in idx_train]

        train_dataset = [pydantic_dataset[i] for i in idx_train]
        val_dataset = [pydantic_dataset[i] for i in idx_val]

        os.makedirs(os.path.join(args.data_path, "polaris"), exist_ok=True)
        data_path = os.path.join(args.data_path, "polaris")
        for p in property_dataset.split("/"):
            os.makedirs(os.path.join(data_path, p), exist_ok=True)
            data_path = os.path.join(data_path, p)

        write_jsonl(
            Path(os.path.join(data_path, "train.jsonl")),
            train_dataset,
        )
        write_jsonl(
            Path(os.path.join(data_path, "eval.jsonl")),
            train_dataset,
        )
        cumulative_length += len(train_dataset)
        print("#" * 10)
        print("Total length: {}".format(cumulative_length))

        # # Plot some stats about the data
        # sns.histplot(final_ys)
        # plt.title(
        #     train_dataset[0].conversations[0].messages[1].content,
        #     fontdict={"fontsize":7},
        # )
        # plt.show()
