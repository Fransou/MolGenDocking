import argparse
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal

import numpy as np
from polaris.hub.client import PolarisHubClient
from tqdm import tqdm

from mol_gen_docking.data.pydantic_dataset import (
    Conversation,
    Message,
    Sample,
    read_jsonl,
    write_jsonl,
)

with PolarisHubClient() as client:
    client.login()

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. "
    "The User asks a question, and the Assistant solves it.\n"
    "The reasoning process is enclosed within <think> </think>"
    " and answer is enclosed within <answer> </answer> tags, "
    "respectively, i.e., <think> reasoning process here </think> "
    "<answer> answer here </answer>."
)


def data_dict_to_pydantic(
    data_dict: dict, key: str = "prompt", final_ys: List[float] = []
) -> List[Sample]:
    sample_list: List[Sample] = []
    norm_var: float = 0.0
    if final_ys != []:
        norm_var = float(np.std(final_ys))

    for i in range(len(data_dict[key])):
        messages: List[Message] = [Message(**msg) for msg in data_dict[key][i]]
        metadata = {
            "properties": data_dict["properties"][i],
            "objectives": data_dict["objectives"][i],
            "target": data_dict["target"][i],
            "prompt_id": data_dict["prompt_id"][i],
            "smiles": data_dict["smiles"][i],
        }
        if final_ys != []:
            metadata["norm_var"] = norm_var
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
            meta=metadata,
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
    # "asap-discovery/antiviral-potency-2025-unblinded|CXSMILES|pIC50 (SARS-CoV-2 Mpro)": "pIC50 against SARS-CoV-2 main protease",
    "asap-discovery/antiviral-potency-2025-unblinded|CXSMILES|pIC50 (MERS-CoV Mpro)": "pIC50 against MERS-CoV main protease",
    # "leash-bio/belka-v1|molecule_smiles|binds_HSA": "binding to HSA",
    # "leash-bio/belka-v1|molecule_smiles|binds_sEH": "binding to sEH",
    # "leash-bio/belka-v1|molecule_smiles|binds_BRD4": "binding to BRD4",
    "polaris/az-logd-74-v1|MOL_smiles|LOGD_74": "octan-1-ol/water (pH7.4) distribution coefficent",
    "polaris/az-ppb-clearance-v1|MOL_smiles|log_unbound_PPB": "log percent of compound unbound to whole human plasma",
    "novartis/novartis-cyp3a4-v1|MOL_smiles|log_kobs": "log-inactivation rate constant of CYP enzymes",
    "polaris/drewry2017-pkis2-subset-v2|MOL_smiles|CLS_EGFR": "an inhibator of the EGFR kinase",
    "polaris/drewry2017-pkis2-subset-v2|MOL_smiles|CLS_KIT": "an inhibator of the KIT kinase",
    "polaris/drewry2017-pkis2-subset-v2|MOL_smiles|CLS_RET": "an inhibator of the RET kinase",
    "polaris/drewry2017-pkis2-subset-v2|MOL_smiles|CLS_LOK": "an inhibator of the LOK kinase",
    "polaris/drewry2017-pkis2-subset-v2|MOL_smiles|CLS_SLK": "an inhibator of the SLK kinase",
    "biogen/adme-fang-solu-reg-v1": "log-solubility",
    "biogen/adme-fang-rppb-reg-v1": "log-rat plasma protein binding rate",
    "biogen/adme-fang-hppb-reg-v1": "log-human plasma protein binding rate",
    "biogen/adme-fang-perm-reg-v1": "log-MDR1 MDCK efflux ratio",
    "biogen/adme-fang-hclint-reg-v1": "log-human liver microsomal stability",
    "biogen/adme-fang-rclint-reg-v1": "log-rat liver microsomal stability",
    "tdcommons/pgp-broccatelli": "an inhibitor of the P-glycoprotein",
    "tdcommons/vdss-lombardo": "volume of distribution at steady state",
    "tdcommons/bbb-martins": "able to penetrate the blood-brain barrier",
    "tdcommons/caco2-wang": "rate of compounds passing through the Caco-2 cells",
    "tdcommons/dili": "inducing liver injuries",
    "tdcommons/herg": "a blocker of hERG",
    "tdcommons/ames": "mutagenic",
    "tdcommons/half-life-obach": "duration for the concentration of the molecule in the body to be reduced by half",
    "tdcommons/lipophilicity-astrazeneca": "lipophilicity",
    "tdcommons/clearance-hepatocyte-az": "drug clearance (hepatocyte)",
    "tdcommons/clearance-microsome-az": "drug clearance (microsome)",
    "tdcommons/ld50-zhu": "acute toxicity LD50",
    # "tdcommons/cyp3a4-veith": "inhibiting CYP3A4",
    # "tdcommons/cyp2d6-veith": "inhibiting CYP2D6",
    # "tdcommons/cyp2c9-veith": "inhibiting CYP2C9",
    "tdcommons/cyp2c9-substrate-carbonmangels": "a substrate of CYP2C9",
    "tdcommons/cyp2d6-substrate-carbonmangels": "a substrate of CYP2D6",
    "tdcommons/cyp3a4-substrate-carbonmangels": "a substrate of CYP3A4",
    "tdcommons/solubility-aqsoldb": "solubility",
    # "molecularml/moleculeace-chembl2034-ki": "pKi against the Nuclear hormone receptor subfamily 3 group C member 1",
    # "molecularml/moleculeace-chembl1871-ki": "pKi against the Nuclear hormone receptor subfamily 3 group C member 4",
    # "molecularml/moleculeace-chembl239-ec50": "pEC50 against the Nuclear hormone receptor subfamily 1 group C member 1",
    # "molecularml/moleculeace-chembl3979-ec50": "pEC50 against the Nuclear hormone receptor subfamily 1 group C member 2",
    # "molecularml/moleculeace-chembl235-ec50": "pEC50 against the Nuclear hormone receptor subfamily 1 group C member 3",
    # "molecularml/moleculeace-chembl2047-ec50": "pEC50 against the Nuclear hormone receptor subfamily 1 group H member 4",
    # "molecularml/moleculeace-chembl2147-ki": "pKi against the CAMK protein kinase PIM family",
    # "molecularml/moleculeace-chembl4792-ki": "pKi against the Orexin receptor family",
    # "molecularml/moleculeace-chembl236-ki": "pKi against the Opioid receptor family",
    # "molecularml/moleculeace-chembl234-ki": "pKi against the Dopamine receptor family",
    # "molecularml/moleculeace-chembl1862-ki": "pKi against the Tyrosine protein kinase Abl family",
    # "molecularml/moleculeace-chembl2835-ki": "pKi against the Tyrosine protein kinase JakA family",
    # "molecularml/moleculeace-chembl238-ki": "pKi against the SLC06 neurotransmitter transporter family",
    # "molecularml/moleculeace-chembl204-ki": "pKi against the Serine protease S1A subfamily",
    # "molecularml/moleculeace-chembl214-ki": "pKi against the Serotonin receptor",
    # "molecularml/moleculeace-chembl264-ki": "pKi against the Histamine receptor",
    # "molecularml/moleculeace-chembl4203-ki": "pKi against the CMGC protein kinase CLK family",
    # "molecularml/moleculeace-chembl262-ki": "pKi against the protein kinase GSK family",
    # "molecularml/moleculeace-chembl4616-ec50": "pEC50 against the GRP-related receptor ",
}

DATASETS_CLS = [
    "polaris/drewry2017-pkis2-subset-v2|MOL_smiles|CLS_EGFR",
    "polaris/drewry2017-pkis2-subset-v2|MOL_smiles|CLS_KIT",
    "polaris/drewry2017-pkis2-subset-v2|MOL_smiles|CLS_RET",
    "polaris/drewry2017-pkis2-subset-v2|MOL_smiles|CLS_SLK",
    "polaris/drewry2017-pkis2-subset-v2|MOL_smiles|CLS_LOK",
    "leash-bio/belka-v1|molecule_smiles|binds_HSA",
    "leash-bio/belka-v1|molecule_smiles|binds_sEH",
    "leash-bio/belka-v1|molecule_smiles|binds_BRD4",
]

PROMPT_TEMPLATES = {
    "regression": ["Give me the {property} of the following molecule: {mol}."],
    "classification": ["Tell me if {mol} is {property}."],
}

DOWNSCALE = {
    "leash-bio/belka-v1|molecule_smiles|binds_HSA": 1e-5,
    "leash-bio/belka-v1|molecule_smiles|binds_sEH": 1e-5,
    "leash-bio/belka-v1|molecule_smiles|binds_BRD4": 1e-5,
}

REGRESSION_TRANSFORMER = {
    "tdcommons/vdss-lombardo": "log",
    "tdcommons/half-life-obach": "log",
    "tdcommons/clearance-hepatocyte-az": "log",
    "tdcommons/clearance-microsome-az": "log",
}


def polaris_iterator(
    dataset: Any,
    type: Literal["benchmark", "dataset"],
    smiles_key: str | None = None,
    key: str | None = None,
    downscale: float = 1.0,
    label: bool = True,
) -> Iterator[Any]:
    if type == "benchmark":
        if label:
            for smiles, y in dataset:
                if downscale < 1 and np.random.random() < downscale:
                    continue
                yield smiles, y
        else:
            for smiles in dataset:
                if downscale < 1 and np.random.random() < downscale:
                    continue
                yield smiles, 0.0

    elif type == "dataset":
        for i in range(dataset.size()[0]):
            if downscale < 1 and np.random.random() > downscale:
                yield None, None
                continue
            smiles = dataset.get_data(
                row=dataset.rows[i],
                col=smiles_key,
            ).split(" ")[0]
            y = dataset.get_data(
                row=dataset.rows[i],
                col=key,
            )
            if y is None or np.isnan(y).any():
                continue
            yield smiles, y


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
    length = {}
    for property_dataset_k in args.property_dataset:
        property_dataset_k_split = property_dataset_k.split("|")
        data_dict: Dict[str, Any] = {
            "prompt": [],
            "properties": [],
            "objectives": [],
            "target": [],
            "prompt_id": [],
            "smiles": [],
        }

        final_ys = []
        smiles_key: None | str = None
        key: str | None = None
        dataset_type: Literal["benchmark", "dataset"] = "dataset"
        if len(property_dataset_k_split) == 1:
            property_dataset = property_dataset_k
            benchmark = po.load_benchmark(property_dataset)
            data = list(benchmark.get_train_test_split())
            dataset_type = "benchmark"
            obj: str
            targettype = list(benchmark.target_types.values())[0]
            if targettype == TargetType.REGRESSION:
                obj = "regression"
            elif targettype == TargetType.CLASSIFICATION:
                obj = "classification"
            else:
                raise NotImplementedError(f"{targettype} not covered")

        elif len(property_dataset_k_split) == 3:
            property_dataset = property_dataset_k_split[0]
            smiles_key = property_dataset_k_split[1]
            key = property_dataset_k_split[2]
            dataset_type = "dataset"
            data = [po.load_dataset(property_dataset)]
            if property_dataset_k in DATASETS_CLS:
                obj = "classification"
            else:
                obj = "regression"

        else:
            raise NotImplementedError(f"{property_dataset_k_split} not covered")

        for split, split_name in zip(data, ["train", "test"]):
            print(property_dataset, split_name)
            for i, (smiles, y) in tqdm(
                enumerate(
                    polaris_iterator(
                        split,
                        dataset_type,
                        smiles_key=smiles_key,
                        key=key,
                        downscale=DOWNSCALE.get(property_dataset_k, 1.0),
                        label=split_name == "train",
                    )
                ),
                total=len(split),
            ):
                if smiles is None:
                    continue
                prop_name = POLARIS_DATASET_PROP_NAME[property_dataset_k]
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
                data_dict["properties"].append([property_dataset])
                data_dict["objectives"].append([obj])
                data_dict["target"].append([y])
                data_dict["smiles"].append([smiles])
                data_dict["prompt_id"].append(
                    f"{property_dataset.replace('/', ':')}_{split_name}_{i}"
                )
                final_ys.append(y)

            data_path = os.path.join(args.data_path, "polaris")
            os.makedirs(data_path, exist_ok=True)
            for p in property_dataset.split("/"):
                os.makedirs(os.path.join(data_path, p), exist_ok=True)
                data_path = os.path.join(data_path, p)
            if key is not None:
                data_path = os.path.join(data_path, key)
                os.makedirs(data_path, exist_ok=True)

            n_tot = len(data_dict["prompt"])
            n_train = int(len(data_dict["prompt"]) * 0.8)
            if split_name == "train":
                main_path = Path(os.path.join(data_path, "train.jsonl"))
                if main_path.exists():
                    train_dataset = read_jsonl(main_path)
                    idx_train = [
                        int(sample.identifier.split("_")[-1])
                        for sample in train_dataset
                    ]
                else:
                    idx_train = np.random.choice(n_tot, n_train, replace=False).tolist()
                idx_val = [idx for idx in range(n_tot) if idx not in idx_train]

                pydantic_dataset = data_dict_to_pydantic(data_dict, final_ys=final_ys)

                train_dataset = [pydantic_dataset[i] for i in idx_train]
                val_dataset = [pydantic_dataset[i] for i in idx_val]

                if not main_path.exists():
                    write_jsonl(
                        main_path,
                        train_dataset,
                    )
                write_jsonl(
                    Path(os.path.join(data_path, "eval.jsonl")),
                    val_dataset,
                )
            else:
                main_path = Path(os.path.join(data_path, "test.jsonl"))
                write_jsonl(
                    main_path,
                    val_dataset,
                )
            length[property_dataset] = len(train_dataset)

            # Plot some stats about the data
            # sns.histplot(final_ys)
            # plt.title(
            #     property_dataset
            #     + "\n"
            #     + train_dataset[0].conversations[0].messages[1].content.replace(":", "\n"),
            #     fontdict={"fontsize": 7},
            # )
            # plt.show()

    print("#" * 10)
    print("All length: {}".format("\n".join([f"{k}:{v}" for k, v in length.items()])))
    print("Final Length: ", np.sum(list(length.values())))
