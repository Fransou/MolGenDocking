import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import process_model_name

PROP_NAME = {
    # ASAP
    "asap-discovery/antiviral-potency-2025-unblinded": "antiviral-potency",
    # Biogen ADME (FANG)
    "biogen/adme-fang-hclint-reg-v1": "fang-hclint",
    "biogen/adme-fang-hppb-reg-v1": "fang-hppb",
    "biogen/adme-fang-perm-reg-v1": "fang-perm",
    "biogen/adme-fang-rclint-reg-v1": "fang-rclint",
    "biogen/adme-fang-rppb-reg-v1": "fang-rppb",
    "biogen/adme-fang-solu-reg-v1": "fang-solubility",
    # Novartis
    "novartis/novartis-cyp3a4-v1": "cyp3a4-novartis",
    # Polaris / AZ
    "polaris/az-logd-74-v1": "az-logd",
    "polaris/az-ppb-clearance-v1": "az-ppb-clearance",
    "polaris/drewry2017-pkis2-subset-v2": "pkis2-drewry",
    # Therapeutics Data Commons (TDC)
    "tdcommons/ames": "ames",
    "tdcommons/bbb-martins": "bbb",
    "tdcommons/caco2-wang": "caco2",
    "tdcommons/clearance-hepatocyte-az": "hep-clearance-az",
    "tdcommons/clearance-microsome-az": "mic-clearance-az",
    "tdcommons/cyp2c9-substrate-carbonmangels": "cyp2c9-substrate",
    "tdcommons/cyp2d6-substrate-carbonmangels": "cyp2d6-substrate",
    "tdcommons/cyp3a4-substrate-carbonmangels": "cyp3a4-substrate",
    "tdcommons/dili": "dili",
    "tdcommons/half-life-obach": "half-life",
    "tdcommons/herg": "herg",
    "tdcommons/ld50-zhu": "ld50",
    "tdcommons/lipophilicity-astrazeneca": "lipophilicity",
    "tdcommons/pgp-broccatelli": "pgp",
    "tdcommons/solubility-aqsoldb": "solubility",
    "tdcommons/vdss-lombardo": "vdss",
}


def load_molprop_results(
    filenames: list[Path],
) -> pd.DataFrame:
    generations = []

    for f in tqdm(filenames):
        with f.open("r") as fd:
            for i_l, line in enumerate(fd):
                g = json.loads(line)
                valid = g["reward_meta"].get("extracted_answer", None) is not None
                objective = g["metadata"]["objectives"][0]
                target = float(g["metadata"]["target"][0])
                norm_var = float(g["metadata"]["norm_var"])

                if valid:
                    extracted = g["reward_meta"]["extracted_answer"]
                else:
                    extracted = "invalid"
                if objective == "classification":
                    reward = 0.5 if not valid else float(extracted) == target
                elif objective == "regression":
                    reward = (
                        0.0
                        if not valid
                        else np.clip(
                            1 - ((float(extracted) - target) / norm_var) ** 2,
                            a_max=1,
                            a_min=-100,
                        )
                    )
                else:
                    raise ValueError(f"unknown objective: {objective}")
                reward = float(reward)
                model_name = str(f).split("/")[-1].split("eval")[0][:-1]
                if "scored" in model_name:
                    model_name = str(f).split("/")[-1].split("scored")[0][:-2]

                generations.append(
                    {
                        "prompt_id": g["metadata"]["prompt_id"],
                        "reward": reward,
                        "gt": target,
                        "y": extracted,
                        "norm_var": norm_var,
                        "model": model_name,
                        "n_props": len(g["metadata"]["properties"]),
                        "properties": ",".join(g["metadata"]["properties"]),
                        "objectives": objective,
                        "validity": valid,
                        "Task": PROP_NAME[",".join(g["metadata"]["properties"])],
                    }
                )

    df = pd.DataFrame(generations)
    df["gen_id"] = df.index % 5

    df["Model"] = df["model"].apply(process_model_name)
    return df
