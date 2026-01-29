import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .utils import process_model_name


def load_synth_results(
    filenames: list[Path],
) -> pd.DataFrame:
    generations = []
    for f in tqdm(filenames):
        with f.open("r") as fd:
            for i_l, line in enumerate(fd):
                g = json.loads(line)
                target = g["metadata"]["target"][0]
                reward = g["reward"]
                reward = float(reward)
                valid = reward > 0
                prop_valid = float(g["reward_meta"].get("prop_valid", 0.0))
                model_name = str(f).split("/")[-2]
                generations.append(
                    {
                        "prompt_id": g["metadata"]["prompt_id"],
                        "reward": reward,
                        "model": model_name,
                        "prop_valid": prop_valid,
                        "validity": valid,
                        "target": target,
                        "tanim_sim": 0.0
                        if prop_valid == 0
                        else (reward / prop_valid**2) ** (1 / 3),
                    }
                )

    df = pd.DataFrame(generations)
    df["Model"] = df["model"].apply(process_model_name)
    return df
