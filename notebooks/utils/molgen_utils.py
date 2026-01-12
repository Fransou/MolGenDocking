import json
import re
from pathlib import Path
from typing import Any, Callable, List

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from scipy.spatial.distance import squareform
from tqdm import tqdm

from mol_gen_docking.reward.diversity_aware_top_k import diversity_aware_top_k

RDLogger.DisableLog("rdApp.*")


# LOADING
def load_molgen_results(
    filenames: List[Path],
) -> pd.DataFrame:
    generations = []
    for f in tqdm(filenames):
        with f.open("r") as fd:
            for i_l, line in enumerate(fd):
                g = json.loads(line)
                all_smis = g["reward_meta"].get("all_smi", [""])
                fail_reason = "valid"
                if len(all_smis) == 0:
                    valid = 0
                    smiles = ""
                    reward = 0.0
                    fail_reason = (
                        g["reward_meta"]
                        .get("smiles_extraction_failure", "unknown")
                        .replace("_", " ")
                        .replace("smiles", "SMILES")
                    )
                elif len(all_smis) > 1:
                    valid = 1
                    smiles = all_smis[-1]
                    reward = float(g["reward_meta"]["all_smi_rewards"][-1])
                else:
                    valid = 1
                    smiles = all_smis[0]
                    reward = float(g["reward"])
                model_name = str(f).split("/")[-1].split("eval")[0][:-1]
                if "scored" in model_name:
                    model_name = str(f).split("/")[-1].split("scored")[0][:-2]
                generations.append(
                    {
                        "prompt_id": g["metadata"]["prompt_id"],
                        "reward": reward,
                        "model": model_name,
                        "n_props": len(g["metadata"]["properties"]),
                        "properties": ",".join(g["metadata"]["properties"]),
                        "objectives": ",".join(g["metadata"]["objectives"]),
                        "smiles": smiles,
                        "validity": valid,
                        "valid": fail_reason,
                    }
                )

    df = pd.DataFrame(generations)

    df["Model"] = df["model"].apply(
        lambda x: re.sub(r"-\d+(B|b)", "", x[:-1])
        .replace("-2507", "")
        .replace("Distill", "D.")
        .replace("-it", "")
        .replace("Thinking", "Think.")
    )

    return df


# AGGREGATION


def fp_name_to_fn(
    fp_name: str,
) -> Callable[[Chem.Mol], DataStructs.cDataStructs.ExplicitBitVect]:
    """
    Names in:
        - ecfp{diam}-{Nbits}
        - maccs
        - rdkit
        - Gobbi2d
        - Avalon
    """

    if fp_name.startswith("ecfp"):
        d = int(fp_name[4])
        n_bits = int(fp_name.split("-")[1])
        assert fp_name == f"ecfp{d}-{n_bits}", f"Invalid fingerprint name: {fp_name}"

        def fp_fn(mol: Chem.Mol) -> DataStructs.cDataStructs.ExplicitBitVect:
            return AllChem.GetMorganFingerprintAsBitVect(mol, d // 2, n_bits)

        return fp_fn
    elif fp_name == "maccs":

        def fp_fn(mol: Chem.Mol) -> DataStructs.cDataStructs.ExplicitBitVect:
            return Chem.rdMolDescriptors.GetMACCSKeysFingerprint(mol)

        return fp_fn
    elif fp_name == "rdkit":

        def fp_fn(mol: Chem.Mol) -> DataStructs.cDataStructs.ExplicitBitVect:
            return Chem.RDKFingerprint(mol)

        return fp_fn
    elif fp_name == "Gobbi2d":
        from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

        def fp_fn(mol: Chem.Mol) -> DataStructs.cDataStructs.ExplicitBitVect:
            return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)

        return fp_fn
    elif fp_name == "Avalon":
        from rdkit.Avalon import pyAvalonTools

        def fp_fn(mol: Chem.Mol) -> DataStructs.cDataStructs.ExplicitBitVect:
            return pyAvalonTools.GetAvalonFP(mol)

        return fp_fn
    else:
        raise ValueError(f"Unknown fingerprint name: {fp_name}")


def agg_topk(k: int = 100, n_rollout: int = 2) -> Callable[[pd.Series], float]:
    def w_fn(x: pd.Series) -> float:
        # print(len(x))
        x = x[:n_rollout]
        x = x.sort_values(ascending=False)
        # Pad with 0s
        x_np = np.pad(x, (0, 100), "constant")
        out_val: float = x_np[:k].mean()
        return out_val

    return w_fn


def uniqueness_(k: int = 100) -> Callable[[pd.Series], float]:
    def w_fn(x: pd.Series) -> float:
        x = x[:k]
        tot = len(x)
        return len(x.drop_duplicates()) / tot

    return w_fn


def tanim_sim_(
    k: int = 100, fp_name: str = "ecfp4-2048"
) -> Callable[[pd.Series], float]:
    fp_fn = fp_name_to_fn(fp_name)

    def w_fn(x: pd.Series) -> float:
        x = x[:k]
        if len(x) == 1:
            return 1.0
        mols = [Chem.MolFromSmiles(smi) for smi in x]
        fps = [fp_fn(m) for m in mols]
        # Compute pairwise tanimoto similarity
        dist = [
            1 - np.array(DataStructs.BulkTanimotoSimilarity(fp, fps[i + 1 :]))
            for i, fp in enumerate(fps[:-1])
        ]
        dist = np.concatenate(dist)
        dist_npy: np.ndarray = squareform(dist)
        for i in range(dist_npy.shape[0]):
            dist_npy[i, i] = np.nan
        out_val: float = np.nanmean(dist_npy, axis=0).mean()
        return out_val

    return w_fn


def aggregate_molgen_fn(fn_name: str, k: int, **kwargs: Any) -> Callable:
    if fn_name == "topk":
        return agg_topk(k=k, **kwargs)
    elif fn_name == "uniqueness":
        return uniqueness_(k=k)
    elif fn_name == "murcko_sim":
        return tanim_sim_(k=k, **kwargs)
    raise ValueError(f"Unknown fn_name: {fn_name}")


# Diversity-aware top-k selection


def sim_topk(
    k: int = 100, div: float = 0.7, n_rollout: int = 100, fp_name: str = "ecfp4-2048"
) -> Callable[[pd.DataFrame], float]:
    fp_fn = fp_name_to_fn(fp_name)

    def w_fn(df: pd.DataFrame) -> float:
        x = df["smiles"].to_numpy()[:n_rollout]
        rewards = df["reward"].to_numpy()[:n_rollout]

        if len(x) == 1:
            cluster_rewards = [rewards[0]]
        else:
            mols = [Chem.MolFromSmiles(smi) for smi in x]
            fps = [fp_fn(m) for m in mols]
            # Compute pairwise tanimoto similarity
            dist_l = [
                1 - np.array(DataStructs.BulkTanimotoSimilarity(fp, fps[i + 1 :]))
                for i, fp in enumerate(fps[:-1])
            ]
            dist = np.concatenate(dist_l)
            idxs = diversity_aware_top_k(dist=dist, weights=rewards, k=k, t=div)
            cluster_rewards = [rewards[i] for i in idxs]
        cluster_rewards_npy = np.array(cluster_rewards)
        cluster_rewards_npy = np.sort(cluster_rewards_npy)[::-1]
        cluster_rewards_npy = np.pad(cluster_rewards_npy, (0, k), "constant")[:k]
        out_val: float = cluster_rewards_npy.mean()
        return out_val

    return w_fn


def get_top_k_div_df(
    df: pd.DataFrame,
    div_values: List[float] = [0.01, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    rollouts: List[int] = [50, 75, 100],
    ks: List[int] = [5, 10, 20, 30],
    fp_name: str = "ecfp4-2048",
) -> pd.DataFrame:
    div_clus_df_list = []

    pbar = tqdm(total=len(div_values) * len(rollouts) * len(ks))
    for div in div_values:
        for n_rollout in rollouts:
            new_pbar_desc = f"div: {div}, n_rollout: {n_rollout}"
            pbar.set_description(new_pbar_desc)
            pbar.refresh()
            for k in ks:
                div_clus_df_single = (
                    df.groupby(["model", "prompt_id"])
                    .apply(sim_topk(k=k, div=div, n_rollout=n_rollout, fp_name=fp_name))
                    .to_frame("value")
                    .reset_index()
                )
                div_clus_df_single["k"] = k
                div_clus_df_single["n_rollout"] = n_rollout
                div_clus_df_single["div"] = div
                div_clus_df_list.append(div_clus_df_single)
                pbar.update(1)
    pbar.close()

    div_clus_df = pd.concat(div_clus_df_list).reset_index()
    div_clus_df = (
        div_clus_df.groupby(["model", "n_rollout", "div", "k"])["value"]
        .mean()
        .reset_index()
    )
    div_clus_df["sim"] = 1 - div_clus_df["div"]
    div_clus_df["Model"] = div_clus_df["model"].apply(
        lambda x: re.sub(r"-\d+(B|b)", "", x[:-1])
        .replace("-2507", "")
        .replace("Distill", "D.")
        .replace("-it", "")
        .replace("Thinking", "Think.")
    )

    return div_clus_df
