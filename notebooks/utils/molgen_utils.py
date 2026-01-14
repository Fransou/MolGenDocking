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

tqdm.pandas()


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
        -
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


def agg_topk(
    k: int = 100, n_rollout: int = 2, pbar: None | tqdm = None
) -> Callable[[pd.Series], float]:
    def w_fn(x: pd.Series) -> float:
        # print(len(x))
        x = x[:n_rollout]
        x = x.sort_values(ascending=False)
        # Pad with 0s
        x_np = np.pad(x, (0, 100), "constant")
        out_val: float = x_np[:k].mean()
        if pbar is not None:
            pbar.update(1)
        return out_val

    return w_fn


def uniqueness_(k: int = 100, pbar: None | tqdm = None) -> Callable[[pd.Series], float]:
    def w_fn(x: pd.Series) -> float:
        x = x[:k]
        tot = len(x)
        if pbar is not None:
            pbar.update(1)
        return len(x.drop_duplicates()) / tot

    return w_fn


def tanim_sim_(
    k: int = 100,
    fp_name: str = "ecfp4-2048",
    is_fp: bool = False,
    pbar: None | tqdm = None,
) -> Callable[[pd.Series], float]:
    fp_fn = fp_name_to_fn(fp_name)

    def w_fn(x: pd.Series) -> float:
        x = x[:k]
        if len(x) == 1:
            return 0.0
        if not is_fp:
            mols = [Chem.MolFromSmiles(smi) for smi in x]
            fps = [fp_fn(m) for m in mols]
        else:
            fps = x.to_list()
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
        if pbar is not None:
            pbar.update(1)
        return out_val

    return w_fn


def aggregate_molgen_fn(
    fn_name: str, k: int, pbar: None | tqdm = None, **kwargs: Any
) -> Callable:
    if fn_name == "topk":
        return agg_topk(k=k, pbar=pbar, **kwargs)
    elif fn_name == "uniqueness":
        return uniqueness_(
            k=k,
            pbar=pbar,
        )
    elif fn_name == "diversity":
        return tanim_sim_(k=k, pbar=pbar, **kwargs)
    raise ValueError(f"Unknown fn_name: {fn_name}")


# Diversity-aware top-k selection


def sim_topk(
    k: int = 100, sim: float = 0.7, n_rollout: int = 100, fp_name: str = "ecfp4-2048"
) -> Callable[[pd.DataFrame], List[float]]:
    def w_fn(df: pd.DataFrame) -> List[float]:
        fps = df[f"fp-{fp_name}"].to_numpy()[:n_rollout]
        rewards = df["reward"].to_numpy()[:n_rollout]

        if len(fps) == 1:
            cluster_rewards = [rewards[0]]
        else:
            # Compute pairwise tanimoto similarity
            dist_l = [
                1 - np.array(DataStructs.BulkTanimotoSimilarity(fp, fps[i + 1 :]))
                for i, fp in enumerate(fps[:-1])
            ]
            dist = np.concatenate(dist_l)
            idxs = diversity_aware_top_k(dist=dist, weights=rewards, k=k, t=1 - sim)
            cluster_rewards = [rewards[i] for i in idxs]
        cluster_rewards_npy = np.array(cluster_rewards)
        cluster_rewards_npy = np.sort(cluster_rewards_npy)[::-1]
        cluster_rewards_npy = np.pad(cluster_rewards_npy, (0, k), "constant")[:k]

        # Average reward in the topk (div_top_k[k] = top-k score)
        div_top_k: List[float] = (
            cluster_rewards_npy.cumsum() / np.ones_like(cluster_rewards_npy).cumsum()
        ).tolist()
        return div_top_k

    return w_fn


SIM_VALUES_DEF = [0.01] + [0.05 * i for i in range(1, 20)] + [0.99]


def get_top_k_div_df(
    df: pd.DataFrame,
    sim_values: List[float] = SIM_VALUES_DEF,
    rollouts: List[int] = [50, 75, 100],
    k_max: int = 30,
    fp_name: str = "ecfp4-2048",
) -> pd.DataFrame:
    div_clus_df_list = []
    if f"fp-{fp_name}" not in df.columns:
        fp_fn = fp_name_to_fn(fp_name)
        df[f"fp-{fp_name}"] = df["smiles"].progress_apply(
            lambda x: fp_fn(Chem.MolFromSmiles(x))
        )
    pbar = tqdm(total=len(sim_values) * len(rollouts))
    for sim in sim_values:
        for n_rollout in rollouts:
            new_pbar_desc = f"sim: {sim}, n_rollout: {n_rollout}"
            pbar.set_description(new_pbar_desc)
            pbar.refresh()
            div_clus_df_single = (
                df.groupby(["Model", "prompt_id"])
                .apply(sim_topk(k=k_max, sim=sim, n_rollout=n_rollout, fp_name=fp_name))
                .to_frame("value")
                .reset_index()
            )
            div_clus_df_single["k"] = df.Model.apply(
                lambda x: [i + 1 for i in range(k_max)]
            )
            div_clus_df_single["n_rollout"] = n_rollout
            div_clus_df_single["sim"] = sim

            div_clus_df_single = div_clus_df_single.explode(["k", "value"]).reset_index(
                drop=True
            )
            div_clus_df_list.append(div_clus_df_single)
            pbar.update(1)
    pbar.close()

    div_clus_df = pd.concat(div_clus_df_list).reset_index()
    div_clus_df = (
        div_clus_df.groupby(["Model", "n_rollout", "sim", "k"])["value"]
        .mean()
        .reset_index()
    )

    return div_clus_df
