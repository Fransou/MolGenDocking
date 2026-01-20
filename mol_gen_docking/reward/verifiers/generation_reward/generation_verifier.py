import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import ray

from mol_gen_docking.reward.verifiers.abstract_verifier import Verifier
from mol_gen_docking.reward.verifiers.generation_reward.oracle_wrapper import (
    OracleWrapper,
    get_oracle,
)
from mol_gen_docking.reward.verifiers.generation_reward.property_utils import (
    rescale_property_values,
)


class GenerationVerifier(Verifier):
    """From a list of smiles and a metadata dict, returns a reward based
    on how well the proposed molecules meet the criterias"""

    def __init__(
        self,
        path_to_mappings: Optional[str] = None,
        reward: Literal["property", "valid_smiles"] = "property",
        rescale: bool = True,
        oracle_kwargs: Dict[str, Any] = {},
        docking_concurrency_per_gpu: int = 2,
    ):
        super().__init__()
        self.logger = logging.getLogger("GenerationVerifier")
        if path_to_mappings is not None:
            with open(os.path.join(path_to_mappings, "names_mapping.json")) as f:
                property_name_mapping = json.load(f)
            with open(os.path.join(path_to_mappings, "docking_targets.json")) as f:
                docking_target_list = json.load(f)
        self.docking_concurrency_per_gpu = docking_concurrency_per_gpu
        self.property_name_mapping = property_name_mapping
        self.docking_target_list = docking_target_list
        self.path_to_mappings = path_to_mappings

        self.slow_props = docking_target_list  # + ["GSK3B", "JNK3", "DRD2"]

        self.rescale = rescale
        self.oracle_kwargs = oracle_kwargs

        self.oracles: Dict[str, OracleWrapper] = {}
        self.debug = False  # Only for tests

    def fill_df_properties(self, df_properties: pd.DataFrame) -> None:
        def _get_property(
            smiles: List[str],
            prop: str,
            rescale: bool = True,
            kwargs: Dict[str, Any] = {},
        ) -> List[float]:
            """
            Get property reward
            """
            oracle_fn = self.oracles.get(
                prop,
                get_oracle(
                    prop,
                    path_to_data=self.path_to_mappings if self.path_to_mappings else "",
                    docking_target_list=self.docking_target_list,
                    property_name_mapping=self.property_name_mapping,
                    **kwargs,
                ),
            )
            if prop not in self.oracles:
                self.oracles[prop] = oracle_fn
            property_reward: np.ndarray | float = oracle_fn(smiles, rescale=rescale)
            assert isinstance(property_reward, np.ndarray)

            return [float(p) for p in property_reward]

        _get_property_fast = ray.remote(num_cpus=0)(_get_property)
        _get_property_long = ray.remote(
            num_cpus=1,
            num_gpus=float("gpu" in self.oracle_kwargs.get("docking_oracle", ""))
            / self.docking_concurrency_per_gpu,
        )(_get_property)

        all_properties = df_properties["property"].unique().tolist()
        prop_smiles = {
            p: df_properties[df_properties["property"] == p]["smiles"].unique().tolist()
            for p in all_properties
        }

        values_job = []
        for p in all_properties:
            # If the reward is long to compute, use ray
            smiles = prop_smiles[p]
            if p in self.slow_props:
                _get_property_remote = _get_property_long
            else:
                _get_property_remote = _get_property_fast

            values_job.append(
                _get_property_remote.remote(
                    smiles,
                    p,
                    rescale=self.rescale,
                    kwargs=self.oracle_kwargs,
                )
            )
        all_values = ray.get(values_job)
        for idx_p, p in enumerate(all_properties):
            values = all_values[idx_p]
            smiles = prop_smiles[p]
            for s, v in zip(smiles, values):
                df_properties.loc[
                    (df_properties["smiles"] == s) & (df_properties["property"] == p),
                    "value",
                ] = v

    def get_reward(self, row: pd.Series) -> float:
        reward: float = 0
        obj = row["obj"]
        mol_prop = row["value"]
        target_value = row["target_value"]
        prop = row["property"]
        is_docking = prop in self.docking_target_list
        # Replace 0 docking score by the worst outcome
        if is_docking and prop == 0.0:
            return 0.0
        if self.rescale:
            target_value = rescale_property_values(
                prop, target_value, docking=is_docking
            )
        if obj == "below":
            reward += float(mol_prop <= target_value)
        elif obj == "above":
            reward += float(mol_prop >= target_value)
        elif obj == "maximize":
            reward += mol_prop
        elif obj == "minimize":
            reward += 1 - mol_prop
        elif obj == "equal":
            reward += np.clip(1 - 100 * (mol_prop - target_value) ** 2, 0, 1)
        return float(reward)

    def _get_prop_to_smiles_dataframe(
        self,
        smiles_list_per_completion: List[List[str]],
        objectives: List[dict[str, Tuple[str, float]]],
    ) -> pd.DataFrame:
        df_properties = pd.DataFrame(
            [
                (s, p, None, obj, target_value, i)
                for i, (props, smiles_list) in enumerate(
                    zip(objectives, smiles_list_per_completion)
                )
                for s in smiles_list
                for p, (obj, target_value) in props.items()
            ],
            columns=[
                "smiles",
                "property",
                "value",
                "obj",
                "target_value",
                "id_completion",
            ],
        )
        return df_properties

    def get_score(
        self, smiles_per_completion: List[List[str]], metadata: List[Dict[str, Any]]
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        assert metadata is not None and (
            all(
                [
                    p in m
                    for m in metadata
                    for p in ["properties", "objectives", "target"]
                ]
            )
        )
        objectives = []
        for m in metadata:
            props = {}
            for p, obj, target in zip(m["properties"], m["objectives"], m["target"]):
                props[p] = (obj, float(target))
            objectives.append(props)

        df_properties = self._get_prop_to_smiles_dataframe(
            smiles_per_completion, objectives
        )
        self.fill_df_properties(df_properties)
        df_properties["reward"] = df_properties.apply(
            lambda x: self.get_reward(x), axis=1
        )

        rewards = []
        rewards_meta = []
        for id_completion, smiles in enumerate(smiles_per_completion):
            meta: Dict[str, List[Any]] = {"properties": [], "individual_rewards": []}
            compl_reward: List[float] = []
            if len(smiles) > 0:
                for idx_s, s in enumerate(smiles):
                    rows_completion = df_properties[
                        (df_properties["id_completion"] == id_completion)
                        & (df_properties["smiles"] == s)
                    ]
                    rewards_l = rows_completion["reward"].to_numpy()
                    reward = np.power(
                        rewards_l.prod(), (1 / len(rewards_l))
                    )  # Geometric mean
                    if idx_s == 0:
                        for i in range(len(rows_completion["smiles"])):
                            meta["properties"].append(
                                rows_completion["property"].iloc[i]
                            )
                            meta["individual_rewards"].append(
                                rows_completion["reward"].iloc[i]
                            )

                    if self.rescale and not self.debug:
                        reward = np.clip(reward, 0, 1)
                    compl_reward.append(float(reward))
            else:
                reward = 0
                compl_reward = [0.0]
            meta["all_smi_rewards"] = compl_reward
            meta["all_smi"] = smiles
            rewards_meta.append(meta)

            if np.isnan(reward) or reward is None:
                self.logger.warning(
                    f"Warning: Reward is None or NaN for completion id {id_completion} with smiles {smiles}\n"
                )
                reward = 0
            if len(smiles) > 1:
                reward = 0.0
            rewards.append(float(reward))
        return rewards, rewards_meta
