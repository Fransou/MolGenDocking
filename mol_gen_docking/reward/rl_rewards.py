import logging
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import ray
from ray.experimental import tqdm_ray
from rdkit import Chem, RDLogger

from mol_gen_docking.reward.verifiers import (
    GenerationVerifier,
    MolPropVerifier,
    ReactionVerifier,
)

RDLogger.DisableLog("rdApp.*")


def has_bridged_bond(mol: Chem.Mol) -> bool:
    """
    Returns True if the molecule contains a bridged ring system.
    A bridged system is defined as two rings sharing more than two atoms.
    """
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    # Compare all ring pairs
    for i in range(len(atom_rings)):
        for j in range(i + 1, len(atom_rings)):
            shared_atoms = set(atom_rings[i]) & set(atom_rings[j])
            if len(shared_atoms) > 2:  # more than 2 shared atoms → bridged
                return True
    return False


class RewardScorer:
    def __init__(
        self,
        path_to_mappings: Optional[str] = None,
        reward: Literal["property", "valid_smiles", "MolFilters"] = "property",
        rescale: bool = True,
        parse_whole_completion: bool = False,
        reaction_matrix_path: str | None = "data/rxn_matrix.pkl",
        oracle_kwargs: Dict[str, Any] = {},
        docking_concurrency_per_gpu: int = 2,  # Takes 1Gb*4 on 80Gb we allow 10% of a GPU to keep a margin
    ):
        self.path_to_mappings = path_to_mappings
        self.reward = reward
        self.rescale = rescale
        self.parse_whole_completion = parse_whole_completion
        self.reaction_matrix_path = reaction_matrix_path
        self.oracle_kwargs = oracle_kwargs
        self.docking_concurrency_per_gpu = docking_concurrency_per_gpu

        self.__name__ = f"RewardScorer/{reward}"
        self.remote_tqdm = ray.remote(tqdm_ray.tqdm)

        self._generation_verifier: None | GenerationVerifier = None
        self._mol_prop_verifier: None | MolPropVerifier = None
        self._reaction_verifier: None | ReactionVerifier = None
        self.logger = logging.getLogger("RewardScorer")

        if not ray.is_initialized():
            ray.init()

    @property
    def generation_verifier(self) -> GenerationVerifier:
        if self._generation_verifier is not None:
            return self._generation_verifier
        self._generation_verifier = GenerationVerifier(
            path_to_mappings=self.path_to_mappings,
            reward=self.reward,
            rescale=self.rescale,
            oracle_kwargs=self.oracle_kwargs,
            docking_concurrency_per_gpu=self.docking_concurrency_per_gpu,
        )
        return self._generation_verifier

    @property
    def mol_prop_verifier(self) -> MolPropVerifier:
        if self._mol_prop_verifier is not None:
            return self._mol_prop_verifier
        self._mol_prop_verifier = MolPropVerifier(reward=self.reward)
        return self._mol_prop_verifier

    @property
    def reaction_verifier(self) -> ReactionVerifier:
        if self._reaction_verifier is not None:
            return self._reaction_verifier
        self._reaction_verifier = ReactionVerifier(
            reward=self.reward, rxn_matrix_path=self.reaction_matrix_path
        )
        return self._reaction_verifier

    def get_smiles_from_completion(self, comp: str) -> Tuple[List[str], str]:
        """
        Get smiles from completion
        """
        comp = comp.strip()
        reason: str = ""
        if not self.parse_whole_completion:
            matches = re.findall(
                r"(?:<answer>|<\|answer_start\|>)(.*?)(?:</answer>|<\|answer_end\|>)",
                comp,
                flags=re.DOTALL,
            )
            if len(matches) > 0:
                comp = matches[-1]
            else:
                comp = ""
                reason = "no_answer"
        # Now we identify which elements are possibly SMILES
        # First we split the completion by newlines and spaces
        # Then we filter by removing any string that does not contain "C"
        valid_smiles_pattern = re.compile(r"^[A-Za-z0-9=#:\+\-\[\]\(\)/\\@.%]+$")
        mkd_pattern = re.compile(r"^(\*\*|[-*'])(.+)\1$")

        def filter_smiles(x: str) -> str:
            x = x.replace("<|im_end|>", "")
            if len(x) < 3:
                return ""
            # Check if the string is encapsulated in some kind of markdown
            m = mkd_pattern.match(x)
            x = m.group(2) if m else x
            if "e" in x or len(x) < 3:
                return ""
            if (
                "C" in x
                or x.count("c") > 2
                and valid_smiles_pattern.fullmatch(x) is not None
            ):
                return x
            return ""

        # Finally we remove any string that is not a valid SMILES
        def test_is_valid_batch(smis: list[str]) -> list[bool]:
            RDLogger.DisableLog("rdApp.*")
            results = []
            for smi in smis:
                if len(smi) >= 130:
                    results.append(False)
                    continue
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        results.append(False)
                        continue
                    if has_bridged_bond(mol):  ### WE REMOVE BRIDGED MOLS
                        results.append(False)
                        continue
                    Chem.MolToMolBlock(mol)
                    results.append(True)
                except Exception:
                    results.append(False)
            return results

        s_poss = [filter_smiles(x) for x in re.split("\n| |\\.|\t|:|`|'", comp)]
        s_poss = [x for x in s_poss if x != ""]
        s_poss = list(set(s_poss))

        if len(s_poss) == 0:
            if reason == "":
                reason = "no_smiles"
            return [], reason

        if len(s_poss) > 1:
            reason = "multiple_smiles"
        is_valid: List[bool] = test_is_valid_batch(s_poss)

        s_spl = [x for (x, val) in zip(s_poss, is_valid) if val]
        if s_spl == [] and reason == "":
            reason = "no_valid_smiles"
        elif reason == "":
            reason = s_spl[0]
        return s_spl, reason

    def get_all_completions_smiles(
        self, completions: Any
    ) -> Tuple[List[List[str]], List[str]]:
        smiles = []
        failures = []
        for completion in completions:
            if isinstance(completion, list):
                assert len(completion) == 1
                completion = completion[0]
            if isinstance(completion, dict):
                assert "content" in completion
                completion = completion["content"]
            smi, failure = self.get_smiles_from_completion(completion)
            smiles.append(smi)
            failures.append(failure)
        return smiles, failures

    def _get_generation_score(
        self,
        completions: List[Any],
        metadata: List[Dict[str, Any]],
        debug: bool = False,
        use_pbar: bool = False,
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Get reward for molecular properties
        """
        smiles_per_completion, failures = self.get_all_completions_smiles(completions)
        if (
            self.reward == "valid_smiles"
        ):  # TODO: Currently always return 1 if at least one valid smiles
            return [float(len(smis) > 0) for smis in smiles_per_completion], [
                {"failure": fail} for fail in failures
            ]
        if debug:
            self.generation_verifier.debug = True
        elif self._generation_verifier is not None:
            self.generation_verifier.debug = False
        scores, metadata = self.generation_verifier.get_score(
            smiles_per_completion, metadata
        )
        for meta, fail in zip(metadata, failures):
            meta["smiles_extraction_failure"] = fail
        return scores, metadata

    def _get_prop_pred_score(
        self,
        completions: List[Any],
        metadata: List[Dict[str, Any]],
        debug: bool = False,
        use_pbar: bool = False,
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        return self.mol_prop_verifier.get_score(completions, metadata)

    def _get_reaction_score(
        self,
        completions: List[Any],
        metadata: List[Dict[str, Any]],
        debug: bool = False,
        use_pbar: bool = False,
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        return self.reaction_verifier.get_score(completions, metadata)

    def get_score(
        self,
        completions: List[Any],
        metadata: List[Dict[str, Any]],
        debug: bool = False,
        use_pbar: bool = False,
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        assert len(completions) == len(metadata)
        obj_to_fn: Dict[
            str,
            Callable[
                [List[Any], List[dict[str, Any]], bool, bool],
                Tuple[List[float], List[Dict[str, Any]]],
            ],
        ] = {
            "docking": self._get_generation_score,
            "prop_pred": self._get_prop_pred_score,
            "reaction": self._get_reaction_score,
        }
        idxs: Dict[str, List[int]] = {"docking": [], "prop_pred": [], "reaction": []}
        completions_per_obj: Dict[str, List[str]] = {
            "docking": [],
            "prop_pred": [],
            "reaction": [],
        }
        metadata_per_obj: Dict[str, List[Dict[str, Any]]] = {
            "docking": [],
            "prop_pred": [],
            "reaction": [],
        }
        for i, meta in enumerate(metadata):
            if meta["objectives"][0] in ["regression", "classification"]:
                idxs["prop_pred"].append(i)
                completions_per_obj["prop_pred"].append(completions[i])
                metadata_per_obj["prop_pred"].append(meta)
            elif meta["objectives"][0] in [
                "final_product",
                "reactant",
                "all_reactants",
                "all_reactants_bb_ref",
                "smarts",
                "full_path",
                "full_path_bb_ref",
                "full_path_smarts_ref",
                "full_path_smarts_bb_ref",
                "analog_gen",
            ]:
                idxs["reaction"].append(i)
                completions_per_obj["reaction"].append(completions[i])
                metadata_per_obj["reaction"].append(meta)
            elif meta["objectives"][0] in [
                "maximize",
                "minimize",
                "above",
                "below",
                "equal",
            ]:
                idxs["docking"].append(i)
                completions_per_obj["docking"].append(completions[i])
                metadata_per_obj["docking"].append(meta)
            else:
                raise NotImplementedError(
                    "Unrecognized objective: {}".format(meta["objectives"])
                )
        rewards = [0.0 for _ in range(len(metadata))]
        metadata = [{} for _ in range(len(metadata))]
        for key, fn in obj_to_fn.items():
            if len(completions_per_obj[key]) > 0:
                rewards_obj, metadata_obj = fn(
                    completions_per_obj[key],
                    metadata_per_obj[key],
                    debug,
                    use_pbar,
                )
                for i, r, m in zip(idxs[key], rewards_obj, metadata_obj):
                    rewards[i] = r
                    metadata[i] = m
        self.logger.info(f"Rewards total for given batch: {rewards}")
        return rewards, metadata

    def __call__(
        self,
        completions: List[Any],
        metadata: List[Dict[str, Any]],
        debug: bool = False,
        use_pbar: bool = False,
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Call the scorer to get the rewards.
        """
        return self.get_score(
            completions=completions, metadata=metadata, debug=debug, use_pbar=use_pbar
        )
