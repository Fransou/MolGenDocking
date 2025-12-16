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


def template_to_regex(template: str) -> str:
    """
    Convert a single template string into a regex pattern.
    """
    # Escape special regex characters that might appear in text
    pattern = re.escape(template)
    # Replace escaped {prop} and {val} with regex groups
    pattern = pattern.replace(r"\{prop\}", r"(?P<prop>.+?)")
    if r"\{val\}" in pattern:
        pattern = pattern.replace(r"\{val\}", r"(?P<val>[-+]?\d+\.?\d*)")
    else:
        pattern += r"(?:[.!?]+|$)"
    return pattern


def generate_regex_patterns(templates: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    """
    Converts OBJECTIVES_TEMPLATES to a list of regex patterns with associated objective type.
    Returns: List of (regex_pattern, objective_type)
    """
    pattern_list = []
    for obj_type, template_list in templates.items():
        for tmpl in template_list:
            regex = template_to_regex(tmpl)
            pattern_list.append((regex, obj_type))
    return pattern_list


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
        gpu_utilization_gpu_docking: float = 0.10,  # Takes 1Gb*4 on 80Gb we allow 10% of a GPU to keep a margin
    ):
        self.gpu_utilization_gpu_docking = gpu_utilization_gpu_docking
        self.reward = reward
        self.parse_whole_completion = parse_whole_completion
        self.__name__ = f"RewardScorer/{reward}"
        self.remote_tqdm = ray.remote(tqdm_ray.tqdm)

        self.generation_verifier = GenerationVerifier(
            path_to_mappings=path_to_mappings,
            reward=reward,
            rescale=rescale,
            oracle_kwargs=oracle_kwargs,
            gpu_utilization_gpu_docking=gpu_utilization_gpu_docking,
        )
        self.mol_prop_verifier = MolPropVerifier(reward=reward)
        self.reaction_verifier = ReactionVerifier(
            reward=reward, rxn_matrix_path=reaction_matrix_path
        )

        if not ray.is_initialized():
            ray.init()

    def get_smiles_from_completion(self, comp: str) -> List[str]:
        """
        Get smiles from completion
        """
        if not self.parse_whole_completion:
            matches = re.findall(r"<answer>(.*?)</answer>", comp, flags=re.DOTALL)
            if len(matches) > 0:
                comp = matches[-1]
            else:
                comp = ""
        # Now we identify which elements are possibly SMILES
        # First we split the completion by newlines and spaces
        re.split("\n| |.|\t|:", comp)
        # Then we filter by removing any string that does not contain "C"
        valid_smiles_pattern = re.compile(r"^[A-Za-z0-9=#:\+\-\[\]\(\)/\\@.%]+$")

        def filter_smiles(x: str) -> bool:
            if "e" in x or len(x) < 3:
                return False
            if "C" in x or x.count("c") > 2:
                return valid_smiles_pattern.fullmatch(x) is not None
            return False

        # Finally we remove any string that is not a valid SMILES
        @ray.remote(num_cpus=1)
        def test_is_valid_batch(smis: list[str]) -> list[bool]:
            RDLogger.DisableLog("rdApp.*")
            results = []
            for smi in smis:
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

        s_poss = [x for x in comp.split() if filter_smiles(x)]
        chunk_size = 4
        tasks = [
            test_is_valid_batch.remote(s_poss[i : i + chunk_size])
            for i in range(0, len(s_poss), chunk_size)
        ]
        is_valid: List[bool] = sum(ray.get(tasks), [])
        s_spl = [
            x for (x, val) in zip(s_poss, is_valid) if val
        ]  ### TODO: Maybe do not return the mean if mutliple molecules
        return s_spl

    def get_all_completions_smiles(self, completions: Any) -> List[List[str]]:
        smiles = []
        for completion in completions:
            if isinstance(completion, list):
                assert len(completion) == 1
                completion = completion[0]
            if isinstance(completion, dict):
                assert "content" in completion
                completion = completion["content"]

            smiles.append(self.get_smiles_from_completion(completion))
        return smiles

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
        smiles_per_completion = self.get_all_completions_smiles(completions)
        if (
            self.reward == "valid_smiles"
        ):  # TODO: Currently always return 1 if at least one valid smiles
            return [float(len(smis) > 0) for smis in smiles_per_completion], [
                {} for _ in smiles_per_completion
            ]
        if debug:
            self.generation_verifier.debug = True
        else:
            self.generation_verifier.debug = False
        scores, metadata = self.generation_verifier.get_score(
            smiles_per_completion, metadata
        )
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
            rewards_obj, metadata_obj = fn(
                completions_per_obj[key],
                metadata_per_obj[key],
                debug,
                use_pbar,
            )
            for i, r, m in zip(idxs[key], rewards_obj, metadata_obj):
                rewards[i] = r
                metadata[i] = m
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
