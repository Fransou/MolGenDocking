"""Reaction verifier for chemical reaction and retro-synthesis tasks.

This module provides the ReactionVerifier class which computes rewards for
chemical reaction tasks including retro-synthesis planning, SMARTS prediction,
and reaction product verification.
"""

import logging
import pickle
import re
from functools import reduce
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
from rdkit import DataStructs, RDLogger
from rdkit.Chem import AllChem

from mol_gen_docking.data.reactions.mol import Molecule
from mol_gen_docking.data.reactions.reaction import Reaction, ReactionContainer
from mol_gen_docking.data.reactions.reaction_matrix import ReactantReactionMatrix
from mol_gen_docking.reward.verifiers.abstract_verifier import (
    Verifier,
)
from mol_gen_docking.reward.verifiers.abstract_verifier_pydantic_model import (
    BatchVerifiersInputModel,
)
from mol_gen_docking.reward.verifiers.reaction_reward.input_metadata import (
    ReactionVerifierInputMetadataModel,
)
from mol_gen_docking.reward.verifiers.reaction_reward.reaction_verifier_pydantic_model import (
    ReactionVerifierConfigModel,
    ReactionVerifierMetadataModel,
    ReactionVerifierOutputModel,
)

RDLogger.DisableLog("rdApp.*")


class ReactionVerifier(Verifier):
    """Verifier for chemical reaction and retro-synthesis tasks.

    This verifier computes rewards for various reaction-related tasks including:
    - Final product prediction
    - Reactant identification
    - SMARTS pattern prediction
    - Full retro-synthesis path validation

    The verifier uses a reaction matrix to validate synthesis steps and supports
    both binary and Tanimoto-based reward computation.

    Attributes:
        verifier_config: Configuration for the reaction verifier.
        rxn_matrix: Pre-loaded reaction matrix for validation.
        check_ground_truth_tasks: List of task types requiring ground truth comparison.
        run_validation_tasks: List of task types requiring path validation.
        logger: Logger instance for the verifier.
    """

    def __init__(
        self,
        verifier_config: ReactionVerifierConfigModel,
    ):
        """Initialize the ReactionVerifier.

        Args:
            verifier_config: Configuration containing reaction matrix path
                and reward type settings.
        """
        super().__init__()
        self.verifier_config = verifier_config
        self.rxn_matrix: ReactantReactionMatrix
        with open(verifier_config.reaction_matrix_path, "rb") as f:
            self.rxn_matrix = pickle.load(f)
        self.check_ground_truth_tasks = [
            "final_product",
            "reactant",
            "all_reactants",
            "all_reactants_bb_ref",
        ]
        self.run_validation_tasks = [
            "full_path",
            "full_path_bb_ref",
            "full_path_smarts_ref",
            "full_path_smarts_bb_ref",
        ]
        self.logger = logging.getLogger("ReactionVerifier")

    def r_ground_truth_mols(
        self, mol_y: List[Molecule], mol_label: List[Molecule]
    ) -> float:
        """Compute reward for molecule prediction against ground truth.

        The reward is computed as:
        - 0.1 base reward if molecules are valid
        - +0.4 * IoU for partial overlap
        - +0.5 if molecules match exactly in order

        Args:
            mol_y: List of predicted molecules.
            mol_label: List of ground truth molecules.

        Returns:
            Reward value between 0.1 and 1.0.
        """
        self.logger.info(
            f"Computed molecules: {[mol.smiles for mol in mol_y]} vs labels: {[mol.smiles for mol in mol_label]}"
        )
        smi_y = [mol.csmiles for mol in mol_y]
        smi_y_true = [mol.csmiles for mol in mol_label]
        intersection = set(smi_y_true).intersection(set(smi_y))
        union = set(smi_y_true).union(set(smi_y))
        reward = (
            0.1
            + 0.4 * len(intersection) / len(union)
            + 0.5 * float(smi_y == smi_y_true)
        )
        self.logger.info(f"Intersection: {intersection}, reward : {reward}")
        return reward

    def ground_truth_reward_mol(
        self, completion: str, labels: List[str], impossible: bool
    ) -> float:
        """Compute reward for molecule prediction tasks.

        Args:
            completion: Model completion containing the answer.
            labels: List of ground truth SMILES strings.
            impossible: If True, expects "impossible" as the answer.

        Returns:
            Reward value between 0.0 and 1.0.
        """
        matches = re.findall(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
        self.logger.info(f"Matches for ground truth mols: {matches}")
        if len(matches) != 1:
            return 0.0
        match: str = (
            matches[-1].split("<answer>")[-1].split("<|answer_start|>")[-1]
        )  # In case of nested tags
        if impossible:
            return float(match == "impossible")
        mol_label = [Molecule(smi) for smi in labels]
        if not all([m.is_valid for m in mol_label]):
            self.logger.error("Invalid ground truth molecule")
            return 0.0
        mols = [
            Molecule(smi)
            for smi in match.replace(", ", " and ")
            .replace(" + ", " and ")
            .strip()
            .split("and")
        ]

        if any([not m.is_valid for m in mols]):
            self.logger.info("Invalid molecule found in prediction")
            return 0.0

        return self.r_ground_truth_mols(mols, mol_label)

    def reward_smarts(
        self,
        completion: str,
        labels: List[str],
        reactants: List[str],
        product: str,
        impossible: bool,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute reward for SMARTS prediction tasks.

        Args:
            completion: Model completion containing the SMARTS answer.
            labels: List containing the ground truth SMARTS string.
            reactants: List of reactant SMILES strings.
            product: Expected product SMILES string.
            impossible: If True, expects "impossible" as the answer.

        Returns:
            Tuple of (reward, metadata_dict) where metadata contains
            'Reactants_contained' and 'Products_contained' flags.
        """
        gt_smarts = labels[0]
        matches = re.findall(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
        if len(matches) != 1:
            return 0.0, {"Reactants_contained": False, "Products_contained": False}
        match: str = (
            matches[-1].split("<answer>")[-1].split("<|answer_start|>")[-1]
        )  # In case of nested tags
        if impossible:
            return float(match == "impossible"), {
                "Reactants_contained": True,
                "Products_contained": True,
            }
        if match.strip() == gt_smarts:
            return 1.0, {"Reactants_contained": True, "Products_contained": True}
        self.logger.info(
            f"Proposed SMARTS: {match.strip()} | GT SMARTS: {gt_smarts}, checking reaction..."
        )
        try:
            rxnB = Reaction(match.strip())
            p = rxnB([Molecule(r) for r in reactants])
            reward = 0.0
            if product in p:
                reward = 0.1
            return reward, {
                "Reactants_contained": True,
                "Products_contained": reward == 0.1,
            }
        except Exception as e:
            self.logger.info(
                f"Error in reaction SMARTS parsing: {e} (proposed: {match} | gt: {gt_smarts})"
            )
            return 0.0, {"Reactants_contained": False, "Products_contained": False}

    def _find_reaction_smarts(
        self,
        reactants_step: List[Molecule],
        products_step: List[Molecule],
        allowed_smarts: ReactionContainer,
    ) -> List[Reaction]:
        """Find valid reaction SMARTS that can produce products from reactants.

        Args:
            reactants_step: List of reactant molecules for this step.
            products_step: List of expected product molecules.
            allowed_smarts: Container of allowed reaction SMARTS patterns.

        Returns:
            List of Reaction objects that successfully produce the expected products.
        """
        found_reactions: List[Reaction] = []
        id_poss_smarts: List[List[int]] = []
        for r in reactants_step:
            id_poss_smarts.append(list(allowed_smarts.match_reactions(r).keys()))
        possible = reduce(
            np.intersect1d, tuple(np.array(ids) for ids in id_poss_smarts)
        )
        possible = [  # type:ignore
            p
            for p in possible
            if allowed_smarts[p].num_reactants == len(reactants_step)
        ]
        for id_reaction in possible:
            run_product = allowed_smarts[id_reaction](reactants_step)
            all_found: bool = True
            for p in products_step:
                if p not in run_product:
                    all_found = False
                    break
            if all_found:
                found_reactions.append(allowed_smarts[id_reaction])
        return found_reactions

    def _check_valid_step(
        self,
        reactants_step: List[Molecule],
        products_step: List[Molecule],
        possible_reactants: List[Molecule],
        allowed_smarts: ReactionContainer,
    ) -> Tuple[bool, str]:
        """Check if a synthesis step is valid.

        Validates that:
        1. All reactants and products are valid molecules
        2. All reactants are in building blocks or previous products
        3. At least one reaction can produce the products from the reactants

        Args:
            reactants_step: List of reactant molecules for this step.
            products_step: List of expected product molecules.
            possible_reactants: List of valid starting materials (building blocks + previous products).
            allowed_smarts: Container of allowed reaction SMARTS patterns.

        Returns:
            Tuple of (is_valid, fail_reason) where fail_reason is empty string if valid,
            or one of "reactants", "products", "reaction" indicating what failed.
        """
        # 1. Check that all reactants and products are valid molecules
        if not all([r.is_valid for r in reactants_step]):
            self.logger.info(
                "Reactants not valid in {}".format([r.smiles for r in reactants_step])
            )
            return False, "reactants"
        if not all([p.is_valid for p in products_step]):
            self.logger.info(
                "Products not valid in {}".format([p.smiles for p in products_step])
            )
            return False, "products"
        # 2. Check that all reactants are in building blocks or previous products
        for r in reactants_step:
            if r not in possible_reactants:
                self.logger.info(
                    "Reactant {} not in building blocks or previous products".format(
                        r.smiles
                    )
                )
                return False, "reactants"
        if products_step == []:
            self.logger.info("No products in step")
            return False, "products"

        # 3. Check that there is at least one reaction that can produce the products from the reactants
        found_reactions = self._find_reaction_smarts(
            reactants_step, products_step, allowed_smarts
        )
        if len(found_reactions) == 0:
            self.logger.info(
                "No reaction found for step: {} -> {}".format(
                    [r.smiles for r in reactants_step],
                    [p.smiles for p in products_step],
                )
            )
            return False, "reaction"
        # Log success
        self.logger.info(
            "Found valid reaction for step: {} -> {}".format(
                [r.smiles for r in reactants_step],
                [p.smiles for p in products_step],
            )
        )
        return True, ""

    def reward_run_path(
        self,
        completion: str,
        label: str,
        building_blocks: List[str],
        smarts: List[str],
        n_steps_max: int,
        impossible: bool,
        reward_type: Literal["binary", "tanimoto"] = "binary",
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute reward for retro-synthesis path validation.

        Validates a multi-step synthesis path by checking:
        1. All reactants are valid building blocks or previous products
        2. Each reaction step has a valid SMARTS pattern
        3. The final product matches the target (exactly or by Tanimoto similarity)

        Args:
            completion: Model completion containing the synthesis path.
            label: Target product SMILES string.
            building_blocks: List of valid starting building block SMILES.
            smarts: List of allowed SMARTS patterns (empty = use reaction matrix).
            n_steps_max: Maximum allowed number of synthesis steps.
            impossible: If True, expects "impossible" as the answer.
            reward_type: "binary" for exact match or "tanimoto" for similarity-based.

        Returns:
            Tuple of (reward, metadata_dict) containing validation results.

        Notes:
            - Path format: "reactant1 + reactant2 -> product" per line
            - Reward is scaled by (n_valid/n_total)^2 for partial paths
            - Tanimoto similarity is cubed (sim^3) for reward scaling
        """
        matches = re.findall(
            r"(?:<answer>|<\|answer_start\|>)((?:(?!<answer>|<\|answer_start\|>).)*?)(?:</answer>|<\|answer_end\|>)",
            completion,
            flags=re.DOTALL,
        )
        if len(matches) == 0:
            return 0.0, {
                "valid": 0.0,
                "correct_product": 0.0,
                "correct_reactant": False,
            }
        match: str = (
            matches[-1].split("<answer>")[-1].split("<|answer_start|>")[-1]
        )  # In case of nested tags
        steps: List[str] = match.split("\n")
        if impossible and match == "impossible":
            self.logger.info("Reaction predicted impossible correctly")
            return 0.9, {
                "valid": 1.0,
                "correct_product": 0.0,
                "correct_reactant": True,
            }
        # Ensure one reaction per line
        self.logger.info("Running reaction verifier on: {}".format(match))
        steps = [step for step in steps if not step.strip() == "" and "->" in step]
        if len(steps) == 0 or not all([len(step.split("->")) == 2 for step in steps]):
            self.logger.info("Template error")
            return 0.0, {
                "valid": 0.0,
                "correct_product": 0.0,
                "correct_reactant": False,
            }
        basic_smiles_pattern = re.compile(r"^[A-Za-z0-9=#:\+\-\[\]\(\)/\\@.%]+$")
        if not all(
            all(
                [
                    basic_smiles_pattern.match(smi.strip())
                    for smi in re.split(r"\+|->", step)
                ]
            )
            for step in steps
        ):
            self.logger.info("Invalid SMILES pattern found in reaction steps")
            return 0.0, {
                "valid": 0.0,
                "correct_product": 0.0,
                "correct_reactant": False,
            }
        reactants = [
            [Molecule(smi.strip()) for smi in step.split("->")[0].split(" + ")]
            for step in steps
        ]
        products = [
            [Molecule(smi.strip()) for smi in step.split("->")[1].split(" + ")]
            for step in steps
        ]
        n_steps = len(reactants)
        label_mol = Molecule(label)
        if n_steps > n_steps_max:
            self.logger.info("Too many steps for synthesis")
            return 0.0, {
                "valid": 0.0,
                "correct_product": False,
                "correct_reactant": False,
            }
        reward_mult: List[float] = [1.0 for _ in products]
        if reward_type == "binary" and not any(
            [label_mol == last_p for last_p in products[-1]]
        ):
            self.logger.info("Product not found")
            return 0.0, {
                "valid": 0.0,
                "correct_last_product": 0.0,
                "correct_reactant": False,
            }
        elif reward_type == "tanimoto":
            # Compute the tanimoto similarity between the label and products at each step
            label_fp = AllChem.GetMorganFingerprintAsBitVect(
                label_mol._rdmol, 2, nBits=2048
            )
            for i, product in enumerate(products):
                all_sims = []
                for p in product:
                    if p.is_valid:
                        fp_p = AllChem.GetMorganFingerprintAsBitVect(
                            p._rdmol, 2, nBits=2048
                        )
                        all_sims.append(DataStructs.TanimotoSimilarity(label_fp, fp_p))
                    else:
                        all_sims.append(0.0)
                reward_mult[i] = max(all_sims) ** 3

        reactions: ReactionContainer
        if smarts == []:
            reactions = self.rxn_matrix.reactions
        else:
            reactions = ReactionContainer([Reaction(sma) for sma in smarts])

        if building_blocks == []:
            building_blocks_mol = list(self.rxn_matrix.reactants)
        else:
            building_blocks_mol = [Molecule(smi) for smi in building_blocks]

        n_valid = 0
        fail_reason = ""
        for i_reac, (reactant, product) in enumerate(zip(reactants, products)):
            is_valid, fail_reason = self._check_valid_step(
                reactant,
                product,
                building_blocks_mol + [p for step in products[:i_reac] for p in step],
                reactions,
            )
            if not is_valid:
                self.logger.info(f"Invalid step at index {i_reac} due to {fail_reason}")
                break
            else:
                n_valid += 1
        if n_valid < n_steps:
            return reward_mult[n_valid - 1] * (n_valid / n_steps) ** 2, {
                "valid": n_valid / n_steps,
                "correct_product": reward_mult[n_valid - 1],
                "correct_reactant": fail_reason != "reactants",
            }

        return reward_mult[n_valid - 1], {
            "valid": 1.0,
            "correct_product": reward_mult[n_valid - 1],
            "correct_reactant": True,
        }

    def get_score(
        self, inputs: BatchVerifiersInputModel
    ) -> List[ReactionVerifierOutputModel]:
        """Compute reaction rewards for a batch of completions.

        This method routes each completion to the appropriate reward function
        based on the objective type specified in the metadata.

        Args:
            inputs: Batch of completions and metadata for verification.

        Returns:
            List of ReactionVerifierOutputModel containing rewards and metadata.

        Notes:
            - Ground truth tasks: final_product, reactant, all_reactants
            - SMARTS tasks: smarts prediction with reaction validation
            - Path tasks: full_path with step-by-step validation
        """
        completions = inputs.completions
        assert all(
            isinstance(meta, ReactionVerifierInputMetadataModel)
            for meta in inputs.metadatas
        )
        metadatas: List[ReactionVerifierInputMetadataModel] = inputs.metadatas

        output_models = []
        for answer, meta in zip(completions, metadatas):
            objective = meta.objectives[0]
            impossible: bool = meta.impossible
            reward = 0.0
            reward_metadata = {
                "valid": 0.0,
                "correct_product": 0.0,
                "correct_reactant": False,
            }
            if objective in self.check_ground_truth_tasks:
                reward = self.ground_truth_reward_mol(
                    answer, meta.target, impossible=impossible
                )
                reward_metadata = {
                    "valid": float(reward > 0.0),
                    "correct_product": reward > 0.0,
                    "correct_reactant": reward > 0.0,
                }
            elif objective == "smarts":
                assert len(meta.reactants) > 0, (
                    "Reactants must be provided for SMARTS objective"
                )
                assert len(meta.products) > 0, (
                    "Product must be provided for SMARTS objective"
                )
                reward, raw_metadata = self.reward_smarts(
                    answer,
                    meta.target,
                    meta.reactants[0],
                    meta.products[0],
                    impossible=impossible,
                )
                reward_metadata = {
                    "valid": reward,
                    "correct_product": raw_metadata.get("Products_contained", False),
                    "correct_reactant": raw_metadata.get("Reactants_contained", False),
                }
            elif objective in self.run_validation_tasks:
                assert len(meta.target) > 0, (
                    "Target must be provided for run validation tasks"
                )
                reward, raw_metadata = self.reward_run_path(
                    answer,
                    meta.target[0],
                    meta.building_blocks if meta.building_blocks else [],
                    meta.smarts if meta.smarts else [],
                    n_steps_max=meta.n_steps_max,
                    impossible=False,  # We always try to generate a compound
                    reward_type=self.verifier_config.reaction_reward_type,
                )
                reward_metadata = raw_metadata
            elif objective == "analog_gen":
                assert len(meta.target) > 0, (
                    "Target must be provided for analog generation task"
                )
                reward, raw_metadata = self.reward_run_path(
                    answer,
                    meta.target[0],
                    meta.building_blocks if meta.building_blocks else [],
                    meta.smarts if meta.smarts else [],
                    n_steps_max=meta.n_steps_max,
                    reward_type="tanimoto",
                    impossible=False,
                )
                reward_metadata = raw_metadata

            if self.verifier_config.reward == "valid_smiles":
                reward = float(reward > 0.0)

            # Create the output model
            output_model = ReactionVerifierOutputModel(
                reward=reward,
                verifier_metadata=ReactionVerifierMetadataModel(
                    valid=reward_metadata["valid"],
                    correct_product=reward_metadata["correct_product"],
                    correct_reactant=reward_metadata["correct_reactant"],
                ),
            )
            output_models.append(output_model)

        return output_models
