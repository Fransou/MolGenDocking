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
    VerifierInputBatchModel,
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
    def __init__(
        self,
        verifier_config: ReactionVerifierConfigModel,
    ):
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
        """Returns 0.1 if the molecules are valid, 0.1 + 0.4*iou if the molecules are not in the same order, 1 otherwise."""
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
        matches = re.findall(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
        self.logger.info(f"Matches for ground truth mols: {matches}")
        if len(matches) != 1:
            return 0.0
        if impossible:
            return float(matches[0] == "impossible")

        mol_label = [Molecule(smi) for smi in labels]
        if not all([m.is_valid for m in mol_label]):
            self.logger.error("Invalid ground truth molecule")
            return 0.0
        mols = [
            Molecule(smi)
            for smi in matches[0]
            .replace(", ", " and ")
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
        gt_smarts = labels[0]
        matches = re.findall(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
        if len(matches) != 1:
            return 0.0, {"Reactants_contained": False, "Products_contained": False}
        if impossible:
            return float(matches[0] == "impossible"), {
                "Reactants_contained": True,
                "Products_contained": True,
            }
        if matches[0].strip() == gt_smarts:
            return 1.0, {"Reactants_contained": True, "Products_contained": True}
        self.logger.info(
            f"Proposed SMARTS: {matches[0].strip()} | GT SMARTS: {gt_smarts}, checking reaction..."
        )
        try:
            rxnB = Reaction(matches[0].strip())
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
                f"Error in reaction SMARTS parsing: {e} (proposed: {matches[0]} | gt: {gt_smarts})"
            )
            return 0.0, {"Reactants_contained": False, "Products_contained": False}

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
        """
        Returns 0.9 if the synthesis is deemed impossible, and the model returns 'impossible'.
        Test all steps with all smarts allowed, and returns the reward as :
        sum(n_valid)**2 / n_total**2
        if n_total <= n_steps_max otherwise returns 0.0
        Valid steps are considered as such:
            If the first step, uses building blocks and the reaction appears in the Reaction Matrix, and the product is correct.
            Otherwise, there exist a SMARTS describing the reaction.
        Finally, if the last product is not the target, or some reactants are unknown, return the original reward * the tanimoto similarity**3
        """
        matches = re.findall(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
        if len(matches) != 1:
            return 0.0, {
                "valid": 0.0,
                "correct_product": 0.0,
                "correct_reactant": False,
            }
        if impossible and matches[0] == "impossible":
            self.logger.info("Reaction predicted impossible correctly")
            return 0.9, {
                "valid": 1.0,
                "correct_product": 0.0,
                "correct_reactant": True,
            }
        # Ensure one reaction per line
        self.logger.info("Running reaction verifier on: {}".format(matches[0]))
        steps: List[str] = matches[0].split("\n")
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
        if any([not r.is_valid for r in sum(reactants, [])]) or any(
            [not p.is_valid for p in sum(products, [])]
        ):
            invalid_reactants = [r.smiles for r in sum(reactants, []) if not r.is_valid]
            invalid_products = [p.smiles for p in sum(products, []) if not p.is_valid]
            self.logger.info(
                f"Invalid molecule found in synthesis path : reactants {invalid_reactants}, products {invalid_products}"
            )
            return 0.0, {
                "valid": 0.0,
                "correct_product": 0.0,
                "correct_reactant": False,
            }
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
                    fp_p = AllChem.GetMorganFingerprintAsBitVect(
                        p._rdmol, 2, nBits=2048
                    )
                    all_sims.append(DataStructs.TanimotoSimilarity(label_fp, fp_p))
                reward_mult[i] = max(all_sims) ** 3

        if building_blocks == []:
            building_blocks_mol = list(self.rxn_matrix.reactants)
        else:
            building_blocks_mol = [Molecule(smi) for smi in building_blocks]

        error = False
        for reactant, product in zip(reactants, products):
            for r in reactant:
                if r not in building_blocks_mol:
                    self.logger.info("Using unkown molecule: {}".format(r.smiles))
                    error = True
                    break
            if error:
                break
            if product == []:
                self.logger.info("Missing product")
                error = True
                break
            for p in product:
                building_blocks_mol.append(p)
        if error:
            return 0.0, {
                "valid": 0.0,
                "correct_product": reward_mult[-1],
                "correct_reactant": False,
            }
        reactions: ReactionContainer
        if smarts == []:
            reactions = self.rxn_matrix.reactions
        else:
            reactions = ReactionContainer([Reaction(sma) for sma in smarts])

        n_valid = 0
        for i_reac, (reactant, product) in enumerate(zip(reactants, products)):
            id_poss_smarts: List[List[int]] = []
            for r in reactant:
                id_poss_smarts.append(list(reactions.match_reactions(r).keys()))
            possible = reduce(
                np.intersect1d, tuple(np.array(ids) for ids in id_poss_smarts)
            )
            possible = [
                p for p in possible if reactions[p].num_reactants == len(reactant)
            ]  # type:ignore
            if len(possible) == 0:
                break
            all_found: bool
            for id_reaction in possible:
                run_product = reactions[id_reaction](reactant)
                for p in product:
                    all_found = True
                    if p not in run_product:
                        all_found = False
                        break
                if all_found:
                    self.logger.info(
                        "Found correct smart: {}".format(reactions[id_reaction].smarts)
                    )
                    break
            if not all_found:
                self.logger.info(
                    "No reaction found for: {} -> {}".format(
                        [r.smiles for r in reactant], [p.smiles for p in product]
                    )
                )
                break
            else:
                n_valid += 1
        if n_valid < n_steps:
            return reward_mult[n_valid - 1] * (n_valid / n_steps) ** 2, {
                "valid": n_valid / n_steps,
                "correct_product": reward_mult[n_valid - 1],
                "correct_reactant": True,
            }

        return reward_mult[n_valid - 1], {
            "valid": 1.0,
            "correct_product": reward_mult[n_valid - 1],
            "correct_reactant": True,
        }

    def get_score(
        self, inputs: VerifierInputBatchModel
    ) -> List[ReactionVerifierOutputModel]:
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
