import logging
import pickle
import re
from functools import reduce
from typing import Any, Dict, List, Literal

import numpy as np
from rdkit import RDLogger

from mol_gen_docking.data.reactions.mol import Molecule
from mol_gen_docking.data.reactions.reaction import Reaction, ReactionContainer
from mol_gen_docking.data.reactions.reaction_matrix import ReactantReactionMatrix

RDLogger.DisableLog("rdApp.*")


class ReactionVerifier:
    def __init__(
        self,
        reward: Literal["property", "valid_smiles", "MolFilters"] = "property",
        rxn_matrix_path: str | None = None,
    ):
        self.rxn_matrix: ReactantReactionMatrix
        if rxn_matrix_path is not None:
            with open(rxn_matrix_path, "rb") as f:
                self.rxn_matrix = pickle.load(f)
        else:
            self.rxn_matrix = ReactantReactionMatrix([], [], np.array([]))
        self.reward = reward
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

    @staticmethod
    def r_ground_truth_mols(mol_y: List[Molecule], mol_label: List[Molecule]) -> float:
        """Returns 0.1 if the molecules are valid, 0.1 + 0.4*iou if the molecules are not in the same order, 1 otherwise."""
        smi_y = [mol.csmiles for mol in mol_y]
        smi_y_true = [mol.csmiles for mol in mol_label]
        intersection = set(smi_y_true).intersection(set(smi_y))
        union = set(smi_y_true).union(set(smi_y))

        return (
            0.1
            + 0.4 * len(intersection) / len(union)
            + 0.5 * float(smi_y == smi_y_true)
        )

    def ground_truth_reward_mol(
        self, completion: str, labels: List[str], impossible: bool
    ) -> float:
        matches = re.findall(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
        if len(matches) != 1:
            return 0.0
        if impossible:
            return float(matches[0] == "impossible")

        mol_label = [Molecule(smi) for smi in labels]
        assert all([m.is_valid for m in mol_label])
        mols = [
            Molecule(smi)
            for smi in matches[0]
            .replace(", ", "and")
            .replace(" + ", "and")
            .strip()
            .split("and")
        ]
        if any([not m.is_valid for m in mols]):
            return 0.0

        return self.r_ground_truth_mols(mols, mol_label)

    def reward_smarts(
        self, completion: str, labels: List[str], impossible: bool
    ) -> float:  # TODO: run the predicted smarts if non-equal
        gt_smarts = labels[0]
        matches = re.findall(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
        if len(matches) != 1:
            return 0.0
        if impossible:
            return float(matches[0] == "impossible")
        return float(matches[0].strip() == gt_smarts)

    def reward_run_path(
        self,
        completion: str,
        label: str,
        building_blocks: List[str],
        smarts: List[str],
        n_steps_max: int,
        impossible: bool,
    ) -> float:
        """
        Returns 0.9 if the synthesis is deemed impossible, and the model retruns 'impossible'.
        Test all steps with all smarts allowed, and returns the reward as :
        sum(n_valid)**2 / n_total**2
        if n_total <= n_steps_max otherwise returns 0.0
        Valid steps are considered as such:
            If the first step, uses building blocks and the reaction appears in the Reaction Matrix, and the product is correct.
            Otherwise, there exist a SMARTS describing the reaction.
        Finally, if the last product is not the target, or some reactants are unknown, return 0.
        """
        self.logger.info("Running reaction verifier on: {}".format(completion))
        matches = re.findall(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
        if len(matches) != 1:
            return 0.0
        if impossible and matches[0] == "impossible":
            self.logger.info("Reaction predicted impossible correctly")
            return 0.9
        # Ensure one reaction per line

        steps: List[str] = matches[0].split("\n")
        steps = [s for s in steps if not s.strip() == ""]
        if not all([len(step.split("->")) == 2 for step in steps]):
            self.logger.info("Template error")
            return 0.0
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
            return 0.0
        n_steps = len(reactants)

        if n_steps > n_steps_max or not any(
            [Molecule(label) == last_p for last_p in products[-1]]
        ):
            self.logger.info("Too many steps for synthesis")
            return 0.0
        if building_blocks == []:
            building_blocks_mol = list(self.rxn_matrix.reactants)
        else:
            building_blocks_mol = [Molecule(smi) for smi in building_blocks]
        for reactant, product in zip(reactants, products):
            for r in reactant:
                if r not in building_blocks_mol:
                    self.logger.info("Using unkown molecule: {}".format(r.smiles))
                    return 0.0
                if product == []:
                    self.logger.info("Missing product")
                    return 0.0
                for p in product:
                    building_blocks_mol.append(p)
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

        return (n_valid / n_steps) ** 2

    def get_score(
        self, completions: List[Any], metadata: List[Dict[str, Any]]
    ) -> List[float]:
        rewards = []
        for meta, answer in zip(metadata, completions):
            objective = meta["objectives"][0]
            impossible: bool = meta["impossible"]
            if objective in self.check_ground_truth_tasks:
                rewards.append(
                    self.ground_truth_reward_mol(
                        answer, meta["target"], impossible=impossible
                    )
                )
            elif objective == "smarts":
                rewards.append(
                    self.reward_smarts(answer, meta["target"], impossible=impossible)
                )
            elif objective in self.run_validation_tasks:
                rewards.append(
                    self.reward_run_path(
                        answer,
                        meta["target"][0],
                        meta["building_blocks"],
                        meta["smarts"],
                        n_steps_max=meta["n_steps_max"],
                        impossible=impossible,
                    )
                )

        if self.reward == "valid_smiles":
            return [float(r > 0.0) for r in rewards]
        return rewards

    def __call__(
        self, completions: List[Any], metadata: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Call the scorer to get the rewards.
        """
        return self.get_score(completions=completions, metadata=metadata)
