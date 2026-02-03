import argparse
from typing import List, Tuple

import numpy as np

from mol_gen_docking.data.reactions.mol import Molecule
from mol_gen_docking.data.reactions.reaction_matrix import ReactantReactionMatrix
from mol_gen_docking.data.reactions.utils import PROMPT_TEMPLATES


class ReactionTaskSampler:
    def __init__(
        self,
        args: argparse.Namespace,
        reaction_matrix: ReactantReactionMatrix,
    ) -> None:
        self.args = args
        self.reaction_matrix = reaction_matrix
        self.all_reactants = reaction_matrix.reactants
        self.all_reactantas_csmi = [r.csmiles for r in self.all_reactants]
        self.all_reactions = reaction_matrix.reactions

    def sample_reactants_nreacs(self, n: int) -> List[Molecule]:
        n_reacts = (self.reaction_matrix.matrix > 0).sum(1)
        n_reacts = n_reacts / n_reacts.sum()
        idx_selected = np.random.choice(
            list(range(len(self.all_reactants))), n, p=n_reacts, replace=False
        )
        return [self.all_reactants[i] for i in idx_selected]

    def get_eval_obj_and_label(self, product: str) -> Tuple[str, int, List[str]]:
        prop = "full_path_bb_ref" if self.args.n_bb_max > 0 else "full_path"
        label = [product]
        idx_chosen = 0
        return prop, idx_chosen, label

    def get_training_obj_and_label(
        self, reactants: List[List[str]], products: List[str], or_smarts: List[str]
    ) -> Tuple[str, int, List[str]]:
        prop = np.random.choice(list(PROMPT_TEMPLATES.keys()), p=self.args.proba_obj)
        idx_chosen: int = 0  # If objective is SMARTS we select a random step, if it is reactants we select the first step
        if (
            "full_path" not in prop and prop != "final_product"
        ):  # Reactant or smarts objective
            if prop == "smarts":
                idx_chosen = int(np.random.choice(list(range(len(reactants)))))
            reactants = [reactants[idx_chosen]]
            products = [products[idx_chosen]]
            or_smarts = [or_smarts[idx_chosen]]
        label: List[str] = ["n/a"]
        if prop == "final_product" or prop.startswith("full_path"):
            label = [products[-1]]
        elif prop == "reactant":
            label = [np.random.choice(reactants[0])]
        elif prop == "smarts":
            label = [or_smarts[0]]
        elif prop in ["all_reactants", "all_reactants_bb_ref"]:
            label = reactants[0]
        return prop, idx_chosen, label

    def get_building_blocks_ref(
        self, prop: str, reactants: List[List[str]], target_smi: str
    ) -> Tuple[List[str], List[str]]:
        original_building_blocks = []
        for l_reactants in reactants:
            for smi in l_reactants:
                mol = Molecule(smi)
                if mol.csmiles in self.all_reactantas_csmi:
                    original_building_blocks.append(smi)
        assert len(original_building_blocks) > 0 or reactants == []
        if prop not in [
            "all_reactants_bb_ref",
            "full_path_bb_ref",
            "full_path_smarts_bb_ref",
        ]:
            return [], original_building_blocks

        if prop == "all_reactants_bb_ref":
            # We sample less building blocks
            n_bb_max = np.random.choice(
                [
                    self.args.n_bb_max,
                    self.args.n_bb_max // 2,
                    self.args.n_bb_max // 4,
                    self.args.n_bb_max // 8,
                ],
                p=[0.125, 0.125, 0.25, 0.5],
            )
        else:
            n_bb_max = np.random.choice(
                [self.args.n_bb_max, self.args.n_bb_max // 2, self.args.n_bb_max // 4],
                p=[0.5, 0.3, 0.2],
            )
        bb = self.get_building_blocks_tanim_sim(
            target_smi,
            n_bb_max,
            [Molecule(smi) for smi in original_building_blocks],
        )

        if prop == "all_reactants_bb_ref":
            # We make sure to always have the original building blocks
            bb = original_building_blocks + bb
            bb = bb[:n_bb_max]
            bb = list(set(bb))
        np.random.shuffle(bb)

        return bb, original_building_blocks

    def get_building_blocks_tanim_sim(
        self,
        target_smi: str,
        n_bb_max: int,
        or_bb: List[Molecule],
    ) -> List[str]:
        # Find the most similar building blocks
        target_mol = Molecule(target_smi)

        ### We only add similar building blocks that can be part of the last reaction
        # Find all reactions where the target can be obtained
        poss_reactions = self.all_reactions.match_product_reactions(target_mol)
        if len(poss_reactions) == 0:
            idx_poss_last = np.array(list(range(len(self.all_reactants))))
        else:
            # Only keep the reactants that can participate in one of these reactions
            idx_poss_last = np.where(
                (self.reaction_matrix.matrix[:, np.array(poss_reactions)] > 0).any(1)
            )[0]
        reactants_for_tanimoto_sim = [self.all_reactants[i] for i in idx_poss_last]
        # Compute Tanimoto similarity
        tanimoto_sim = target_mol.tanimoto_similarity(reactants_for_tanimoto_sim)
        sorted_idx = np.argsort(tanimoto_sim)[::-1]

        idx_or_bb = np.where(
            np.isin(
                [r.csmiles for r in reactants_for_tanimoto_sim],
                [r.csmiles for r in or_bb],
            )
        )[0]
        print(np.where(np.isin(sorted_idx, idx_or_bb))[0])

        # Get the most similar building blocks for 50% of n_bb_max
        building_blocks = [
            reactants_for_tanimoto_sim[i] for i in sorted_idx[: n_bb_max // 2]
        ]
        building_blocks = list(
            set(building_blocks + self.sample_reactants_nreacs(n_bb_max // 2))
        )

        return [bb.smiles for bb in building_blocks]

        return building_blocks

    def get_smarts(self, or_smarts: List[str], prop: str) -> List[str]:
        if prop not in ["full_path_smarts_ref", "full_path_smarts_bb_ref"]:
            return or_smarts
        print(len(or_smarts))
        n_smarts_max = np.random.randint(
            low=len(or_smarts), high=self.args.n_smarts_max - len(or_smarts)
        )
        idx_random_reactions = np.random.choice(
            list(range(len(self.all_reactions))), n_smarts_max, replace=False
        )
        smarts = [
            self.all_reactions._reactions[i].smarts for i in idx_random_reactions
        ] + or_smarts
        smarts = list(set(smarts))
        np.random.shuffle(smarts)
        return smarts

    def sample(
        self,
        reactants: List[List[str]],
        products: List[str],
        or_smarts: List[str],
    ) -> Tuple[str, int, List[str], List[str], List[str], List[str]]:
        prop, idx_chosen, label = self.get_training_obj_and_label(
            reactants, products, or_smarts
        )
        bb, or_bb = self.get_building_blocks_ref(prop, reactants, products[-1])
        smarts = self.get_smarts(or_smarts, prop)
        return prop, idx_chosen, label, bb, smarts, or_bb

    def sample_eval(
        self,
        product: str,
    ) -> Tuple[str, int, List[str], List[str]]:
        prop, idx_chosen, label = self.get_eval_obj_and_label(product)
        bb, or_bb = self.get_building_blocks_ref(prop, [], product)
        return prop, idx_chosen, label, bb
