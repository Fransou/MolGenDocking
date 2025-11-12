import copy
import dataclasses
import itertools
import random
from typing import Any, TypeAlias

import numpy as np
import ray
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from mol_gen_docking.data.reactions.mol import Molecule
from mol_gen_docking.data.reactions.reaction import Reaction
from mol_gen_docking.data.reactions.reaction_matrix import ReactantReactionMatrix

RDLogger.DisableLog("rdApp.*")

_NumReactants: TypeAlias = int
_MolOrRxnIndex: TypeAlias = int
_TokenType: TypeAlias = tuple[_NumReactants, _MolOrRxnIndex]


def pass_filters(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    descriptors = Descriptors.CalcMolDescriptors(mol)
    allowed = {
        "qed": (1, 0.3),
        "ExactMolWt": (600, 0),
        "TPSA": (200, 0),
        "NumHAcceptors": (10, 0),
        "NumHDonors": (10, 0),
        "NumRotatableBonds": (15, 1),
        "RingCount": (7, 0),
    }
    return all(
        [
            descriptors[k] > v_min and descriptors[k] < v_max
            for k, (v_max, v_min) in allowed.items()
        ]
    )


@dataclasses.dataclass
class _Node:
    mol: Molecule
    rxn: Reaction | None
    token: _TokenType
    children: list["_Node"]

    def to_str(self, depth: int) -> str:
        pad = " " * depth * 2
        lines = [f"{pad}{self.mol.smiles}"]
        if self.rxn is not None:
            for c in self.children:
                lines.append(f"{c.to_str(depth + 1)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Node(\n{self.to_str(1)}\n)"


class Stack:
    def __init__(self) -> None:
        super().__init__()
        self._mols: list[Molecule] = []
        self._rxns: list[Reaction | None] = []

    @property
    def mols(self) -> tuple[Molecule, ...]:
        return tuple(self._mols)

    @property
    def rxns(self) -> tuple[Reaction | None, ...]:
        return tuple(self._rxns)

    @staticmethod
    def push_rxn(
        reactants: list[Molecule],
        rxn: Reaction,
        max_num_atoms: int = 80,
    ) -> tuple[
        Molecule, bool, bool
    ]:  # Is the reaction correct and was there a filter issue
        if len(reactants) < rxn.num_reactants:
            return Molecule(""), False, False

        prods: list[Molecule] = []
        for r_ in itertools.permutations(reactants):
            prods += rxn(list(r_))
        if len(prods) == 0:
            return Molecule(""), False, False
        prods_pass_filters = [pass_filters(p.smiles) for p in prods]
        if any(prods_pass_filters):
            prods = [p for p, filt in zip(prods, prods_pass_filters) if filt]

        prods_len = [p.num_atoms for p in prods]
        if any([n <= max_num_atoms for n in prods_len]):
            prods = [p for p, n in zip(prods, prods_len) if n <= max_num_atoms]
        prod: Molecule = random.choice(prods)
        return prod, True, any(prods_pass_filters)

    def add_reactants(self, mol: Molecule) -> None:
        self._mols.append(mol)
        self._rxns.append(None)

    def add_products(self, mol: Molecule, rxn: Reaction) -> None:
        self._mols.append(mol)
        self._rxns.append(rxn)


def select_random_reaction(indices: list[int], matrix: ReactantReactionMatrix) -> int:
    return np.random.choice(indices)  # type: ignore


def create_init_stack(
    matrix: ReactantReactionMatrix,
    weighted_ratio: float = 0.0,
    n_attempts_per_reaction: int = 100,
    n_retry: int = 10,
) -> Stack:
    stack = Stack()
    pass_filter = False
    for _ in range(n_retry):
        rxn_index = select_random_reaction(matrix.seed_reaction_indices, matrix)
        rxn_col = matrix.matrix[:, rxn_index]
        rxn = matrix.reactions[rxn_index]
        for _ in range(n_attempts_per_reaction):
            if rxn.num_reactants == 2:
                m1 = matrix.sample_reactant(
                    np.bitwise_and(rxn_col, 0b01).nonzero()[0],
                )
                m2 = matrix.sample_reactant(
                    np.bitwise_and(rxn_col, 0b10).nonzero()[0],
                )
                reactants = [
                    matrix.reactants[m1],
                    matrix.reactants[m2],
                ]
            elif rxn.num_reactants == 1:
                m = matrix.sample_reactant(rxn_col.nonzero()[0])  # type: ignore
                reactants = [
                    matrix.reactants[m],
                ]
            elif rxn.num_reactants == 3:
                m1 = matrix.sample_reactant(
                    np.bitwise_and(rxn_col, 0b001).nonzero()[0],
                )
                m2 = matrix.sample_reactant(
                    np.bitwise_and(rxn_col, 0b010).nonzero()[0],
                )
                m3 = matrix.sample_reactant(
                    np.bitwise_and(rxn_col, 0b100).nonzero()[0],
                )
                reactants = [
                    matrix.reactants[m1],
                    matrix.reactants[m2],
                    matrix.reactants[m3],
                ]

            prod, success, pass_filter = stack.push_rxn(reactants, rxn)
            if pass_filter:
                for r in reactants:
                    stack.add_reactants(r)
                stack.add_products(prod, rxn)
                break
        if pass_filter:
            break
    if not pass_filter:
        # If no reaction could be applied that passes we return the last tried stack
        for r in reactants:
            stack.add_reactants(r)
        stack.add_products(prod, rxn)
    return stack


def expand_stack_one_reaction(
    stack: Stack,
    matrix: ReactantReactionMatrix,
    forbidden_reactions: list[int] = [],
    max_num_atoms: int = 80,
    n_attempts_per_reaction: int = 100,
) -> tuple[Stack, bool, int, bool]:
    last_product = stack.mols[-1]
    matches = matrix.reactions.match_reactions(last_product)

    if forbidden_reactions is None:
        forbidden_reactions = []
    for forb_index in forbidden_reactions:
        if forb_index in matches:
            del matches[forb_index]

    if len(matches) == 0:
        return stack, False, -1, False
    rxn_index = select_random_reaction(list(matches.keys()), matrix)
    # Position of the last product in the reaction
    reactant_flag = 1 << matches[rxn_index][0]
    rxn_col = matrix.matrix[:, rxn_index]
    for _ in range(n_attempts_per_reaction):
        if np.any(rxn_col >= 4):
            # Case of tri-mol reaction
            all_reactants = 0b111
            remaining_reactants = all_reactants ^ reactant_flag
            reactant_1 = remaining_reactants & 0b001  # Isolate the 001 bit
            reactant_2 = remaining_reactants & 0b010  # Isolate the 010 bit
            reactant_3 = remaining_reactants & 0b100  # Isolate the 100 bit
            valid_reactants = [
                reactant
                for reactant in [reactant_1, reactant_2, reactant_3]
                if reactant != 0
            ]
            s_indices_1 = np.logical_and(
                rxn_col != 0, (rxn_col & valid_reactants[0]) == valid_reactants[0]
            ).nonzero()[0]
            s_indices_2 = np.logical_and(
                rxn_col != 0, (rxn_col & valid_reactants[1]) == valid_reactants[1]
            ).nonzero()[0]
            s_index1 = matrix.sample_reactant(
                s_indices_1,
            )
            s_index2 = matrix.sample_reactant(
                s_indices_2,
            )
            reactants = [
                last_product,
                matrix.reactants[s_index1],
                matrix.reactants[s_index2],
            ]
        else:
            # case of uni- and bi-mol reaction
            s_indices = np.logical_and(
                rxn_col != 0, rxn_col != reactant_flag
            ).nonzero()[0]
            # Case of uni-mol reaction
            if len(s_indices) == 0:
                reactants = [last_product]
            # Case of bi-mol reaction
            else:
                s_index = matrix.sample_reactant(s_indices)
                reactants = [
                    last_product,
                    matrix.reactants[s_index],
                ]
        prod, rxn_success, filter_success = stack.push_rxn(
            reactants,
            matrix.reactions[rxn_index],
            max_num_atoms=max_num_atoms,
        )
        if filter_success:
            for r in reactants[1:]:
                stack.add_reactants(r)
            stack.add_products(prod, matrix.reactions[rxn_index])
            return stack, rxn_success, rxn_index, filter_success
    else:
        for r in reactants[1:]:
            stack.add_reactants(r)
        stack.add_products(prod, matrix.reactions[rxn_index])
    return stack, rxn_success, rxn_index, filter_success


def expand_stack(
    stack: Stack,
    matrix: ReactantReactionMatrix,
    max_num_atoms: int = 80,
    n_retry: int = 10,
    n_attempts_per_reaction: int = 1,
) -> tuple[Stack, bool, int]:
    forbidden_reactions: list[int] = []
    for _ in range(n_retry):
        new_stack, rxn_success, rxn_index, filter_pass = expand_stack_one_reaction(
            copy.deepcopy(stack),
            matrix,
            forbidden_reactions=forbidden_reactions,
            max_num_atoms=max_num_atoms,
            n_attempts_per_reaction=n_attempts_per_reaction,
        )
        if filter_pass:
            return new_stack, rxn_success, rxn_index
        else:
            forbidden_reactions.append(rxn_index)

        if not rxn_success:
            break
    return stack, False, rxn_index


def create_stack(
    matrix: ReactantReactionMatrix,
    max_num_reactions: int = 5,
    max_num_atoms: int = 80,
    init_stack_weighted_ratio: float = 0.0,
    n_attempts_per_reaction: int = 10,
    n_retry: int = 10,
) -> Stack:
    stack = create_init_stack(
        matrix, n_attempts_per_reaction=n_attempts_per_reaction, n_retry=n_retry
    )
    for _ in range(1, max_num_reactions):
        stack, changed, _ = expand_stack(
            stack,
            matrix,
            n_attempts_per_reaction=n_attempts_per_reaction,
            n_retry=n_retry,
        )
        if not changed:
            break
        assert len(stack.mols) > 0
        if stack.mols[-1].num_atoms > max_num_atoms:
            break
    return stack


@ray.remote(num_cpus=1)
def create_stack_ray(
    matrix: Any,
    max_num_reactions: int = 5,
    max_num_atoms: int = 80,
    init_stack_weighted_ratio: float = 0.0,
    n_attempts_per_reaction: int = 100,
    n_retry: int = 10,
    pbar: Any = None,
) -> Stack:
    out = create_stack(
        matrix,
        max_num_reactions=max_num_reactions,
        max_num_atoms=max_num_atoms,
        init_stack_weighted_ratio=init_stack_weighted_ratio,
        n_attempts_per_reaction=n_attempts_per_reaction,
        n_retry=n_retry,
    )
    if pbar is not None:
        pbar.update.remote(1)
    return out
