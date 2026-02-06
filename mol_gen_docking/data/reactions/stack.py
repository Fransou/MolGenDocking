import itertools
from typing import Any, TypeAlias

import numpy as np
import ray
from rdkit import Chem, RDLogger

from mol_gen_docking.data.reactions.mol import Molecule
from mol_gen_docking.data.reactions.reaction import Reaction
from mol_gen_docking.data.reactions.reaction_matrix import ReactantReactionMatrix
from mol_gen_docking.data.reactions.utils import (
    PROP_RANGE,
    PROP_TARGET_DISTRIB_FN,
    get_prop,
)

RDLogger.DisableLog("rdApp.*")

_NumReactants: TypeAlias = int
_MolOrRxnIndex: TypeAlias = int
_TokenType: TypeAlias = tuple[_NumReactants, _MolOrRxnIndex]


def pass_filters_p(smiles: str) -> tuple[bool, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, -float("inf")
    descriptors = {k: get_prop(k, mol) for k in PROP_RANGE.keys()}
    filter_pass = all(
        [
            descriptors[k] > v_min and descriptors[k] < v_max
            for k, (v_max, v_min) in PROP_RANGE.items()
        ]
    )
    if filter_pass:
        logprob = [
            PROP_TARGET_DISTRIB_FN[k](descriptors[k])
            for k in PROP_TARGET_DISTRIB_FN.keys()
        ]
        prob = float(np.exp(sum(logprob)))
    else:
        prob = 0
    return filter_pass, prob


class Stack:
    def __init__(self) -> None:
        super().__init__()
        self._mols: list[Molecule] = []
        self._rxns: list[Reaction | None] = []
        self.last_prod_prob: float = 0.0

    @property
    def mols(self) -> tuple[Molecule, ...]:
        return tuple(self._mols)

    @property
    def rxns(self) -> tuple[Reaction | None, ...]:
        return tuple(self._rxns)

    @staticmethod
    def push_rxn(
        reactants: list[Molecule] | tuple[Molecule],
        rxn: Reaction,
        max_num_atoms: int = 80,
    ) -> tuple[
        Molecule, bool, float
    ]:  # Is the reaction correct and was there a filter issue
        if len(reactants) < rxn.num_reactants:
            return Molecule(""), False, False

        prods: list[Molecule] = []
        for r_ in itertools.permutations(reactants):
            prods.extend(rxn(list(r_)))
        if len(prods) == 0:
            return Molecule(""), False, -float("inf")
        fp = [pass_filters_p(p.smiles) for p in prods]

        prods_pass_filters = [f for f, _ in fp]
        probs = [lp for _, lp in fp]

        if any(prods_pass_filters):
            prods = [p for p, filt in zip(prods, prods_pass_filters) if filt]
            probs = [lp for lp, filt in zip(probs, prods_pass_filters) if filt]
        prods_len = [p.num_atoms for p in prods]
        if any([n <= max_num_atoms for n in prods_len]):
            prods = [p for p, n in zip(prods, prods_len) if n <= max_num_atoms]
            probs = [lp for lp, n in zip(probs, prods_len) if n <= max_num_atoms]
        proba = np.array(probs)
        if proba.sum() == 0:
            proba = proba + 1.0
            proba = proba / proba.sum()
        else:
            proba = proba / (proba.sum())
        idx: int = np.random.choice(list(range(len(prods))), p=proba.tolist())

        prod = prods[idx]
        prob = probs[idx]
        return prod, any(prods_pass_filters), prob

    def add_reactants(self, mol: Molecule) -> None:
        self._mols.append(mol)
        self._rxns.append(None)

    def add_products(self, mol: Molecule, rxn: Reaction) -> None:
        self._mols.append(mol)
        self._rxns.append(rxn)

    def add_new_step(
        self, reactants: list[Molecule], rxn: Reaction, prod: Molecule, prod_prob: float
    ) -> None:
        for r in reactants:
            self.add_reactants(r)
        self.add_products(prod, rxn)
        self.last_prod_prob = prod_prob


def select_random_reaction(
    indices: list[int], matrix: ReactantReactionMatrix, k: int = 2
) -> list[int]:
    return np.random.choice(indices, size=k, replace=False).tolist()  # type: ignore


def create_init_stack(
    matrix: ReactantReactionMatrix,
    weighted_ratio: float = 0.0,
    n_attempts_per_reaction: int = 100,
) -> Stack:
    stack = Stack()
    rxn_index = select_random_reaction(matrix.seed_reaction_indices, matrix)[0]
    rxn_index_to_rp: dict[int, tuple[list[list[Molecule]], list[Molecule]]] = {}
    probs: list[float] = []
    rxn_col = matrix.matrix[:, rxn_index]
    rxn = matrix.reactions[rxn_index]
    reactants_avail: list[np.ndarray[int]]
    if rxn.num_reactants == 1:
        reactants_avail = [rxn_col.nonzero()[0]]
    elif rxn.num_reactants == 2:
        reactants_avail = [
            np.bitwise_and(rxn_col, 0b01).nonzero()[0],
            np.bitwise_and(rxn_col, 0b10).nonzero()[0],
        ]
    elif rxn.num_reactants == 3:
        reactants_avail = [
            np.bitwise_and(rxn_col, 0b001).nonzero()[0],
            np.bitwise_and(rxn_col, 0b010).nonzero()[0],
            np.bitwise_and(rxn_col, 0b100).nonzero()[0],
        ]
    n_to_sample = min(
        n_attempts_per_reaction, *[len(reactants) for reactants in reactants_avail]
    )
    all_reactants: list[list[Molecule]] = [
        np.random.choice(r, n_to_sample, replace=False).tolist()
        for r in reactants_avail
    ]
    for reactants_idx in zip(*all_reactants):
        reactants = [matrix.reactants[idx] for idx in reactants_idx]
        prod, success, prob = stack.push_rxn(reactants, rxn)
        if success:
            if rxn_index not in rxn_index_to_rp:
                rxn_index_to_rp[rxn_index] = ([], [])
            rxn_index_to_rp[rxn_index][0].append(reactants)
            rxn_index_to_rp[rxn_index][1].append(prod)
            probs.append(prob)

    success = choose_reaction_from_candidates(
        rxn_index_to_rp, probs, stack, matrix, init_step=True
    )
    if not success:
        # If no reaction could be applied that passes we return the last tried stack
        stack.add_new_step(reactants, rxn, prod, prob)

    return stack


@ray.remote(num_cpus=1)
def find_products_reactants(
    stack: Stack,
    matrix: ReactantReactionMatrix,
    last_product: Molecule,
    matches: dict[int, tuple[int, ...]],
    rxn_index: int,
    max_num_atoms: int = 80,
    n_attempts_per_reaction: int = 100,
) -> tuple[list[list[Molecule]], list[Molecule], list[float], bool]:
    found_reactants: list[list[Molecule]] = []
    found_products: list[Molecule] = []
    probs: list[float] = []
    # Position of the last product in the reaction
    reactant_flag = 1 << matches[rxn_index][0]
    rxn_col = matrix.matrix[:, rxn_index]
    reactants_avail: list[np.ndarray[int]]

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
        reactants_avail = [s_indices_1, s_indices_2]
    else:
        # case of uni- and bi-mol reaction
        s_indices = np.logical_and(rxn_col != 0, rxn_col != reactant_flag).nonzero()[0]
        # Case of uni-mol reaction
        if len(s_indices) == 0:
            reactants_avail = []
        # Case of bi-mol reaction
        else:
            reactants_avail = [s_indices]

    poss_reactants: list[list[int]]
    if len(reactants_avail) == 0:
        poss_reactants = [[-1]]
    else:
        n_to_sample = min(
            n_attempts_per_reaction, *[len(reactants) for reactants in reactants_avail]
        )
        poss_reactants = [
            np.random.choice(r, n_to_sample, replace=False).tolist()
            for r in reactants_avail
        ]

    for reactants_idx in zip(*poss_reactants):
        if reactants_idx == (-1,):
            reactants = [last_product]
        else:
            reactants = [last_product] + [
                matrix.reactants[idx] for idx in reactants_idx
            ]
        prod, rxn_success, prob = stack.push_rxn(
            reactants,
            matrix.reactions[rxn_index],
            max_num_atoms=max_num_atoms,
        )
        if rxn_success:
            found_reactants.append(reactants)
            found_products.append(prod)
            probs.append(prob)
    if found_reactants == []:
        rxn_success = False
    assert len(found_reactants) == len(found_products)
    assert len(found_reactants) == len(probs)
    return found_reactants, found_products, probs, rxn_success


def expand_stack(
    stack: Stack,
    matrix: ReactantReactionMatrix,
    max_num_atoms: int = 80,
    n_retry: int = 10,
    n_attempts_per_reaction: int = 1,
) -> tuple[Stack, bool]:
    last_product = stack.mols[-1]
    matches = matrix.reactions.match_reactions(last_product)
    if len(matches) == 0:
        return stack, False
    rxn_indexes = select_random_reaction(
        list(matches.keys()), matrix, k=min(n_retry, len(matches))
    )
    rxn_index_to_rp: dict[int, tuple[list[list[Molecule]], list[Molecule]]] = {}

    probs: list[float] = []

    # Get all possible products at this stage for n_retry reactions
    react_prods_prob_success = ray.get(
        [
            find_products_reactants.remote(  # type: ignore
                stack,
                matrix,
                last_product=last_product,
                matches=matches,
                rxn_index=rxn_index,
                max_num_atoms=max_num_atoms,
                n_attempts_per_reaction=n_attempts_per_reaction,
            )
            for rxn_index in rxn_indexes
        ]
    )
    for rxn_index, (reactants, prods, prob, success) in zip(
        rxn_indexes, react_prods_prob_success
    ):
        if success:
            rxn_index_to_rp[rxn_index] = (reactants, prods)
            for i, p in enumerate(prob):
                probs.append(p)
                assert i < len(rxn_index_to_rp[rxn_index][0])

    changed = choose_reaction_from_candidates(rxn_index_to_rp, probs, stack, matrix)

    return stack, changed


def choose_reaction_from_candidates(
    rxn_index_to_rp: dict[int, tuple[list[list[Molecule]], list[Molecule]]],
    probs: list[float],
    stack: Stack,
    matrix: ReactantReactionMatrix,
    init_step: bool = False,
) -> bool:
    # Add token corresponding to last prod but with 50% chance
    probs.append(stack.last_prod_prob / 2)
    probs_array = np.array(probs)
    if len(rxn_index_to_rp) == 0 or probs_array.sum() == 0:
        return False

    probs_array = probs_array / probs_array.sum()

    rxn_idx_flatten: list[int] = []
    idx_flatten: list[int] = []
    for rxn_idx in rxn_index_to_rp:
        for i in range(len(rxn_index_to_rp[rxn_idx][0])):
            rxn_idx_flatten.append(rxn_idx)
            idx_flatten.append(i)

    idx_chosen = np.random.choice(list(range(len(rxn_idx_flatten) + 1)), p=probs_array)
    if idx_chosen == probs_array.shape[0] - 1:  # We stop the reaction here
        return False

    prob = probs[idx_chosen]
    rxn_index = rxn_idx_flatten[idx_chosen]
    rp_idx = idx_flatten[idx_chosen]

    reactants_list, products = rxn_index_to_rp[rxn_index]

    assert len(reactants_list) > rp_idx

    final_reactant: list[Molecule] = reactants_list[rp_idx]
    final_prod: Molecule = products[rp_idx]

    # Add rxn, reactants and products to the stack
    rxn = matrix.reactions[rxn_index]
    if not init_step:
        stack.add_new_step(final_reactant[1:], rxn, final_prod, prob)
    else:
        stack.add_new_step(final_reactant, rxn, final_prod, prob)
    return True


def create_stack(
    matrix: ReactantReactionMatrix,
    max_num_reactions: int = 5,
    max_num_atoms: int = 80,
    init_stack_weighted_ratio: float = 0.0,
    n_attempts_per_reaction: int = 10,
    n_retry: int = 10,
) -> Stack:
    stack = create_init_stack(matrix, n_attempts_per_reaction=n_attempts_per_reaction)
    for _ in range(1, max_num_reactions):
        stack, changed = expand_stack(
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
