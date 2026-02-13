import itertools
from typing import Any, Dict, List, Tuple, TypeAlias

import numpy as np
import ray
import torch
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


def pass_filters_p(smiles: str) -> Tuple[bool, float, Dict[str, float]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, -float("inf"), {}
    descriptors = {k: get_prop(k, mol) for k in PROP_RANGE.keys()}
    filter_pass = all(
        [
            descriptors[k] > v_min and descriptors[k] < v_max
            for k, (v_max, v_min) in PROP_RANGE.items()
        ]
    )
    if filter_pass:
        logprob_l = [
            PROP_TARGET_DISTRIB_FN[k](descriptors[k])
            for k in PROP_TARGET_DISTRIB_FN.keys()
        ]
        logprob = float(sum(logprob_l))
    else:
        logprob = (
            -12
        )  # Arbitrary low logprob for molecules that do not pass the filters
    return filter_pass, logprob, descriptors


class Stack:
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self._mols: List[Molecule] = []
        self._rxns: List[Reaction | None] = []

    @property
    def mols(self) -> tuple[Molecule, ...]:
        return tuple(self._mols)

    @property
    def rxns(self) -> tuple[Reaction | None, ...]:
        return tuple(self._rxns)

    def push_rxn(
        self,
        reactants: List[Molecule] | tuple[Molecule],
        rxn: Reaction,
        max_num_atoms: int = 80,
    ) -> tuple[
        List[Molecule], List[bool], List[float], List[Dict[str, float]]
    ]:  # Returns all valid products with their filter status, logprobs, and properties
        if len(reactants) < rxn.num_reactants:
            return [], [], [], []

        prods: List[Molecule] = []
        for r_ in itertools.permutations(reactants):
            prods.extend(rxn(list(r_)))
        if len(prods) == 0:
            return [], [], [], []
        fp = [pass_filters_p(p.smiles) for p in prods]

        prods_pass_filters = [f for f, _, _ in fp]
        logprobs = [lp for _, lp, _ in fp]
        properties = [prop for _, _, prop in fp]

        # Filter to keep only products that pass filters (if any do)
        if any(prods_pass_filters):
            filtered_data = [
                (p, f, lp, prop)
                for p, f, lp, prop in zip(
                    prods, prods_pass_filters, logprobs, properties
                )
                if f
            ]
            prods = [d[0] for d in filtered_data]
            prods_pass_filters = [d[1] for d in filtered_data]
            logprobs = [d[2] for d in filtered_data]
            properties = [d[3] for d in filtered_data]

        # Filter by max_num_atoms
        prods_len = [p.num_atoms for p in prods]
        if any([n <= max_num_atoms for n in prods_len]):
            filtered_data_max_natoms = [
                (p, f, lp, prop)
                for p, f, lp, prop, n in zip(
                    prods, prods_pass_filters, logprobs, properties, prods_len
                )
                if n <= max_num_atoms
            ]
            prods = [d[0] for d in filtered_data_max_natoms]
            prods_pass_filters = [d[1] for d in filtered_data_max_natoms]
            logprobs = [d[2] for d in filtered_data_max_natoms]
            properties = [d[3] for d in filtered_data_max_natoms]
        return prods, prods_pass_filters, logprobs, properties

    def add_reactants(self, mol: Molecule) -> None:
        self._mols.append(mol)
        self._rxns.append(None)

    def add_products(self, mol: Molecule, rxn: Reaction) -> None:
        self._mols.append(mol)
        self._rxns.append(rxn)

    def add_new_step(
        self,
        reactants: List[Molecule],
        rxn: Reaction,
        prod: Molecule,
    ) -> None:
        for r in reactants:
            self.add_reactants(r)
        self.add_products(prod, rxn)


def select_random_reaction(
    indices: List[int], matrix: ReactantReactionMatrix, k: int = 2
) -> List[int]:
    return np.random.choice(indices, size=k, replace=False).tolist()  # type: ignore


class StackSampler:
    """
    Module responsible for sampling reaction stacks using the reaction
    matrix, using multiple building blocks and reactions.
    """

    def __init__(
        self,
        matrix: ReactantReactionMatrix,
        max_num_reactions: int = 5,
        max_num_atoms: int = 80,
        init_stack_weighted_ratio: float = 0.0,
        n_attempts_per_reaction: int = 10,
        n_retry: int = 10,
        decreas_only_rand: float = 0.5,
    ) -> None:
        self.matrix = matrix
        self.max_num_reactions = max_num_reactions
        self.max_num_atoms = max_num_atoms
        self.n_attempts_per_reaction = n_attempts_per_reaction
        self.n_retry = n_retry
        self.stack = Stack()
        self.i_step = 0

    def init_stack(self) -> List[int]:
        """
        Initialize the sampling by finding the first reaction to
        perform, and adds to the stack a random building block
        matching the reaction.

        Returns:
            List containing the reaction indice to perform at the first step.
        """
        rxn_index = select_random_reaction(
            self.matrix.seed_reaction_indices, self.matrix
        )[0]
        rxn_col = self.matrix.matrix[:, rxn_index]
        rxn = self.matrix.reactions[rxn_index]
        reactants_avail: List[np.ndarray[int]]
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
        all_possible_reactants = np.unique(sum(reactants_avail, start=[]))
        chosen_reactant_idx = np.random.choice(all_possible_reactants, 1)[0]
        chosen_reactant = self.matrix.reactants[chosen_reactant_idx]
        self.stack.add_reactants(chosen_reactant)
        return [rxn_index]

    def sample_reactants_products(
        self,
        last_product: Molecule,
        matches: Dict[int, Tuple[int, ...]],
        rxn_indexes: List[int],
        no_filters: bool = False,
    ) -> List[
        Tuple[
            List[List[Molecule]],
            List[Molecule],
            List[float],
            bool,
            List[Dict[str, float]],
        ]
    ]:
        if no_filters:
            return [
                find_products_reactants(
                    self.stack,
                    self.matrix,
                    last_product=last_product,
                    matches=matches,
                    rxn_index=rxn_index,
                    max_num_atoms=self.max_num_atoms,
                    n_attempts_per_reaction=1,
                    use_filters=not no_filters,
                )
                for rxn_index in rxn_indexes
            ]
        # If we are using filters, we can parallelize the sampling of reactants/products
        # for each reaction index as it is the bottleneck step
        remote_fn = ray.remote(find_products_reactants).options(num_cpus=1)
        return ray.get(
            [
                remote_fn.remote(  # type: ignore
                    self.stack,
                    self.matrix,
                    last_product=last_product,
                    matches=matches,
                    rxn_index=rxn_index,
                    max_num_atoms=self.max_num_atoms,
                    n_attempts_per_reaction=self.n_attempts_per_reaction,
                    use_filters=not no_filters,
                )
                for rxn_index in rxn_indexes
            ]
        )

    def expand_stack(
        self, no_filters: bool, rxn_indexes_constraint: List[int] | None = None
    ) -> bool:
        last_product: Molecule = self.stack.mols[-1]
        matches = self.matrix.reactions.match_reactions(last_product)
        if len(matches) == 0:
            return False
        if rxn_indexes_constraint is not None:
            matches = {
                rxn_idx: matches[rxn_idx]
                for rxn_idx in rxn_indexes_constraint
                if rxn_idx in matches
            }
            if len(matches) == 0:
                return False
        rxn_indexes = select_random_reaction(
            list(matches.keys()),
            self.matrix,
            k=min(self.n_retry, len(matches)),
        )
        react_prods_prob_success_list = self.sample_reactants_products(
            last_product, matches, rxn_indexes, no_filters
        )

        # For each rxn_index, list of reactants explored, list of corresponding products
        rxn_index_to_rp: dict[
            int,
            Tuple[
                List[
                    List[Molecule]
                ],  # Reactants (list of lists to account for multiple reactant combinations)
                List[Molecule],
            ],  # Products
            List[float],  # Logprobs of the products
            List[Dict[str, float]],  # Properties of the products
        ] = {}

        for rxn_index, react_prods_prob_success in zip(
            rxn_indexes, react_prods_prob_success_list
        ):
            reactants, prods, logprob, success, prod_properties = (
                react_prods_prob_success
            )
            if success:
                rxn_index_to_rp[rxn_index] = (
                    reactants,
                    prods,
                    logprob,
                    prod_properties,
                )
        changed = self.choose_reaction_from_candidates(rxn_index_to_rp, no_filters)
        return changed

    def get_probs_from_candidates(
        self,
        rxn_index_to_rp: dict[
            int,
            Tuple[
                List[List[Molecule]],
                List[Molecule],
                List[float],
                List[Dict[str, float]],
            ],
        ],
        no_filters: bool,
        init_step: bool = False,
    ) -> Tuple[List[float], List[int], List[int]]:
        # Step 1: Flatten all candidates
        rxn_idx_flatten: List[int] = []
        idx_flatten: List[int] = []
        logprobs = []
        properties = []
        for rxn_idx in rxn_index_to_rp:
            for i in range(len(rxn_index_to_rp[rxn_idx][0])):
                rxn_idx_flatten.append(rxn_idx)
                idx_flatten.append(i)
                logprobs.append(rxn_index_to_rp[rxn_idx][2][i])
                properties.append(rxn_index_to_rp[rxn_idx][3][i])

        if no_filters:
            probs_array = np.array([1.0 / len(logprobs) for _ in logprobs])
            probs_array = probs_array / probs_array.sum()
            return probs_array, rxn_idx_flatten, idx_flatten

        # Step 2: Only keep top-10 candidates to avoid too much noise from low-probability candidates
        if len(logprobs) > 10 and not init_step:
            logprobs_array = np.array(logprobs)
            top_indices = np.argsort(logprobs_array)[-10:]
            logprobs = [logprobs[i] for i in top_indices]
            rxn_idx_flatten = [rxn_idx_flatten[i] for i in top_indices]
            idx_flatten = [idx_flatten[i] for i in top_indices]

        assert len(logprobs) == len(rxn_idx_flatten)
        logprobs_tensor = torch.tensor(logprobs)
        if len(rxn_index_to_rp) == 0 or logprobs_tensor.sum() == 0:
            return np.array([]), [], []

        if not init_step:
            temperature = 1.0
        else:
            temperature = 0.1  # We want to be more greedy for the first step to ensure exploration
        probs_array = torch.softmax(temperature * logprobs_tensor, dim=0).numpy()
        probs_array = probs_array / probs_array.sum()

        return probs_array, rxn_idx_flatten, idx_flatten

    def choose_reaction_from_candidates(
        self,
        rxn_index_to_rp: dict[
            int,
            tuple[
                List[List[Molecule]],
                List[Molecule],
                List[float],
                List[Dict[str, float]],
            ],
        ],
        no_filters: bool = False,
    ) -> bool:
        init_step = self.i_step == 0

        probs_array, rxn_idx_flatten, idx_flatten = self.get_probs_from_candidates(
            rxn_index_to_rp, no_filters, init_step=init_step
        )
        if len(probs_array) == 0:
            return False
        idx_chosen = np.random.choice(list(range(len(probs_array))), p=probs_array)

        rxn_index = rxn_idx_flatten[idx_chosen]
        rp_idx = idx_flatten[idx_chosen]

        reactants_list, products, *_ = rxn_index_to_rp[rxn_index]

        assert len(reactants_list) > rp_idx

        final_reactant: List[Molecule] = reactants_list[rp_idx]
        final_prod: Molecule = products[rp_idx]
        # Add rxn, reactants and products to the stack
        rxn = self.matrix.reactions[rxn_index]
        if not init_step:
            self.stack.add_new_step(final_reactant[1:], rxn, final_prod)
        else:
            self.stack.add_new_step(final_reactant, rxn, final_prod)
        return True

    def sample_stack(self) -> Stack | None:
        # Pre define a number of reaction steps
        prob_n_step = np.array([i + 1 for i in range(self.max_num_reactions)])
        prob_n_step = prob_n_step / prob_n_step.sum()
        n_steps = np.random.choice(
            [i + 1 for i in range(self.max_num_reactions)],
            p=prob_n_step,
        )
        first_rxn_idxs = self.init_stack()

        if n_steps == 1:
            n_steps_no_filters = 0
        else:
            n_steps_no_filters = np.random.randint(0, (n_steps + 1) // 2)

        for _ in range(0, n_steps):
            changed = self.expand_stack(
                no_filters=self.i_step <= n_steps_no_filters,
                rxn_indexes_constraint=first_rxn_idxs if self.i_step == 0 else None,
            )
            self.i_step += 1
            if not changed:
                return None
            assert len(self.stack.mols) > 0
        if (
            self.stack.mols[-1].num_atoms > self.max_num_atoms
            and not self.i_step <= n_steps_no_filters
        ):
            return None
        if not pass_filters_p(self.stack.mols[-1].smiles)[0]:
            return None
        return self.stack


def sample_from_cart_product(n: int, *lists: List[Any]) -> List[Tuple[Any, ...]]:
    cart_product_size = np.prod([len(lis) for lis in lists])
    n = min(n, cart_product_size)
    if n == 0:
        return []
    elif len(lists) == 1:
        return [(item,) for item in lists[0]]
    else:
        chosen_samples: set[Tuple[Any, ...]] = set()
        idx_try = 0
        while len(chosen_samples) < n and idx_try < n * 10:  # Add a max number of tries
            sample = tuple(np.random.choice(lis, 1)[0] for lis in lists)
            chosen_samples.add(sample)
            idx_try += 1
        return list(chosen_samples)


def find_products_reactants(
    stack: Stack,
    matrix: ReactantReactionMatrix,
    last_product: Molecule,
    matches: Dict[int, Tuple[int, ...]],
    rxn_index: int,
    max_num_atoms: int = 80,
    n_attempts_per_reaction: int = 100,
    use_filters: bool = True,
) -> Tuple[
    List[List[Molecule]], List[Molecule], List[float], bool, List[Dict[str, float]]
]:
    found_reactants: List[
        List[Molecule]
    ] = []  # List of lists of reactants (possibly repeating)
    found_products: List[Molecule] = []
    logprobs: List[float] = []
    properties: List[Dict[str, float]] = []
    # Position of the last product in the reaction
    reactant_flag = 1 << matches[rxn_index][0]
    rxn_col = matrix.matrix[:, rxn_index]
    reactants_avail: List[np.ndarray[int]]

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

    poss_reactants: List[List[int]]
    if len(reactants_avail) == 0:
        poss_reactants = [[-1]]
    else:
        n_to_sample = min(
            n_attempts_per_reaction, *[len(reactants) for reactants in reactants_avail]
        )
        if n_to_sample < n_attempts_per_reaction // 8:
            chosen_reactants_comb = sample_from_cart_product(
                n_attempts_per_reaction, *reactants_avail
            )
            poss_reactants = [
                [chos_reac[j] for chos_reac in chosen_reactants_comb]
                for j in range(len(reactants_avail))
            ]
        else:
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
        prods, rxn_successes, logprob_list, property_list = stack.push_rxn(
            reactants,
            matrix.reactions[rxn_index],
            max_num_atoms=max_num_atoms,
        )
        # Add all valid products from this reactant combination
        for prod, rxn_success, logprob, prop in zip(
            prods, rxn_successes, logprob_list, property_list
        ):
            found_reactants.append(reactants)
            found_products.append(prod)
            logprobs.append(logprob)
            properties.append(prop)
    if found_reactants == []:
        rxn_success = False
    else:
        rxn_success = True
    assert len(found_reactants) == len(found_products)
    assert len(found_reactants) == len(logprobs)
    return found_reactants, found_products, logprobs, rxn_success, properties


@ray.remote(num_cpus=1)
def create_stack_ray(
    matrix: Any,
    max_num_reactions: int = 5,
    max_num_atoms: int = 80,
    init_stack_weighted_ratio: float = 0.0,
    n_attempts_per_reaction: int = 100,
    n_retry: int = 10,
    pbar: Any = None,
) -> Stack | None:
    stack_sampler = StackSampler(
        matrix,
        max_num_reactions=max_num_reactions,
        max_num_atoms=max_num_atoms,
        init_stack_weighted_ratio=init_stack_weighted_ratio,
        n_attempts_per_reaction=n_attempts_per_reaction,
        n_retry=n_retry,
    )
    out = stack_sampler.sample_stack()
    if pbar is not None:
        pbar.update.remote(1)
    return out
