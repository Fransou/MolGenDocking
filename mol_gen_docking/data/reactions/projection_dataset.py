from itertools import permutations, product
from typing import Iterator

from torch.utils.data import IterableDataset

from mol_gen_docking.data.reactions.mol import Molecule
from mol_gen_docking.data.reactions.reaction import Reaction
from mol_gen_docking.data.reactions.reaction_matrix import ReactantReactionMatrix
from mol_gen_docking.data.reactions.stack import create_stack_step_by_step


class TextualProjectionDataset(IterableDataset[ReactantReactionMatrix]):
    def __init__(
        self,
        reaction_matrix: ReactantReactionMatrix,
        max_num_atoms: int = 80,
        max_smiles_len: int = 192,
        max_num_reactions: int = 5,
        init_stack_weighted_ratio: float = 0.0,
        virtual_length: int = 65536,
    ) -> None:
        super().__init__()
        self._reaction_matrix = reaction_matrix
        self._max_num_atoms = max_num_atoms
        self._max_smiles_len = max_smiles_len
        self._max_num_reactions = max_num_reactions
        self._init_stack_weighted_ratio = init_stack_weighted_ratio
        self._virtual_length = virtual_length

    def __len__(self) -> int:
        return self._virtual_length

    def __iter__(self) -> Iterator[tuple[list[list[str]], list[str], list[str]]]:
        for _ in range(self._virtual_length):
            for stack in create_stack_step_by_step(
                self._reaction_matrix,
                max_num_reactions=self._max_num_reactions,
                max_num_atoms=self._max_num_atoms,
                init_stack_weighted_ratio=self._init_stack_weighted_ratio,
            ):
                rxn_smarts = [
                    rxn.smarts for rxn in stack.rxns if rxn is not None
                ]  # TODO find how to handle this better
                is_product = [rxn is not None for rxn in stack.rxns]
                reactants: list[list[Molecule]] = [[]]
                products: list[Molecule] = []
                for mol, is_prod in zip(stack.mols, is_product):
                    if not is_prod:
                        reactants[-1].append(mol)
                    else:
                        products.append(mol)
                        reactants.append([mol])
            reactants = reactants[:-1]
            assert len(reactants) == len(products)
            assert len(reactants) == len(rxn_smarts)

            reactants_smiles = [
                self.find_mol_order(r, p, smarts)
                for r, p, smarts in zip(reactants, products, rxn_smarts)
            ]
            product_smiles = [p.smiles for p in products]

            yield reactants_smiles, product_smiles, rxn_smarts

    @staticmethod
    def find_mol_order(
        reactants: list[Molecule], prod: Molecule, smarts: str
    ) -> list[str]:
        reaction = Reaction(smarts)
        for reactants_order in permutations(reactants):
            prods = reaction(reactants_order)
            if prod in prods:
                return [r.smiles for r in reactants_order]
        raise ValueError(f"{product} not in {reactants}")
