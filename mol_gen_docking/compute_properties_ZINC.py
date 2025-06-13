import argparse
from typing import List

import numpy as np
from multiprocess import Pool
from tdc.generation import MolGen
from tqdm import tqdm

from mol_gen_docking.reward.oracle_wrapper import get_oracle
from mol_gen_docking.reward.property_utils.classical_properties import (
    CLASSICAL_PROPERTIES_NAMES,
)


def get_args() -> argparse.Namespace:
    """Get the arguments for the script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default="ZINC",
        help="Name of the dataset to use for the generation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for the property calculation",
    )
    parser.add_argument(
        "--i-start", type=int, default=0, help="Start index for the batch"
    )
    parser.add_argument(
        "--i-end",
        type=int,
        default=10000000,
        help="End index for the batch (0 for all)",
    )
    parser.add_argument(
        "--sub-sample",
        type=int,
        default=None,
        help="Subsample the dataset to this number of samples",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    """Create a dataset with molecules and the property they could be optimizing."""
    args = get_args()
    molgen = MolGen(name=args.name).get_data()
    if args.sub_sample:
        molgen = molgen.sample(args.sub_sample)
    # Limits the dataframe to a multiple of the batch size
    i_end = min(len(molgen), args.i_end)
    molgen = molgen.iloc[args.i_start : i_end - (i_end % args.batch_size)]

    smiles_batches = [
        molgen["smiles"].tolist()[i * args.batch_size : (i + 1) * args.batch_size]
        for i in range(len(molgen) // args.batch_size)
    ]
    for i_name, oracle_name in enumerate(CLASSICAL_PROPERTIES_NAMES.values()):
        oracle_name = CLASSICAL_PROPERTIES_NAMES.get(oracle_name, oracle_name)

        oracle = get_oracle(
            oracle_name,
            property_name_mapping=CLASSICAL_PROPERTIES_NAMES,
            docking_target_list=[],
        )
        pool = Pool(16)

        def get_property(batch: List[str]) -> dict:
            """Get the property for a batch of SMILES strings."""
            props: np.ndarray = oracle(batch)
            return {smi: prop for smi, prop in zip(batch, props)}

        props_pbar = tqdm(
            pool.imap_unordered(get_property, smiles_batches),
            total=len(smiles_batches),
            desc=f"[{i_name}/{len(CLASSICAL_PROPERTIES_NAMES)}] | Calculating {oracle_name}",
        )

        props = {k: v for d in props_pbar for k, v in d.items()}

        molgen[oracle_name] = molgen["smiles"].map(props)

    if args.i_end > molgen.shape[0] and args.i_start == 0:
        path = "mol_gen_docking/reward/oracles/properties.csv"
    else:
        path = (
            f"mol_gen_docking/reward/oracles/properties_{args.i_start}_{args.i_end}.csv"
        )
    molgen.to_csv(path, index=False)
