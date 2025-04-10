import argparse

from multiprocessing import Pool
from typing import List
from tqdm import tqdm

from tdc.generation import MolGen

from mol_gen_docking.reward.oracles import PROPERTIES_NAMES_SIMPLE, get_oracle


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
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for the property calculation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers for the property calculation",
    )
    parser.add_argument("--sub-sample", type=int, default=None, help="Sub-sample size")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    """Create a dataset with molecules and the property they could be optimizing."""
    args = get_args()
    molgen = MolGen(name=args.name).get_data()
    if args.sub_sample:
        molgen = molgen.sample(args.sub_sample)
    # Limits the dataframe to a multiple of the batch size
    molgen = molgen.iloc[: len(molgen) - (len(molgen) % args.batch_size)]

    smiles_batches = [
        molgen["smiles"].tolist()[i * args.batch_size : (i + 1) * args.batch_size]
        for i in range(len(molgen) // args.batch_size)
    ]

    for i_name, oracle_name in enumerate(PROPERTIES_NAMES_SIMPLE.values()):
        oracle_name = PROPERTIES_NAMES_SIMPLE.get(oracle_name, oracle_name)
        oracle = get_oracle(oracle_name)

        p_bar = tqdm(
            total=len(molgen),
            desc=f"[{i_name}/{len(PROPERTIES_NAMES_SIMPLE)}] Calculating {oracle_name}",
        )

        def get_property(batch: List[str]) -> dict:
            """Get the property for a batch of SMILES strings."""
            props = oracle(batch)
            return {smi: prop for smi, prop in zip(batch, props)}

        pool = Pool(args.num_workers)
        props_pbar = tqdm(
            pool.imap_unordered(get_property, smiles_batches),
            total=len(smiles_batches),
            desc=f"[{i_name}/{len(PROPERTIES_NAMES_SIMPLE)}] | Calculating {oracle_name}",
        )

        props = {k: v for d in props_pbar for k, v in d.items()}
        molgen[oracle_name] = molgen["smiles"].map(props)

    print(molgen.sample(10))

    path = "mol_gen_docking/reward/oracles/propeties.csv"
    molgen.to_csv(path, index=False)
