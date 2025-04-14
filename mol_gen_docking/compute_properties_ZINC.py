import argparse

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
        if "docking" in oracle_name:
            oracle = get_oracle(oracle_name, n_cpus=64)
        else:
            oracle = get_oracle(oracle_name)

        p_bar = tqdm(
            total=len(molgen),
            desc=f"[{i_name}/{len(PROPERTIES_NAMES_SIMPLE)}] Calculating {oracle_name}",
        )

        props = {}
        for batch in smiles_batches:
            props_batch = oracle(batch)
            for smiles, prop in zip(batch, props_batch):
                props[smiles] = prop
            p_bar.update(len(batch))

        molgen[oracle_name] = molgen["smiles"].map(props)

    print(molgen.sample(10))
    if args.i_end > molgen.shape[0] and args.i_start == 0:
        path = "mol_gen_docking/reward/oracles/properties.csv"
    else:
        path = (
            f"mol_gen_docking/reward/oracles/properties_{args.i_start}_{args.i_end}.csv"
        )
    molgen.to_csv(path, index=False)
