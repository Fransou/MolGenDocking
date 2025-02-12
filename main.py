import json
import argparse
from tdc import Oracle
from mol_rewards import OracleWrapper


smis = [
    "O=C(NCCCc1ccccc1)NCCc1cccs1",
    "CCCCOc1ccccc1C[C@H]1COC(=O)[C@@H]1Cc1ccc(Cl)c(Cl)c1",
    "O=c1[nH]nc2n1-c1ccc(OCc3ccc(F)cc3)cc1CCC2",
    "CCN1CCN(c2ccc(C3=CC4(CCc5cc(O)ccc54)c4ccc(O)cc43)cc2)CC1",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles', type=str, default=smis)
    parser.add_argument("--oracle", type=str, default='JNK3')
    parser.add_argument("--max-oracle-calls", type=int, default=100)
    parser.add_argument("--freq-log", type=int, default=10)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()


    oracle = OracleWrapper(args)
    if not args.oracle.endswith('docking'):
        oracle.assign_evaluator(
            Oracle(name=args.oracle),
        )
    else:
        oracle.assign_evaluator(
            Oracle(name=args.oracle, ncpus=1),
        )
    rewards = oracle(smis)
    print(rewards)




