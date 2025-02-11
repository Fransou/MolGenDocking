from mol_rewards import OracleWrapper
from tdc import Oracle
import argparse

smis = [
    "Cc1cccc(C)c1N1CCC(Oc2ccc(N3N=C(C(F)(F)F)[C@@H](C)[C@@H]3CC(=O)O)cc2)CC1",
    "O=C(NCCCc1ccccc1)NCCc1cccs1",
    "CC[C@H](C)[C@H](NC(=O)[C@H](CC(N)=O)NC(=O)[C@@H](NC(=O)[C@H](Cc1cnc[nH]1)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H](NC(=O)[C@H](CCCCN)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](NC(=O)CN)C(C)C)[C@@H](C)O)C(C)C)C(=O)N[C@@H](CC(C)C)C(=O)NC(CCC(C)C)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCSC)C(=O)NCC(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CO)C(=O)N[C@H](C(=O)O)[C@@H](C)O",
    "CCCCOc1ccccc1C[C@H]1COC(=O)[C@@H]1Cc1ccc(Cl)c(Cl)c1",
    "CO[C@]1(C)C[C@H](O[C@H]2[C@H](C)[C@@H](O[C@@H]3O[C@H](C)C[C@H](N(C)C)[C@H]3O)[C@](C)(O)C[C@@H](C)CN(C)C(=O)C[C@H](Cc3ccc(NC(=O)[C@H](Cc4ccccc4Cl)NC(=O)Cc4ccccc4)cc3)NC(=O)[C@H](C(C)C)NC(=O)[C@@H]2C)O[C@@H](C)[C@@H]1O",
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
    oracle.assign_evaluator(
        Oracle(name=args.oracle),
    )
    rewards = oracle(smis)
    print(rewards)




