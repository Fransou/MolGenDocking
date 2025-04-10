import os
from typing import List

from tdc.metadata import docking_target_info


class PyscreenerOracle:
    def __init__(
        self,
        target_name: str,
        software_class: str = "vina",
        ncpu: int = 4,
        **kwargs,
    ):
        self.name = "[DOCKING]-" + target_name
        if not os.path.isfile(target_name):
            pdbid = target_name.split("_")[0]
            receptor_pdb_file = "./oracle/" + pdbid + ".pdbqt"
            box_center = docking_target_info[pdbid]["center"]
            box_size = docking_target_info[pdbid]["size"]
        else:
            raise NotImplementedError
        try:
            import ray

            try:
                ray.init()
            except Exception as e:
                print(e)
                ray.shutdown()
                ray.init()
            import pyscreener as ps
        except Exception as e:
            print(e)
            raise ImportError(
                "Please install PyScreener following guidance in https://github.com/coleygroup/pyscreener"
            )

        try:
            metadata = ps.build_metadata(software_class, metadata={"exhaustivness": 1})
        except Exception as e:
            print(e)
            raise ValueError(
                'The value of software_class is not implemented. Currently available:["vina", "qvina", "smina", "psovina", "dock", "dock6", "ucsfdock"]'
            )
        print(metadata)
        self.scorer = ps.virtual_screen(
            software_class,
            [receptor_pdb_file],
            box_center,
            box_size,
            metadata,
            ncpu=ncpu,
        )

    def __call__(self, test_smiles: str | List[str], error_value=None):
        final_score = self.scorer(test_smiles)
        if isinstance(test_smiles, str):
            return list(final_score)[0]
        else:  ## list
            # dict: {'O=C(/C=C/c1ccc([N+](=O)[O-])o1)c1ccc(-c2ccccc2)cc1': -9.9, 'CCOc1cc(/C=C/C(=O)C(=Cc2ccc(O)c(OC)c2)C(=O)/C=C/c2ccc(O)c(OCC)c2)ccc1O': -9.1}
            # return [list(i.values())[0] for i in final_score]
            score_lst = []
            for smiles in test_smiles:
                score = final_score[smiles]
                if score is None:
                    score = error_value
                score_lst.append(score)
            return score_lst
