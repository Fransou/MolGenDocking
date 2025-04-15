import os
from typing import List
import ray

import pyscreener as ps

from tdc.metadata import docking_target_info

from pyscreener.docking.vina.utils import Software



class PyscreenerOracle:
    def __init__(
        self,
        target_name: str,
        software_class: str = "qvina",
        ncpu: int = 4,
        **kwargs,
    ):
        if software_class not in [
            "vina",
            "qvina",
            "smina",
            "psovina",
            "dock",
            "dock6",
            "ucsfdock",
        ]:
            raise ValueError(
                'The value of software_class is not implemented. Currently available:["vina", "qvina", "smina", "psovina", "dock", "dock6", "ucsfdock"]'
            )

        self.name = "[DOCKING]-" + target_name
        if not os.path.isfile(target_name):
            pdbid = target_name.split("_")[0]
            receptor_pdb_file = "./oracle/" + pdbid + ".pdbqt"
            box_center = docking_target_info[pdbid]["center"]
            box_size = docking_target_info[pdbid]["size"]
        else:
            raise NotImplementedError

        if not ray.is_initialized():
            ray.init()
        
        metadata = ps.build_metadata(software_class, metadata={"exhaustiveness": 8})
        
        if software_class == "qvina":
          metadata.software = Software.QVINA
        
        self.scorer = ps.virtual_screen(
            software_class,
            [receptor_pdb_file],
            box_center,
            box_size,
            metadata,
            ncpu=ncpu,
        )

    def __call__(self, test_smiles: str | List[str], error_value=None):
        if isinstance(test_smiles, str):
            final_score = self.scorer(test_smiles)
            return list(final_score)[0]
        else:
            final_score = self.scorer(test_smiles)
            score_lst = []
            for i, smiles in enumerate(test_smiles):
                score = final_score[i]
                if score is None:
                    score = error_value
                score_lst.append(score)
            return score_lst
