import os
import warnings
from typing import List
import ray

import pyscreener as ps

from tdc.metadata import docking_target_info
from tdc.utils import receptor_load

from pyscreener.docking.vina.utils import Software

from .utils import POCKETS_SIU, SIU_PATH


class PyscreenerOracle:
    def __init__(
        self,
        target_name: str,
        software_class: str = "qvina",
        ncpu: int = 16,
        exhaustiveness: int = 8,
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
        if (
            not os.path.isfile(target_name)
            and target_name.endswith("docking")
            or target_name.endswith("docking_vina")
        ):
            pdbid = target_name.split("_")[0]
            receptor_load(pdbid)
            receptor_pdb_file = "./oracle/" + pdbid + ".pdbqt"
            box_center = docking_target_info[pdbid]["center"]
            box_size = tuple([s for s in docking_target_info[pdbid]["size"]])
        else:
            pdb_id = target_name
            assert pdb_id in POCKETS_SIU
            receptor_pdb_file = os.path.join(
                SIU_PATH, "pdb_files", f"{target_name}.pdb"
            )
            box_center = tuple(POCKETS_SIU[pdb_id]["center"])
            box_size = tuple(POCKETS_SIU[pdb_id]["size"])

        if not ray.is_initialized():
            ray.init()

        if hasattr(ps, "build_metadata"):
            metadata = ps.build_metadata(
                software_class, metadata={"exhaustiveness": exhaustiveness}
            )
        else:
            raise OSError(
                "Pyscreener version is not compatible. Please update to the latest version."
            )

        if software_class == "qvina" and os.system("qvina --help") != 32512:
            metadata.software = Software.QVINA

        if hasattr(ps, "virtual_screen"):
            self.scorer = ps.virtual_screen(  # type: ignore
                software_class,
                [receptor_pdb_file],
                box_center,
                box_size,
                metadata,
                ncpu=ncpu,
            )
        else:
            raise OSError(
                "Pyscreener version is not compatible. Please update to the latest version."
            )

    def __call__(self, test_smiles: str | List[str], error_value=0):
        if isinstance(test_smiles, str):
            final_score = self.scorer(test_smiles)
            return list(final_score)[0]
        else:
            final_score = self.scorer(test_smiles)
            score_lst = []
            for i, smiles in enumerate(test_smiles):
                score = final_score[i]
                if score is None:
                    warnings.warn(f"Docking failed for {smiles}.")
                    score = error_value
                score_lst.append(score)
            return score_lst
