import json
import os
import warnings
from typing import Any, List

import pyscreener as ps
from pyscreener.docking.vina.utils import Software
from tdc.metadata import docking_target_info
from tdc.utils import receptor_load

import ray


class PyscreenerOracle:
    def __init__(
        self,
        target_name: str,
        path_to_data: str,
        software_class: str = "vina",
        ncpu: int = 16,
        exhaustiveness: int = 8,
        **kwargs: Any,
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
            receptor_pdb_file = os.path.join(
                path_to_data, "pdb_files", f"{target_name}_processed.pdb"
            )
            with open(os.path.join(path_to_data, "pockets_info.json")) as f:
                pockets_info = json.load(f)
            box_center = tuple(pockets_info[pdb_id]["center"])
            box_size = tuple(pockets_info[pdb_id]["size"])

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
            self.scorer = ps.virtual_screen(
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

    def __call__(self, test_smiles: str | List[str], error_value: float = 0.0) -> Any:
        final_score = self.scorer(test_smiles)

        if isinstance(test_smiles, str):
            return list(final_score)[0]
        else:
            score_lst = []
            for i, smiles in enumerate(test_smiles):
                score = final_score[i]
                if score is None:
                    warnings.warn(f"Docking failed for {smiles}.")
                    score = error_value
                score_lst.append(score)
            return score_lst
