"""A script to process receptors using Meeko in order to prepare them for molecular docking with AutoDock-GPU."""

import argparse
import json
import logging
import os
import re
import subprocess as sp
from typing import Any, Dict, Tuple

import numpy as np
from Bio.PDB import PDBParser

import ray
from ray.experimental import tqdm_ray


def residue_in_box(
    residue: Any, box_center: list[float], box_size: list[float]
) -> bool:
    """
    Check if any atom of the residue is inside the box.
    box_center: (x, y, z)
    box_size: (sx, sy, sz)
    """
    box_center_arr = np.array(box_center)
    box_min = box_center_arr - np.array(box_size) / 2
    box_max = box_center_arr + np.array(box_size) / 2

    for atom in residue.get_atoms():
        coord = atom.get_coord()
        if np.all(coord >= box_min) and np.all(coord <= box_max):
            return True
    return False


def check_failed_residues_in_box(
    pdb_file: str,
    failed_residues: set[str],
    box_center: list[float],
    box_size: list[float],
) -> list[str]:
    """
    Return the list of failed residues that are inside the docking box.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_file)
    residues_in_box = []

    for chain in structure.get_chains():
        for res in chain:
            res_id = f"{chain.id}:{res.get_id()[1]}"
            if res_id in failed_residues and residue_in_box(res, box_center, box_size):
                residues_in_box.append(res_id)

    return residues_in_box


class ReceptorProcess:
    def __init__(self, data_path: str) -> None:
        self.data_path: str = data_path
        self.logger = logging.getLogger(
            __name__ + "/" + self.__class__.__name__,
        )
        self.receptor_path = os.path.join(self.data_path, "pdb_files")

        assert os.path.exists(data_path), f"Data path {data_path} does not exist."
        assert os.path.exists(self.receptor_path), (
            f"Receptor path {self.receptor_path} does not exist."
        )
        assert os.path.exists(os.path.join(data_path, "pockets_info.json")), (
            f"Pockets info file does not exist in {data_path}."
        )

        with open(os.path.join(data_path, "pockets_info.json")) as f:
            self.pockets: Dict[str, Dict[str, Any]] = json.load(f)
        self.cmd = "mk_prepare_receptor.py -i {INPUT} -o {OUTPUT} -p -g --box_size {BOX_SIZE} --box_center {BOX_CENTER}"

    def _run_meeko(
        self, input_path: str, receptor: str, bad_res: bool = False
    ) -> Tuple[str, int, str]:
        output_path = input_path.replace(".pdb", "_ag")

        box_center = " ".join([str(x) for x in self.pockets[receptor]["center"]])
        box_size = " ".join([str(x) for x in self.pockets[receptor]["size"]])

        command = self.cmd.format(
            INPUT=input_path,
            OUTPUT=output_path,
            BOX_SIZE=box_size,
            BOX_CENTER=box_center,
        )
        if bad_res:
            command += " --allow_bad_res"

        self.logger.info(f"Running command: {command}")
        process = sp.Popen(
            command,
            shell=True,
            stdout=sp.DEVNULL,
            stderr=sp.PIPE,
            preexec_fn=os.setpgrp,
        )
        _, stderr = process.communicate(
            timeout=300,
        )

        stderr_text = stderr.decode("utf-8")

        return stderr_text, process.returncode, output_path

    def meeko_process(
        self, receptor: str, allow_bad_res: bool = False
    ) -> Tuple[int, str]:
        input_path = os.path.join(self.receptor_path, f"{receptor}.pdb")
        stderr_text, returncode, output_path = self._run_meeko(
            input_path, receptor, allow_bad_res
        )

        if returncode != 0:
            # Check if failing resiudes are inside the docking box
            failed_residues = set()
            for line in stderr_text.splitlines():
                match = re.search(
                    r"No template matched for residue_key='(\w+:\d+)'", line
                )
                if match:
                    failed_residues.add(match.group(1))

            if failed_residues:
                residues_in_box = check_failed_residues_in_box(
                    input_path,
                    failed_residues,
                    self.pockets[receptor]["center"],
                    self.pockets[receptor]["size"],
                )
                if residues_in_box == []:
                    _, returncode, output_path = self._run_meeko(
                        input_path, receptor, bad_res=True
                    )
                    return returncode, output_path
                else:
                    _, returncode, output_path = self._run_meeko(
                        input_path, receptor, bad_res=True
                    )
                    return 1, output_path
            else:
                raise ValueError(
                    f"Error in Meeko processing {receptor}:\n{stderr_text}"
                )
        return 0, output_path

        self.logger.info(f"Successfully processed {receptor} to {output_path}")
        return output_path

    def _run_autogrid(self, path: str) -> None:
        grid_command = f"autogrid4 -p {path}.gpf -l {path}.glg -d"
        self.logger.info(f"Running command: {grid_command}")
        process = sp.Popen(
            grid_command,
            shell=True,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            cwd=self.receptor_path,
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            self.logger.error(
                f"Error in running AutoGrid on {path}:\n{stderr.decode()}"
            )
            raise RuntimeError(f"AutoGrid failed for {path}")
        self.logger.info(f"Successfully ran AutoGrid on {path}")

    def process_receptors(
        self, receptors: list[str] = [], allow_bad_res: bool = False
    ) -> Tuple[list[str], list[str]]:
        @ray.remote(num_cpus=4)
        def process_receptors(
            receptor: str, pbar: Any, allow_bad_res: bool = False
        ) -> int:
            """
            Outputs the level of the error:
            0: Success
            1: Failed residues inside the box
            2: Other errors (critical)
            """
            try:
                result, processed_path = self.meeko_process(receptor, allow_bad_res)
                self._run_autogrid(processed_path)
                pbar.update.remote(1)
            except Exception as e:
                self.logger.error(f"Error processing {receptor}: {e}")
                pbar.update.remote(1)
                return 2
            return result

        if receptors == []:
            receptors = list(self.pockets.keys())
        else:
            assert all(r in self.pockets for r in receptors), (
                "Some receptors are not in pockets_info.json"
            )

        remote_tqdm = ray.remote(tqdm_ray.tqdm)
        pbar = remote_tqdm.remote(total=len(receptors), desc="Processing receptors")  # type: ignore

        # Find receptors that already have a _ag.pdbqt and_ag.maps.fld file
        receptors_to_process = []
        for receptor in receptors:
            ag_pdbqt_path = os.path.join(self.receptor_path, f"{receptor}_ag.pdbqt")
            ag_maps_fld_path = os.path.join(
                self.receptor_path, f"{receptor}_ag.maps.fld"
            )
            if not (os.path.exists(ag_pdbqt_path) and os.path.exists(ag_maps_fld_path)):
                receptors_to_process.append(receptor)
            else:
                self.logger.info(f"Receptor {receptor} already processed. Skipping.")
                pbar.update.remote(1)  # type: ignore
        if len(receptors_to_process) == 0:
            self.logger.info("All receptors already processed.")
            pbar.close.remote()  # type: ignore
            return [], []

        self.logger.info(f"Processing {len(receptors_to_process)} receptors.")
        futures = [
            process_receptors.remote(receptor, pbar, allow_bad_res)
            for receptor in receptors_to_process
        ]
        results = ray.get(futures)

        missed_receptors_1 = [
            receptor
            for receptor, success in zip(receptors_to_process, results)
            if success == 1
        ]
        missed_receptors_2 = [
            receptor
            for receptor, success in zip(receptors_to_process, results)
            if success == 2
        ]
        return missed_receptors_1, missed_receptors_2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process receptors using Meeko for AutoDock-GPU."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the data directory containing pdb_files and pockets_info.json",
    )
    args = parser.parse_args()

    processor = ReceptorProcess(data_path=args.data_path)
    missed_receptors_1, missed_receptors_2 = processor.process_receptors()
    print("###########\n Missed receptors with critical errors: \n", missed_receptors_2)

    # Remove receptors with critical errors from pockets_info.json and docking_targets.json
    if len(missed_receptors_2) > 0:
        with open(os.path.join(args.data_path, "pockets_info.json")) as f:
            pockets_info = json.load(f)
        for receptor in missed_receptors_2:
            if receptor in pockets_info:
                del pockets_info[receptor]
        with open(os.path.join(args.data_path, "pockets_info.json"), "w") as f:
            json.dump(pockets_info, f, indent=4)

        if os.path.exists(os.path.join(args.data_path, "docking_targets.json")):
            with open(os.path.join(args.data_path, "docking_targets.json")) as f:
                docking_targets = json.load(f)
            docking_targets = [
                target for target in docking_targets if target not in missed_receptors_2
            ]
            with open(os.path.join(args.data_path, "docking_targets.json"), "w") as f:
                json.dump(docking_targets, f, indent=4)
