from Bio.PDB import PDBParser, PDBIO, Select
import argparse
import json
import os

class PocketSelect(Select):
    def __init__(self, center, half_box):
        self.center = center
        self.half_box = half_box

    def accept_residue(self, residue):
        # Keep residue if ANY of its atoms are inside the box
        for atom in residue:
            x, y, z = atom.get_coord()
            if (abs(x - self.center[0]) <= self.half_box[0] and
                abs(y - self.center[1]) <= self.half_box[1] and
                abs(z - self.center[2]) <= self.half_box[2]):
                return True
        return False



def get_target_id(prompt_id, data_path="data"):
    # Open jsonl file and read the prompt data
    if not os.path.exists("targets_id.json"):
        import jsonlines
        targets_id = {}
        id=0
        with jsonlines.open(f"{data_path}/test_data/test_prompts_ood.jsonl") as reader:
            for obj in reader:
                if obj["conversations"][0]["meta"]["n_props"] == 1 and obj["conversations"][0]["meta"]["n_docking_props"] == 1:
                    targets_id[id] = (obj["identifier"], obj["conversations"][0]["meta"]["properties"][0])


        # Save targets_id
        with open(f"targets_id.json", "w") as f:
            json.dump(targets_id, f, indent=4)
    else:
        with open(f"targets_id.json", "r") as f:
            targets_id = json.load(f)

    return targets_id[int(prompt_id)][1]


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Extract pocket from PDB file.")
    parser.add_argument("prompt_id", help="prompt id for the pocket")
    parser.add_argument("output_pdb", help="Output PDB file for the pocket")
    parser.add_argument("--data_path", type=str, default="data", help="Path to the data directory")

    args = parser.parse_args()

    prot_id = get_target_id(args.prompt_id, args.data_path)

    # Open pocket metadata
    with open(f"data/molgendata/pockets_info.json", "r") as f:
        pockets_info = json.load(f)

    assert prot_id in pockets_info, f"Protein ID {prot_id} not found in pockets_info.json"
    center = pockets_info[prot_id]["center"]
    half_box = [d/2 for d in pockets_info[prot_id]["size"]]

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", f"{args.data_path}/pdb_files/{prot_id}.pdb")

    io = PDBIO()
    io.set_structure(structure)
    io.save(args.output_pdb, PocketSelect(center, half_box))




