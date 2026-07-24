from Bio.PDB import PDBParser, PDBIO, Select
import argparse
import json
import os



def get_prompt(prompt_id, data_path="data"):
    # Open jsonl file and read the prompt data
    import jsonlines
    targets_id = {}
    id=0
    with jsonlines.open(f"{data_path}/test_data/test_prompts_ood.jsonl") as reader:
        for obj in reader:
            if obj["conversations"][0]["meta"]["n_props"] == 1 and obj["conversations"][0]["meta"]["n_docking_props"] == 1:
                targets_id[id] = (obj["identifier"], obj["conversations"][0]["meta"]["properties"][0])
                if id == int(prompt_id):
                    return obj
                id += 1
    raise ValueError(f"Prompt ID {prompt_id} not found in test_prompts_ood.jsonl")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Extract pocket from PDB file.")
    parser.add_argument("prompt_id", help="prompt id for the pocket")
    parser.add_argument("smiles_path", help="Path to the SMILES file for the generated molecule")
    parser.add_argument("output_dir", help="Output dir file for the generated pocket")
    parser.add_argument("--data_path", type=str, default="data", help="Path to the data directory")

    args = parser.parse_args()

    line = get_prompt(args.prompt_id, args.data_path)
    smiles = []
    # Open the .smi file and read the SMILES string
    with open(args.smiles_path, "r") as f:
        for line in f:
            smiles.append(line.strip())
    print(smiles)

    completions = []
    for smi in smiles:
        completions.append(
            {
                "output": smi,
                "metadata": line["conversations"][0]["meta"]
            }
        )

    # Save to output_dir with the name <prompt_id>.jsonl
    output_path = os.path.join(args.output_dir, f"{args.prompt_id}.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)
    import jsonlines
    with jsonlines.open(output_path, "w") as writer:
        writer.write_all(completions)





