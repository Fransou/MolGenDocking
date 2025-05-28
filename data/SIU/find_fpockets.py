import os
import json
import argparse


def process_pockets(file_list):
    for f in file_list:
        os.system("fpocket -f " + f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a list of PDB files with fpocket."
    )
    parser.add_argument(
        "file_list",
        type=str,
        help="Path to the JSON file containing the list of PDB files.",
    )

    args = parser.parse_args()

    # Load the file list from the JSON file
    with open(args.file_list) as f:
        file_list = json.load(f)

    # Process the pockets
    process_pockets(file_list)
