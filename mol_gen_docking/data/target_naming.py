import re
from multiprocessing import Pool
from typing import Dict, List

import requests
from tqdm import tqdm

from mol_gen_docking.reward.property_utils.classical_properties import (
    CLASSICAL_PROPERTIES_NAMES,
)


def fetch_uniprot_id_from_pdbid(pdb_id: str) -> str | None:
    """
    Fetches the UniProt accession ID corresponding to a PDB ID using the UniProt API.
    """
    pdb_id = pdb_id.replace("_docking", "")
    url = f"https://rest.uniprot.org/uniprotkb/search?query=(xref:pdb-{pdb_id})"

    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise ValueError(f"Failed to query UniProt for PDB ID {pdb_id}")

    data = response.json()
    if not data.get("results"):
        raise ValueError(f"No UniProt mapping found for PDB ID {pdb_id}")
    uniprot_id: str = data["results"][0]["primaryAccession"]
    return uniprot_id


def fetch_uniprot_info(uniprot_id: str) -> str:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise ValueError(f"UniProt ID {uniprot_id} not found.")
    data = response.json()

    # Get protein name
    protein_name = (
        data.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value")
    )
    if not protein_name:
        submitted = data.get("proteinDescription", {}).get("submissionNames", [])
        protein_name = submitted[0]["fullName"]["value"] if submitted else "protein"

    # Get species name
    species_data = data.get("organism", {}).get("scientificName", "organism")
    species = species_data.lower()
    if "homo sapiens" in species:
        species = "human"
    elif "mus musculus" in species:
        species = "mouse"
    elif "rattus norvegicus" in species:
        species = "rat"
    else:
        species = species.split()[0]  # fallback: genus only

    return f"{species} {protein_name.lower()}"


def clean_protein_name(name: str) -> str:
    name = name.replace("β", "beta").replace("α", "alpha")
    name = re.sub(r"\b(chain|isoform.*|fragment)\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def get_pdb_description(pdb_id: str) -> str | None:
    try:
        uniprot_id = fetch_uniprot_id_from_pdbid(pdb_id)
        assert isinstance(uniprot_id, str)
        description = fetch_uniprot_info(uniprot_id)
        return clean_protein_name(description)
    except Exception as e:
        print(f"[Warning] UniProt fallback: {e}")
        print(f"[Warning]: pdb_id: {pdb_id}")

        return None


def get_names_mapping(docking_targets: List[str], n_proc: int = 8) -> Dict[str, str]:
    names_mapping: Dict[str, str] = {}
    pool = Pool(64)
    docking_desc = list(
        tqdm(
            pool.imap(get_pdb_description, docking_targets),
            total=len(docking_targets),
            desc="Adding descriptions to docking targets",
        )
    )
    for pdb_id, desc in zip(docking_targets, docking_desc):
        if desc is not None:
            names_mapping[f"Binding affinity against {desc} ({pdb_id})"] = pdb_id

    pool.close()

    print("Final number of targets: {}".format(len(docking_targets)))
    # Add classical properties
    for k, v in CLASSICAL_PROPERTIES_NAMES.items():
        names_mapping[k] = v
    return names_mapping
