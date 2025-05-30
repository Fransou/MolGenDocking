import re
from multiprocessing import Pool
from typing import Dict, List

import requests
from tqdm import tqdm

from mol_gen_docking.reward.property_utils.classical_properties import (
    CLASSICAL_PROPERTIES_NAMES,
)

IS_CONNECTED = True


def get_pdb_description(pdb_id: str) -> str | None:
    def fetch_uniprot_id_from_pdb(pdb_id):
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            raise ValueError(f"PDB ID {pdb_id} not found.")
        data = response.json()

        # Get polymer entity IDs (usually 1 per chain)
        polymer_ids = data.get("rcsb_entry_container_identifiers", {}).get(
            "polymer_entity_ids", []
        )
        if not polymer_ids:
            raise ValueError("No polymer entities found in PDB entry.")

        # Try the first polymer entity to get UniProt mapping
        entity_id = polymer_ids[0]
        mapping_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id.lower()}/{entity_id}"
        mapping_response = requests.get(mapping_url, timeout=10)
        if mapping_response.status_code != 200:
            raise ValueError("Unable to fetch polymer entity data.")

        mapping_data = mapping_response.json()
        references = mapping_data.get(
            "rcsb_polymer_entity_container_identifiers", {}
        ).get("reference_sequence_identifiers", [])

        for ref in references:
            if ref.get("database_name") == "UniProt":
                return ref.get("database_accession")

        raise ValueError("No UniProt mapping found for the given PDB entry.")

    def fetch_uniprot_info(uniprot_id):
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

    def clean_protein_name(name):
        name = name.replace("β", "beta").replace("α", "alpha")
        name = re.sub(r"\b(chain|isoform.*|fragment)\b", "", name, flags=re.IGNORECASE)
        name = re.sub(r"\s+", " ", name)
        return name.strip()

    try:
        uniprot_id = fetch_uniprot_id_from_pdb(pdb_id)
        description = fetch_uniprot_info(uniprot_id)
        return clean_protein_name(description)
    except Exception as e:
        print(f"[Warning] UniProt fallback: {e}")
        print(f"[Warning]: pdb_id: {pdb_id}")
        return None


def get_names_mapping(docking_targets: List[str], n_proc: int = 8) -> Dict[str, str]:
    names_mapping: Dict[str, str] = {}
    pool = Pool(8)
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
    # Add classical properties
    for k, v in CLASSICAL_PROPERTIES_NAMES.items():
        names_mapping[k] = v
    return names_mapping
