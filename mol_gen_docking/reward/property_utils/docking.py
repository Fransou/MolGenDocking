import os
from typing import List
import pandas as pd
import requests


import re


SIU_PATH = os.environ.get("SIU_DATA_PATH", os.path.join("data", "SIU"))


with open(os.path.join(SIU_PATH, "pockets_info.json")) as f:
    POCKETS_SIU = pd.read_json(f)


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


with open(os.path.join(SIU_PATH, "pockets_info.json")) as f:
    POCKETS_SIU = pd.read_json(f)


DOCKING_TARGETS: List[str] = [
    "3pbl_docking",
    "1iep_docking",
    "2rgp_docking",
    "3eml_docking",
    "3ny8_docking",
    "4rlu_docking",
    "4unn_docking",
    "5mo4_docking",
    "7l11_docking",
] + [k for k in POCKETS_SIU.keys()]
