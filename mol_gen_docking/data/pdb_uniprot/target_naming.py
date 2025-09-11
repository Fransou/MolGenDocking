import re
from multiprocessing import Pool
from typing import Dict, List

from tqdm import tqdm

from data.pdb_uniprot.api_requests import (
    fetch_uniprot_id_from_pdbid,
    fetch_uniprot_info,
)
from mol_gen_docking.reward.property_utils import (
    CLASSICAL_PROPERTIES_NAMES,
)


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


def get_unip_description(uniprot_id: str) -> str | None:
    try:
        assert isinstance(uniprot_id, str)
        description = fetch_uniprot_info(uniprot_id)
        return clean_protein_name(description)
    except Exception as e:
        print(f"[Warning] UniProt fallback: {e}")
        print(f"[Warning]: uniprot_id: {uniprot_id}")

        return None


def get_names_mapping_uniprot(
    docking_targets: List[str], n_proc: int = 8, names_override: List[str] = []
) -> Dict[str, str]:
    assert len(names_override) == 0 or len(names_override) == len(docking_targets), (
        "If names_override is provided, it must match the length of docking_targets"
    )
    names_mapping: Dict[str, str] = {}
    pool = Pool(16)
    docking_desc = list(
        tqdm(
            pool.imap(get_unip_description, docking_targets),
            total=len(docking_targets),
            desc="Adding descriptions to docking targets",
        )
    )

    pool.close()

    if len(names_override) == 0:
        for uniprot_id, desc in zip(docking_targets, docking_desc):
            if desc is not None:
                names_mapping[f"Docking score against {desc}"] = uniprot_id
    else:
        for name, desc in zip(docking_targets, docking_desc):
            if desc is not None:
                names_mapping[f"Docking score against {desc}"] = name

    print("Final number of targets: {}".format(len(docking_targets)))
    # Add classical properties
    for k, v in CLASSICAL_PROPERTIES_NAMES.items():
        names_mapping[k] = v
    return names_mapping


def get_names_mapping(docking_targets: List[str], n_proc: int = 8) -> Dict[str, str]:
    names_mapping: Dict[str, str] = {}
    pool = Pool(16)
    docking_desc = list(
        tqdm(
            pool.imap(get_pdb_description, docking_targets),
            total=len(docking_targets),
            desc="Adding descriptions to docking targets",
        )
    )
    for pdb_id, desc in zip(docking_targets, docking_desc):
        if desc is not None:
            names_mapping[f"Docking score against {desc} ({pdb_id})"] = pdb_id

    pool.close()

    print("Final number of targets: {}".format(len(docking_targets)))
    # Add classical properties
    for k, v in CLASSICAL_PROPERTIES_NAMES.items():
        names_mapping[k] = v
    return names_mapping
