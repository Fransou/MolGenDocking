import logging

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from mol_gen_docking.data.api_requests import get_data_from_pdb_id

logger = logging.getLogger(__name__)


class ProteinStructureEmbeddingExtractor:
    def __init__(
        self,
        model: str = "facebook/esm2_t6_8M_UR50D",
        data_dir: str = "data/mol_orz",
        layer_embs: int = 3,
    ) -> None:
        self.data_dir = data_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForMaskedLM.from_pretrained(model)
        self.layer_embs = layer_embs

    def generate_sequence(self, pdb_id: str) -> str:
        data = get_data_from_pdb_id(pdb_id)
        sequence: str = data["results"][0]["sequence"]["value"]
        if not sequence:
            raise ValueError(f"No sequence found for PDB ID {pdb_id}")
        return sequence

    def extract_embeddings(self, pdb_path: str, output_path: str) -> None:
        """Extract embeddings for a given PDB file using the specified model."""

        pdb_id = pdb_path.split("/")[-1].split("_")[0]
        sequence = self.generate_sequence(pdb_id)

        device = "cuda" if self.model.device.type == "cuda" else "cpu"
        self.model = self.model.to(device)
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = self.model(output_hidden_states=True, **inputs)
        hidden_states = outputs.hidden_states[self.layer_embs][0].mean(0).cpu()
        torch.save(hidden_states, output_path)
