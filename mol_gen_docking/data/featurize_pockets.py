from typing import Literal
from subprocess import DEVNULL, STDOUT, check_call

import logging

logger = logging.getLogger(__name__)

class ProteinStructureEmbeddingExtractor:
    def __init__(self, model: Literal["GSnet", "aLCnet"]):
        self.call_template = f"python embed_{model}.py "
        self.call_template += "{{PDBPATH}} {{OUTPUTPATH}}"

    def extract_embeddings(self, pdb_path: str, output_path: str) -> None:
        """Extract embeddings for a given PDB file using the specified model."""

        command = self.call_template.format(PDBPATH=pdb_path, OUTPUTPATH=output_path)
        print(f"Running command: {command}")
        check_call(command.split(), stdout=DEVNULL, stderr=STDOUT)
        logger.info(f"Embeddings saved to {output_path}")



