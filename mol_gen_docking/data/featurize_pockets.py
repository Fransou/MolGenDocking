from typing import Literal
from subprocess import DEVNULL, STDOUT, check_call



class ProteinStructureEmbeddingExtractor:
    def __init__(self, model: Literal["GSnet", "aLCnet"]):
        self.call_template = f"python embed_{model}.py "
        self.call_template += "{{PDBPATH}} {{OUTPUTPATH}}"

