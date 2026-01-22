from .abstract_verifier import Verifier, VerifierInputBatchModel, VerifierOutputModel
from .generation_reward.generation_verifier import GenerationVerifier
from .generation_reward.generation_verifier_pydantic_model import (
    GenerationVerifierConfigModel,
    GenerationVerifierMetadataModel,
    GenerationVerifierOutputModel,
)
from .generation_reward.input_metadata import GenerationVerifierInputMetadataModel
from .mol_prop_reward.input_metadata import MolPropVerifierInputMetadataModel
from .mol_prop_reward.mol_prop_verifier import MolPropVerifier
from .mol_prop_reward.mol_prop_verifier_pydantic_model import (
    MolPropVerifierConfigModel,
    MolPropVerifierMetadataModel,
    MolPropVerifierOutputModel,
)
from .reaction_reward.input_metadata import ReactionVerifierInputMetadataModel
from .reaction_reward.reaction_verifier import ReactionVerifier
from .reaction_reward.reaction_verifier_pydantic_model import (
    ReactionVerifierConfigModel,
    ReactionVerifierMetadataModel,
    ReactionVerifierOutputModel,
)


def assign_to_inputs(completion: str, metadata: dict) -> str:
    for b_model_cls, label in [
        (GenerationVerifierInputMetadataModel, "generation"),
        (MolPropVerifierInputMetadataModel, "mol_prop"),
        (ReactionVerifierInputMetadataModel, "reaction"),
    ]:
        try:
            b_model_cls.model_validate({"completion": completion, **metadata})  # type: ignore
            return label
        except Exception:
            continue
    raise NotImplementedError(
        f"Input metadata does not match any verifier input model for completion: {completion} with metadata: {metadata}"
    )


__all__ = [
    "Verifier",
    "GenerationVerifier",
    "MolPropVerifier",
    "ReactionVerifier",
    ###
    "GenerationVerifierConfigModel",
    "MolPropVerifierConfigModel",
    "ReactionVerifierConfigModel",
    ###,
    "VerifierOutputModel",
    "GenerationVerifierOutputModel",
    "MolPropVerifierOutputModel",
    "ReactionVerifierOutputModel",
    ###
    "GenerationVerifierMetadataModel",
    "MolPropVerifierMetadataModel",
    "ReactionVerifierMetadataModel",
    ###
    "GenerationVerifierInputMetadataModel",
    "MolPropVerifierInputMetadataModel",
    "ReactionVerifierInputMetadataModel",
    ###
    "VerifierInputBatchModel",
]
