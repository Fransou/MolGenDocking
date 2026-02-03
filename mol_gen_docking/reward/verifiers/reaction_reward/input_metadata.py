from typing import List, Literal

from pydantic import BaseModel, Field

ReactionObjT = Literal[
    "final_product",
    "reactant",
    "all_reactants",
    "all_reactants_bb_ref",
    "smarts",
    "full_path",
    "full_path_bb_ref",
    "full_path_smarts_ref",
    "full_path_smarts_bb_ref",
    "analog_gen",
]


class ReactionVerifierInputMetadataModel(BaseModel):
    """
    Input metadata model for reaction verifier.

    """

    objectives: List[ReactionObjT] = Field(
        ...,
        description="The type of objective for the reaction verification.",
    )
    target: List[str] = Field(
        default_factory=list,
        description="The target molecule or SMARTS string for verification.",
    )
    reactants: List[List[str]] = Field(
        default_factory=list,
        description="List of reactants in a reaction.",
    )
    products: List[str] = Field(
        default_factory=list,
        description="The product molecule of the reaction.",
    )
    building_blocks: List[str] | None = Field(
        None,
        description="List of valid building blocks for the reaction.",
    )
    smarts: List[str] = Field(
        ...,
        description="Reference SMARTS strings for the reaction steps.",
    )
    or_smarts: List[str] = Field(
        ...,
        description="Original Reference SMARTS strings for the reaction steps.",
    )
    n_steps_max: int = Field(
        default=5,
        gt=0,
        description="Maximum number of reaction steps allowed in the synthesis route.",
    )
    idx_chosen: int = Field(
        ...,
        description="Index of the chosen reaction.",
    )
