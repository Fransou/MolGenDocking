# Reaction Verifier

The `ReactionVerifier` computes rewards for chemical reaction and retro-synthesis tasks, validating synthesis paths, SMARTS predictions, and reaction product verification.

## Overview

The Reaction Verifier supports various reaction-related tasks:

- **Retro-synthesis Planning**: Validate multi-step synthesis routes
- **SMARTS Prediction**: Evaluate predicted reaction SMARTS patterns
- **Product/Reactant Prediction**: Compare predicted molecules to ground truth
- **Analog Generation**: Generate molecular analogs via synthesis

??? Note "Supported Task Types"
    The following task types are supported by the Reaction Verifier:

    | Task Type | Description |
    |-----------|-------------|
    | `final_product` | Predict the final product of a reaction |
    | `reactant` | Predict a single reactant |
    | `all_reactants` | Predict all reactants for a reaction |
    | `smarts` | Predict the SMARTS pattern for a reaction |
    | `full_path` | Provide a complete retro-synthesis path |
    | `full_path_bb_ref` | Synthesis path with building block constraints |
    | `full_path_smarts_ref` | Synthesis path with SMARTS constraints |
    | `analog_gen` | Generate molecular analogs |



::: mol_gen_docking.reward.verifiers.reaction_reward.reaction_verifier_pydantic_model
    handler: python
    heading_level: 3
    options:
        show_root_toc_entry: false
        members:
            - ReactionVerifierConfigModel


::: mol_gen_docking.reward.verifiers.reaction_reward.input_metadata
    handler: python
    heading_level: 3
    options:
        show_root_toc_entry: false
        members:
            - ReactionVerifierInputMetadataModel

::: mol_gen_docking.reward.verifiers.reaction_reward.reaction_verifier_pydantic_model
    handler: python
    heading_level: 3
    options:
        show_root_toc_entry: false
        members:
            - ReactionVerifierOutputModel
            - ReactionVerifierMetadataModel

::: mol_gen_docking.reward.verifiers.reaction_reward.reaction_verifier
    handler: python
    heading_level: 3
    options:
        show_root_toc_entry: false
        members:
            - ReactionVerifier



!!! note "Synthesis Path Format"

    Synthesis paths must follow this format:

    ```
    reactant1 + reactant2 -> product1
    product1 + reactant3 -> product2
    product2 + reactant4 -> final_product
    ```
    example:
    ```
    <answer>
    CC(=O)O + CCO -> CC(=O)OCC
    CC(=O)OCC + c1ccccc1N -> CC(=O)Nc1ccccc1
    </answer>
    ```


## Related

- [Molecular Verifier](molecular_verifier.md) - Main orchestrator
- [Generation Verifier](generation_verifier.md) - De novo generation tasks
- [Molecular Property Verifier](mol_prop_verifier.md) - Molecular property prediction tasks
