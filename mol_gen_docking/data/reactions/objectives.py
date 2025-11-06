PROMPT_TEMPLATES = {
    "final_product": [
        "Give me the final product of the following multi-step synthesis:\n{reaction}\nwhere the SMARTS reprentation of the reaction is:\n{smarts}\nIf the reaction is impossible, return 'impossible'."
    ],
    "reactant": [
        "What is the missing reactant of this synthesis step:\n{reaction}\nwhere the SMARTS reprentation of the reaction is:\n{smarts}\nIf the reaction is impossible, return 'impossible'."
    ],
    "all_reactants": [
        "Given the following reaction in the SMARTS format:\n{smarts}\nprovide one or multiple reactants to obtain the following product: {product}\nIf such a reaction does not exist, return 'impossible'."
    ],
    "all_reactants_bb_ref": [
        "Given the following reaction in the SMARTS format:\n{smarts}\nprovide one molecule and one or more building blocks from:\n{building_blocks}\nto obtain the following product: {product}\nIf such a reaction does not exist, return 'impossible'."
    ],
    "smarts": [
        "Provide the SMARTS representation of the following synthesis step:\n{reaction}\nIf the reaction is impossible, return 'impossible'."
    ],
    "full_path": [
        "Propose a synthesis pathway to generate {product}. Your answer must include at most {n_reaction} reactions. Provide your answer in the following format:\nA + B -> C\n C -> D"
    ],
    "full_path_bb_ref": [
        "Propose a synthesis pathway to generate {product} using building blocks among:\n{building_blocks}\nYour answer must include at most {n_reaction} reactions. Provide your answer in the following format:\nA + B -> C\nC -> D\nIf it is impossible, return 'impossible'."
    ],
    "full_path_smarts_ref": [
        "Propose a synthesis pathway to generate {product} using reactions among the following SMARTS:\n{smarts}\nYour answer must include at most {n_reaction} reactions. Provide your answer in the following format:\nA + B -> C\nC -> D\nIf it is impossible, return 'impossible'."
    ],
    "full_path_smarts_bb_ref": [
        "Propose a synthesis pathway to generate {product} using building blocks among:\n{building_blocks}\n and reactions among the following SMARTS:\n{smarts}\nYour answer must include at most {n_reaction} reactions. Provide your answer in the following format:\nA + B -> C\nC -> D\nIf it is impossible, return 'impossible'."
    ],
}
