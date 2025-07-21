def register() -> None:
    from vllm import ModelRegistry

    print("Registering DockGenModels in vLLM ModelRegistry...")

    ModelRegistry.register_model(
        "DockGenModel",
        "mol_gen_docking.models.vllm_model.modeling_vllm_dockgen:DockGenModel",
    )

    ModelRegistry.register_model(
        "DockGenModelBase",
        "mol_gen_docking.models.vllm_model.modeling_vllm_dockgen:VllmDockGenModelBase",
    )
