from typing import Any, Optional

import torch
from torch import nn
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from transformers.models.qwen3.modeling_qwen3 import (
    KwargsForCausalLM,
    Qwen3ForCausalLM,
    Qwen3Model,
)
from transformers.processing_utils import Unpack

from .configuration_dockgen import DockGenConfig


class DockGenModelBase(Qwen3Model):
    config_class = DockGenConfig

    def __init__(self, config: DockGenConfig) -> None:
        super().__init__(config)
        self.aligner = nn.Linear(
            self.config.prot_embedding_dim, self.config.hidden_size, bias=True
        )

    def get_multimodal_embeddings(
        self, pixel_values: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> torch.Tensor:
        inp = kwargs.get("pixel_values", None)
        if inp is None:
            return None
        # Run multimodal inputs through encoder and projector
        embeddings = self.aligner(inp)
        return embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        special_image_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        # Handle multimodal inputs
        multimodal_embeddings = self.get_multimodal_embeddings(
            pixel_values=pixel_values, **kwargs
        )
        inputs_embeds = self.embed_tokens(input_ids)
        if not (multimodal_embeddings is None or special_image_mask is None):
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
                inputs_embeds.device
            )
            mm_values = multimodal_embeddings.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, mm_values)
        return super().forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    @classmethod
    def from_language_model(
        cls,
        language_model: Qwen3Model,
        prot_embedding_dim: int = 1024,
    ) -> "DockGenModelBase":
        """Create a DockGenModel from a Qwen3ForCausalLM model."""
        dock_gen_config = DockGenConfig.from_qwen3_config(
            language_model.config,
            prot_embedding_dim=prot_embedding_dim,
        )
        model = cls(dock_gen_config)
        model.load_state_dict(language_model.state_dict(), strict=False)

        return model


class DockGenModel(Qwen3ForCausalLM):
    config_class = DockGenConfig

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: DockGenConfig) -> None:
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = DockGenModelBase(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        special_image_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Optional[int] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

    @classmethod
    def from_language_model(
        cls,
        language_model: Qwen3ForCausalLM,
        prot_embedding_dim: int = 1024,
    ) -> "DockGenModel":
        """Create a DockGenModel from a Qwen3ForCausalLM model."""
        base_model = DockGenModelBase.from_language_model(
            language_model.model,
            prot_embedding_dim=prot_embedding_dim,
        )

        dock_gen_config = DockGenConfig.from_qwen3_config(
            language_model.config,
            prot_embedding_dim=prot_embedding_dim,
        )
        model = cls(dock_gen_config)
        model.model = base_model
        return model
