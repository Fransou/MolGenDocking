from typing import Any, Optional, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
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

    @classmethod
    def from_language_model(cls, language_model: Qwen3Model) -> "DockGenModelBase":
        """Create a DockGenModelBase from a Qwen3Model."""
        base_model = language_model
        dock_gen_config = DockGenConfig.from_qwen3_config(
            language_model.config,
        )
        model = cls(dock_gen_config)
        model.load_state_dict(base_model.state_dict(), strict=True)
        return model


class DockGenModel(Qwen3ForCausalLM):
    config_class = DockGenConfig

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: DockGenConfig) -> None:
        super(Qwen3ForCausalLM, self).__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model = DockGenModelBase(config)
        self.vocab_size = config.vocab_size
        self.aligner = nn.Linear(
            self.config.prot_embedding_dim, self.config.hidden_size, bias=True
        )
        self.post_init()

    def get_multimodal_embeddings(
        self, pixel_values: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if pixel_values is None:
            return None
        # Run multimodal inputs through encoder and projector
        embeddings = self.aligner(pixel_values)
        return embeddings

    def get_input_embed_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[Any] = None,
    ) -> torch.Tensor:
        # `get_input_embeddings` should already be implemented for the language
        # model as one of the requirements of basic vLLM model implementation.
        inputs_embeds = self.model.embed_tokens(input_ids)
        print(inputs_embeds.shape)
        if multimodal_embeddings is not None:
            if input_ids is None:
                special_mm_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(
                        self.config.mm_token_id,
                        dtype=torch.long,
                        device=inputs_embeds.device,
                    )
                )
                special_mm_mask = special_mm_mask.all(-1)
            else:
                special_mm_mask = input_ids == self.config.mm_token_id

            special_mm_mask = (
                special_mm_mask.unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            assert special_mm_mask.all(-1).sum() == multimodal_embeddings.shape[0], (
                "The number of multimodal embeddings should match the number of "
                "special multimodal tokens in the input_ids."
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_mm_mask, multimodal_embeddings
            )

        return inputs_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        if inputs_embeds is None:
            multimodal_embeddings = self.get_multimodal_embeddings(pixel_values)
            inputs_embeds = self.get_input_embed_embeddings(
                input_ids=input_ids, multimodal_embeddings=multimodal_embeddings
            )

        return self.legacy_forward(
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
        mm_token_id: int = 151655,
    ) -> "DockGenModel":
        """Create a DockGenModel from a Qwen3ForCausalLM model."""
        base_model = DockGenModelBase.from_language_model(language_model.model)

        dock_gen_config = DockGenConfig.from_qwen3_config(
            language_model.config,
            prot_embedding_dim=prot_embedding_dim,
        )
        model = cls(dock_gen_config)
        model.model = base_model
        return model

    def legacy_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[Any],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
