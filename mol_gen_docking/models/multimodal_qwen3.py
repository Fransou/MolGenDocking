from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, KwargsForCausalLM


from typing import Callable, Optional, Tuple, Union
import torch

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache

PROT_FEATURE_DIM = 1024  # Example dimension for protein features, adjust as needed

class MultiModalQwen3ForCausalLM(Qwen3ForCausalLM):

    def __init__(self, *args, **kwargs):
        super.__init__(*args, **kwargs)

        self.aligner = torch.nn.Linear(
            PROT_FEATURE_DIM,
            self.get_input_embeddings.shape[1],
            bias=True
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        special_image_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:

        mm_values = self.aligner(pixel_values) if pixel_values is not None else None
        if inputs_embeds is None:
            if input_ids is not None:
                inputs_embeds = self.embed_tokens(input_ids)
            else:
                raise ValueError("Either `input_ids` or `pixel_values` must be provided.")

        special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
        mm_values = mm_values.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, mm_values)
        # Extract the embedding of the multimodal object, and aligns it


        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs
        )