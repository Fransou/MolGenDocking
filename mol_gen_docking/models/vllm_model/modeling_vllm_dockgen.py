from typing import Any, Iterable, Optional

import torch
from torch import nn
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import (
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.qwen3 import (
    Qwen3DecoderLayer,
    Qwen3Model,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    maybe_prefix,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import merge_multimodal_embeddings
from vllm.multimodal import MULTIMODAL_REGISTRY

from .dummy_inputs import DockGenDummyInputsBuilder
from .processing_vllm_dockgen import DockGenProcessingInfo, VllmMultiModalProcessor


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class VllmDockGenModelBase(Qwen3Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, decoder_layer_type=Qwen3DecoderLayer
        )


@MULTIMODAL_REGISTRY.register_processor(
    VllmMultiModalProcessor,
    info=DockGenProcessingInfo,
    dummy_inputs=DockGenDummyInputsBuilder,
)
class DockGenModel(nn.Module, SupportsPP, SupportsLoRA, SupportsMultiModal):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.language_model = VllmDockGenModelBase(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.language_model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.aligner = RowParallelLinear(
            config.prot_embedding_dim, config.hidden_size, bias=True
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def get_multimodal_embeddings(
        self, pixel_values: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if pixel_values is None:
            return None
        # Run multimodal inputs through encoder and projector
        embeddings = self.aligner(pixel_values)
        return embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[Any] = None,
    ) -> torch.Tensor:
        # `get_input_embeddings` should already be implemented for the language
        # model as one of the requirements of basic vLLM model implementation.
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)

        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                placeholder_token_id=self.config.mm_token_id,
            )

        return inputs_embeds

    def get_language_model(self) -> torch.nn.Module:
        # Change `language_model` according to your implementation.
        return self.language_model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        special_image_mask: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        if inputs_embeds is None:
            multimodal_embeddings = self.get_multimodal_embeddings(pixel_values)
            inputs_embeds = self.get_input_embeddings(
                input_ids=input_ids,
                multimodal_embeddings=multimodal_embeddings,
            )

        hidden_states = self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            pixel_values,
            special_image_mask,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Any:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "<|image_pad|>"

        raise ValueError("Only image modality is supported")
