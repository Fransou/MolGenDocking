"""To minimize modifications to pre-existing code,
we store all multimodal informations as image embeddings.
"""

from typing import Mapping, Optional, Sequence

from transformers import PreTrainedTokenizer
from transformers.feature_extraction_utils import BatchFeature
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargs,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    MultiModalDataItems,
    PromptReplacement,
    PromptUpdate,
)


class DockGenProcessingInfo(BaseProcessingInfo):
    """
    Processing info for the DockGen model.
    This class is used to store processing information for the DockGen model.
    It extends the BaseProcessingInfo class and provides methods to validate images and text inputs.
    """

    def get_tokenizer(self) -> PreTrainedTokenizer:
        tokenizer = self.ctx.tokenizer
        return tokenizer

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": 16}


class VllmMultiModalProcessor(BaseMultiModalProcessor[DockGenProcessingInfo]):
    """
    A processor for multimodal inputs in vLLM.
    This processor is used to handle multimodal inputs in vLLM models.
    It extends the BaseMultiModalProcessor and provides methods to process images and text.
    """

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object] = {},
        tokenization_kwargs: Mapping[str, object] = {},
    ) -> bool:
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        num_image_tokens = 1
        return [
            PromptReplacement(
                modality="image",
                target="<|image_pad|>",
                replacement="<|image_pad|>" * num_image_tokens,
            )
        ]
