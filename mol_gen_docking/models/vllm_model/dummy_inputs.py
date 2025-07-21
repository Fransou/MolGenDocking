from typing import Any, Mapping

import numpy as np
import torch
from vllm.multimodal import MultiModalDataDict
from vllm.multimodal.profiling import BaseDummyInputsBuilder

PROT_FEATURE_DIM = 320


class DockGenDummyInputsBuilder(BaseDummyInputsBuilder):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        image = np.random.randn(num_images, PROT_FEATURE_DIM).astype(np.float32)
        return {
            "image": torch.tensor(image, dtype=torch.float32),
        }

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token: str = processor.image_token

        return image_token * num_images
