from typing import Any, Dict

import torch
import torch.utils.data
from trl import GRPOTrainer


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


class VanillaReinventTrainer(GRPOTrainer):
    def _compute_loss(self, model: Any, inputs: Dict[str, Any], outputs: Any) -> Any:
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        # Get the reward
        reward = inputs["reward"]
        # Compute the Diff

        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_diff = ref_per_token_logps - per_token_logps
        diff = per_token_diff.mean()
        print(diff.shape)

        if self.beta != 0.0:
            loss = (per_token_diff.mean(-1) - self.beta * reward) ** 2
        else:
            loss = per_token_diff.mean(-1) ** 2

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            self._metrics[mode]["NLL_diff"].append(
                self.accelerator.gather(diff).nanmean().item()
            )

        mean_entropy = entropies.mean()
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather(mean_entropy).nanmean().item()
        )
        return loss
