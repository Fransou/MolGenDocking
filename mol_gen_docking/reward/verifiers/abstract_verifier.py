"""Abstract base classes for molecular verifiers.

This module provides the base classes and input/output models used by all
verifiers in the reward system. It defines the common interface that all
verifiers must implement.
"""

import re
from typing import Any, List

from pydantic import BaseModel

from mol_gen_docking.reward.verifiers.abstract_verifier_pydantic_model import (
    BatchVerifiersInputModel,
)


class Verifier:
    """Abstract base class for all molecular verifiers.

    This class defines the interface that all verifiers must implement.
    Subclasses should override the get_score method to provide specific
    verification logic.

    Example:
        ```python
        class MyVerifier(Verifier):
            def get_score(self, inputs: BatchVerifiersInputModel) -> List[VerifierOutputModel]:
                # Implement verification logic
                return [VerifierOutputModel(reward=1.0, verifier_metadata={})]
        ```
    """

    ENTRY_ANSWER: List[str] = ["<answer>", r"<\|answer_start\|>"]
    EXIT_ANSWER: List[str] = ["</answer>", r"<\|answer_end\|>"]

    def __init__(self, verifier_config: BaseModel) -> None:
        """Initialize the base verifier."""
        self.verifier_config = verifier_config
        pass

    def parse_none(self, completion: str) -> str:
        """Parse the answer using no special parsing.
        Simply returns the completion with special tokens removed.

        Args:
            completion: The full text completion from the model.
        Returns:
            The extracted answer string.
        """
        comp = re.sub(r"<|>", " ", completion)
        return comp.strip()

    def parse_answer_tags(self, completion: str) -> str:
        """Parse the answer using <answer>...</answer> tags."""
        entry_pattern = "|".join(self.ENTRY_ANSWER)
        exit_pattern = "|".join(self.EXIT_ANSWER)

        full_pattern = (
            rf"(?:{entry_pattern})((?:(?!{entry_pattern}).)*?)(?:{exit_pattern})"
        )
        matches: List[str] = re.findall(
            full_pattern,
            completion,
            flags=re.DOTALL,
        )
        if len(matches) > 0:
            for entry_tag in self.ENTRY_ANSWER:
                entry_tag_no_escape = entry_tag.replace(r"\\", "")
                assert entry_tag not in matches[-1], (
                    f"Entry tag {entry_tag} found in extracted answer."
                )
                assert entry_tag_no_escape not in matches[-1], (
                    f"Entry tag {entry_tag_no_escape} found in extracted answer."
                )
            return matches[-1]
        return ""

    def parse_boxed(self, completion: str) -> str:
        """Parse the answer enclosed `\boxed{...}`."""

        boxed_pattern = r"\\boxed\{((?:(?!\\boxed\{).)*?)\}"
        matches: List[str] = re.findall(
            boxed_pattern,
            completion,
            flags=re.DOTALL,
        )
        if len(matches) > 0:
            return matches[-1]
        return ""

    def parse_answer(self, completion: str) -> str:
        """Parse the answer from a model completion.

        Args:
            completion: The full text completion from the model.
        Returns:
            The extracted answer string.
        """
        if (
            hasattr(self.verifier_config, "parsing_method")
            and self.verifier_config.parsing_method == "none"
        ):
            # We just need to not match any special token (which we will assume to be in the format: <...>) so we
            # replace < and > by spaces
            return self.parse_none(completion)

        tags_extraction = self.parse_answer_tags(completion)
        if hasattr(self.verifier_config, "parsing_method"):
            if self.verifier_config.parsing_method == "answer_tags":
                return tags_extraction
            elif self.verifier_config.parsing_method == "boxed":
                boxed_extraction = self.parse_boxed(tags_extraction)
                return boxed_extraction
            else:
                raise ValueError(
                    f"Unknown parsing method: {self.verifier_config.parsing_method}"
                )
        return tags_extraction

    def get_score(self, inputs: BatchVerifiersInputModel) -> List[Any]:
        """Compute scores for a batch of inputs.

        Args:
            inputs: Batch of completions and metadata to verify.

        Returns:
            List of VerifierOutputModel containing rewards and metadata.

        Raises:
            NotImplementedError: This method must be overridden by subclasses.
        """
        raise NotImplementedError
