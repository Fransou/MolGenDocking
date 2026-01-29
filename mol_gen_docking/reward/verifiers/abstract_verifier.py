"""Abstract base classes for molecular verifiers.

This module provides the base classes and input/output models used by all
verifiers in the reward system. It defines the common interface that all
verifiers must implement.
"""

import re
from typing import List

from mol_gen_docking.reward.verifiers.abstract_verifier_pydantic_model import (
    BatchVerifiersInputModel,
    VerifierOutputModel,
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

    def __init__(self) -> None:
        """Initialize the base verifier."""
        pass

    def parse_answer(self, completion: str) -> str:
        """Parse the answer from a model completion.

        Args:
            completion: The full text completion from the model.
        Returns:
            The extracted answer string.
        """
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

    def get_score(self, inputs: BatchVerifiersInputModel) -> List[VerifierOutputModel]:
        """Compute scores for a batch of inputs.

        Args:
            inputs: Batch of completions and metadata to verify.

        Returns:
            List of VerifierOutputModel containing rewards and metadata.

        Raises:
            NotImplementedError: This method must be overridden by subclasses.
        """
        raise NotImplementedError
