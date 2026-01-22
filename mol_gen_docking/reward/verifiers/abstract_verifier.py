"""Abstract base classes for molecular verifiers.

This module provides the base classes and input/output models used by all
verifiers in the reward system. It defines the common interface that all
verifiers must implement.
"""

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

    def __init__(self) -> None:
        """Initialize the base verifier."""
        pass

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
