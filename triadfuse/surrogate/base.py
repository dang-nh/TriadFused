"""Base class for surrogate models"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


class SurrogateModel(nn.Module, ABC):
    """
    Abstract base class for surrogate VLM models

    All surrogate models must implement methods for:
    - Computing task-specific loss with gradients
    - Making predictions for evaluation
    """

    @abstractmethod
    def forward_task_loss(
        self, image: torch.Tensor, prompt: str, target: str
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute task-specific loss for optimization

        Args:
            image: Input image tensor (B, C, H, W) in range [0, 1]
            prompt: Task prompt or question
            target: Target answer or output

        Returns:
            Tuple of (loss tensor, auxiliary features dict)
        """
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def predict(self, image: torch.Tensor, prompt: str) -> str:
        """
        Generate prediction for evaluation

        Args:
            image: Input image tensor
            prompt: Task prompt or question

        Returns:
            Generated text response
        """
        raise NotImplementedError

    def tokenize(self, text: str) -> torch.Tensor:
        """
        Tokenize text input (optional, model-specific)

        Args:
            text: Input text

        Returns:
            Token tensor
        """
        raise NotImplementedError("Tokenization is model-specific")

    def get_features(self, image: torch.Tensor, layer: str | None = None) -> torch.Tensor:
        """
        Extract intermediate features (optional, for feature-level losses)

        Args:
            image: Input image tensor
            layer: Name of layer to extract features from

        Returns:
            Feature tensor
        """
        raise NotImplementedError("Feature extraction is model-specific")
