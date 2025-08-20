"""
Expectation-over-Transformations (EOT) module for robustness

Applies differentiable transformations to simulate real-world document processing:
- JPEG compression
- Random resizing  
- Gaussian blur
- Gamma correction
- Brightness/contrast adjustments
"""

from __future__ import annotations

import random
from typing import Callable

import albumentations as A
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class EOT:
    """
    Expectation-over-Transformations sampler for robust adversarial attacks

    Simulates realistic document processing pipeline with differentiable transforms
    """

    def __init__(
        self,
        n: int = 4,
        jpeg_q: tuple[int, int] = (40, 85),
        resize: tuple[float, float] = (0.8, 1.2),
        blur_p: float = 0.5,
        gamma: tuple[float, float] = (0.9, 1.1),
        seed: int = 1337,
    ):
        """
        Initialize EOT sampler

        Args:
            n: Number of transformation samples for expectation
            jpeg_q: JPEG quality range (min, max)
            resize: Random resize scale range
            blur_p: Probability of applying Gaussian blur
            gamma: Gamma correction range
            seed: Random seed for reproducibility
        """
        self.n = n
        self.resize_range = resize
        random.seed(seed)

        # Albumentations for realistic image corruptions
        self.jpeg = A.ImageCompression(
            quality_lower=jpeg_q[0], quality_upper=jpeg_q[1], compression_type="jpeg", p=1.0
        )

        self.photo = A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.05, contrast_limit=0.05, p=0.5
                ),
                A.GaussianBlur(blur_limit=(3, 5), p=blur_p),
                A.RandomGamma(
                    gamma_limit=(int(gamma[0] * 100), int(gamma[1] * 100)), p=0.5
                ),
            ]
        )

    def _alb_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Albumentations transforms to tensor"""
        b, c, h, w = x.shape
        transformed = []

        for i in range(b):
            # Convert to HWC uint8 format for Albumentations
            img = x[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            img = (img * 255).astype("uint8")

            # Apply JPEG compression
            img = self.jpeg(image=img)["image"]

            # Apply photometric transforms
            img = self.photo(image=img)["image"]

            # Convert back to CHW float tensor
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            transformed.append(img_tensor)

        return torch.stack(transformed, 0).to(x.device)

    def _random_resize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random resizing within specified range"""
        scale = random.uniform(*self.resize_range)
        h, w = x.shape[-2:]
        new_h, new_w = int(h * scale), int(w * scale)

        # Use bicubic interpolation for smooth resizing
        resized = F.interpolate(
            x, size=(new_h, new_w), mode="bicubic", align_corners=False
        )

        # Resize back to original dimensions for consistent processing
        output = F.interpolate(resized, size=(h, w), mode="bicubic", align_corners=False)

        return output.clamp(0, 1)

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a single random transformation sample

        Args:
            x: Input tensor of shape (B, C, H, W) in range [0, 1]

        Returns:
            Transformed tensor of same shape
        """
        # Apply random resize
        x = self._random_resize(x)

        # Apply Albumentations transforms
        x = self._alb_transform(x)

        return x.clamp(0, 1)

    def expectation(
        self, x: torch.Tensor, loss_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute expectation of loss over transformation distribution

        Args:
            x: Input tensor to transform
            loss_fn: Loss function to evaluate on transformed inputs

        Returns:
            Mean loss over n transformation samples
        """
        losses = []

        for _ in range(self.n):
            x_transformed = self.sample(x)
            loss = loss_fn(x_transformed)
            losses.append(loss)

        return torch.stack(losses).mean()

    def __call__(
        self, x: torch.Tensor, loss_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """Alias for expectation method"""
        return self.expectation(x, loss_fn)
