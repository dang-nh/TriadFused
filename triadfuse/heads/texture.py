"""
Texture head: Low-frequency imperceptible perturbations

Generates smooth, blur-robust perturbations by optimizing a low-resolution
parameter field that is bicubically upsampled to image resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureHead(nn.Module):
    """
    Low-frequency texture perturbation generator

    Creates imperceptible perturbations robust to downsampling/compression
    by parameterizing in low-frequency space
    """

    def __init__(
        self,
        img_hw: tuple[int, int],
        scale: float = 0.01,
        lowres: int = 64,
        init_std: float = 1e-3,
    ):
        """
        Initialize texture head

        Args:
            img_hw: Target image dimensions (height, width)
            scale: Maximum perturbation magnitude
            lowres: Resolution of low-frequency parameter grid
            init_std: Standard deviation for parameter initialization
        """
        super().__init__()
        h, w = img_hw
        self.h, self.w = h, w
        self.scale = scale
        self.lowres = lowres

        # Low-resolution parameter field
        self.param = nn.Parameter(torch.zeros(1, 3, lowres, lowres))
        nn.init.normal_(self.param, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply texture perturbation to input

        Args:
            x: Input tensor of shape (B, C, H, W) in range [0, 1]

        Returns:
            Perturbed tensor clamped to [0, 1]
        """
        batch_size = x.shape[0]

        # Expand parameters to batch size if needed
        params = self.param.expand(batch_size, -1, -1, -1)

        # Upsample low-res parameters to image resolution
        # Bicubic interpolation creates smooth, low-frequency perturbations
        perturbation = F.interpolate(
            params, size=(self.h, self.w), mode="bicubic", align_corners=False
        )

        # Apply tanh to bound perturbations and scale
        perturbation = perturbation.tanh() * self.scale

        # Add perturbation and clamp to valid image range
        perturbed = x + perturbation
        return perturbed.clamp(0, 1)

    def get_perturbation(self) -> torch.Tensor:
        """Get the current perturbation pattern"""
        perturbation = F.interpolate(
            self.param, size=(self.h, self.w), mode="bicubic", align_corners=False
        )
        return (perturbation.tanh() * self.scale).squeeze(0)

    def reset_parameters(self):
        """Reset parameters to initial random state"""
        nn.init.normal_(self.param, mean=0.0, std=1e-3)

    def project_constraints(self, eps: float | None = None):
        """
        Project parameters to satisfy L∞ constraint

        Args:
            eps: Maximum L∞ perturbation (overrides self.scale if provided)
        """
        if eps is not None:
            self.scale = min(self.scale, eps)

        # Clamp parameters to reasonable range
        with torch.no_grad():
            self.param.data = self.param.data.clamp(-3, 3)


# TODO: Implement DCT-based parameterization for even better frequency control
class DCTTextureHead(nn.Module):
    """
    DCT-based texture head for precise frequency control (TODO for v2)

    Uses Discrete Cosine Transform basis functions for parameterization,
    allowing direct control over frequency components
    """

    def __init__(self, img_hw: tuple[int, int], num_frequencies: int = 16):
        super().__init__()
        # TODO: Implement DCT basis parameterization
        # - Create DCT basis functions
        # - Optimize coefficients for low frequencies only
        # - Use inverse DCT to generate perturbations
        raise NotImplementedError("DCT texture head planned for v2")
