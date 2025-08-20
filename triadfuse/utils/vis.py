"""Visualization utilities"""

from __future__ import annotations

import torch
import torchvision.utils as vutils
from PIL import Image


def save_image_grid(
    images: torch.Tensor | list[torch.Tensor],
    path: str,
    nrow: int = 4,
    normalize: bool = True,
    labels: list[str] | None = None,
):
    """Save a grid of images with optional labels"""
    if isinstance(images, list):
        images = torch.stack(images)

    vutils.save_image(images, path, nrow=nrow, normalize=normalize, value_range=(0, 1))

    # TODO: Add label overlay support in v2
    if labels:
        pass


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor to PIL Image"""
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)

    img = (tensor.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(img)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to tensor"""
    import torchvision.transforms as T

    transform = T.ToTensor()
    return transform(img).unsqueeze(0)
