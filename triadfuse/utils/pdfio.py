"""PDF I/O utilities for document processing"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def pdf_to_images(
    pdf_path: str | Path, dpi: int = 150, max_pages: int | None = None
) -> list[np.ndarray]:
    """
    Convert PDF pages to images

    TODO: Implement full PyMuPDF integration in v2
    Currently returns empty list as stub
    """
    # TODO: Implement using PyMuPDF
    # import fitz
    # doc = fitz.open(pdf_path)
    # images = []
    # for i, page in enumerate(doc):
    #     if max_pages and i >= max_pages:
    #         break
    #     pix = page.get_pixmap(dpi=dpi)
    #     img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    #     images.append(img)
    # return images
    return []


def rasterize_at_dpi(img: torch.Tensor | np.ndarray, source_dpi: int, target_dpi: int) -> torch.Tensor:
    """
    Simulate DPI change for print/scan robustness

    TODO: Implement proper DPI-aware resizing in v2
    """
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)

    scale = target_dpi / source_dpi
    if scale == 1.0:
        return img

    # Simple resize simulation
    import torch.nn.functional as F

    h, w = img.shape[-2:]
    new_h, new_w = int(h * scale), int(w * scale)

    if img.dim() == 3:
        img = img.unsqueeze(0)

    resized = F.interpolate(img, size=(new_h, new_w), mode="bicubic", align_corners=False)

    # Resize back to original for consistent tensor size
    output = F.interpolate(resized, size=(h, w), mode="bicubic", align_corners=False)

    return output.squeeze(0) if img.dim() == 3 else output
