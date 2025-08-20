"""
Constraint enforcement and perceptual metrics

Provides utilities for:
- L∞ and L2 norm projection
- LPIPS perceptual distance
- SSIM structural similarity
- Constraint satisfaction checking
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as compute_ssim

# Global LPIPS model (lazy loading)
_lpips_model = None


def get_lpips_model(device: torch.device | str = "cuda"):
    """Get or initialize LPIPS model (lazy loading)"""
    global _lpips_model
    if _lpips_model is None:
        import lpips

        _lpips_model = lpips.LPIPS(net="alex", verbose=False)
        _lpips_model.to(device)
        _lpips_model.eval()
    return _lpips_model


def lpips_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute LPIPS perceptual distance

    Args:
        x, y: Image tensors of shape (B, C, H, W) in range [0, 1]

    Returns:
        Mean LPIPS distance
    """
    lpips_fn = get_lpips_model(x.device)
    with torch.no_grad():
        # LPIPS expects inputs in [-1, 1]
        x_norm = 2 * x - 1
        y_norm = 2 * y - 1
        distance = lpips_fn(x_norm, y_norm)
    return distance.mean()


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute SSIM structural similarity

    Args:
        x, y: Image tensors of shape (B, C, H, W) in range [0, 1]

    Returns:
        Mean SSIM value (higher is more similar)
    """
    return compute_ssim(x, y, data_range=1.0, size_average=True)


def ssim_ok(x: torch.Tensor, y: torch.Tensor, threshold: float = 0.95) -> bool:
    """
    Check if SSIM constraint is satisfied

    Args:
        x, y: Image tensors
        threshold: Minimum SSIM value required

    Returns:
        True if SSIM >= threshold
    """
    return float(ssim(x, y)) >= threshold


def project_linf(
    x_adv: torch.Tensor, x_clean: torch.Tensor, eps: float
) -> torch.Tensor:
    """
    Project perturbation to L∞ ball

    Args:
        x_adv: Adversarial image
        x_clean: Clean image
        eps: Maximum L∞ perturbation

    Returns:
        Projected adversarial image
    """
    delta = x_adv - x_clean
    delta_clipped = delta.clamp(-eps, eps)
    x_projected = x_clean + delta_clipped
    return x_projected.clamp(0, 1)


def project_l2(
    x_adv: torch.Tensor, x_clean: torch.Tensor, eps: float
) -> torch.Tensor:
    """
    Project perturbation to L2 ball

    Args:
        x_adv: Adversarial image
        x_clean: Clean image
        eps: Maximum L2 perturbation

    Returns:
        Projected adversarial image
    """
    delta = x_adv - x_clean
    delta_flat = delta.view(delta.shape[0], -1)
    l2_norm = delta_flat.norm(dim=1, keepdim=True)

    # Scale down if exceeds budget
    scale = torch.minimum(torch.ones_like(l2_norm), eps / (l2_norm + 1e-12))
    delta_scaled = delta_flat * scale
    delta_scaled = delta_scaled.view_as(delta)

    x_projected = x_clean + delta_scaled
    return x_projected.clamp(0, 1)


def check_constraints(
    x_adv: torch.Tensor,
    x_clean: torch.Tensor,
    eps_linf: float | None = None,
    eps_l2: float | None = None,
    ssim_min: float | None = None,
    lpips_max: float | None = None,
) -> dict[str, bool]:
    """
    Check if all constraints are satisfied

    Args:
        x_adv: Adversarial image
        x_clean: Clean image
        eps_linf: Maximum L∞ perturbation
        eps_l2: Maximum L2 perturbation
        ssim_min: Minimum SSIM value
        lpips_max: Maximum LPIPS distance

    Returns:
        Dictionary of constraint names to satisfaction status
    """
    results = {}

    if eps_linf is not None:
        delta = (x_adv - x_clean).abs()
        results["linf"] = float(delta.max()) <= eps_linf

    if eps_l2 is not None:
        delta = x_adv - x_clean
        l2_norm = delta.view(delta.shape[0], -1).norm(dim=1).max()
        results["l2"] = float(l2_norm) <= eps_l2

    if ssim_min is not None:
        ssim_val = ssim(x_adv, x_clean)
        results["ssim"] = float(ssim_val) >= ssim_min

    if lpips_max is not None:
        lpips_val = lpips_loss(x_adv, x_clean)
        results["lpips"] = float(lpips_val) <= lpips_max

    return results


def project_all_constraints(
    x_adv: torch.Tensor,
    x_clean: torch.Tensor,
    eps_linf: float | None = None,
    eps_l2: float | None = None,
    ssim_min: float = 0.95,
) -> torch.Tensor:
    """
    Project to satisfy all constraints

    Applies projections in order of strictness.
    Note: SSIM constraint is checked but not directly projected.

    Args:
        x_adv: Adversarial image
        x_clean: Clean image
        eps_linf: Maximum L∞ perturbation
        eps_l2: Maximum L2 perturbation
        ssim_min: Minimum SSIM (checked, not projected)

    Returns:
        Projected adversarial image
    """
    x_proj = x_adv

    # Apply norm projections
    if eps_linf is not None:
        x_proj = project_linf(x_proj, x_clean, eps_linf)

    if eps_l2 is not None:
        x_proj = project_l2(x_proj, x_clean, eps_l2)

    # Check SSIM and fall back if violated
    if not ssim_ok(x_proj, x_clean, ssim_min):
        # If SSIM is violated after projection, reduce perturbation
        alpha = 0.5  # Reduce perturbation by half
        x_proj = x_clean + alpha * (x_proj - x_clean)

    return x_proj.clamp(0, 1)


# TODO: Add more sophisticated perceptual constraints for v2
class PerceptualConstraint:
    """
    Advanced perceptual constraint manager (TODO for v2)

    Will support:
    - Multi-scale SSIM
    - Feature-based perceptual metrics
    - Semantic preservation constraints
    """

    def __init__(self):
        raise NotImplementedError("Advanced constraints planned for v2")
