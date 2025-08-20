"""
Composite loss functions for TriadFuse attacks

Combines task-specific losses with perceptual and regularization terms
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .constraints import lpips_loss


def composite_loss(
    task_loss: torch.Tensor,
    x_adv: torch.Tensor,
    x_clean: torch.Tensor,
    lam_lpips: float = 0.0,
    lam_tv: float = 0.0,
    lam_l2: float = 0.0,
) -> torch.Tensor:
    """
    Compute composite adversarial loss

    Args:
        task_loss: Task-specific loss (e.g., cross-entropy)
        x_adv: Adversarial image
        x_clean: Clean image
        lam_lpips: Weight for LPIPS perceptual loss
        lam_tv: Weight for total variation regularization
        lam_l2: Weight for L2 regularization

    Returns:
        Combined loss scalar
    """
    total_loss = task_loss

    # Add perceptual loss
    if lam_lpips > 0:
        perceptual = lpips_loss(x_adv, x_clean)
        total_loss = total_loss + lam_lpips * perceptual

    # Add total variation for smoothness
    if lam_tv > 0:
        tv = total_variation(x_adv - x_clean)
        total_loss = total_loss + lam_tv * tv

    # Add L2 regularization
    if lam_l2 > 0:
        l2_norm = (x_adv - x_clean).pow(2).mean()
        total_loss = total_loss + lam_l2 * l2_norm

    return total_loss


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """
    Compute total variation for smoothness regularization

    Args:
        x: Input tensor (B, C, H, W)

    Returns:
        Mean total variation
    """
    diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
    diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]

    tv_h = diff_h.abs().mean()
    tv_w = diff_w.abs().mean()

    return tv_h + tv_w


def feature_matching_loss(
    features_adv: dict[str, torch.Tensor],
    features_clean: dict[str, torch.Tensor],
    layers: list[str] | None = None,
    distance: str = "l2",
) -> torch.Tensor:
    """
    Compute feature matching loss between clean and adversarial features

    Args:
        features_adv: Dictionary of adversarial features by layer
        features_clean: Dictionary of clean features by layer
        layers: Specific layers to use (None = all)
        distance: Distance metric ("l2", "l1", or "cosine")

    Returns:
        Mean feature distance
    """
    if layers is None:
        layers = list(features_adv.keys())

    losses = []
    for layer in layers:
        if layer not in features_adv or layer not in features_clean:
            continue

        feat_adv = features_adv[layer]
        feat_clean = features_clean[layer]

        if distance == "l2":
            loss = F.mse_loss(feat_adv, feat_clean)
        elif distance == "l1":
            loss = F.l1_loss(feat_adv, feat_clean)
        elif distance == "cosine":
            loss = 1 - F.cosine_similarity(
                feat_adv.flatten(1), feat_clean.flatten(1)
            ).mean()
        else:
            raise ValueError(f"Unknown distance metric: {distance}")

        losses.append(loss)

    return torch.stack(losses).mean() if losses else torch.tensor(0.0)


def targeted_misclassification_loss(
    logits: torch.Tensor,
    target_class: int,
    confidence: float = 0.9,
    untargeted: bool = False,
) -> torch.Tensor:
    """
    Loss for targeted or untargeted misclassification

    Args:
        logits: Model output logits
        target_class: Target class index (or true class for untargeted)
        confidence: Desired confidence level
        untargeted: If True, minimize target class probability

    Returns:
        Classification loss
    """
    probs = F.softmax(logits, dim=-1)

    if untargeted:
        # Minimize probability of true class
        loss = probs[:, target_class].mean()
    else:
        # Maximize probability of target class
        loss = -torch.log(probs[:, target_class] + 1e-12).mean()

        # Add confidence margin
        if confidence > 0:
            margin = confidence - probs[:, target_class]
            loss = loss + F.relu(margin).mean()

    return loss


# TODO: Implement advanced losses for v2
class ContrastiveLoss:
    """
    Contrastive loss for cross-modal attacks (TODO for v2)

    Will support:
    - Image-text contrastive objectives
    - Hard negative mining
    - Temperature-scaled similarities
    """

    def __init__(self):
        raise NotImplementedError("Contrastive loss planned for v2")


class SemanticConsistencyLoss:
    """
    Semantic consistency loss for layout attacks (TODO for v2)

    Will support:
    - Layout-aware objectives
    - Reading order preservation
    - Spatial relationship constraints
    """

    def __init__(self):
        raise NotImplementedError("Semantic consistency loss planned for v2")


def attack_success_rate(
    predictions: list[str], targets: list[str], mode: str = "exact"
) -> float:
    """
    Compute attack success rate

    Args:
        predictions: Model predictions
        targets: Target outputs
        mode: Comparison mode ("exact", "contains", "different")

    Returns:
        Success rate (0 to 1)
    """
    successes = 0
    for pred, target in zip(predictions, targets):
        if mode == "exact":
            success = pred.strip().lower() == target.strip().lower()
        elif mode == "contains":
            success = target.lower() in pred.lower()
        elif mode == "different":
            success = pred.strip().lower() != target.strip().lower()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        successes += int(success)

    return successes / len(predictions) if predictions else 0.0
