from __future__ import annotations

import torch


def balanced_engagement_loss(
    predictions: torch.Tensor,
    oracle_labels: torch.Tensor,
    *,
    margin_weight: float = 0.0,
    margin: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError("predictions must have shape [groups, 2]")
    labels = oracle_labels.long().reshape(-1)
    if labels.shape[0] != predictions.shape[0]:
        raise ValueError("oracle_labels must align with predictions")
    if margin_weight < 0.0 or margin <= 0.0:
        raise ValueError("margin_weight must be non-negative and margin positive")
    valid = labels >= 0
    if not bool(torch.any(valid)):
        raise ValueError("At least one reliable oracle label is required")
    labels = labels[valid]
    if not bool(torch.any(labels == 0)) or not bool(torch.any(labels == 1)):
        raise ValueError("Both engagement classes are required")
    logits = predictions[valid, 1] - predictions[valid, 0]
    targets = labels.to(predictions.dtype)
    positive_count = torch.sum(labels == 1).to(predictions.dtype)
    negative_count = torch.sum(labels == 0).to(predictions.dtype)
    weights = torch.where(
        labels == 1,
        0.5 / positive_count,
        0.5 / negative_count,
    )
    elementwise_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    bce = torch.sum(weights * elementwise_bce)
    signs = targets * 2.0 - 1.0
    margin_loss = torch.sum(
        weights * torch.relu(torch.as_tensor(margin, device=logits.device) - signs * logits)
    )
    total = bce + margin_weight * margin_loss
    return total, {
        "bce": bce,
        "margin": margin_loss,
        "positive_count": positive_count,
        "negative_count": negative_count,
    }
