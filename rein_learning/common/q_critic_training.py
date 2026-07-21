from __future__ import annotations

from itertools import combinations
from typing import Hashable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def action_group_ids(
    state_ids: Sequence[Hashable], unit_indices: Sequence[int]
) -> np.ndarray:
    states = np.asarray(state_ids)
    units = np.asarray(unit_indices)
    if states.ndim != 1 or states.shape != units.shape or states.size == 0:
        raise ValueError("state_ids and unit_indices must be non-empty and aligned")
    return np.asarray(
        [f"{state_id}/unit{int(unit)}" for state_id, unit in zip(states, units)]
    )


def integer_group_codes(group_ids: Sequence[Hashable]) -> np.ndarray:
    groups = np.asarray(group_ids)
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError("group_ids must be a non-empty one-dimensional sequence")
    _, codes = np.unique(groups, return_inverse=True)
    return codes.astype(np.int64)


def center_by_group(values: torch.Tensor, group_codes: torch.Tensor) -> torch.Tensor:
    values = values.reshape(-1)
    codes = group_codes.long().reshape(-1)
    if values.shape != codes.shape or values.numel() == 0:
        raise ValueError("values and group_codes must be non-empty and aligned")
    if bool(torch.any(codes < 0)):
        raise ValueError("group codes must be non-negative")
    group_count = int(codes.max().item()) + 1
    sums = torch.zeros(group_count, device=values.device, dtype=values.dtype)
    counts = torch.zeros(group_count, device=values.device, dtype=values.dtype)
    sums.scatter_add_(0, codes, values)
    counts.scatter_add_(0, codes, torch.ones_like(values))
    return values - (sums / counts.clamp_min(1.0))[codes]


def build_pairwise_training_data(
    labels: Sequence[float],
    group_ids: Sequence[Hashable],
    return_samples: np.ndarray,
    *,
    reliability_floor: float = 0.25,
    reliability_ceiling: float = 4.0,
    uncertainty_z: float = 1.96,
) -> dict[str, np.ndarray]:
    truth = np.asarray(labels, dtype=np.float64)
    groups = np.asarray(group_ids)
    samples = np.asarray(return_samples, dtype=np.float64)
    if truth.ndim != 1 or truth.shape != groups.shape or truth.size == 0:
        raise ValueError("labels and group_ids must be non-empty and aligned")
    if samples.ndim != 2 or samples.shape[0] != truth.size:
        raise ValueError("return_samples must have shape [rows, rollouts]")
    if reliability_floor <= 0.0 or reliability_ceiling < reliability_floor:
        raise ValueError("invalid reliability bounds")

    left_indices: list[int] = []
    right_indices: list[int] = []
    standard_errors: list[float] = []
    reliabilities: list[float] = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        for left, right in combinations(indices.tolist(), 2):
            finite = np.isfinite(samples[left]) & np.isfinite(samples[right])
            paired = samples[left, finite] - samples[right, finite]
            if len(paired) < 2:
                raise ValueError("Every training pair needs at least two paired returns")
            standard_error = float(np.std(paired, ddof=1) / np.sqrt(len(paired)))
            difference = float(truth[left] - truth[right])
            denominator = uncertainty_z * standard_error
            if denominator == 0.0:
                reliability = reliability_ceiling if difference != 0.0 else reliability_floor
            else:
                reliability = abs(difference) / denominator
            left_indices.append(left)
            right_indices.append(right)
            standard_errors.append(standard_error)
            reliabilities.append(
                float(np.clip(reliability, reliability_floor, reliability_ceiling))
            )
    if not left_indices:
        raise ValueError("At least one group must contain two candidate actions")
    return {
        "left": np.asarray(left_indices, dtype=np.int64),
        "right": np.asarray(right_indices, dtype=np.int64),
        "standard_error": np.asarray(standard_errors, dtype=np.float32),
        "reliability": np.asarray(reliabilities, dtype=np.float32),
    }


def q_critic_training_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    group_codes: torch.Tensor,
    *,
    pair_left: torch.Tensor | None = None,
    pair_right: torch.Tensor | None = None,
    pair_weights: torch.Tensor | None = None,
    centered_weight: float = 0.0,
    pairwise_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)
    if predictions.shape != labels.shape:
        raise ValueError("predictions and labels must align")
    absolute = F.mse_loss(predictions, labels)
    centered = F.mse_loss(
        center_by_group(predictions, group_codes),
        center_by_group(labels, group_codes),
    )
    pairwise = torch.zeros((), device=predictions.device, dtype=predictions.dtype)
    if pairwise_weight > 0.0:
        if pair_left is None or pair_right is None or pair_weights is None:
            raise ValueError("pairwise loss requires indices and weights")
        predicted_difference = predictions[pair_left] - predictions[pair_right]
        label_difference = labels[pair_left] - labels[pair_right]
        raw_pairwise = F.smooth_l1_loss(
            predicted_difference, label_difference, reduction="none"
        )
        weights = pair_weights.to(predictions.dtype)
        pairwise = torch.sum(raw_pairwise * weights) / weights.sum().clamp_min(1e-8)
    total = absolute + centered_weight * centered + pairwise_weight * pairwise
    return total, {
        "absolute": absolute,
        "centered": centered,
        "pairwise": pairwise,
    }


def validation_difference_score(
    labels: Sequence[float],
    predictions: Sequence[float],
    group_ids: Sequence[Hashable],
    *,
    scale: float,
) -> dict[str, float]:
    truth = np.asarray(labels, dtype=np.float64)
    estimate = np.asarray(predictions, dtype=np.float64)
    groups = np.asarray(group_ids)
    if truth.shape != estimate.shape or truth.shape != groups.shape or truth.size == 0:
        raise ValueError("validation arrays must be non-empty and aligned")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    absolute_mae = float(np.mean(np.abs(estimate - truth)) / scale)
    centered_errors: list[np.ndarray] = []
    for group in np.unique(groups):
        selected = groups == group
        centered_truth = truth[selected] - np.mean(truth[selected])
        centered_estimate = estimate[selected] - np.mean(estimate[selected])
        centered_errors.append(np.abs(centered_estimate - centered_truth))
    centered_mae = float(np.mean(np.concatenate(centered_errors)) / scale)
    return {
        "absolute_mae": absolute_mae,
        "centered_mae": centered_mae,
        "score": absolute_mae + centered_mae,
    }
