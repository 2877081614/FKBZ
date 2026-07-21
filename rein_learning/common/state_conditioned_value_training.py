from __future__ import annotations

from typing import Mapping, NamedTuple, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .balanced_engagement_training import balanced_engagement_loss


class ConstrainedValueLossOutput(NamedTuple):
    safety_gain: torch.Tensor
    cost_delta: torch.Tensor
    budget_multiplier: torch.Tensor
    score: torch.Tensor


def engagement_delta_targets(
    components: Mapping[str, np.ndarray],
    *,
    damage_weight: float = 30.0,
    high_threat_leak_weight: float = 20.0,
) -> dict[str, np.ndarray]:
    damage = np.asarray(components["damage_samples"], dtype=np.float64)
    leaks = np.asarray(components["high_threat_leak_samples"], dtype=np.float64)
    costs = np.asarray(components["resource_cost_samples"], dtype=np.float64)
    if damage.shape != leaks.shape or damage.shape != costs.shape:
        raise ValueError("Component samples must have matching shapes")
    if damage.ndim != 3 or damage.shape[1] != 2 or damage.shape[2] < 2:
        raise ValueError("Components must have shape [groups, 2, rollouts>=2]")
    safety_samples = (
        damage_weight * (damage[:, 0] - damage[:, 1])
        + high_threat_leak_weight * (leaks[:, 0] - leaks[:, 1])
    )
    cost_samples = costs[:, 1] - costs[:, 0]
    return {
        "safety_gain": np.mean(safety_samples, axis=1).astype(np.float32),
        "cost_delta": np.mean(cost_samples, axis=1).astype(np.float32),
        "safety_gain_samples": safety_samples.astype(np.float32),
        "cost_delta_samples": cost_samples.astype(np.float32),
    }


def state_conditioned_value_loss(
    output: ConstrainedValueLossOutput,
    safety_targets: torch.Tensor,
    cost_targets: torch.Tensor,
    oracle_labels: torch.Tensor,
    *,
    classification_weight: float = 1.0,
    margin_weight: float = 0.25,
    budget_regularization: float = 0.001,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    safety_targets = safety_targets.reshape(-1)
    cost_targets = cost_targets.reshape(-1)
    if output.safety_gain.shape != safety_targets.shape:
        raise ValueError("Safety targets must align with model output")
    if output.cost_delta.shape != cost_targets.shape:
        raise ValueError("Cost targets must align with model output")
    if classification_weight < 0.0 or budget_regularization < 0.0:
        raise ValueError("Loss weights must be non-negative")
    safety_loss = F.smooth_l1_loss(output.safety_gain, safety_targets)
    cost_loss = F.smooth_l1_loss(output.cost_delta, cost_targets)
    classification_logits = torch.stack(
        (torch.zeros_like(output.score), output.score), dim=1
    )
    classification, classification_parts = balanced_engagement_loss(
        classification_logits,
        oracle_labels,
        margin_weight=margin_weight,
    )
    budget_penalty = torch.mean(output.budget_multiplier)
    total = (
        safety_loss
        + cost_loss
        + classification_weight * classification
        + budget_regularization * budget_penalty
    )
    return total, {
        "safety": safety_loss,
        "cost": cost_loss,
        "classification": classification,
        "classification_bce": classification_parts["bce"],
        "classification_margin": classification_parts["margin"],
        "budget_penalty": budget_penalty,
    }


def paired_delta_reliability(
    samples: np.ndarray,
    *,
    uncertainty_z: float = 1.645,
    floor: float = 0.25,
    ceiling: float = 4.0,
) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("samples must have shape [groups, rollouts>=2]")
    if uncertainty_z <= 0.0 or floor <= 0.0 or ceiling < floor:
        raise ValueError("Reliability parameters are invalid")
    means = np.mean(values, axis=1)
    standard_errors = np.std(values, axis=1, ddof=1) / np.sqrt(values.shape[1])
    reliability = np.empty(len(values), dtype=np.float64)
    deterministic = standard_errors <= 1e-12
    reliability[deterministic & (np.abs(means) > 1e-12)] = ceiling
    reliability[deterministic & (np.abs(means) <= 1e-12)] = floor
    stochastic = ~deterministic
    reliability[stochastic] = np.abs(means[stochastic]) / (
        uncertainty_z * standard_errors[stochastic]
    )
    reliability = np.clip(reliability, floor, ceiling)
    return (reliability / np.mean(reliability)).astype(np.float32)


def scenario_class_balanced_loss(
    scores: torch.Tensor,
    oracle_labels: torch.Tensor,
    scenarios: Sequence[object],
    *,
    margin_weight: float = 0.25,
    margin: float = 1.0,
    worst_block_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scores = scores.reshape(-1)
    labels = oracle_labels.long().reshape(-1)
    scenario_values = np.asarray(scenarios).reshape(-1)
    if scores.shape != labels.shape or len(scenario_values) != len(scores):
        raise ValueError("Scores, labels, and scenarios must align")
    if margin_weight < 0.0 or margin <= 0.0 or worst_block_weight < 0.0:
        raise ValueError("Robust classification weights are invalid")
    valid = labels >= 0
    if not bool(torch.any(valid)):
        raise ValueError("At least one reliable oracle label is required")
    block_losses: list[torch.Tensor] = []
    bce_losses: list[torch.Tensor] = []
    margin_losses: list[torch.Tensor] = []
    targets = labels.to(scores.dtype)
    signs = targets * 2.0 - 1.0
    for scenario in np.unique(scenario_values):
        scenario_mask = torch.as_tensor(
            scenario_values == scenario, device=scores.device
        )
        for class_label in (0, 1):
            selected = valid & scenario_mask & (labels == class_label)
            if not bool(torch.any(selected)):
                continue
            bce = F.binary_cross_entropy_with_logits(
                scores[selected], targets[selected]
            )
            margin_loss = torch.mean(
                torch.relu(
                    torch.as_tensor(margin, device=scores.device)
                    - signs[selected] * scores[selected]
                )
            )
            bce_losses.append(bce)
            margin_losses.append(margin_loss)
            block_losses.append(bce + margin_weight * margin_loss)
    if len(block_losses) < 2:
        raise ValueError("Robust classification requires at least two blocks")
    blocks = torch.stack(block_losses)
    mean_block = torch.mean(blocks)
    worst_block = torch.max(blocks)
    total = mean_block + worst_block_weight * worst_block
    return total, {
        "mean_block": mean_block,
        "worst_block": worst_block,
        "bce": torch.mean(torch.stack(bce_losses)),
        "margin": torch.mean(torch.stack(margin_losses)),
        "block_count": torch.as_tensor(
            float(len(block_losses)), device=scores.device
        ),
    }


def robust_state_conditioned_value_loss(
    output: ConstrainedValueLossOutput,
    safety_targets: torch.Tensor,
    cost_targets: torch.Tensor,
    oracle_labels: torch.Tensor,
    scenarios: Sequence[object],
    *,
    cost_reliability: torch.Tensor | None = None,
    classification_weight: float = 1.0,
    margin_weight: float = 0.25,
    worst_block_weight: float = 0.5,
    budget_regularization: float = 0.001,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    safety_targets = safety_targets.reshape(-1)
    cost_targets = cost_targets.reshape(-1)
    if output.safety_gain.shape != safety_targets.shape:
        raise ValueError("Safety targets must align with model output")
    if output.cost_delta.shape != cost_targets.shape:
        raise ValueError("Cost targets must align with model output")
    safety_loss = F.smooth_l1_loss(output.safety_gain, safety_targets)
    raw_cost = F.smooth_l1_loss(
        output.cost_delta, cost_targets, reduction="none"
    )
    if cost_reliability is None:
        cost_loss = torch.mean(raw_cost)
    else:
        weights = cost_reliability.reshape(-1).to(raw_cost.dtype)
        if weights.shape != raw_cost.shape or bool(torch.any(weights <= 0.0)):
            raise ValueError("Cost reliability must be positive and aligned")
        cost_loss = torch.sum(raw_cost * weights) / weights.sum()
    classification, classification_parts = scenario_class_balanced_loss(
        output.score,
        oracle_labels,
        scenarios,
        margin_weight=margin_weight,
        worst_block_weight=worst_block_weight,
    )
    budget_penalty = torch.mean(output.budget_multiplier)
    total = (
        safety_loss
        + cost_loss
        + classification_weight * classification
        + budget_regularization * budget_penalty
    )
    return total, {
        "safety": safety_loss,
        "cost": cost_loss,
        "classification": classification,
        "classification_bce": classification_parts["bce"],
        "classification_margin": classification_parts["margin"],
        "worst_block": classification_parts["worst_block"],
        "budget_penalty": budget_penalty,
    }


def constrained_value_metrics(
    safety_targets: np.ndarray,
    cost_targets: np.ndarray,
    safety_predictions: np.ndarray,
    cost_predictions: np.ndarray,
) -> dict[str, float]:
    safety = np.asarray(safety_targets, dtype=np.float64)
    cost = np.asarray(cost_targets, dtype=np.float64)
    predicted_safety = np.asarray(safety_predictions, dtype=np.float64)
    predicted_cost = np.asarray(cost_predictions, dtype=np.float64)
    if not (
        safety.shape == cost.shape == predicted_safety.shape == predicted_cost.shape
    ) or safety.ndim != 1:
        raise ValueError("Value metric arrays must be aligned vectors")

    def correlation(left: np.ndarray, right: np.ndarray) -> float:
        if np.std(left) == 0.0 or np.std(right) == 0.0:
            return float("nan")
        return float(np.corrcoef(left, right)[0, 1])

    nonzero_safety = np.abs(safety) > 1e-8
    safety_sign_accuracy = (
        float(
            np.mean(
                np.sign(safety[nonzero_safety])
                == np.sign(predicted_safety[nonzero_safety])
            )
        )
        if np.any(nonzero_safety)
        else float("nan")
    )
    return {
        "safety_mae": float(np.mean(np.abs(predicted_safety - safety))),
        "cost_mae": float(np.mean(np.abs(predicted_cost - cost))),
        "safety_correlation": correlation(safety, predicted_safety),
        "cost_correlation": correlation(cost, predicted_cost),
        "safety_sign_accuracy": safety_sign_accuracy,
    }
