from __future__ import annotations

from typing import Any

import numpy as np


def engagement_threshold_grid(
    start: float = 0.10,
    stop: float = 0.90,
    step: float = 0.05,
) -> np.ndarray:
    """Return an inclusive, numerically stable engagement-threshold grid."""

    if not 0.0 <= start <= stop <= 1.0:
        raise ValueError("Threshold bounds must satisfy 0 <= start <= stop <= 1")
    if step <= 0.0:
        raise ValueError("Threshold step must be positive")
    count = int(np.floor((stop - start) / step + 1e-9)) + 1
    values = start + step * np.arange(count, dtype=np.float64)
    if values[-1] < stop - 1e-9:
        values = np.append(values, stop)
    return np.round(values, 10)


def binary_calibration_metrics(
    probabilities: np.ndarray | list[float],
    outcomes: np.ndarray | list[float],
    *,
    num_bins: int = 10,
) -> dict[str, Any]:
    """Compute Brier score, ECE, and reliability-bin statistics."""

    predicted = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    observed = np.asarray(outcomes, dtype=np.float64).reshape(-1)
    if predicted.shape != observed.shape or predicted.size == 0:
        raise ValueError("Probabilities and outcomes must be non-empty and aligned")
    if np.any((predicted < 0.0) | (predicted > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1]")
    if np.any((observed != 0.0) & (observed != 1.0)):
        raise ValueError("Outcomes must be binary")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.minimum(np.digitize(predicted, edges[1:-1]), num_bins - 1)
    bins: list[dict[str, float | int]] = []
    ece = 0.0
    for bin_index in range(num_bins):
        selected = indices == bin_index
        count = int(np.sum(selected))
        mean_probability = float(np.mean(predicted[selected])) if count else float("nan")
        observed_rate = float(np.mean(observed[selected])) if count else float("nan")
        if count:
            ece += count / predicted.size * abs(mean_probability - observed_rate)
        bins.append(
            {
                "bin_index": bin_index,
                "lower": float(edges[bin_index]),
                "upper": float(edges[bin_index + 1]),
                "count": count,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
            }
        )
    return {
        "count": int(predicted.size),
        "brier_score": float(np.mean(np.square(predicted - observed))),
        "ece": float(ece),
        "mean_probability": float(np.mean(predicted)),
        "observed_rate": float(np.mean(observed)),
        "bins": bins,
    }


def hierarchical_counterfactual_advantages(
    *,
    q_noop: np.ndarray,
    q_targets: np.ndarray,
    engage_probabilities: np.ndarray,
    target_probabilities: np.ndarray,
    legal_target_mask: np.ndarray,
    selected_actions: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Decompose masked action values into engagement and target advantages.

    Arrays use leading dimensions for samples/units and a final target axis.
    Target probabilities are renormalized over the dynamic legal set.
    """

    noop = np.asarray(q_noop, dtype=np.float64)
    targets = np.asarray(q_targets, dtype=np.float64)
    engage = np.asarray(engage_probabilities, dtype=np.float64)
    target_policy = np.asarray(target_probabilities, dtype=np.float64)
    legal = np.asarray(legal_target_mask, dtype=bool)
    if targets.shape != target_policy.shape or targets.shape != legal.shape:
        raise ValueError("Target values, probabilities, and masks must align")
    if targets.shape[:-1] != noop.shape or engage.shape != noop.shape:
        raise ValueError("Leading sample/unit dimensions must align")
    if np.any((engage < 0.0) | (engage > 1.0)):
        raise ValueError("Engagement probabilities must lie in [0, 1]")
    if np.any(target_policy < 0.0):
        raise ValueError("Target probabilities cannot be negative")

    masked_policy = np.where(legal, target_policy, 0.0)
    mass = masked_policy.sum(axis=-1, keepdims=True)
    actionable = legal.any(axis=-1)
    if np.any(actionable & (mass.squeeze(-1) <= 0.0)):
        raise ValueError("Every actionable row needs positive legal target mass")
    normalized_policy = np.divide(
        masked_policy,
        mass,
        out=np.zeros_like(masked_policy),
        where=mass > 0.0,
    )
    effective_engage = np.where(actionable, engage, 0.0)
    q_engage = np.sum(normalized_policy * targets, axis=-1)
    q_engage = np.where(actionable, q_engage, noop)
    baseline = effective_engage * q_engage + (1.0 - effective_engage) * noop
    engagement_advantage_engage = q_engage - baseline
    engagement_advantage_noop = noop - baseline
    target_advantages = np.where(legal, targets - q_engage[..., None], np.nan)

    result = {
        "normalized_target_probabilities": normalized_policy,
        "q_engage": q_engage,
        "counterfactual_baseline": baseline,
        "engagement_advantage_engage": engagement_advantage_engage,
        "engagement_advantage_noop": engagement_advantage_noop,
        "target_advantages": target_advantages,
    }
    if selected_actions is not None:
        actions = np.asarray(selected_actions, dtype=np.int64)
        if actions.shape != noop.shape:
            raise ValueError("Selected actions must match sample/unit dimensions")
        noop_action = targets.shape[-1]
        if np.any((actions < 0) | (actions > noop_action)):
            raise ValueError("Selected action is outside the target/no-op range")
        selected_is_noop = actions == noop_action
        clipped = np.minimum(actions, noop_action - 1)
        selected_target_advantage = np.take_along_axis(
            np.nan_to_num(target_advantages, nan=0.0), clipped[..., None], axis=-1
        ).squeeze(-1)
        selected_engagement_advantage = np.where(
            selected_is_noop,
            engagement_advantage_noop,
            engagement_advantage_engage,
        )
        result["selected_engagement_advantage"] = selected_engagement_advantage
        result["selected_target_advantage"] = np.where(
            selected_is_noop, 0.0, selected_target_advantage
        )
        result["selected_total_advantage"] = (
            result["selected_engagement_advantage"]
            + result["selected_target_advantage"]
        )
    return result


def one_step_td_error(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    *,
    gamma: float,
) -> np.ndarray:
    """Compute one-step TD residuals with terminal bootstrapping disabled."""

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")
    reward_array = np.asarray(rewards, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    next_value_array = np.asarray(next_values, dtype=np.float64)
    terminal_array = np.asarray(terminated, dtype=bool)
    if not (
        reward_array.shape
        == value_array.shape
        == next_value_array.shape
        == terminal_array.shape
    ):
        raise ValueError("TD arrays must have identical shapes")
    return reward_array + gamma * (~terminal_array) * next_value_array - value_array
