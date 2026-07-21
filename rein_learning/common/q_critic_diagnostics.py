from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Hashable, Sequence

import numpy as np


def grouped_state_split(
    state_ids: Sequence[Hashable],
    *,
    strata: Sequence[Hashable] | None = None,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 14,
) -> np.ndarray:
    """Split rows by state id, optionally stratified by state-level labels."""

    ids = np.asarray(state_ids)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("state_ids must be a non-empty one-dimensional sequence")
    if validation_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("Validation and test fractions must be positive")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("Validation and test fractions must sum to less than one")
    if strata is None:
        stratum_values = np.full(ids.shape, "all", dtype=object)
    else:
        stratum_values = np.asarray(strata, dtype=object)
        if stratum_values.shape != ids.shape:
            raise ValueError("strata must align with state_ids")

    state_to_stratum: dict[Hashable, Hashable] = {}
    for state_id, stratum in zip(ids.tolist(), stratum_values.tolist()):
        previous = state_to_stratum.setdefault(state_id, stratum)
        if previous != stratum:
            raise ValueError("Each state_id must belong to exactly one stratum")

    grouped: dict[Hashable, list[Hashable]] = defaultdict(list)
    for state_id, stratum in state_to_stratum.items():
        grouped[stratum].append(state_id)
    rng = np.random.default_rng(seed)
    assignments: dict[Hashable, str] = {}
    for state_group in grouped.values():
        shuffled = np.asarray(state_group, dtype=object)
        rng.shuffle(shuffled)
        count = len(shuffled)
        if count < 3:
            raise ValueError("Every stratum needs at least three distinct states")
        validation_count = max(1, int(round(count * validation_fraction)))
        test_count = max(1, int(round(count * test_fraction)))
        while validation_count + test_count >= count:
            if validation_count >= test_count and validation_count > 1:
                validation_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                raise ValueError("A stratum is too small for a three-way split")
        for state_id in shuffled[:test_count]:
            assignments[state_id] = "test"
        for state_id in shuffled[test_count : test_count + validation_count]:
            assignments[state_id] = "validation"
        for state_id in shuffled[test_count + validation_count :]:
            assignments[state_id] = "train"
    return np.asarray([assignments[state_id] for state_id in ids.tolist()])


def regression_metrics(labels: Sequence[float], predictions: Sequence[float]) -> dict[str, float]:
    truth = np.asarray(labels, dtype=np.float64)
    estimate = np.asarray(predictions, dtype=np.float64)
    if truth.shape != estimate.shape or truth.size == 0:
        raise ValueError("Labels and predictions must be non-empty and aligned")
    errors = estimate - truth
    return {
        "count": int(truth.size),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "bias": float(np.mean(errors)),
    }


def pairwise_ranking_accuracy(
    labels: Sequence[float],
    predictions: Sequence[float],
    group_ids: Sequence[Hashable],
    *,
    standard_errors: Sequence[float] | None = None,
    return_samples: np.ndarray | None = None,
    candidate_actions: Sequence[int] | None = None,
    noop_action: int | None = None,
    target_only: bool = False,
    uncertainty_z: float = 1.96,
) -> dict[str, float | int]:
    truth = np.asarray(labels, dtype=np.float64)
    estimate = np.asarray(predictions, dtype=np.float64)
    groups = np.asarray(group_ids)
    if truth.shape != estimate.shape or truth.shape != groups.shape or truth.size == 0:
        raise ValueError("Ranking arrays must be non-empty and aligned")
    errors = (
        np.zeros_like(truth)
        if standard_errors is None
        else np.asarray(standard_errors, dtype=np.float64)
    )
    if errors.shape != truth.shape or np.any(errors < 0.0):
        raise ValueError("standard_errors must be non-negative and aligned")
    samples = None if return_samples is None else np.asarray(return_samples, dtype=np.float64)
    if samples is not None and (
        samples.ndim != 2 or samples.shape[0] != truth.shape[0] or samples.shape[1] < 2
    ):
        raise ValueError("return_samples must have shape [rows, rollouts>=2]")
    actions = None if candidate_actions is None else np.asarray(candidate_actions)
    if target_only and (actions is None or noop_action is None):
        raise ValueError("target_only ranking requires actions and noop_action")

    correct = 0.0
    count = 0
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        for left, right in combinations(indices.tolist(), 2):
            if target_only and (
                actions[left] == noop_action or actions[right] == noop_action
            ):
                continue
            difference = truth[left] - truth[right]
            if samples is None:
                difference_se = np.sqrt(errors[left] ** 2 + errors[right] ** 2)
            else:
                paired = samples[left] - samples[right]
                difference_se = float(
                    np.std(paired, ddof=1) / np.sqrt(samples.shape[1])
                )
            uncertainty = uncertainty_z * difference_se
            if abs(difference) <= uncertainty:
                continue
            predicted_difference = estimate[left] - estimate[right]
            if predicted_difference == 0.0:
                correct += 0.5
            elif np.sign(predicted_difference) == np.sign(difference):
                correct += 1.0
            count += 1
    return {
        "count": count,
        "accuracy": float(correct / count) if count else float("nan"),
    }


def top_action_accuracy(
    labels: Sequence[float],
    predictions: Sequence[float],
    group_ids: Sequence[Hashable],
    *,
    standard_errors: Sequence[float] | None = None,
    return_samples: np.ndarray | None = None,
    uncertainty_z: float = 1.96,
) -> dict[str, float | int]:
    truth = np.asarray(labels, dtype=np.float64)
    estimate = np.asarray(predictions, dtype=np.float64)
    groups = np.asarray(group_ids)
    errors = (
        np.zeros_like(truth)
        if standard_errors is None
        else np.asarray(standard_errors, dtype=np.float64)
    )
    if not (truth.shape == estimate.shape == groups.shape == errors.shape):
        raise ValueError("Top-action arrays must align")
    samples = None if return_samples is None else np.asarray(return_samples, dtype=np.float64)
    if samples is not None and (
        samples.ndim != 2 or samples.shape[0] != truth.shape[0] or samples.shape[1] < 2
    ):
        raise ValueError("return_samples must have shape [rows, rollouts>=2]")
    correct = 0
    count = 0
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        if len(indices) < 2:
            continue
        ordered = indices[np.argsort(truth[indices])[::-1]]
        best, second = ordered[0], ordered[1]
        if samples is None:
            difference_se = np.sqrt(errors[best] ** 2 + errors[second] ** 2)
        else:
            paired = samples[best] - samples[second]
            difference_se = float(
                np.std(paired, ddof=1) / np.sqrt(samples.shape[1])
            )
        uncertainty = uncertainty_z * difference_se
        if truth[best] - truth[second] <= uncertainty:
            continue
        correct += int(indices[np.argmax(estimate[indices])] == best)
        count += 1
    return {
        "count": count,
        "accuracy": float(correct / count) if count else float("nan"),
    }


def engagement_sign_accuracy(
    labels: Sequence[float],
    predictions: Sequence[float],
    group_ids: Sequence[Hashable],
    candidate_actions: Sequence[int],
    conditional_target_probabilities: Sequence[float],
    *,
    noop_action: int,
    standard_errors: Sequence[float] | None = None,
    return_samples: np.ndarray | None = None,
    uncertainty_z: float = 1.96,
) -> dict[str, float | int]:
    """Compare policy-weighted engage value against no-op within each group."""

    truth = np.asarray(labels, dtype=np.float64)
    estimate = np.asarray(predictions, dtype=np.float64)
    groups = np.asarray(group_ids)
    actions = np.asarray(candidate_actions, dtype=np.int64)
    probabilities = np.asarray(conditional_target_probabilities, dtype=np.float64)
    errors = (
        np.zeros_like(truth)
        if standard_errors is None
        else np.asarray(standard_errors, dtype=np.float64)
    )
    if not (
        truth.shape
        == estimate.shape
        == groups.shape
        == actions.shape
        == probabilities.shape
        == errors.shape
    ):
        raise ValueError("Engagement-sign arrays must align")
    samples = None if return_samples is None else np.asarray(return_samples, dtype=np.float64)
    if samples is not None and (
        samples.ndim != 2 or samples.shape[0] != truth.shape[0] or samples.shape[1] < 2
    ):
        raise ValueError("return_samples must have shape [rows, rollouts>=2]")
    correct = 0
    count = 0
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        noop_indices = indices[actions[indices] == noop_action]
        target_indices = indices[actions[indices] != noop_action]
        if len(noop_indices) != 1 or len(target_indices) == 0:
            continue
        weights = probabilities[target_indices]
        if np.sum(weights) <= 0.0:
            continue
        weights = weights / np.sum(weights)
        noop_index = noop_indices[0]
        true_engage = float(np.sum(weights * truth[target_indices]))
        predicted_engage = float(np.sum(weights * estimate[target_indices]))
        true_difference = true_engage - truth[noop_index]
        if samples is None:
            difference_se = np.sqrt(
                errors[noop_index] ** 2
                + float(np.sum(np.square(weights * errors[target_indices])))
            )
        else:
            engage_samples = np.sum(
                weights[:, None] * samples[target_indices], axis=0
            )
            paired = engage_samples - samples[noop_index]
            difference_se = float(
                np.std(paired, ddof=1) / np.sqrt(samples.shape[1])
            )
        uncertainty = uncertainty_z * difference_se
        if abs(true_difference) <= uncertainty:
            continue
        predicted_difference = predicted_engage - estimate[noop_index]
        correct += int(
            predicted_difference != 0.0
            and np.sign(predicted_difference) == np.sign(true_difference)
        )
        count += 1
    return {
        "count": count,
        "accuracy": float(correct / count) if count else float("nan"),
    }
