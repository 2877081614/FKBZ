from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Protocol, Sequence, TypeVar

import numpy as np

from .air_defense_v1_decision_metrics import validate_unit_order


class DynamicSupportError(ValueError):
    """Base error for invalid or undefined dynamic-support calculations."""


class DynamicSupportNotApplicableError(DynamicSupportError):
    """Raised when a DS comparison has no downstream decision position."""


class IllegalPrefixError(DynamicSupportError):
    """Raised when a supplied prefix violates the official conditional mask."""


class IllegalCurrentActionError(DynamicSupportError):
    """Raised when a compared current action is illegal under its prefix."""


class EmptyFeasibleSuffixError(DynamicSupportError):
    """Raised when a legal current action has no feasible joint completion."""


class EmptySupportUnionError(DynamicSupportError):
    """Raised when Jaccard distance is requested for two empty supports."""


class _ActionMaskProvider(Protocol):
    num_defense_units: int
    noop_action: int

    def action_mask(self) -> np.ndarray: ...


@dataclass(frozen=True)
class DynamicSupportCostMatrix:
    """Exact pairwise DS costs for all feasible current actions."""

    actions: tuple[int, ...]
    suffix_counts: tuple[int, ...]
    costs: np.ndarray

    def __post_init__(self) -> None:
        expected = (len(self.actions), len(self.actions))
        if self.costs.shape != expected:
            raise ValueError(
                f"costs must have shape {expected}, got {self.costs.shape}"
            )
        self.costs.setflags(write=False)


_T = TypeVar("_T", bound=Hashable)


def enumerate_feasible_suffixes(
    state: _ActionMaskProvider | np.ndarray,
    prefix: Sequence[int],
    unit_order: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate exact legal completions after an ordered action prefix.

    ``state`` is either an AirDefense-v1 environment exposing the official
    ``action_mask`` or an already frozen two-dimensional base action mask.
    Prefix and returned suffix actions are ordered by ``unit_order`` rather
    than by raw unit index.
    """

    base_mask, noop_action = _resolve_action_mask(state)
    order = validate_unit_order(unit_order, base_mask.shape[0])
    normalized_prefix, used_targets = _validate_prefix(
        base_mask,
        noop_action,
        prefix,
        order,
    )
    remaining_units = order[len(normalized_prefix) :]
    suffixes: list[tuple[int, ...]] = []

    def visit(
        remaining_index: int,
        occupied_targets: frozenset[int],
        suffix: tuple[int, ...],
    ) -> None:
        if remaining_index == len(remaining_units):
            suffixes.append(suffix)
            return
        unit_index = remaining_units[remaining_index]
        for action in np.flatnonzero(base_mask[unit_index]):
            candidate = int(action)
            if candidate != noop_action and candidate in occupied_targets:
                continue
            next_occupied = (
                occupied_targets
                if candidate == noop_action
                else occupied_targets | {candidate}
            )
            visit(remaining_index + 1, next_occupied, suffix + (candidate,))

    visit(0, frozenset(used_targets), ())
    return tuple(suffixes)


def suffix_count(
    state: _ActionMaskProvider | np.ndarray,
    prefix: Sequence[int],
    action: int,
    unit_order: Sequence[int] | None = None,
) -> int:
    """Count exact downstream completions for one current action."""

    suffixes = _suffixes_after_current_action(
        state,
        prefix,
        action,
        unit_order,
    )
    return len(suffixes)


def dynamic_support_jaccard(
    state: _ActionMaskProvider | np.ndarray,
    prefix: Sequence[int],
    action_a: int,
    action_b: int,
    unit_order: Sequence[int] | None = None,
) -> float:
    """Return the frozen Jaccard distance between two feasible suffix sets."""

    suffixes_a = _suffixes_after_current_action(
        state,
        prefix,
        action_a,
        unit_order,
    )
    suffixes_b = _suffixes_after_current_action(
        state,
        prefix,
        action_b,
        unit_order,
    )
    return jaccard_distance(suffixes_a, suffixes_b)


def dynamic_support_cost_matrix(
    state: _ActionMaskProvider | np.ndarray,
    prefix: Sequence[int],
    unit_order: Sequence[int] | None = None,
) -> DynamicSupportCostMatrix:
    """Build exact DS costs over current actions with feasible completions."""

    base_mask, noop_action = _resolve_action_mask(state)
    order = validate_unit_order(unit_order, base_mask.shape[0])
    normalized_prefix, used_targets = _validate_prefix(
        base_mask,
        noop_action,
        prefix,
        order,
    )
    _require_downstream_position(len(normalized_prefix), len(order))
    current_unit = order[len(normalized_prefix)]
    supports: list[tuple[int, frozenset[tuple[int, ...]]]] = []
    for raw_action in np.flatnonzero(base_mask[current_unit]):
        action = int(raw_action)
        if action != noop_action and action in used_targets:
            continue
        suffixes = frozenset(
            enumerate_feasible_suffixes(
                base_mask,
                normalized_prefix + (action,),
                order,
            )
        )
        if suffixes:
            supports.append((action, suffixes))

    if not supports:
        raise EmptyFeasibleSuffixError(
            "No current action has a feasible downstream completion"
        )
    actions = tuple(action for action, _ in supports)
    suffix_counts = tuple(len(suffixes) for _, suffixes in supports)
    costs = np.zeros((len(actions), len(actions)), dtype=np.float64)
    for row, (_, first) in enumerate(supports):
        for column in range(row + 1, len(actions)):
            distance = jaccard_distance(first, supports[column][1])
            costs[row, column] = distance
            costs[column, row] = distance
    return DynamicSupportCostMatrix(
        actions=actions,
        suffix_counts=suffix_counts,
        costs=costs,
    )


def jaccard_distance(
    first: Sequence[_T] | set[_T] | frozenset[_T],
    second: Sequence[_T] | set[_T] | frozenset[_T],
) -> float:
    """Return set Jaccard distance and reject the undefined empty union."""

    first_set = frozenset(first)
    second_set = frozenset(second)
    union = first_set | second_set
    if not union:
        raise EmptySupportUnionError(
            "Dynamic-support Jaccard is undefined for an empty union"
        )
    return float(1.0 - len(first_set & second_set) / len(union))


def old_policy_structural_risk(
    pairwise_costs: np.ndarray,
    old_probabilities: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Compute ``r_old(a)=sum_b pi_old(b)c_DS(a,b)``."""

    costs = _validate_cost_matrix(pairwise_costs)
    probabilities = _validate_probability_vector(
        old_probabilities,
        expected_size=costs.shape[0],
        name="old_probabilities",
    )
    risk = costs @ probabilities
    if bool(np.any(risk < -1e-12) or np.any(risk > 1.0 + 1e-12)):
        raise ValueError("Structural risk falls outside [0, 1]")
    return np.clip(risk, 0.0, 1.0)


def dynamic_support_policy_distance(
    new_probabilities: Sequence[float] | np.ndarray,
    old_probabilities: Sequence[float] | np.ndarray,
    structural_risk: Sequence[float] | np.ndarray,
) -> float:
    """Compute the frozen old-policy-weighted total-variation distance."""

    old = _validate_probability_vector(
        old_probabilities,
        expected_size=None,
        name="old_probabilities",
    )
    new = _validate_probability_vector(
        new_probabilities,
        expected_size=old.size,
        name="new_probabilities",
    )
    risk = np.asarray(structural_risk, dtype=np.float64).reshape(-1)
    if risk.size != old.size:
        raise ValueError("structural_risk must match the probability vectors")
    if not bool(np.all(np.isfinite(risk))):
        raise ValueError("structural_risk must contain only finite values")
    if bool(np.any(risk < 0.0) or np.any(risk > 1.0)):
        raise ValueError("structural_risk must lie in [0, 1]")
    distance = 0.5 * float(np.sum(np.abs(new - old) * risk))
    if not -1e-12 <= distance <= 1.0 + 1e-12:
        raise ValueError("Dynamic-support policy distance falls outside [0, 1]")
    return float(np.clip(distance, 0.0, 1.0))


def _suffixes_after_current_action(
    state: _ActionMaskProvider | np.ndarray,
    prefix: Sequence[int],
    action: int,
    unit_order: Sequence[int] | None,
) -> tuple[tuple[int, ...], ...]:
    base_mask, noop_action = _resolve_action_mask(state)
    order = validate_unit_order(unit_order, base_mask.shape[0])
    normalized_prefix, used_targets = _validate_prefix(
        base_mask,
        noop_action,
        prefix,
        order,
    )
    _require_downstream_position(len(normalized_prefix), len(order))
    current_unit = order[len(normalized_prefix)]
    current_action = _normalize_action(action, base_mask.shape[1])
    if not base_mask[current_unit, current_action]:
        raise IllegalCurrentActionError(
            f"Action {current_action} is illegal for unit {current_unit}"
        )
    if current_action != noop_action and current_action in used_targets:
        raise IllegalCurrentActionError(
            f"Action {current_action} is occupied by the supplied prefix"
        )
    suffixes = enumerate_feasible_suffixes(
        base_mask,
        normalized_prefix + (current_action,),
        order,
    )
    if not suffixes:
        raise EmptyFeasibleSuffixError(
            f"Action {current_action} has no feasible downstream completion"
        )
    return suffixes


def _resolve_action_mask(
    state: _ActionMaskProvider | np.ndarray,
) -> tuple[np.ndarray, int]:
    if isinstance(state, np.ndarray):
        mask = np.asarray(state, dtype=bool)
        noop_action = mask.shape[1] - 1 if mask.ndim == 2 else -1
    else:
        if not hasattr(state, "action_mask"):
            raise TypeError(
                "state must be a 2D base action mask or expose action_mask()"
            )
        mask = np.asarray(state.action_mask(), dtype=bool)
        noop_action = int(state.noop_action)
        if int(state.num_defense_units) != mask.shape[0]:
            raise ValueError("Environment unit count does not match action mask")
    if mask.ndim != 2 or min(mask.shape) <= 0:
        raise ValueError("Base action mask must be a nonempty 2D array")
    if not 0 <= noop_action < mask.shape[1]:
        raise ValueError("No-op action is outside the action-mask width")
    if noop_action != mask.shape[1] - 1:
        raise ValueError("AirDefense-v1 no-op action must be the final action")
    if not bool(np.all(np.any(mask, axis=1))):
        raise ValueError("Every unit must have at least one legal base action")
    return mask.copy(), noop_action


def _validate_prefix(
    base_mask: np.ndarray,
    noop_action: int,
    prefix: Sequence[int],
    unit_order: tuple[int, ...],
) -> tuple[tuple[int, ...], set[int]]:
    normalized = tuple(
        _normalize_action(action, base_mask.shape[1]) for action in prefix
    )
    if len(normalized) > len(unit_order):
        raise IllegalPrefixError("Prefix is longer than the unit order")
    used_targets: set[int] = set()
    for position, action in enumerate(normalized):
        unit_index = unit_order[position]
        if not base_mask[unit_index, action]:
            raise IllegalPrefixError(
                f"Prefix action {action} is illegal for unit {unit_index}"
            )
        if action == noop_action:
            continue
        if action in used_targets:
            raise IllegalPrefixError(
                f"Prefix assigns target {action} more than once"
            )
        used_targets.add(action)
    return normalized, used_targets


def _normalize_action(action: int, num_actions: int) -> int:
    if isinstance(action, (bool, np.bool_)):
        raise DynamicSupportError("Boolean values are not valid action ids")
    try:
        normalized = int(action)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DynamicSupportError(f"Invalid action id: {action!r}") from exc
    if normalized != action or not 0 <= normalized < num_actions:
        raise DynamicSupportError(
            f"Action {action!r} is outside [0, {num_actions})"
        )
    return normalized


def _require_downstream_position(prefix_length: int, num_units: int) -> None:
    if prefix_length >= num_units - 1:
        raise DynamicSupportNotApplicableError(
            "Dynamic support is not_applicable at the last unit position"
        )


def _validate_cost_matrix(pairwise_costs: np.ndarray) -> np.ndarray:
    costs = np.asarray(pairwise_costs, dtype=np.float64)
    if costs.ndim != 2 or costs.shape[0] != costs.shape[1] or costs.size == 0:
        raise ValueError("pairwise_costs must be a nonempty square matrix")
    if not bool(np.all(np.isfinite(costs))):
        raise ValueError("pairwise_costs must contain only finite values")
    if bool(np.any(costs < 0.0) or np.any(costs > 1.0)):
        raise ValueError("pairwise_costs must lie in [0, 1]")
    return costs


def _validate_probability_vector(
    probabilities: Sequence[float] | np.ndarray,
    *,
    expected_size: int | None,
    name: str,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if values.size == 0 or (expected_size is not None and values.size != expected_size):
        raise ValueError(f"{name} has the wrong size")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError(f"{name} must contain only finite values")
    if bool(np.any(values < 0.0)):
        raise ValueError(f"{name} must be nonnegative")
    if not np.isclose(values.sum(), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(f"{name} must sum to 1")
    return values
