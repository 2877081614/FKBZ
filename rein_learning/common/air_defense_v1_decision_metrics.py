from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..envs.air_defense_v1.centralized_env import (
    AirDefenseResourceAssignmentEnvV1,
)
from ..simulators import euclidean_distance
from .air_defense_v1_metrics import DEFAULT_HIGH_THREAT_THRESHOLD


LEAK_ATTRIBUTION_CATEGORIES = (
    "never_legal",
    "unassigned",
    "prefix_denied",
    "mismatched_resource",
    "attempted_miss",
    "resource_exhausted",
)


@dataclass
class _TargetDecisionTrace:
    ever_geometrically_reachable: bool = False
    ever_blocked_by_unavailability: bool = False
    ever_legal: bool = False
    ever_assigned: bool = False
    prefix_denied_better_resource: bool = False
    mismatched_resource: bool = False


def validate_unit_order(
    unit_order: Sequence[int] | None,
    num_units: int,
) -> tuple[int, ...]:
    order = (
        tuple(range(num_units))
        if unit_order is None
        else tuple(int(value) for value in unit_order)
    )
    if len(order) != num_units or set(order) != set(range(num_units)):
        raise ValueError("unit_order must be a permutation of all unit indices")
    return order


def classify_high_threat_leak(
    *,
    ever_legal: bool,
    ever_assigned: bool,
    prefix_denied_better_resource: bool,
    mismatched_resource: bool,
    ever_geometrically_reachable: bool,
    ever_blocked_by_unavailability: bool,
) -> str:
    """Assign one mutually exclusive cause using a frozen priority order."""

    if ever_assigned:
        if prefix_denied_better_resource:
            return "prefix_denied"
        if mismatched_resource:
            return "mismatched_resource"
        return "attempted_miss"
    if ever_legal:
        return "unassigned"
    if ever_geometrically_reachable and ever_blocked_by_unavailability:
        return "resource_exhausted"
    return "never_legal"


class AirDefenseV1DecisionTracker:
    """Capture unit-level choices without changing environment execution."""

    def __init__(
        self,
        *,
        unit_order: Sequence[int] | None,
        num_units: int,
        num_targets: int,
        high_threat_threshold: float = DEFAULT_HIGH_THREAT_THRESHOLD,
    ) -> None:
        if not 0.0 <= high_threat_threshold <= 1.0:
            raise ValueError("high_threat_threshold must be between 0 and 1")
        self.unit_order = validate_unit_order(unit_order, num_units)
        self.num_units = num_units
        self.num_targets = num_targets
        self.high_threat_threshold = high_threat_threshold
        self.rows: list[dict[str, Any]] = []
        self._target_traces = [
            _TargetDecisionTrace() for _ in range(num_targets)
        ]

    def before_step(
        self,
        env: AirDefenseResourceAssignmentEnvV1,
        joint_action: Sequence[int] | np.ndarray,
    ) -> list[dict[str, Any]]:
        actions = np.asarray(joint_action, dtype=np.int64).reshape(-1)
        if actions.size != self.num_units:
            raise ValueError(
                f"Expected {self.num_units} unit actions, got {actions.size}"
            )
        base_mask = env.action_mask().astype(bool, copy=False)
        self._update_target_opportunities(env, base_mask)

        rows: list[dict[str, Any]] = []
        selected_targets: set[int] = set()
        selected_expected_reduction: dict[int, float] = {}
        order_label = "-".join(str(value) for value in self.unit_order)

        for order_position, unit_index in enumerate(self.unit_order):
            unit = env.defense_units[unit_index]
            unit_base_mask = base_mask[unit_index].copy()
            conditional_mask = unit_base_mask.copy()
            if selected_targets:
                conditional_mask[list(selected_targets)] = False

            prefix_denied_targets = [
                target_index
                for target_index in selected_targets
                if unit_base_mask[target_index]
            ]
            for target_index in prefix_denied_targets:
                later_value = self._expected_damage_reduction(
                    env,
                    unit_index,
                    target_index,
                )
                if later_value > selected_expected_reduction[target_index] + 1e-12:
                    self._target_traces[
                        target_index
                    ].prefix_denied_better_resource = True

            selected_action = int(actions[unit_index])
            selected_noop = selected_action == env.noop_action
            selected_target = None if selected_noop else selected_action
            selected_legal = bool(conditional_mask[selected_action])
            legal_targets = np.flatnonzero(conditional_mask[: self.num_targets])
            high_threat_legal_targets = [
                int(target_index)
                for target_index in legal_targets
                if env.targets[int(target_index)].threat
                >= self.high_threat_threshold
            ]
            best_expected_reduction = max(
                (
                    self._expected_damage_reduction(
                        env,
                        unit_index,
                        int(target_index),
                    )
                    for target_index in legal_targets
                ),
                default=0.0,
            )

            row: dict[str, Any] = {
                "step_index": int(env.current_step),
                "unit_index": unit_index,
                "resource_type": unit.resource_type,
                "unit_order": order_label,
                "unit_order_position": order_position,
                "selected_action": selected_action,
                "selected_target": selected_target,
                "selected_noop": selected_noop,
                "base_action_legal": bool(unit_base_mask[selected_action]),
                "conditional_action_legal": selected_legal,
                "ammo_before": int(unit.ammo),
                "cooldown_before": int(unit.cooldown),
                "energy_before": float(unit.energy),
                "num_base_legal_targets": int(
                    np.count_nonzero(unit_base_mask[: self.num_targets])
                ),
                "num_conditional_legal_targets": int(legal_targets.size),
                "num_conditional_high_threat_targets": len(
                    high_threat_legal_targets
                ),
                "prefix_denied_target_count": len(prefix_denied_targets),
                "avoidable_noop": bool(selected_noop and legal_targets.size > 0),
                "selected_high_threat": False,
                "target_alive": None,
                "target_threat": None,
                "target_payload": None,
                "target_time_to_impact": None,
                "target_distance": None,
                "hit_probability": None,
                "damage_potential": None,
                "expected_damage_reduction": None,
                "best_expected_damage_reduction": best_expected_reduction,
                "matching_efficiency": None,
                "target_already_selected_by_prefix": bool(
                    selected_target in selected_targets
                    if selected_target is not None
                    else False
                ),
                "shot_fired": False,
                "hit": False,
                "target_intercepted": False,
                "target_status_after": None,
            }

            if selected_target is not None and 0 <= selected_target < self.num_targets:
                target = env.targets[selected_target]
                hit_probability = (
                    env.hit_probability(unit_index, selected_target)
                    if target.alive
                    else 0.0
                )
                damage_potential = env.target_damage_potential(selected_target)
                expected_reduction = hit_probability * damage_potential
                row.update(
                    {
                        "selected_high_threat": bool(
                            target.threat >= self.high_threat_threshold
                        ),
                        "target_alive": bool(target.alive),
                        "target_threat": float(target.threat),
                        "target_payload": float(target.payload),
                        "target_time_to_impact": float(target.time_to_impact),
                        "target_distance": float(
                            euclidean_distance(unit.position, target.position)
                        ),
                        "hit_probability": float(hit_probability),
                        "damage_potential": float(damage_potential),
                        "expected_damage_reduction": float(expected_reduction),
                        "matching_efficiency": (
                            float(expected_reduction / best_expected_reduction)
                            if best_expected_reduction > 0.0
                            else 0.0
                        ),
                    }
                )
                trace = self._target_traces[selected_target]
                trace.ever_assigned = True
                best_unit_value = self._best_unit_value_for_target(
                    env,
                    base_mask,
                    selected_target,
                )
                if expected_reduction + 1e-12 < best_unit_value:
                    trace.mismatched_resource = True
                if selected_target not in selected_targets:
                    selected_expected_reduction[selected_target] = expected_reduction
                selected_targets.add(selected_target)

            rows.append(row)

        return rows

    def after_step(
        self,
        env: AirDefenseResourceAssignmentEnvV1,
        info: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> None:
        result_by_unit = {
            int(result["unit_index"]): result
            for result in info.get("unit_results", [])
        }
        for row in rows:
            unit_index = int(row["unit_index"])
            result = result_by_unit.get(unit_index)
            if result is not None:
                row["shot_fired"] = bool(
                    result["action_type"] == "engage" and result["legal"]
                )
                row["hit"] = bool(result["hit"])
            target_index = row["selected_target"]
            if target_index is not None and 0 <= int(target_index) < self.num_targets:
                target = env.targets[int(target_index)]
                row["target_intercepted"] = target.status == "intercepted"
                row["target_status_after"] = target.status
            self.rows.append(row)

    def finalize_leak_attributions(
        self,
        env: AirDefenseResourceAssignmentEnvV1,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for target_index, (target, trace) in enumerate(
            zip(env.targets, self._target_traces)
        ):
            if target.status != "leaked" or target.threat < self.high_threat_threshold:
                continue
            output.append(
                {
                    "target_index": target_index,
                    "target_threat": float(target.threat),
                    "target_payload": float(target.payload),
                    "target_class": target.target_class,
                    "attribution": classify_high_threat_leak(
                        ever_legal=trace.ever_legal,
                        ever_assigned=trace.ever_assigned,
                        prefix_denied_better_resource=(
                            trace.prefix_denied_better_resource
                        ),
                        mismatched_resource=trace.mismatched_resource,
                        ever_geometrically_reachable=(
                            trace.ever_geometrically_reachable
                        ),
                        ever_blocked_by_unavailability=(
                            trace.ever_blocked_by_unavailability
                        ),
                    ),
                }
            )
        return output

    def _update_target_opportunities(
        self,
        env: AirDefenseResourceAssignmentEnvV1,
        base_mask: np.ndarray,
    ) -> None:
        for target_index, target in enumerate(env.targets):
            if not target.alive:
                continue
            trace = self._target_traces[target_index]
            trace.ever_legal |= bool(np.any(base_mask[:, target_index]))
            for unit in env.defense_units:
                reachable = (
                    euclidean_distance(unit.position, target.position)
                    <= unit.max_range
                )
                trace.ever_geometrically_reachable |= reachable
                trace.ever_blocked_by_unavailability |= (
                    reachable and not unit.available
                )

    @staticmethod
    def _expected_damage_reduction(
        env: AirDefenseResourceAssignmentEnvV1,
        unit_index: int,
        target_index: int,
    ) -> float:
        return float(
            env.hit_probability(unit_index, target_index)
            * env.target_damage_potential(target_index)
        )

    def _best_unit_value_for_target(
        self,
        env: AirDefenseResourceAssignmentEnvV1,
        base_mask: np.ndarray,
        target_index: int,
    ) -> float:
        return max(
            (
                self._expected_damage_reduction(env, unit_index, target_index)
                for unit_index in range(self.num_units)
                if base_mask[unit_index, target_index]
            ),
            default=0.0,
        )


def aggregate_decision_rows(
    rows: Sequence[dict[str, Any]],
    *,
    group_keys: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)

    output: list[dict[str, Any]] = []
    for key, group in grouped.items():
        assignments = [row for row in group if not bool(row["selected_noop"])]
        actionable = [
            row for row in group if int(row["num_conditional_legal_targets"]) > 0
        ]
        matching_values = [
            float(row["matching_efficiency"])
            for row in assignments
            if row["matching_efficiency"] is not None
        ]
        expected_values = [
            float(row["expected_damage_reduction"])
            for row in assignments
            if row["expected_damage_reduction"] is not None
        ]
        threat_values = [
            float(row["target_threat"])
            for row in assignments
            if row["target_threat"] is not None
        ]
        high_opportunities = sum(
            int(row["num_conditional_high_threat_targets"]) for row in group
        )
        prefix_denials = sum(
            int(row["prefix_denied_target_count"]) for row in group
        )
        base_opportunities = sum(
            int(row["num_base_legal_targets"]) for row in group
        )
        summary = {
            group_key: value for group_key, value in zip(group_keys, key)
        }
        summary.update(
            {
                "decision_opportunities": len(group),
                "assignments": len(assignments),
                "assignment_rate": _safe_ratio(len(assignments), len(group)),
                "actionable_decisions": len(actionable),
                "avoidable_noops": sum(bool(row["avoidable_noop"]) for row in group),
                "avoidable_noop_rate": _safe_ratio(
                    sum(bool(row["avoidable_noop"]) for row in group),
                    len(actionable),
                ),
                "high_threat_assignments": sum(
                    bool(row["selected_high_threat"]) for row in assignments
                ),
                "high_threat_legal_target_opportunities": high_opportunities,
                "high_threat_assignment_rate": _safe_ratio(
                    sum(bool(row["selected_high_threat"]) for row in assignments),
                    high_opportunities,
                ),
                "mean_assigned_threat": _safe_mean(threat_values),
                "mean_expected_damage_reduction": _safe_mean(expected_values),
                "mean_matching_efficiency": _safe_mean(matching_values),
                "prefix_denied_target_opportunities": prefix_denials,
                "base_legal_target_opportunities": base_opportunities,
                "prefix_denial_rate": _safe_ratio(
                    prefix_denials,
                    base_opportunities,
                ),
                "collapsed_unit": (
                    len(actionable) >= 100
                    and _safe_ratio(len(assignments), len(group)) < 0.01
                ),
            }
        )
        output.append(summary)
    return output


def aggregate_collapsed_unit_counts(
    rows: Sequence[dict[str, Any]],
    *,
    group_keys: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    return [
        {
            **{
                group_key: value
                for group_key, value in zip(group_keys, key)
            },
            "collapsed_unit_count": sum(
                bool(row["collapsed_unit"]) for row in group
            ),
            "evaluated_unit_count": len(group),
        }
        for key, group in grouped.items()
    ]


def aggregate_leak_attributions(
    rows: Sequence[dict[str, Any]],
    *,
    group_keys: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)

    output: list[dict[str, Any]] = []
    for key, group in grouped.items():
        total = len(group)
        for category in LEAK_ATTRIBUTION_CATEGORIES:
            count = sum(row["attribution"] == category for row in group)
            summary = {
                group_key: value for group_key, value in zip(group_keys, key)
            }
            summary.update(
                {
                    "attribution": category,
                    "count": count,
                    "total_high_threat_leaks": total,
                    "rate": _safe_ratio(count, total),
                }
            )
            output.append(summary)
    return output


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0
