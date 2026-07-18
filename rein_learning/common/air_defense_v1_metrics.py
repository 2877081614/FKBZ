from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..envs import AirDefenseResourceAssignmentEnvV1


DEFAULT_HIGH_THREAT_THRESHOLD = 0.8

DIAGNOSTIC_AGGREGATE_METRICS = (
    "high_threat_leak_rate",
    "avg_zone_weighted_damage",
    "assignment_conflict_rate",
    "overkill_rate",
    "damage_reduction_per_ammo",
    "avg_resource_cost",
)


@dataclass
class AirDefenseV1DiagnosticsTracker:
    """Accumulate diagnostic counts without changing environment dynamics."""

    high_threat_threshold: float = DEFAULT_HIGH_THREAT_THRESHOLD
    engaged_target_events: int = 0
    conflict_target_events: int = 0
    overkill_assignments: int = 0
    legal_shots: int = 0
    resource_cost: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.high_threat_threshold <= 1.0:
            raise ValueError("high_threat_threshold must be between 0 and 1")

    def record_step(self, info: Mapping[str, Any]) -> None:
        assignments_per_target: dict[int, int] = {}
        for result in info.get("unit_results", ()):
            target_index = result.get("target_index")
            if (
                result.get("action_type") == "engage"
                and bool(result.get("legal"))
                and target_index is not None
            ):
                target_index = int(target_index)
                assignments_per_target[target_index] = (
                    assignments_per_target.get(target_index, 0) + 1
                )

        self.engaged_target_events += len(assignments_per_target)
        self.conflict_target_events += sum(
            assignment_count > 1
            for assignment_count in assignments_per_target.values()
        )
        self.overkill_assignments += sum(
            max(0, assignment_count - 1)
            for assignment_count in assignments_per_target.values()
        )
        self.legal_shots += int(info.get("shots", 0))
        reward_breakdown = info.get("reward_breakdown", {})
        self.resource_cost += max(0.0, -float(reward_breakdown.get("cost", 0.0)))

    def finalize(
        self,
        env: AirDefenseResourceAssignmentEnvV1,
        *,
        ammo_used: int,
    ) -> dict[str, float | int]:
        high_threat_targets = [
            target
            for target in env.targets
            if target.threat >= self.high_threat_threshold
        ]
        num_high_threat_leaked = sum(
            target.status == "leaked" for target in high_threat_targets
        )
        zone_weighted_damage = float(
            sum(target.leaked_damage for target in env.targets)
        )
        intercepted_damage_potential = float(
            sum(
                env.target_damage_potential(target_index)
                for target_index, target in enumerate(env.targets)
                if target.status == "intercepted"
            )
        )

        return {
            "high_threat_threshold": float(self.high_threat_threshold),
            "num_high_threat_targets": len(high_threat_targets),
            "num_high_threat_leaked": int(num_high_threat_leaked),
            "high_threat_leak_rate": _safe_ratio(
                num_high_threat_leaked,
                len(high_threat_targets),
            ),
            "zone_weighted_damage": zone_weighted_damage,
            "engaged_target_events": self.engaged_target_events,
            "conflict_target_events": self.conflict_target_events,
            "assignment_conflict_rate": _safe_ratio(
                self.conflict_target_events,
                self.engaged_target_events,
            ),
            "overkill_assignments": self.overkill_assignments,
            "overkill_rate": _safe_ratio(
                self.overkill_assignments,
                self.legal_shots,
            ),
            "intercepted_damage_potential": intercepted_damage_potential,
            "damage_reduction_per_ammo": _safe_ratio(
                intercepted_damage_potential,
                ammo_used,
            ),
            "resource_cost": float(self.resource_cost),
        }


def aggregate_air_defense_v1_episode_metrics(
    episode_metrics: Sequence[Mapping[str, float | int | bool]],
) -> dict[str, float]:
    """Aggregate raw episodes while preserving the established metric meanings."""

    if not episode_metrics:
        raise ValueError("episode_metrics must not be empty")

    rewards = _values(episode_metrics, "total_reward")
    steps = _values(episode_metrics, "steps")
    intercepted = _values(episode_metrics, "num_intercepted")
    leaked = _values(episode_metrics, "num_leaked")
    targets = _values(episode_metrics, "num_targets")
    damage = _values(episode_metrics, "total_damage")
    ammo_used = _values(episode_metrics, "ammo_used")
    shots = _values(episode_metrics, "shots")
    hits = _values(episode_metrics, "hits")
    invalid_actions = _values(episode_metrics, "invalid_actions")
    success = _values(episode_metrics, "success")
    decision_time_seconds = _values(
        episode_metrics,
        "decision_time_seconds",
        default=0.0,
    )

    high_threat_targets = _values(
        episode_metrics,
        "num_high_threat_targets",
        default=0.0,
    )
    high_threat_leaked = _values(
        episode_metrics,
        "num_high_threat_leaked",
        default=0.0,
    )
    zone_weighted_damage = np.asarray(
        [
            float(metrics.get("zone_weighted_damage", metrics["total_damage"]))
            for metrics in episode_metrics
        ],
        dtype=np.float64,
    )
    engaged_target_events = _values(
        episode_metrics,
        "engaged_target_events",
        default=0.0,
    )
    conflict_target_events = _values(
        episode_metrics,
        "conflict_target_events",
        default=0.0,
    )
    overkill_assignments = _values(
        episode_metrics,
        "overkill_assignments",
        default=0.0,
    )
    intercepted_damage_potential = _values(
        episode_metrics,
        "intercepted_damage_potential",
        default=0.0,
    )
    resource_cost = _values(
        episode_metrics,
        "resource_cost",
        default=0.0,
    )
    unit_decisions = _values(
        episode_metrics, "unit_decisions", default=0.0
    )
    actionable_decisions = _values(
        episode_metrics, "actionable_decisions", default=0.0
    )
    engagements = _values(episode_metrics, "engagements", default=0.0)
    actionable_engagements = _values(
        episode_metrics, "actionable_engagements", default=0.0
    )
    all_noop_episodes = _values(
        episode_metrics, "all_noop_episode", default=0.0
    )

    return {
        "episodes": float(len(episode_metrics)),
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_steps": float(np.mean(steps)),
        "success_rate": float(np.mean(success)),
        "intercept_rate": _safe_ratio(np.sum(intercepted), np.sum(targets)),
        "leak_rate": _safe_ratio(np.sum(leaked), np.sum(targets)),
        "avg_total_damage": float(np.mean(damage)),
        "avg_ammo_used": float(np.mean(ammo_used)),
        "avg_shots": float(np.mean(shots)),
        "hit_rate_per_shot": _safe_ratio(np.sum(hits), np.sum(shots)),
        "avg_invalid_actions": float(np.mean(invalid_actions)),
        "avg_decision_time_ms": _safe_ratio(
            1_000.0 * np.sum(decision_time_seconds),
            np.sum(steps),
        ),
        "high_threat_leak_rate": _safe_ratio(
            np.sum(high_threat_leaked),
            np.sum(high_threat_targets),
        ),
        "avg_zone_weighted_damage": float(np.mean(zone_weighted_damage)),
        "assignment_conflict_rate": _safe_ratio(
            np.sum(conflict_target_events),
            np.sum(engaged_target_events),
        ),
        "overkill_rate": _safe_ratio(
            np.sum(overkill_assignments),
            np.sum(shots),
        ),
        "damage_reduction_per_ammo": _safe_ratio(
            np.sum(intercepted_damage_potential),
            np.sum(ammo_used),
        ),
        "avg_resource_cost": float(np.mean(resource_cost)),
        "engagement_rate": _safe_ratio(
            np.sum(engagements), np.sum(unit_decisions)
        ),
        "actionable_engagement_rate": _safe_ratio(
            np.sum(actionable_engagements), np.sum(actionable_decisions)
        ),
        "all_noop_episode_rate": float(np.mean(all_noop_episodes)),
    }


def _values(
    episode_metrics: Sequence[Mapping[str, float | int | bool]],
    key: str,
    *,
    default: float | None = None,
) -> np.ndarray:
    if default is None:
        values = [float(metrics[key]) for metrics in episode_metrics]
    else:
        values = [float(metrics.get(key, default)) for metrics in episode_metrics]
    return np.asarray(values, dtype=np.float64)


def _safe_ratio(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if denominator <= 0.0:
        return 0.0
    return float(numerator) / denominator
