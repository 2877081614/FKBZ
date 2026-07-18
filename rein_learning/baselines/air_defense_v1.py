from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable, Protocol

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..common import (
    DEFAULT_HIGH_THREAT_THRESHOLD,
    AirDefenseV1DecisionTracker,
    AirDefenseV1DiagnosticsTracker,
    aggregate_air_defense_v1_episode_metrics,
)
from ..envs.air_defense_v1 import AirDefenseResourceAssignmentEnvV1
from ..simulators import euclidean_distance


class AirDefenseV1Policy(Protocol):
    def select_action(self, env: AirDefenseResourceAssignmentEnvV1) -> np.ndarray:
        ...


@dataclass(frozen=True)
class AirDefenseV1EpisodeMetrics:
    total_reward: float
    steps: int
    num_targets: int
    num_intercepted: int
    num_leaked: int
    total_damage: float
    ammo_used: int
    shots: int
    hits: int
    invalid_actions: int
    unit_decisions: int
    actionable_decisions: int
    engagements: int
    actionable_engagements: int
    all_noop_episode: bool
    success: bool
    decision_time_seconds: float
    decision_time_ms: float
    high_threat_threshold: float
    num_high_threat_targets: int
    num_high_threat_leaked: int
    high_threat_leak_rate: float
    zone_weighted_damage: float
    engaged_target_events: int
    conflict_target_events: int
    assignment_conflict_rate: float
    overkill_assignments: int
    overkill_rate: float
    intercepted_damage_potential: float
    damage_reduction_per_ammo: float
    resource_cost: float


class RandomLegalJointPolicy:
    """Sample one legal action independently for each defense unit."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def select_action(self, env: AirDefenseResourceAssignmentEnvV1) -> np.ndarray:
        mask = env.action_mask()
        actions = []
        for unit_index in range(env.num_defense_units):
            legal_actions = np.flatnonzero(mask[unit_index])
            actions.append(int(self.rng.choice(legal_actions)))
        return np.asarray(actions, dtype=np.int64)


class NearestTargetJointPolicy:
    """Assign each available unit to its closest legal target, avoiding duplicates."""

    def select_action(self, env: AirDefenseResourceAssignmentEnvV1) -> np.ndarray:
        return _greedy_joint_assignment(
            env,
            score_fn=lambda unit_index, target_index: -_unit_target_distance(
                env,
                unit_index,
                target_index,
            ),
        )


class HighestThreatJointPolicy:
    """Assign resources to alive legal targets with highest damage-weighted threat."""

    def select_action(self, env: AirDefenseResourceAssignmentEnvV1) -> np.ndarray:
        return _greedy_joint_assignment(
            env,
            score_fn=lambda unit_index, target_index: _target_priority(env, target_index),
        )


class TimeToImpactJointPolicy:
    """Prioritize legal targets with the smallest estimated time-to-impact."""

    def select_action(self, env: AirDefenseResourceAssignmentEnvV1) -> np.ndarray:
        return _greedy_joint_assignment(
            env,
            score_fn=lambda unit_index, target_index: _target_urgency(env, target_index),
        )


class GreedyDamageReductionPolicy:
    """Choose assignments with highest immediate expected damage reduction."""

    def __init__(self, fire_only_if_positive: bool = True) -> None:
        self.fire_only_if_positive = fire_only_if_positive

    def select_action(self, env: AirDefenseResourceAssignmentEnvV1) -> np.ndarray:
        return _greedy_joint_assignment(
            env,
            score_fn=lambda unit_index, target_index: expected_damage_reduction_score(
                env,
                unit_index,
                target_index,
            ),
            fire_only_if_positive=self.fire_only_if_positive,
        )


class HungarianDamageReductionPolicy:
    """Globally maximize immediate expected damage reduction one-to-one."""

    def select_action(self, env: AirDefenseResourceAssignmentEnvV1) -> np.ndarray:
        real_scores = build_expected_damage_reduction_matrix(env)
        num_units, num_targets = real_scores.shape
        augmented_scores = np.full(
            (num_units, num_targets + num_units),
            -np.inf,
            dtype=np.float64,
        )

        positive_legal = np.isfinite(real_scores) & (real_scores > 0.0)
        augmented_scores[:, :num_targets][positive_legal] = real_scores[positive_legal]
        for unit_index in range(num_units):
            augmented_scores[unit_index, num_targets + unit_index] = 0.0

        row_indices, column_indices = linear_sum_assignment(-augmented_scores)
        actions = np.full(num_units, env.noop_action, dtype=np.int64)
        for unit_index, column_index in zip(row_indices, column_indices):
            if column_index < num_targets:
                actions[unit_index] = column_index
        return actions


def build_expected_damage_reduction_matrix(
    env: AirDefenseResourceAssignmentEnvV1,
) -> np.ndarray:
    """Return unit-target scores, using ``-inf`` for illegal assignments."""

    scores = np.full(
        (env.num_defense_units, env.num_targets),
        -np.inf,
        dtype=np.float64,
    )
    for unit_index in range(env.num_defense_units):
        for target_index in range(env.num_targets):
            if env.is_unit_target_action_legal(unit_index, target_index):
                scores[unit_index, target_index] = expected_damage_reduction_score(
                    env,
                    unit_index,
                    target_index,
                )
    return scores


def expected_damage_reduction_score(
    env: AirDefenseResourceAssignmentEnvV1,
    unit_index: int,
    target_index: int,
) -> float:
    """Score one legal assignment as expected avoided loss minus resource cost."""

    unit = env.defense_units[unit_index]
    hit_probability = env.hit_probability(unit_index, target_index)
    target_priority = _target_priority(env, target_index)
    avoided_damage_reward = (
        hit_probability
        * env.config.damage_penalty_weight
        * env.target_damage_potential(target_index)
    )
    intercept_reward = (
        hit_probability * env.config.intercept_reward_weight * target_priority
    )
    return float(avoided_damage_reward + intercept_reward - unit.cost)


def run_air_defense_v1_episode(
    env: AirDefenseResourceAssignmentEnvV1,
    policy: AirDefenseV1Policy,
    seed: int | None = None,
    high_threat_threshold: float = DEFAULT_HIGH_THREAT_THRESHOLD,
    decision_trace_callback: Callable[[dict[str, Any]], None] | None = None,
    leak_attribution_callback: Callable[[dict[str, Any]], None] | None = None,
) -> AirDefenseV1EpisodeMetrics:
    _, initial_info = env.reset(seed=seed)
    initial_ammo = int(initial_info["ammo_remaining"])
    terminated = False
    truncated = False
    total_reward = 0.0
    steps = 0
    shots = 0
    hits = 0
    invalid_actions = 0
    unit_decisions = 0
    actionable_decisions = 0
    engagements = 0
    actionable_engagements = 0
    decision_time_seconds = 0.0
    diagnostics = AirDefenseV1DiagnosticsTracker(
        high_threat_threshold=high_threat_threshold
    )
    decision_tracker = AirDefenseV1DecisionTracker(
        unit_order=tuple(range(env.num_defense_units)),
        num_units=env.num_defense_units,
        num_targets=env.num_targets,
        high_threat_threshold=high_threat_threshold,
    )
    info = initial_info

    while not (terminated or truncated):
        base_mask = env.action_mask()
        actionable_units = np.any(base_mask[:, : env.num_targets], axis=1)
        decision_started = perf_counter()
        action = policy.select_action(env)
        decision_time_seconds += perf_counter() - decision_started
        action_array = np.asarray(action, dtype=np.int64).reshape(-1)
        engaged_units = action_array != env.noop_action
        unit_decisions += env.num_defense_units
        actionable_decisions += int(np.sum(actionable_units))
        engagements += int(np.sum(engaged_units))
        actionable_engagements += int(np.sum(engaged_units & actionable_units))
        decision_rows = decision_tracker.before_step(env, action)
        _, reward, terminated, truncated, info = env.step(action)
        decision_tracker.after_step(env, info, decision_rows)
        if decision_trace_callback is not None:
            for row in decision_rows:
                decision_trace_callback(row.copy())
        total_reward += reward
        steps += 1
        shots += int(info["shots"])
        hits += int(info["hits"])
        invalid_actions += int(info["invalid_actions"])
        diagnostics.record_step(info)

    ammo_used = initial_ammo - int(info["ammo_remaining"])
    diagnostic_metrics = diagnostics.finalize(env, ammo_used=ammo_used)
    if leak_attribution_callback is not None:
        for row in decision_tracker.finalize_leak_attributions(env):
            leak_attribution_callback(row.copy())

    return AirDefenseV1EpisodeMetrics(
        total_reward=float(total_reward),
        steps=steps,
        num_targets=env.num_targets,
        num_intercepted=int(info["num_intercepted"]),
        num_leaked=int(info["num_leaked"]),
        total_damage=float(info["total_damage"]),
        ammo_used=ammo_used,
        shots=shots,
        hits=hits,
        invalid_actions=invalid_actions,
        unit_decisions=unit_decisions,
        actionable_decisions=actionable_decisions,
        engagements=engagements,
        actionable_engagements=actionable_engagements,
        all_noop_episode=bool(actionable_decisions > 0 and engagements == 0),
        success=bool(info["num_alive"] == 0 and info["total_damage"] == 0.0),
        decision_time_seconds=decision_time_seconds,
        decision_time_ms=(1_000.0 * decision_time_seconds / steps),
        **diagnostic_metrics,
    )


def evaluate_air_defense_v1_policy(
    env_factory: Callable[[], AirDefenseResourceAssignmentEnvV1],
    policy_factory: Callable[[int], AirDefenseV1Policy],
    episodes: int = 30,
    seed: int = 0,
    high_threat_threshold: float = DEFAULT_HIGH_THREAT_THRESHOLD,
    episode_metrics_callback: Callable[
        [dict[str, float | int | bool]], None
    ]
    | None = None,
    decision_trace_callback: Callable[[int, dict[str, Any]], None] | None = None,
    leak_attribution_callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> dict[str, float]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    episode_metrics: list[dict[str, float | int | bool]] = []
    for episode_index in range(episodes):
        episode_seed = seed + episode_index
        env = env_factory()
        policy = policy_factory(episode_seed)
        metrics = run_air_defense_v1_episode(
            env,
            policy,
            seed=episode_seed,
            high_threat_threshold=high_threat_threshold,
            decision_trace_callback=(
                lambda row, index=episode_index: decision_trace_callback(index, row)
                if decision_trace_callback is not None
                else None
            ),
            leak_attribution_callback=(
                lambda row, index=episode_index: leak_attribution_callback(index, row)
                if leak_attribution_callback is not None
                else None
            ),
        )
        raw_metrics = asdict(metrics)
        episode_metrics.append(raw_metrics)
        if episode_metrics_callback is not None:
            episode_metrics_callback(raw_metrics.copy())
        env.close()

    return aggregate_air_defense_v1_episode_metrics(episode_metrics)


def _greedy_joint_assignment(
    env: AirDefenseResourceAssignmentEnvV1,
    score_fn: Callable[[int, int], float],
    fire_only_if_positive: bool = False,
) -> np.ndarray:
    actions = np.full(env.num_defense_units, env.noop_action, dtype=np.int64)
    assigned_targets: set[int] = set()
    candidates = []
    for unit_index in range(env.num_defense_units):
        for target_index in range(env.num_targets):
            if not env.is_unit_target_action_legal(unit_index, target_index):
                continue
            score = float(score_fn(unit_index, target_index))
            candidates.append((score, unit_index, target_index))

    for score, unit_index, target_index in sorted(candidates, reverse=True):
        if actions[unit_index] != env.noop_action:
            continue
        if target_index in assigned_targets:
            continue
        if fire_only_if_positive and score <= 0.0:
            continue
        actions[unit_index] = target_index
        assigned_targets.add(target_index)
    return actions


def _unit_target_distance(
    env: AirDefenseResourceAssignmentEnvV1,
    unit_index: int,
    target_index: int,
) -> float:
    unit = env.defense_units[unit_index]
    target = env.targets[target_index]
    return euclidean_distance(unit.position, target.position)


def _target_priority(env: AirDefenseResourceAssignmentEnvV1, target_index: int) -> float:
    target = env.targets[target_index]
    zone = env.protected_zones[target.target_zone]
    return float(target.threat * target.payload * zone.value)


def _target_urgency(env: AirDefenseResourceAssignmentEnvV1, target_index: int) -> float:
    target = env.targets[target_index]
    return float(_target_priority(env, target_index) / (target.time_to_impact + 1.0))
