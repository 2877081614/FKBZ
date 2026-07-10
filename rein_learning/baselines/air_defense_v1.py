from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

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
    success: bool


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
            score_fn=lambda unit_index, target_index: _expected_damage_reduction(
                env,
                unit_index,
                target_index,
            ),
            fire_only_if_positive=self.fire_only_if_positive,
        )


def run_air_defense_v1_episode(
    env: AirDefenseResourceAssignmentEnvV1,
    policy: AirDefenseV1Policy,
    seed: int | None = None,
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
    info = initial_info

    while not (terminated or truncated):
        action = policy.select_action(env)
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        shots += int(info["shots"])
        hits += int(info["hits"])
        invalid_actions += int(info["invalid_actions"])

    return AirDefenseV1EpisodeMetrics(
        total_reward=float(total_reward),
        steps=steps,
        num_targets=env.num_targets,
        num_intercepted=int(info["num_intercepted"]),
        num_leaked=int(info["num_leaked"]),
        total_damage=float(info["total_damage"]),
        ammo_used=initial_ammo - int(info["ammo_remaining"]),
        shots=shots,
        hits=hits,
        invalid_actions=invalid_actions,
        success=bool(info["num_alive"] == 0 and info["total_damage"] == 0.0),
    )


def evaluate_air_defense_v1_policy(
    env_factory: Callable[[], AirDefenseResourceAssignmentEnvV1],
    policy_factory: Callable[[int], AirDefenseV1Policy],
    episodes: int = 30,
    seed: int = 0,
) -> dict[str, float]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    episode_metrics = []
    for episode_index in range(episodes):
        episode_seed = seed + episode_index
        env = env_factory()
        policy = policy_factory(episode_seed)
        metrics = run_air_defense_v1_episode(env, policy, seed=episode_seed)
        episode_metrics.append(metrics)
        env.close()

    rewards = np.asarray([metrics.total_reward for metrics in episode_metrics])
    steps = np.asarray([metrics.steps for metrics in episode_metrics])
    intercepted = np.asarray([metrics.num_intercepted for metrics in episode_metrics])
    leaked = np.asarray([metrics.num_leaked for metrics in episode_metrics])
    targets = np.asarray([metrics.num_targets for metrics in episode_metrics])
    damage = np.asarray([metrics.total_damage for metrics in episode_metrics])
    ammo_used = np.asarray([metrics.ammo_used for metrics in episode_metrics])
    shots = np.asarray([metrics.shots for metrics in episode_metrics])
    hits = np.asarray([metrics.hits for metrics in episode_metrics])
    invalid_actions = np.asarray([metrics.invalid_actions for metrics in episode_metrics])
    success = np.asarray([metrics.success for metrics in episode_metrics], dtype=np.float32)

    return {
        "episodes": float(episodes),
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_steps": float(np.mean(steps)),
        "success_rate": float(np.mean(success)),
        "intercept_rate": float(np.sum(intercepted) / np.sum(targets)),
        "leak_rate": float(np.sum(leaked) / np.sum(targets)),
        "avg_total_damage": float(np.mean(damage)),
        "avg_ammo_used": float(np.mean(ammo_used)),
        "avg_shots": float(np.mean(shots)),
        "hit_rate_per_shot": float(np.sum(hits) / max(1, np.sum(shots))),
        "avg_invalid_actions": float(np.mean(invalid_actions)),
    }


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


def _expected_damage_reduction(
    env: AirDefenseResourceAssignmentEnvV1,
    unit_index: int,
    target_index: int,
) -> float:
    unit = env.defense_units[unit_index]
    hit_probability = env.hit_probability(unit_index, target_index)
    avoided_damage_reward = (
        hit_probability
        * env.config.damage_penalty_weight
        * env.target_damage_potential(target_index)
    )
    intercept_reward = (
        hit_probability
        * env.config.intercept_reward_weight
        * _target_priority(env, target_index)
    )
    return float(avoided_damage_reward + intercept_reward - unit.cost)
