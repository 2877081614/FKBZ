from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from ..envs.air_defense import AirDefenseResourceAssignmentEnv
from ..simulators import compute_hit_probability, euclidean_distance


class AirDefensePolicy(Protocol):
    def select_action(self, env: AirDefenseResourceAssignmentEnv) -> int:
        ...


@dataclass(frozen=True)
class AirDefenseEpisodeMetrics:
    total_reward: float
    steps: int
    num_targets: int
    num_intercepted: int
    num_leaked: int
    ammo_used: int
    shots: int
    hits: int
    invalid_actions: int
    success: bool


class RandomLegalPolicy:
    """Sample uniformly from currently legal actions, including no-op."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def select_action(self, env: AirDefenseResourceAssignmentEnv) -> int:
        legal_actions = np.flatnonzero(env.action_mask())
        return int(self.rng.choice(legal_actions))


class NearestTargetPolicy:
    """Assign a legal resource to the alive target closest to the protected asset."""

    def select_action(self, env: AirDefenseResourceAssignmentEnv) -> int:
        legal_pair_actions = _legal_pair_actions(env)
        if not legal_pair_actions:
            return env.noop_action
        return min(
            legal_pair_actions,
            key=lambda action: _target_distance_to_asset(env, action),
        )


class HighestThreatPolicy:
    """Assign a legal resource to the alive target with highest threat."""

    def select_action(self, env: AirDefenseResourceAssignmentEnv) -> int:
        legal_pair_actions = _legal_pair_actions(env)
        if not legal_pair_actions:
            return env.noop_action
        return max(
            legal_pair_actions,
            key=lambda action: _target_threat(env, action),
        )


class GreedyExpectedBenefitPolicy:
    """Choose the legal action with highest immediate expected reward benefit."""

    def __init__(self, fire_only_if_positive: bool = True) -> None:
        self.fire_only_if_positive = fire_only_if_positive

    def select_action(self, env: AirDefenseResourceAssignmentEnv) -> int:
        legal_pair_actions = _legal_pair_actions(env)
        if not legal_pair_actions:
            return env.noop_action

        best_action = max(
            legal_pair_actions,
            key=lambda action: _expected_benefit(env, action),
        )
        if self.fire_only_if_positive and _expected_benefit(env, best_action) <= 0.0:
            return env.noop_action
        return best_action


def run_air_defense_episode(
    env: AirDefenseResourceAssignmentEnv,
    policy: AirDefensePolicy,
    seed: int | None = None,
) -> AirDefenseEpisodeMetrics:
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

        if info["invalid_action"]:
            invalid_actions += 1
        if info["action_type"] == "assign" and not info["invalid_action"]:
            shots += 1
        if info["hit"]:
            hits += 1

    return AirDefenseEpisodeMetrics(
        total_reward=float(total_reward),
        steps=steps,
        num_targets=env.num_targets,
        num_intercepted=int(info["num_intercepted"]),
        num_leaked=int(info["num_leaked"]),
        ammo_used=initial_ammo - int(info["ammo_remaining"]),
        shots=shots,
        hits=hits,
        invalid_actions=invalid_actions,
        success=bool(info["num_leaked"] == 0 and info["num_alive"] == 0),
    )


def evaluate_air_defense_policy(
    env_factory: Callable[[], AirDefenseResourceAssignmentEnv],
    policy_factory: Callable[[int], AirDefensePolicy],
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
        metrics = run_air_defense_episode(env, policy, seed=episode_seed)
        episode_metrics.append(metrics)
        env.close()

    rewards = np.asarray([metrics.total_reward for metrics in episode_metrics])
    steps = np.asarray([metrics.steps for metrics in episode_metrics])
    intercepted = np.asarray([metrics.num_intercepted for metrics in episode_metrics])
    leaked = np.asarray([metrics.num_leaked for metrics in episode_metrics])
    targets = np.asarray([metrics.num_targets for metrics in episode_metrics])
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
        "avg_ammo_used": float(np.mean(ammo_used)),
        "avg_shots": float(np.mean(shots)),
        "hit_rate_per_shot": float(np.sum(hits) / max(1, np.sum(shots))),
        "avg_invalid_actions": float(np.mean(invalid_actions)),
    }


def _legal_pair_actions(env: AirDefenseResourceAssignmentEnv) -> list[int]:
    mask = env.action_mask()
    return [
        int(action)
        for action in np.flatnonzero(mask)
        if int(action) != env.noop_action
    ]


def _target_distance_to_asset(env: AirDefenseResourceAssignmentEnv, action: int) -> float:
    _, target_index = env.decode_action(action)
    target = env.targets[target_index]
    return euclidean_distance(target.position, env.asset_position)


def _target_threat(env: AirDefenseResourceAssignmentEnv, action: int) -> float:
    _, target_index = env.decode_action(action)
    return float(env.targets[target_index].threat)


def _expected_benefit(env: AirDefenseResourceAssignmentEnv, action: int) -> float:
    unit_index, target_index = env.decode_action(action)
    unit = env.defense_units[unit_index]
    target = env.targets[target_index]
    hit_probability = compute_hit_probability(
        defense_position=unit.position,
        target_position=target.position,
        max_range=unit.max_range,
        base_hit_probability=unit.base_hit_probability,
        target_evasion=target.evasion,
    )
    expected_intercept_reward = (
        hit_probability * env.config.intercept_reward_weight * target.threat
    )
    return float(expected_intercept_reward - unit.cost)
