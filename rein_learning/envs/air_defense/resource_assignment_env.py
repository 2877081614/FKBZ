from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ...simulators import (
    advance_position,
    compute_hit_probability,
    euclidean_distance,
    sample_intercept,
    velocity_towards_asset,
)
from .config import (
    RESOURCE_TYPE_IDS,
    AirDefenseEnvConfig,
    DefenseUnitConfig,
    TargetConfig,
)


@dataclass
class DefenseUnitState:
    resource_type: str
    position: np.ndarray
    ammo: int
    max_ammo: int
    max_range: float
    base_hit_probability: float
    cost: float
    cooldown: int
    cooldown_after_fire: int


@dataclass
class TargetState:
    position: np.ndarray
    velocity: np.ndarray
    speed: float
    threat: float
    evasion: float
    status: str = "alive"

    @property
    def alive(self) -> bool:
        return self.status == "alive"


class AirDefenseResourceAssignmentEnv(gym.Env):
    """Single-agent air-defense resource assignment environment v0."""

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    DEFENSE_FEATURES = 7
    TARGET_FEATURES = 7
    GLOBAL_FEATURES = 4

    def __init__(
        self,
        config: AirDefenseEnvConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        self.config = config or AirDefenseEnvConfig()
        self.render_mode = render_mode
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.asset_position = np.asarray(self.config.asset_position, dtype=np.float32)
        self.num_defense_units = len(self.config.defense_units)
        self.num_targets = len(self.config.targets) or self.config.num_random_targets
        if self.num_defense_units <= 0:
            raise ValueError("At least one defense unit is required")
        if self.num_targets <= 0:
            raise ValueError("At least one target is required")

        self.action_space = spaces.Discrete(self.num_defense_units * self.num_targets + 1)
        obs_dim = (
            self.num_defense_units * self.DEFENSE_FEATURES
            + self.num_targets * self.TARGET_FEATURES
            + self.GLOBAL_FEATURES
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.defense_units: list[DefenseUnitState] = []
        self.targets: list[TargetState] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.defense_units = [
            self._create_defense_unit_state(unit_config)
            for unit_config in self.config.defense_units
        ]
        self.targets = self._create_target_states()
        return self._get_observation(), self._get_info()

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        self._decrement_cooldowns()
        reward_breakdown = self._empty_reward_breakdown()
        reward_breakdown["time"] = self.config.time_penalty

        selected_unit = None
        selected_target = None
        hit = False
        hit_probability = 0.0
        invalid_action = False
        action_type = "assign"

        if int(action) == self.noop_action:
            action_type = "noop"
        else:
            selected_unit, selected_target = self.decode_action(int(action))
            unit = self.defense_units[selected_unit]
            target = self.targets[selected_target]

            if not self._is_action_legal(unit, target):
                invalid_action = True
                reward_breakdown["invalid"] = self.config.invalid_action_penalty
            else:
                intercept = sample_intercept(
                    rng=self.np_random,
                    defense_position=unit.position,
                    target_position=target.position,
                    max_range=unit.max_range,
                    base_hit_probability=unit.base_hit_probability,
                    target_evasion=target.evasion,
                )
                hit_probability = intercept.hit_probability
                hit = intercept.hit
                unit.ammo -= 1
                unit.cooldown = unit.cooldown_after_fire
                reward_breakdown["cost"] = -unit.cost

                if hit:
                    target.status = "intercepted"
                    reward_breakdown["intercept"] = (
                        self.config.intercept_reward_weight * target.threat
                    )

        self._advance_targets()
        reward_breakdown["leak"] = self._mark_leaked_targets()
        self.current_step += 1

        terminated = self._is_terminated()
        truncated = self.current_step >= self.config.max_steps and not terminated
        if terminated:
            if self.num_leaked == 0:
                reward_breakdown["terminal"] = self.config.success_bonus
            elif self.num_leaked >= self.max_allowed_leaks:
                reward_breakdown["terminal"] = self.config.failure_penalty

        reward = float(sum(reward_breakdown.values()))
        info = self._get_info()
        info.update(
            {
                "action_type": action_type,
                "selected_defense_unit": selected_unit,
                "selected_target": selected_target,
                "invalid_action": invalid_action,
                "hit": hit,
                "hit_probability": hit_probability,
                "reward_breakdown": reward_breakdown,
            }
        )

        return self._get_observation(), reward, terminated, truncated, info

    @property
    def noop_action(self) -> int:
        return self.num_defense_units * self.num_targets

    @property
    def num_intercepted(self) -> int:
        return sum(target.status == "intercepted" for target in self.targets)

    @property
    def num_leaked(self) -> int:
        return sum(target.status == "leaked" for target in self.targets)

    @property
    def num_alive(self) -> int:
        return sum(target.alive for target in self.targets)

    @property
    def max_allowed_leaks(self) -> int:
        return self.config.max_allowed_leaks or self.num_targets

    def decode_action(self, action: int) -> tuple[int, int]:
        if action == self.noop_action:
            raise ValueError("no-op action does not map to a defense-target pair")
        return action // self.num_targets, action % self.num_targets

    def render(self) -> str | None:
        lines = [
            f"step={self.current_step}",
            f"asset={tuple(self.asset_position.tolist())}, alive={self.num_alive}, "
            f"intercepted={self.num_intercepted}, leaked={self.num_leaked}",
            "defense_units:",
        ]
        for index, unit in enumerate(self.defense_units):
            lines.append(
                f"  D{index}: type={unit.resource_type}, pos={tuple(unit.position.tolist())}, "
                f"ammo={unit.ammo}, cooldown={unit.cooldown}"
            )
        lines.append("targets:")
        for index, target in enumerate(self.targets):
            distance = euclidean_distance(target.position, self.asset_position)
            lines.append(
                f"  T{index}: status={target.status}, pos={tuple(target.position.tolist())}, "
                f"threat={target.threat:.2f}, dist={distance:.2f}"
            )
        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
            print()
            return None
        return output

    def close(self) -> None:
        return None

    def _create_defense_unit_state(
        self,
        unit_config: DefenseUnitConfig,
    ) -> DefenseUnitState:
        if unit_config.resource_type not in RESOURCE_TYPE_IDS:
            raise ValueError(f"Unknown resource type: {unit_config.resource_type}")
        return DefenseUnitState(
            resource_type=unit_config.resource_type,
            position=np.asarray(unit_config.position, dtype=np.float32),
            ammo=unit_config.ammo,
            max_ammo=unit_config.ammo,
            max_range=unit_config.max_range,
            base_hit_probability=unit_config.base_hit_probability,
            cost=unit_config.cost,
            cooldown=0,
            cooldown_after_fire=unit_config.cooldown_after_fire,
        )

    def _create_target_states(self) -> list[TargetState]:
        if self.config.targets:
            return [
                self._create_target_state(target_config)
                for target_config in self.config.targets
            ]
        return [self._sample_random_target() for _ in range(self.num_targets)]

    def _create_target_state(self, target_config: TargetConfig) -> TargetState:
        position = np.asarray(target_config.position, dtype=np.float32)
        velocity = velocity_towards_asset(
            target_position=position,
            asset_position=self.asset_position,
            speed=target_config.speed,
        )
        return TargetState(
            position=position,
            velocity=velocity,
            speed=target_config.speed,
            threat=target_config.threat,
            evasion=target_config.evasion,
        )

    def _sample_random_target(self) -> TargetState:
        angle = float(self.np_random.uniform(0.0, 2.0 * np.pi))
        distance = float(
            self.np_random.uniform(
                self.config.target_spawn_min_distance,
                self.config.target_spawn_max_distance,
            )
        )
        position = self.asset_position + np.asarray(
            [np.cos(angle) * distance, np.sin(angle) * distance],
            dtype=np.float32,
        )
        speed = float(
            self.np_random.uniform(
                self.config.target_min_speed,
                self.config.target_max_speed,
            )
        )
        threat = float(
            self.np_random.uniform(
                self.config.target_min_threat,
                self.config.target_max_threat,
            )
        )
        velocity = velocity_towards_asset(position, self.asset_position, speed)
        return TargetState(
            position=position,
            velocity=velocity,
            speed=speed,
            threat=threat,
            evasion=0.0,
        )

    def _decrement_cooldowns(self) -> None:
        for unit in self.defense_units:
            unit.cooldown = max(0, unit.cooldown - 1)

    def _is_action_legal(self, unit: DefenseUnitState, target: TargetState) -> bool:
        if unit.ammo <= 0 or unit.cooldown > 0 or not target.alive:
            return False
        return euclidean_distance(unit.position, target.position) <= unit.max_range

    def _advance_targets(self) -> None:
        for target in self.targets:
            if not target.alive:
                continue
            target.position = advance_position(
                position=target.position,
                velocity=target.velocity,
                dt=self.config.dt,
                map_size=self.config.map_size,
            )
            target.velocity = velocity_towards_asset(
                target_position=target.position,
                asset_position=self.asset_position,
                speed=target.speed,
            )

    def _mark_leaked_targets(self) -> float:
        penalty = 0.0
        for target in self.targets:
            if not target.alive:
                continue
            if euclidean_distance(target.position, self.asset_position) <= self.config.asset_radius:
                target.status = "leaked"
                penalty -= self.config.leak_penalty_weight * target.threat
        return float(penalty)

    def _is_terminated(self) -> bool:
        if self.num_alive == 0:
            return True
        return self.num_leaked >= self.max_allowed_leaks

    def _get_observation(self) -> np.ndarray:
        values: list[float] = []
        values.extend(self._defense_features())
        values.extend(self._target_features())
        values.extend(self._global_features())
        return np.asarray(values, dtype=np.float32)

    def _defense_features(self) -> list[float]:
        values: list[float] = []
        max_cooldown = max(
            1,
            max(unit.cooldown_after_fire for unit in self.defense_units),
        )
        for unit in self.defense_units:
            values.extend(
                [
                    unit.position[0] / self.config.map_size,
                    unit.position[1] / self.config.map_size,
                    RESOURCE_TYPE_IDS[unit.resource_type],
                    unit.ammo / max(1, unit.max_ammo),
                    unit.cooldown / max_cooldown,
                    unit.max_range / self.config.map_size,
                    unit.base_hit_probability,
                ]
            )
        return values

    def _target_features(self) -> list[float]:
        values: list[float] = []
        max_distance = max(1.0, self.config.target_spawn_max_distance)
        max_speed = max(1.0, self.config.target_max_speed)
        for target in self.targets:
            distance = euclidean_distance(target.position, self.asset_position)
            values.extend(
                [
                    target.position[0] / self.config.map_size,
                    target.position[1] / self.config.map_size,
                    target.velocity[0] / max_speed,
                    target.velocity[1] / max_speed,
                    min(1.0, distance / max_distance),
                    target.threat,
                    1.0 if target.alive else 0.0,
                ]
            )
        return values

    def _global_features(self) -> list[float]:
        available_units = sum(
            unit.ammo > 0 and unit.cooldown == 0 for unit in self.defense_units
        )
        return [
            self.current_step / max(1, self.config.max_steps),
            (self.config.max_steps - self.current_step) / max(1, self.config.max_steps),
            self.num_alive / self.num_targets,
            available_units / self.num_defense_units,
        ]

    def _get_info(self) -> dict[str, Any]:
        return {
            "current_step": self.current_step,
            "num_intercepted": self.num_intercepted,
            "num_leaked": self.num_leaked,
            "num_alive": self.num_alive,
            "ammo_remaining": sum(unit.ammo for unit in self.defense_units),
        }

    def _empty_reward_breakdown(self) -> dict[str, float]:
        return {
            "intercept": 0.0,
            "leak": 0.0,
            "cost": 0.0,
            "invalid": 0.0,
            "time": 0.0,
            "terminal": 0.0,
        }

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        for unit_index, unit in enumerate(self.defense_units):
            for target_index, target in enumerate(self.targets):
                action = unit_index * self.num_targets + target_index
                if self._is_action_legal(unit, target):
                    mask[action] = 1
        mask[self.noop_action] = 1
        return mask
