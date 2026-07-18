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
    TARGET_CLASS_IDS,
    ZONE_TYPE_IDS,
    AirDefenseV1EnvConfig,
    DefenseUnitV1Config,
    ProtectedZoneConfig,
    TargetV1Config,
)
from .entities import DefenseUnitV1State, ProtectedZoneState, TargetV1State
from .scenarios import default_air_defense_v1_config


@dataclass(frozen=True)
class UnitActionResult:
    unit_index: int
    target_index: int | None
    legal: bool
    hit: bool
    hit_probability: float
    action_type: str


class AirDefenseResourceAssignmentEnvV1(gym.Env):
    """Centralized air-defense resource assignment environment v1.0."""

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    ZONE_FEATURES = 7
    TARGET_FEATURES = 15
    UNIT_FEATURES = 15
    GLOBAL_FEATURES = 8

    def __init__(
        self,
        config: AirDefenseV1EnvConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        self.config = config or default_air_defense_v1_config()
        self.render_mode = render_mode
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.num_zones = len(self.config.protected_zones)
        self.num_defense_units = len(self.config.defense_units)
        self.num_targets = len(self.config.targets) or self.config.num_random_targets
        if self.num_zones <= 0:
            raise ValueError("At least one protected zone is required")
        if self.num_defense_units <= 0:
            raise ValueError("At least one defense unit is required")
        if self.num_targets <= 0:
            raise ValueError("At least one target is required")

        self.noop_action = self.num_targets
        self.num_unit_actions = self.num_targets + 1
        self.action_space = spaces.MultiDiscrete(
            np.full(self.num_defense_units, self.num_unit_actions, dtype=np.int64)
        )
        obs_dim = (
            self.num_zones * self.ZONE_FEATURES
            + self.num_targets * self.TARGET_FEATURES
            + self.num_defense_units * self.UNIT_FEATURES
            + self.GLOBAL_FEATURES
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.protected_zones: list[ProtectedZoneState] = []
        self.defense_units: list[DefenseUnitV1State] = []
        self.targets: list[TargetV1State] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.protected_zones = [
            self._create_zone_state(zone_config)
            for zone_config in self.config.protected_zones
        ]
        self.defense_units = [
            self._create_defense_unit_state(unit_config)
            for unit_config in self.config.defense_units
        ]
        self.targets = self._create_target_states()
        return self._get_observation(), self._get_info()

    def step(
        self,
        action: np.ndarray | list[int] | tuple[int, ...],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        joint_action = np.asarray(action, dtype=np.int64)
        if not self.action_space.contains(joint_action):
            raise ValueError(f"Invalid joint action {action}")

        self._decrement_cooldowns()
        reward_breakdown = self._empty_reward_breakdown()
        reward_breakdown["time"] = self.config.time_penalty

        valid_assignments: dict[int, list[int]] = {}
        unit_results: list[UnitActionResult] = []
        invalid_actions = 0
        shots = 0

        for unit_index, unit_action in enumerate(joint_action.tolist()):
            if unit_action == self.noop_action:
                unit_results.append(
                    UnitActionResult(
                        unit_index=unit_index,
                        target_index=None,
                        legal=True,
                        hit=False,
                        hit_probability=0.0,
                        action_type="noop",
                    )
                )
                continue

            target_index = int(unit_action)
            if not self.is_unit_target_action_legal(unit_index, target_index):
                invalid_actions += 1
                unit_results.append(
                    UnitActionResult(
                        unit_index=unit_index,
                        target_index=target_index,
                        legal=False,
                        hit=False,
                        hit_probability=0.0,
                        action_type="engage",
                    )
                )
                continue

            unit = self.defense_units[unit_index]
            unit.ammo -= 1
            unit.cooldown = unit.cooldown_after_fire
            reward_breakdown["cost"] -= unit.cost
            shots += 1
            valid_assignments.setdefault(target_index, []).append(unit_index)

        if invalid_actions:
            reward_breakdown["invalid"] = (
                self.config.invalid_action_penalty * invalid_actions
            )

        hit_results = self._resolve_valid_assignments(
            valid_assignments=valid_assignments,
            reward_breakdown=reward_breakdown,
        )
        unit_results.extend(hit_results)

        self._advance_targets()
        damage_this_step = self._mark_leaked_targets()
        reward_breakdown["damage"] = -self.config.damage_penalty_weight * damage_this_step
        self.current_step += 1

        terminated = self._is_terminated()
        truncated = self.current_step >= self.config.max_steps and not terminated
        if terminated:
            if self.total_damage <= self.config.success_damage_threshold:
                reward_breakdown["terminal"] = self.config.success_bonus
            elif self.total_damage >= self.config.max_allowed_damage:
                reward_breakdown["terminal"] = self.config.failure_penalty

        reward = float(sum(reward_breakdown.values()))
        info = self._get_info()
        info.update(
            {
                "joint_action": joint_action.copy(),
                "unit_results": [result.__dict__ for result in unit_results],
                "invalid_actions": invalid_actions,
                "shots": shots,
                "hits": sum(result.hit for result in unit_results),
                "damage_this_step": damage_this_step,
                "reward_breakdown": reward_breakdown,
            }
        )
        return self._get_observation(), reward, terminated, truncated, info

    @property
    def total_damage(self) -> float:
        return float(sum(zone.damage for zone in self.protected_zones))

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
    def ammo_remaining(self) -> int:
        return sum(unit.ammo for unit in self.defense_units)

    @property
    def max_ammo(self) -> int:
        return sum(unit.max_ammo for unit in self.defense_units)

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(
            (self.num_defense_units, self.num_unit_actions),
            dtype=np.int8,
        )
        for unit_index in range(self.num_defense_units):
            for target_index in range(self.num_targets):
                if self.is_unit_target_action_legal(unit_index, target_index):
                    mask[unit_index, target_index] = 1
            mask[unit_index, self.noop_action] = 1
        return mask

    def action_masks(self) -> np.ndarray:
        return self.action_mask().reshape(-1)

    def is_unit_target_action_legal(self, unit_index: int, target_index: int) -> bool:
        if not (0 <= unit_index < self.num_defense_units):
            return False
        if not (0 <= target_index < self.num_targets):
            return False
        unit = self.defense_units[unit_index]
        target = self.targets[target_index]
        if not unit.available or not target.alive:
            return False
        return euclidean_distance(unit.position, target.position) <= unit.max_range

    def hit_probability(self, unit_index: int, target_index: int) -> float:
        unit = self.defense_units[unit_index]
        target = self.targets[target_index]
        return compute_hit_probability(
            defense_position=unit.position,
            target_position=target.position,
            max_range=unit.max_range,
            base_hit_probability=unit.base_hit_probability,
            target_evasion=target.evasion,
        )

    def target_damage_potential(self, target_index: int) -> float:
        target = self.targets[target_index]
        zone = self.protected_zones[target.target_zone]
        return float(target.payload * target.threat * zone.value)

    def render(self) -> str | None:
        lines = [
            f"step={self.current_step}",
            f"alive={self.num_alive}, intercepted={self.num_intercepted}, "
            f"leaked={self.num_leaked}, total_damage={self.total_damage:.2f}",
            "protected_zones:",
        ]
        for index, zone in enumerate(self.protected_zones):
            lines.append(
                f"  Z{index}: type={zone.zone_type}, pos={tuple(zone.position.tolist())}, "
                f"value={zone.value:.2f}, damage={zone.damage:.2f}"
            )
        lines.append("defense_units:")
        for index, unit in enumerate(self.defense_units):
            lines.append(
                f"  D{index}: type={unit.resource_type}, pos={tuple(unit.position.tolist())}, "
                f"ammo={unit.ammo}, cooldown={unit.cooldown}"
            )
        lines.append("targets:")
        for index, target in enumerate(self.targets):
            zone = self.protected_zones[target.target_zone]
            distance = euclidean_distance(target.position, zone.position)
            lines.append(
                f"  T{index}: status={target.status}, zone=Z{target.target_zone}, "
                f"pos={tuple(target.position.tolist())}, threat={target.threat:.2f}, "
                f"payload={target.payload:.2f}, tti={target.time_to_impact:.1f}, "
                f"dist_to_zone={distance:.2f}"
            )
        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
            print()
            return None
        return output

    def close(self) -> None:
        return None

    def _resolve_valid_assignments(
        self,
        valid_assignments: dict[int, list[int]],
        reward_breakdown: dict[str, float],
    ) -> list[UnitActionResult]:
        results: list[UnitActionResult] = []
        for target_index, unit_indices in valid_assignments.items():
            target = self.targets[target_index]
            if not target.alive:
                continue

            if len(unit_indices) > 1:
                reward_breakdown["conflict"] -= (
                    self.config.assignment_conflict_penalty * (len(unit_indices) - 1)
                )
                reward_breakdown["overkill"] -= (
                    self.config.overkill_penalty
                    * max(0, len(unit_indices) - 1)
                    * target.threat
                )

            probabilities = [
                self.hit_probability(unit_index, target_index)
                for unit_index in unit_indices
            ]
            combined_miss_probability = float(np.prod([1.0 - p for p in probabilities]))
            combined_hit_probability = 1.0 - combined_miss_probability
            hit = bool(self.np_random.random() < combined_hit_probability)

            if hit:
                target.status = "intercepted"
                zone = self.protected_zones[target.target_zone]
                reward_breakdown["intercept"] += (
                    self.config.intercept_reward_weight
                    * target.threat
                    * zone.value
                )

            for unit_index, probability in zip(unit_indices, probabilities):
                results.append(
                    UnitActionResult(
                        unit_index=unit_index,
                        target_index=target_index,
                        legal=True,
                        hit=hit,
                        hit_probability=float(probability),
                        action_type="engage",
                    )
                )
        return results

    def _create_zone_state(
        self,
        zone_config: ProtectedZoneConfig,
    ) -> ProtectedZoneState:
        if zone_config.zone_type not in ZONE_TYPE_IDS:
            raise ValueError(f"Unknown zone type: {zone_config.zone_type}")
        return ProtectedZoneState(
            position=np.asarray(zone_config.position, dtype=np.float32),
            radius=zone_config.radius,
            value=zone_config.value,
            priority=zone_config.priority,
            zone_type=zone_config.zone_type,
        )

    def _create_defense_unit_state(
        self,
        unit_config: DefenseUnitV1Config,
    ) -> DefenseUnitV1State:
        if unit_config.resource_type not in RESOURCE_TYPE_IDS:
            raise ValueError(f"Unknown resource type: {unit_config.resource_type}")
        return DefenseUnitV1State(
            resource_type=unit_config.resource_type,
            position=np.asarray(unit_config.position, dtype=np.float32),
            ammo=unit_config.ammo,
            max_ammo=unit_config.ammo,
            max_range=unit_config.max_range,
            base_hit_probability=unit_config.base_hit_probability,
            cost=unit_config.cost,
            cooldown=0,
            cooldown_after_fire=unit_config.cooldown_after_fire,
            energy=unit_config.energy,
            max_energy=unit_config.energy,
        )

    def _create_target_states(self) -> list[TargetV1State]:
        if self.config.targets:
            return [
                self._create_target_state(target_config)
                for target_config in self.config.targets
            ]
        return [self._sample_random_target() for _ in range(self.num_targets)]

    def _create_target_state(self, target_config: TargetV1Config) -> TargetV1State:
        if not (0 <= target_config.target_zone < self.num_zones):
            raise ValueError(f"Invalid target_zone: {target_config.target_zone}")
        if target_config.target_class not in TARGET_CLASS_IDS:
            raise ValueError(f"Unknown target class: {target_config.target_class}")
        position = np.asarray(target_config.position, dtype=np.float32)
        zone = self.protected_zones[target_config.target_zone]
        velocity = velocity_towards_asset(position, zone.position, target_config.speed)
        time_to_impact = self._estimate_time_to_zone(position, zone, target_config.speed)
        return TargetV1State(
            position=position,
            velocity=velocity,
            speed=target_config.speed,
            threat=target_config.threat,
            target_zone=target_config.target_zone,
            payload=target_config.payload,
            evasion=target_config.evasion,
            target_class=target_config.target_class,
            time_to_impact=time_to_impact,
        )

    def _sample_random_target(self) -> TargetV1State:
        target_zone = int(self.np_random.integers(0, self.num_zones))
        zone = self.protected_zones[target_zone]
        angle = float(self.np_random.uniform(0.0, 2.0 * np.pi))
        distance = float(
            self.np_random.uniform(
                self.config.target_spawn_min_distance,
                self.config.target_spawn_max_distance,
            )
        )
        position = zone.position + np.asarray(
            [np.cos(angle) * distance, np.sin(angle) * distance],
            dtype=np.float32,
        )
        position = np.clip(
            position,
            -self.config.map_size,
            self.config.map_size,
        ).astype(np.float32)
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
        payload = float(
            self.np_random.uniform(
                self.config.target_min_payload,
                self.config.target_max_payload,
            )
        )
        target_class = str(self.np_random.choice(tuple(TARGET_CLASS_IDS)))
        velocity = velocity_towards_asset(position, zone.position, speed)
        return TargetV1State(
            position=position,
            velocity=velocity,
            speed=speed,
            threat=threat,
            target_zone=target_zone,
            payload=payload,
            evasion=0.0,
            target_class=target_class,
            time_to_impact=self._estimate_time_to_zone(position, zone, speed),
        )

    def _decrement_cooldowns(self) -> None:
        for unit in self.defense_units:
            unit.cooldown = max(0, unit.cooldown - 1)

    def _advance_targets(self) -> None:
        for target in self.targets:
            if not target.alive:
                continue
            zone = self.protected_zones[target.target_zone]
            target.position = advance_position(
                position=target.position,
                velocity=target.velocity,
                dt=self.config.dt,
                map_size=self.config.map_size,
            )
            target.velocity = velocity_towards_asset(
                target_position=target.position,
                asset_position=zone.position,
                speed=target.speed,
            )
            target.time_to_impact = self._estimate_time_to_zone(
                target.position,
                zone,
                target.speed,
            )
            target.aoi += self.config.dt

    def _mark_leaked_targets(self) -> float:
        damage = 0.0
        for target_index, target in enumerate(self.targets):
            if not target.alive:
                continue
            zone = self.protected_zones[target.target_zone]
            if euclidean_distance(target.position, zone.position) <= zone.radius:
                target.status = "leaked"
                target_damage = self.target_damage_potential(target_index)
                target.leaked_damage = target_damage
                zone.damage += target_damage
                damage += target_damage
        return float(damage)

    def _is_terminated(self) -> bool:
        if self.num_alive == 0:
            return True
        return self.total_damage >= self.config.max_allowed_damage

    def _estimate_time_to_zone(
        self,
        position: np.ndarray,
        zone: ProtectedZoneState,
        speed: float,
    ) -> float:
        distance = max(0.0, euclidean_distance(position, zone.position) - zone.radius)
        return float(distance / max(speed, 1e-6))

    def _get_observation(self) -> np.ndarray:
        values: list[float] = []
        values.extend(self._zone_features())
        values.extend(self._target_features())
        values.extend(self._unit_features())
        values.extend(self._global_features())
        return np.asarray(values, dtype=np.float32)

    def _zone_features(self) -> list[float]:
        max_zone_value = max(1.0, max(zone.value for zone in self.protected_zones))
        values: list[float] = []
        for zone in self.protected_zones:
            values.extend(
                [
                    zone.position[0] / self.config.map_size,
                    zone.position[1] / self.config.map_size,
                    zone.radius / self.config.map_size,
                    zone.value / max_zone_value,
                    min(1.0, zone.damage / max(1.0, self.config.max_allowed_damage)),
                    zone.priority,
                    ZONE_TYPE_IDS[zone.zone_type] / max(1, len(ZONE_TYPE_IDS) - 1),
                ]
            )
        return values

    def _target_features(self) -> list[float]:
        max_distance = max(1.0, self.config.target_spawn_max_distance)
        max_speed = max(1.0, self.config.target_max_speed)
        max_payload = max(1.0, self.config.target_max_payload)
        target_zone_scale = max(1, self.num_zones - 1)
        class_scale = max(1, len(TARGET_CLASS_IDS) - 1)
        values: list[float] = []
        for target in self.targets:
            zone = self.protected_zones[target.target_zone]
            distance_to_zone = euclidean_distance(target.position, zone.position)
            is_assigned = 0.0
            values.extend(
                [
                    target.position[0] / self.config.map_size,
                    target.position[1] / self.config.map_size,
                    target.velocity[0] / max_speed,
                    target.velocity[1] / max_speed,
                    min(1.0, distance_to_zone / max_distance),
                    min(1.0, target.time_to_impact / max(1, self.config.max_steps)),
                    target.threat,
                    target.payload / max_payload,
                    target.evasion,
                    target.target_zone / target_zone_scale,
                    TARGET_CLASS_IDS[target.target_class] / class_scale,
                    target.track_confidence,
                    min(1.0, target.aoi / max(1, self.config.max_steps)),
                    1.0 if target.alive else 0.0,
                    is_assigned,
                ]
            )
        return values

    def _unit_features(self) -> list[float]:
        max_cooldown = max(
            1,
            max(unit.cooldown_after_fire for unit in self.defense_units),
        )
        max_cost = max(1.0, max(unit.cost for unit in self.defense_units))
        type_scale = max(1, len(RESOURCE_TYPE_IDS) - 1)
        values: list[float] = []
        for unit in self.defense_units:
            values.extend(
                [
                    unit.position[0] / self.config.map_size,
                    unit.position[1] / self.config.map_size,
                    RESOURCE_TYPE_IDS[unit.resource_type] / type_scale,
                    unit.ammo / max(1, unit.max_ammo),
                    unit.energy / max(1e-6, unit.max_energy),
                    unit.cooldown / max_cooldown,
                    unit.max_range / self.config.map_size,
                    0.0,
                    1.0,
                    unit.base_hit_probability,
                    unit.cost / max_cost,
                    1.0,
                    1.0,
                    0.0,
                    1.0 if unit.available else 0.0,
                ]
            )
        return values

    def _global_features(self) -> list[float]:
        available_units = sum(unit.available for unit in self.defense_units)
        return [
            self.current_step / max(1, self.config.max_steps),
            (self.config.max_steps - self.current_step) / max(1, self.config.max_steps),
            self.num_alive / self.num_targets,
            self.num_intercepted / self.num_targets,
            self.num_leaked / self.num_targets,
            min(1.0, self.total_damage / max(1.0, self.config.max_allowed_damage)),
            available_units / self.num_defense_units,
            self.ammo_remaining / max(1, self.max_ammo),
        ]

    def _get_info(self) -> dict[str, Any]:
        return {
            "current_step": self.current_step,
            "num_intercepted": self.num_intercepted,
            "num_leaked": self.num_leaked,
            "num_alive": self.num_alive,
            "total_damage": self.total_damage,
            "zone_damage": [zone.damage for zone in self.protected_zones],
            "ammo_remaining": self.ammo_remaining,
        }

    def _empty_reward_breakdown(self) -> dict[str, float]:
        return {
            "intercept": 0.0,
            "damage": 0.0,
            "cost": 0.0,
            "invalid": 0.0,
            "conflict": 0.0,
            "overkill": 0.0,
            "time": 0.0,
            "terminal": 0.0,
        }
