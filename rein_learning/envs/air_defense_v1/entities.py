from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProtectedZoneState:
    position: np.ndarray
    radius: float
    value: float
    priority: float
    zone_type: str
    damage: float = 0.0


@dataclass
class DefenseUnitV1State:
    resource_type: str
    position: np.ndarray
    ammo: int
    max_ammo: int
    max_range: float
    base_hit_probability: float
    cost: float
    cooldown: int
    cooldown_after_fire: int
    energy: float
    max_energy: float

    @property
    def available(self) -> bool:
        return self.ammo > 0 and self.cooldown == 0 and self.energy > 0.0


@dataclass
class TargetV1State:
    position: np.ndarray
    velocity: np.ndarray
    speed: float
    threat: float
    target_zone: int
    payload: float
    evasion: float
    target_class: str
    status: str = "alive"
    time_to_impact: float = 0.0
    track_confidence: float = 1.0
    aoi: float = 0.0
    leaked_damage: float = 0.0

    @property
    def alive(self) -> bool:
        return self.status == "alive"
