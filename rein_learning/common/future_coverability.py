from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Sequence

import numpy as np

from ..envs.air_defense_v1 import AirDefenseV1StateSnapshot
from ..simulators import compute_hit_probability, euclidean_distance


@dataclass(frozen=True)
class ThreatDemand:
    target_index: int
    deadline: int
    weight: float
    position: tuple[float, float]
    velocity: tuple[float, float]
    evasion: float = 0.0

    def __post_init__(self) -> None:
        values = (*self.position, *self.velocity, self.weight, self.evasion)
        if not all(isfinite(value) for value in values):
            raise ValueError("threat demand values must be finite")
        if self.target_index < 0:
            raise ValueError("target_index must be non-negative")
        if self.deadline <= 0:
            raise ValueError("deadline must be positive")
        if self.weight < 0.0:
            raise ValueError("weight must be non-negative")
        if not 0.0 <= self.evasion <= 1.0:
            raise ValueError("evasion must lie in [0, 1]")


@dataclass(frozen=True)
class ShotOpportunity:
    unit_index: int
    time: int
    position: tuple[float, float]
    max_range: float
    base_hit_probability: float

    def __post_init__(self) -> None:
        values = (*self.position, self.max_range, self.base_hit_probability)
        if not all(isfinite(value) for value in values):
            raise ValueError("shot opportunity values must be finite")
        if self.unit_index < 0:
            raise ValueError("unit_index must be non-negative")
        if self.time < 0:
            raise ValueError("shot time must be non-negative")
        if self.max_range <= 0.0:
            raise ValueError("max_range must be positive")
        if not 0.0 <= self.base_hit_probability <= 1.0:
            raise ValueError("base_hit_probability must lie in [0, 1]")


@dataclass(frozen=True)
class FutureCoverabilityCertificate:
    unit_index: int
    target_index: int
    other_threat_coverability_before: float
    other_threat_coverability_after: float
    externality: float


def opportunity_threat_value(
    opportunity: ShotOpportunity,
    threat: ThreatDemand,
) -> float:
    """Return the weighted one-attempt coverage value for one feasible edge."""

    if opportunity.time >= threat.deadline:
        return 0.0
    target_position = np.asarray(threat.position, dtype=np.float64) + (
        np.asarray(threat.velocity, dtype=np.float64) * opportunity.time
    )
    probability = compute_hit_probability(
        defense_position=np.asarray(opportunity.position, dtype=np.float64),
        target_position=target_position,
        max_range=opportunity.max_range,
        base_hit_probability=opportunity.base_hit_probability,
        target_evasion=threat.evasion,
    )
    return float(threat.weight * probability)


def maximum_weight_coverability(
    opportunities: Sequence[ShotOpportunity],
    threats: Sequence[ThreatDemand],
) -> float:
    """Solve the small one-attempt weighted matching problem exactly.

    Each shot opportunity and each threat can be selected at most once. The
    dynamic program is exponential only in the number of threats, which is five
    in AirDefense v1 and intentionally keeps the certificate auditable.
    """

    target_indices = [threat.target_index for threat in threats]
    if len(set(target_indices)) != len(target_indices):
        raise ValueError("threat target indices must be unique")
    if not threats or not opportunities:
        return 0.0

    values = [
        [
            opportunity_threat_value(opportunity, threat)
            for threat in threats
        ]
        for opportunity in opportunities
    ]
    states: dict[int, float] = {0: 0.0}
    for row in values:
        updated = dict(states)
        for mask, score in states.items():
            for target_offset, edge_value in enumerate(row):
                bit = 1 << target_offset
                if mask & bit or edge_value <= 0.0:
                    continue
                next_mask = mask | bit
                updated[next_mask] = max(
                    updated.get(next_mask, float("-inf")),
                    score + edge_value,
                )
        states = updated
    return float(max(states.values()))


def snapshot_threat_demands(
    snapshot: AirDefenseV1StateSnapshot,
    *,
    excluded_target_index: int | None = None,
) -> tuple[ThreatDemand, ...]:
    demands: list[ThreatDemand] = []
    for target_index, target in enumerate(snapshot.targets):
        if not target.alive or target_index == excluded_target_index:
            continue
        zone = snapshot.protected_zones[target.target_zone]
        deadline = max(1, int(ceil(target.time_to_impact)))
        demands.append(
            ThreatDemand(
                target_index=target_index,
                deadline=deadline,
                weight=float(target.payload * target.threat * zone.value),
                position=tuple(float(value) for value in target.position),
                velocity=tuple(float(value) for value in target.velocity),
                evasion=float(target.evasion),
            )
        )
    return tuple(demands)


def snapshot_shot_opportunities(
    snapshot: AirDefenseV1StateSnapshot,
    *,
    consumed_unit_index: int | None = None,
    horizon: int | None = None,
) -> tuple[ShotOpportunity, ...]:
    if horizon is None:
        alive_deadlines = [
            max(1, int(ceil(target.time_to_impact)))
            for target in snapshot.targets
            if target.alive
        ]
        horizon = max(alive_deadlines, default=1)
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    opportunities: list[ShotOpportunity] = []
    for unit_index, unit in enumerate(snapshot.defense_units):
        if unit.energy <= 0.0:
            continue
        ammo = int(unit.ammo) - int(unit_index == consumed_unit_index)
        if ammo <= 0:
            continue
        interval = max(1, int(unit.cooldown_after_fire))
        if unit_index == consumed_unit_index:
            first_time = max(1, int(unit.cooldown_after_fire))
        else:
            first_time = max(0, int(unit.cooldown))
        for shot_index in range(ammo):
            shot_time = first_time + shot_index * interval
            if shot_time >= horizon:
                break
            opportunities.append(
                ShotOpportunity(
                    unit_index=unit_index,
                    time=shot_time,
                    position=tuple(float(value) for value in unit.position),
                    max_range=float(unit.max_range),
                    base_hit_probability=float(unit.base_hit_probability),
                )
            )
    return tuple(opportunities)


def future_coverability_externality(
    snapshot: AirDefenseV1StateSnapshot,
    *,
    unit_index: int,
    target_index: int,
) -> FutureCoverabilityCertificate:
    """Measure how a legal current assignment harms coverage of other threats."""

    if not 0 <= unit_index < len(snapshot.defense_units):
        raise IndexError("unit_index out of range")
    if not 0 <= target_index < len(snapshot.targets):
        raise IndexError("target_index out of range")
    unit = snapshot.defense_units[unit_index]
    target = snapshot.targets[target_index]
    if not unit.available or not target.alive:
        raise ValueError("the probed assignment must be currently available")
    if euclidean_distance(unit.position, target.position) > unit.max_range:
        raise ValueError("the probed assignment must be currently in range")

    other_threats = snapshot_threat_demands(
        snapshot,
        excluded_target_index=target_index,
    )
    horizon = max((threat.deadline for threat in other_threats), default=1)
    before = maximum_weight_coverability(
        snapshot_shot_opportunities(snapshot, horizon=horizon),
        other_threats,
    )
    after = maximum_weight_coverability(
        snapshot_shot_opportunities(
            snapshot,
            consumed_unit_index=unit_index,
            horizon=horizon,
        ),
        other_threats,
    )
    externality = max(0.0, before - after)
    return FutureCoverabilityCertificate(
        unit_index=unit_index,
        target_index=target_index,
        other_threat_coverability_before=before,
        other_threat_coverability_after=after,
        externality=externality,
    )

