from __future__ import annotations

import numpy as np
import pytest

from rein_learning.common.future_coverability import (
    ShotOpportunity,
    ThreatDemand,
    future_coverability_externality,
    maximum_weight_coverability,
)
from rein_learning.envs.air_defense_v1 import AirDefenseV1StateSnapshot
from rein_learning.envs.air_defense_v1.entities import (
    DefenseUnitV1State,
    ProtectedZoneState,
    TargetV1State,
)


def _zone() -> ProtectedZoneState:
    return ProtectedZoneState(
        position=np.asarray([0.0, 0.0]),
        radius=1.0,
        value=1.0,
        priority=1.0,
        zone_type="command",
    )


def _unit(
    *,
    x: float,
    ammo: int = 1,
    max_range: float = 10.0,
    cooldown: int = 0,
    cooldown_after_fire: int = 1,
) -> DefenseUnitV1State:
    return DefenseUnitV1State(
        resource_type="missile",
        position=np.asarray([x, 0.0]),
        ammo=ammo,
        max_ammo=max(ammo, 1),
        max_range=max_range,
        base_hit_probability=1.0,
        cost=2.0,
        cooldown=cooldown,
        cooldown_after_fire=cooldown_after_fire,
        energy=1.0,
        max_energy=1.0,
    )


def _target(*, x: float, deadline: float = 3.0) -> TargetV1State:
    return TargetV1State(
        position=np.asarray([x, 0.0]),
        velocity=np.asarray([0.0, 0.0]),
        speed=0.0,
        threat=1.0,
        target_zone=0,
        payload=1.0,
        evasion=0.0,
        target_class="uav",
        time_to_impact=deadline,
    )


def _snapshot(
    units: tuple[DefenseUnitV1State, ...],
    targets: tuple[TargetV1State, ...],
) -> AirDefenseV1StateSnapshot:
    return AirDefenseV1StateSnapshot(
        current_step=0,
        protected_zones=(_zone(),),
        defense_units=units,
        targets=targets,
        np_random_state={},
        hit_random_tape=None,
    )


def test_single_threat_has_zero_externality() -> None:
    certificate = future_coverability_externality(
        _snapshot((_unit(x=0.0),), (_target(x=1.0),)),
        unit_index=0,
        target_index=0,
    )
    assert certificate.externality == 0.0


def test_fully_substitutable_unit_has_zero_externality() -> None:
    snapshot = _snapshot(
        (_unit(x=0.0), _unit(x=0.0)),
        (_target(x=1.0), _target(x=2.0)),
    )
    certificate = future_coverability_externality(
        snapshot,
        unit_index=0,
        target_index=0,
    )
    assert certificate.externality == pytest.approx(0.0)


def test_flexible_unit_consumption_exposes_other_threat() -> None:
    snapshot = _snapshot(
        (
            _unit(x=0.0, max_range=10.0),
            _unit(x=1.0, max_range=2.0),
        ),
        (
            _target(x=1.0),
            _target(x=8.0),
        ),
    )
    flexible_takes_specialist_target = future_coverability_externality(
        snapshot,
        unit_index=0,
        target_index=0,
    )
    flexible_takes_distant_target = future_coverability_externality(
        snapshot,
        unit_index=0,
        target_index=1,
    )
    assert flexible_takes_specialist_target.externality > 0.0
    assert flexible_takes_distant_target.externality == pytest.approx(0.0)


def test_cooldown_can_remove_near_deadline_coverage() -> None:
    snapshot = _snapshot(
        (_unit(x=0.0, ammo=2, cooldown_after_fire=2),),
        (_target(x=1.0, deadline=3.0), _target(x=2.0, deadline=1.0)),
    )
    certificate = future_coverability_externality(
        snapshot,
        unit_index=0,
        target_index=0,
    )
    assert certificate.externality > 0.0


def test_more_opportunities_cannot_reduce_coverability() -> None:
    threats = (
        ThreatDemand(0, 3, 1.0, (1.0, 0.0), (0.0, 0.0)),
        ThreatDemand(1, 3, 2.0, (2.0, 0.0), (0.0, 0.0)),
    )
    first = ShotOpportunity(0, 0, (0.0, 0.0), 10.0, 1.0)
    second = ShotOpportunity(1, 0, (0.0, 0.0), 10.0, 1.0)
    one = maximum_weight_coverability((first,), threats)
    two = maximum_weight_coverability((first, second), threats)
    assert two >= one


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ThreatDemand(0, 0, 1.0, (0.0, 0.0), (0.0, 0.0)),
        lambda: ThreatDemand(0, 1, -1.0, (0.0, 0.0), (0.0, 0.0)),
        lambda: ShotOpportunity(0, -1, (0.0, 0.0), 1.0, 1.0),
        lambda: ShotOpportunity(0, 0, (0.0, 0.0), 0.0, 1.0),
    ],
)
def test_invalid_certificate_inputs_are_rejected(factory) -> None:
    with pytest.raises(ValueError):
        factory()
