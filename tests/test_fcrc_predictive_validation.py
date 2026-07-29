from __future__ import annotations

import numpy as np
import pytest

from rein_learning.common import (
    BPCEAuditContext,
    FCRCBranchTrace,
    FCRCPredictiveValidationConfig,
    candidate_harm,
    mean_interval,
    select_fcrc_candidate_pair,
)
from rein_learning.envs.air_defense_v1 import AirDefenseV1StateSnapshot
from rein_learning.envs.air_defense_v1.entities import (
    DefenseUnitV1State,
    ProtectedZoneState,
    TargetV1State,
)


def _unit(*, x: float, max_range: float) -> DefenseUnitV1State:
    return DefenseUnitV1State(
        resource_type="missile",
        position=np.asarray([x, 0.0]),
        ammo=1,
        max_ammo=1,
        max_range=max_range,
        base_hit_probability=1.0,
        cost=2.0,
        cooldown=0,
        cooldown_after_fire=1,
        energy=1.0,
        max_energy=1.0,
    )


def _target(*, x: float, weight: float = 1.0) -> TargetV1State:
    return TargetV1State(
        position=np.asarray([x, 0.0]),
        velocity=np.asarray([0.0, 0.0]),
        speed=0.0,
        threat=weight,
        target_zone=0,
        payload=1.0,
        evasion=0.0,
        target_class="uav",
        time_to_impact=3.0,
    )


def _snapshot() -> AirDefenseV1StateSnapshot:
    zone = ProtectedZoneState(
        position=np.asarray([0.0, 0.0]),
        radius=1.0,
        value=1.0,
        priority=1.0,
        zone_type="command",
    )
    return AirDefenseV1StateSnapshot(
        current_step=0,
        protected_zones=(zone,),
        defense_units=(
            _unit(x=0.0, max_range=10.0),
            _unit(x=1.0, max_range=2.0),
        ),
        targets=(
            _target(x=1.0),
            _target(x=8.0),
            _target(x=2.0, weight=2.0),
        ),
        np_random_state={},
        hit_random_tape=None,
    )


def _context() -> BPCEAuditContext:
    snapshot = _snapshot()
    return BPCEAuditContext(
        context_id="n3_manual",
        scenario="manual",
        policy_seed=17,
        slot="manual",
        episode_index=0,
        environment_seed=1,
        environment_step=0,
        unit_index=0,
        prefix_actions=(),
        original_action=(0, 2),
        observation_hash="manual",
        observation=np.zeros(1, dtype=np.float32),
        action_mask=np.ones(8, dtype=np.int8),
        snapshot=snapshot,
        engage_probability=1.0,
        engagement_margin=1.0,
        legal_targets=(0, 1),
        target_probabilities=(0.5, 0.5),
        argmax_target=0,
        safety_score=1.0,
        resource_score=1.0,
    )


def test_candidate_pair_is_deterministic_and_has_positive_spread() -> None:
    context = _context()
    low, high = select_fcrc_candidate_pair(context)
    repeated_low, repeated_high = select_fcrc_candidate_pair(context)
    assert (low.target_index, high.target_index) == (
        repeated_low.target_index,
        repeated_high.target_index,
    )
    assert low.target_index != high.target_index
    assert high.externality > low.externality


def test_candidate_harm_excludes_current_target() -> None:
    snapshot = _snapshot()
    noop = FCRCBranchTrace(
        target_statuses=("leaked", "intercepted", "leaked"),
        leaked_damage=(9.0, 0.0, 2.0),
        transitions=3,
    )
    engage = FCRCBranchTrace(
        target_statuses=("intercepted", "leaked", "leaked"),
        leaked_damage=(0.0, 4.0, 3.0),
        transitions=3,
    )
    intercept_harm, damage_harm = candidate_harm(
        engage=engage,
        noop=noop,
        snapshot=snapshot,
        target_index=0,
    )
    assert intercept_harm == pytest.approx(1.0)
    assert damage_harm == pytest.approx(5.0)


def test_mean_interval_and_config_validate_inputs() -> None:
    summary = mean_interval([1.0, 1.0, 1.0, 1.0])
    assert summary["mean"] == pytest.approx(1.0)
    assert summary["ci_lower"] == pytest.approx(1.0)
    assert summary["ci_upper"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        mean_interval([1.0])
    with pytest.raises(ValueError):
        FCRCPredictiveValidationConfig(repeats=1)
