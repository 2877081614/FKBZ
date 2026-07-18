import numpy as np
import pytest
from gymnasium import spaces

from rein_learning.common import AirDefenseV1DiagnosticsTracker
from rein_learning.envs import (
    AirDefenseResourceAssignmentEnvV1,
    ConflictFreeJointActionCodec,
    ConflictFreeJointActionWrapper,
    get_air_defense_v1_scenario,
)


def make_wrapped_medium_env() -> ConflictFreeJointActionWrapper:
    return ConflictFreeJointActionWrapper(
        AirDefenseResourceAssignmentEnvV1(
            config=get_air_defense_v1_scenario("medium")
        )
    )


def test_default_codec_has_136_deterministic_conflict_free_actions() -> None:
    first = ConflictFreeJointActionCodec(num_units=3, num_targets=5)
    second = ConflictFreeJointActionCodec(num_units=3, num_targets=5)

    assert len(first) == 136
    assert first.joint_actions == second.joint_actions
    assert first.joint_actions[first.all_noop_index] == (5, 5, 5)
    for index, action in enumerate(first.joint_actions):
        active_targets = [target for target in action if target != 5]
        assert len(active_targets) == len(set(active_targets))
        assert first.encode(action) == index
        assert np.array_equal(first.decode(index), np.asarray(action))


def test_codec_rejects_conflicts_wrong_shapes_and_out_of_range_indices() -> None:
    codec = ConflictFreeJointActionCodec(num_units=3, num_targets=5)

    with pytest.raises(ValueError, match="multiple units"):
        codec.encode([0, 0, 5])
    with pytest.raises(ValueError, match="Expected 3"):
        codec.encode([0, 1])
    with pytest.raises(ValueError, match="outside"):
        codec.decode(136)


def test_wrapper_preserves_observations_and_builds_exact_joint_mask() -> None:
    env = make_wrapped_medium_env()
    observation, _ = env.reset(seed=0)

    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == 136
    assert observation.shape == env.observation_space.shape

    base_mask = env.base_env.action_mask().astype(bool)
    expected = np.asarray(
        [
            all(base_mask[unit_index, action] for unit_index, action in enumerate(joint))
            for joint in env.codec.joint_actions
        ],
        dtype=bool,
    )
    assert np.array_equal(env.action_masks(), expected)
    assert env.action_masks().dtype == np.bool_
    assert env.action_masks()[env.codec.all_noop_index]
    env.close()


def test_joint_mask_tracks_unavailable_units_and_inactive_targets() -> None:
    env = make_wrapped_medium_env()
    env.reset(seed=3)
    env.base_env.defense_units[0].ammo = 0
    env.base_env.targets[0].status = "intercepted"

    valid_actions = np.flatnonzero(env.action_masks())

    assert valid_actions.size > 0
    for encoded_action in valid_actions:
        decoded_action = env.codec.decode(int(encoded_action))
        assert decoded_action[0] == env.noop_action
        assert 0 not in decoded_action
    env.close()


def test_wrapper_decodes_before_step_and_records_both_action_forms() -> None:
    env = make_wrapped_medium_env()
    env.reset(seed=1)
    encoded_action = env.codec.all_noop_index

    _, _, _, _, info = env.step(np.asarray([encoded_action]))

    assert info["encoded_joint_action"] == encoded_action
    assert np.array_equal(
        info["decoded_joint_action"],
        env.codec.decode(encoded_action),
    )
    assert np.array_equal(info["joint_action"], env.codec.decode(encoded_action))
    env.close()


def test_every_masked_action_is_legal_and_episode_conflicts_remain_zero() -> None:
    env = make_wrapped_medium_env()
    env.reset(seed=2)
    tracker = AirDefenseV1DiagnosticsTracker()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        legal_actions = np.flatnonzero(env.action_masks())
        assert legal_actions.size > 0
        encoded_action = int(legal_actions[0])
        decoded_action = env.codec.decode(encoded_action)
        assert all(
            action == env.noop_action
            or env.base_env.is_unit_target_action_legal(unit_index, int(action))
            for unit_index, action in enumerate(decoded_action)
        )
        _, _, terminated, truncated, info = env.step(encoded_action)
        tracker.record_step(info)

    assert tracker.conflict_target_events == 0
    assert tracker.overkill_assignments == 0
    env.close()
