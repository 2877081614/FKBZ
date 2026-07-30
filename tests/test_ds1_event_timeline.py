from __future__ import annotations

import pytest

from rein_learning.common.ds1_event_timeline import (
    evaluation_seed_bank_sha256,
    event_window_indices,
    finalize_event_timeline,
    formal_event_metrics,
)


SEED_HASH = evaluation_seed_bank_sha256()


def _row(index: int, collapsed: bool = False) -> dict:
    row = formal_event_metrics(
        rollout_update_index=index,
        sb3_n_updates=index * 10,
        num_timesteps=index * 256,
        policy_seed=8,
        evaluation_seed_bank_hash=SEED_HASH,
        evaluation_episodes=50,
        all_noop_episodes=49 if collapsed else 0,
        actionable_decisions=1_000,
        actionable_engagements=0 if collapsed else 500,
    )
    return row


def _timeline(states: list[bool]) -> dict:
    return finalize_event_timeline(
        [_row(index, state) for index, state in enumerate(states)],
        expected_seed_bank_sha256=SEED_HASH,
    )


def test_all_noop_49_of_50_triggers_but_48_does_not() -> None:
    triggered = formal_event_metrics(
        rollout_update_index=1,
        sb3_n_updates=10,
        num_timesteps=256,
        policy_seed=8,
        evaluation_seed_bank_hash=SEED_HASH,
        evaluation_episodes=50,
        all_noop_episodes=49,
        actionable_decisions=100,
        actionable_engagements=10,
    )
    not_triggered = formal_event_metrics(
        rollout_update_index=1,
        sb3_n_updates=10,
        num_timesteps=256,
        policy_seed=8,
        evaluation_seed_bank_hash=SEED_HASH,
        evaluation_episodes=50,
        all_noop_episodes=48,
        actionable_decisions=100,
        actionable_engagements=10,
    )
    assert triggered["collapse_event_state"]
    assert not not_triggered["collapse_event_state"]


def test_actionable_rate_0_009_triggers_but_0_01_does_not() -> None:
    triggered = formal_event_metrics(
        rollout_update_index=1,
        sb3_n_updates=10,
        num_timesteps=256,
        policy_seed=8,
        evaluation_seed_bank_hash=SEED_HASH,
        evaluation_episodes=50,
        all_noop_episodes=0,
        actionable_decisions=1_000,
        actionable_engagements=9,
    )
    boundary = formal_event_metrics(
        rollout_update_index=1,
        sb3_n_updates=10,
        num_timesteps=256,
        policy_seed=8,
        evaluation_seed_bank_hash=SEED_HASH,
        evaluation_episodes=50,
        all_noop_episodes=0,
        actionable_decisions=1_000,
        actionable_engagements=10,
    )
    assert triggered["collapse_event_state"]
    assert not boundary["collapse_event_state"]


def test_initially_collapsed_seed_has_no_event_bearing_precursor() -> None:
    result = _timeline([True, True, False, True, True, True])
    assert result["initially_collapsed"]
    assert result["first_collapse_onset_index"] is None
    assert not result["event_bearing_seed"]
    assert all(
        not row["predictor_row_eligible"] for row in result["rows"]
    )


def test_only_first_recovered_then_recollapsed_onset_is_selected() -> None:
    result = _timeline(
        [False, False, True, False, True, False, False, False]
    )
    assert result["first_collapse_onset_index"] == 2
    onset_rows = [
        row["rollout_update_index"]
        for row in result["rows"]
        if row["collapse_event_onset"]
    ]
    assert onset_rows == [2]


def test_forward_labels_use_t_plus_1_through_t_plus_3() -> None:
    result = _timeline(
        [False, False, False, False, False, True, True, True, True]
    )
    by_index = {
        row["rollout_update_index"]: row for row in result["rows"]
    }
    assert by_index[1]["event_within_3_updates"] is False
    assert by_index[2]["event_within_3_updates"] is True
    assert by_index[3]["event_within_3_updates"] is True
    assert by_index[4]["event_within_3_updates"] is True


def test_concurrent_post_event_and_tail_rows_are_excluded() -> None:
    event = _timeline(
        [False, False, False, False, True, True, True, True]
    )
    by_index = {
        row["rollout_update_index"]: row for row in event["rows"]
    }
    assert by_index[4]["exclusion_reason"] == "concurrent_event"
    assert by_index[5]["exclusion_reason"] == "post_event"
    no_event = _timeline([False] * 6)
    no_event_rows = {
        row["rollout_update_index"]: row for row in no_event["rows"]
    }
    assert no_event_rows[3]["exclusion_reason"] == "right_censored_tail"
    assert no_event_rows[5]["event_within_3_updates"] is None


def test_sb3_n_updates_jumps_by_ten_but_windows_use_rollout_rows() -> None:
    result = _timeline(
        [False, False, False, False, True, True, True, True]
    )
    assert [row["sb3_n_updates"] for row in result["rows"]] == list(
        range(0, 80, 10)
    )
    assert result["first_collapse_onset_index"] == 4


def test_baseline_window_requires_six_predictor_updates() -> None:
    assert not event_window_indices(6)["eligible"]
    eligible = event_window_indices(7)
    assert eligible["baseline"] == [1, 2, 3]
    assert eligible["pre_event"] == [4, 5, 6]


def test_formal_row_rejects_any_episode_count_other_than_50() -> None:
    with pytest.raises(ValueError, match="exactly 50"):
        formal_event_metrics(
            rollout_update_index=1,
            sb3_n_updates=10,
            num_timesteps=256,
            policy_seed=8,
            evaluation_seed_bank_hash=SEED_HASH,
            evaluation_episodes=49,
            all_noop_episodes=49,
            actionable_decisions=100,
            actionable_engagements=10,
        )


def test_seed_bank_hash_change_rejects_formal_merge() -> None:
    rows = [_row(0), _row(1)]
    rows[1]["evaluation_seed_bank_sha256"] = "changed"
    with pytest.raises(ValueError, match="hash changed"):
        finalize_event_timeline(
            rows,
            expected_seed_bank_sha256=SEED_HASH,
        )


def test_rollout_index_must_be_contiguous_even_when_sb3_counter_is_raw() -> None:
    rows = [_row(0), _row(2)]
    with pytest.raises(ValueError, match="consecutive"):
        finalize_event_timeline(
            rows,
            expected_seed_bank_sha256=SEED_HASH,
        )
