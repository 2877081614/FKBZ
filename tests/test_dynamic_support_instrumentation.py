from __future__ import annotations

import numpy as np
import pytest

from rein_learning.common.dynamic_support_instrumentation import (
    PolicyProbeSnapshot,
    build_frozen_dynamic_support_probe,
    compute_dynamic_support_update_metrics,
)
from rein_learning.common.policy_probe import PolicyProbeCorpus


def _tiny_corpus() -> PolicyProbeCorpus:
    observations = np.zeros((1, 97), dtype=np.float32)
    target_start = 2 * 7
    observations[0, target_start + 6] = 0.9
    observations[0, target_start + 13] = 1.0
    observations[0, target_start + 15 + 6] = 0.4
    observations[0, target_start + 15 + 13] = 1.0
    return PolicyProbeCorpus(
        observations=observations,
        action_masks=np.ones((1, 9), dtype=np.bool_),
        scenarios=np.asarray(["heterogeneity_pressure"]),
        sources=np.asarray(["frozen_fixture"]),
        phases=np.asarray(["initial"]),
        environment_seeds=np.asarray([1], dtype=np.int64),
        episode_indices=np.asarray([0], dtype=np.int64),
        step_indices=np.asarray([0], dtype=np.int64),
    )


def _snapshot(
    grid,
    *,
    favor_noop: bool,
) -> PolicyProbeSnapshot:
    probabilities = np.zeros(
        (grid.size, grid.num_actions), dtype=np.float64
    )
    actions = np.empty(grid.size, dtype=np.int16)
    engage = np.empty(grid.size, dtype=np.float64)
    entropies = np.empty(grid.size, dtype=np.float64)
    base_mask = grid.corpus.action_masks[0].reshape(
        grid.num_units, grid.num_actions
    )
    for index in range(grid.size):
        position = int(grid.unit_positions[index])
        current_unit = grid.unit_order[position]
        conditional = base_mask[current_unit].copy()
        for action in grid.prefixes[index]:
            if 0 <= int(action) < grid.noop_action:
                conditional[int(action)] = False
        legal_targets = np.flatnonzero(conditional[: grid.noop_action])
        if len(legal_targets) == 0:
            probabilities[index, grid.noop_action] = 1.0
            actions[index] = grid.noop_action
        elif favor_noop:
            probabilities[index, legal_targets] = 0.05 / len(legal_targets)
            probabilities[index, grid.noop_action] = 0.95
            actions[index] = grid.noop_action
        else:
            probabilities[index, legal_targets] = 0.9 / len(legal_targets)
            probabilities[index, grid.noop_action] = 0.1
            actions[index] = int(legal_targets[0])
        engage[index] = probabilities[index, : grid.noop_action].sum()
        positive = probabilities[index] > 0
        entropies[index] = -np.sum(
            probabilities[index, positive]
            * np.log(probabilities[index, positive])
        )
    state_id = grid.state_ids[0]
    joint = (
        np.asarray([[grid.noop_action] * grid.num_units], dtype=np.int16)
        if favor_noop
        else np.asarray([[0, 1, grid.noop_action]], dtype=np.int16)
    )
    return PolicyProbeSnapshot(
        grid_sha256=grid.content_sha256(),
        context_ids=grid.context_ids.copy(),
        context_probabilities=probabilities,
        context_actions=actions,
        context_engage_probabilities=engage,
        context_entropies=entropies,
        state_ids=np.asarray([state_id]),
        joint_actions=joint,
    )


def test_frozen_prefix_grid_is_deterministic_unique_and_covers_all_positions() -> None:
    corpus = _tiny_corpus()
    first = build_frozen_dynamic_support_probe(corpus)
    second = build_frozen_dynamic_support_probe(corpus)

    assert first.size == 11
    assert int(first.eligible_ds.sum()) == 4
    assert set(first.unit_positions.tolist()) == {0, 1, 2}
    assert len(set(first.context_ids.tolist())) == first.size
    assert first.content_sha256() == second.content_sha256()
    assert np.array_equal(first.context_ids, second.context_ids)
    assert first.high_threat_reachable.any()
    assert (~first.high_threat_reachable).any()


def test_identity_update_has_zero_churn_and_reconstructs_required_fields() -> None:
    grid = build_frozen_dynamic_support_probe(_tiny_corpus())
    snapshot = _snapshot(grid, favor_noop=False)

    metrics = compute_dynamic_support_update_metrics(
        grid,
        snapshot,
        snapshot,
        update_id=7,
        approx_kl=0.001,
        clip_fraction=0.02,
        entropy=0.5,
    )

    required = {
        "update_id",
        "ppo_update",
        "approx_kl",
        "clip_fraction",
        "entropy",
        "engagement_margin_crossings",
        "engage_to_noop_flips",
        "noop_to_engage_flips",
        "joint_argmax_flips",
        "unweighted_prefix_flip_rate",
        "ds_weighted_flip_mass",
        "suffix_count_change",
        "probe_all_noop_rate",
        "probe_high_engagement_rate",
        "probe_high_threat_unassigned_rate",
    }
    assert required <= set(metrics)
    assert metrics["update_id"] == 7
    assert metrics["ppo_update"] == 7
    assert metrics["unweighted_prefix_flip_rate"] == 0.0
    assert metrics["ds_weighted_flip_mass"] == 0.0
    assert metrics["ds_policy_distance"] == 0.0
    assert metrics["joint_argmax_flips"] == 0


def test_ds_weighted_flip_is_context_level_and_nonzero_for_structural_change() -> None:
    grid = build_frozen_dynamic_support_probe(_tiny_corpus())
    old = _snapshot(grid, favor_noop=False)
    new = _snapshot(grid, favor_noop=True)

    first = compute_dynamic_support_update_metrics(
        grid,
        old,
        new,
        update_id=1,
        approx_kl=0.01,
        clip_fraction=0.1,
        entropy=0.4,
    )
    second = compute_dynamic_support_update_metrics(
        grid,
        old,
        new,
        update_id=1,
        approx_kl=0.01,
        clip_fraction=0.1,
        entropy=0.4,
    )

    assert first == second
    assert first["unweighted_prefix_flip_rate"] == 1.0
    assert 0.0 < first["ds_weighted_flip_mass"] <= 1.0
    assert 0.0 < first["ds_policy_distance"] <= 1.0
    assert first["engagement_margin_crossings"] > 0
    assert first["engage_to_noop_flips"] > 0
    assert first["joint_argmax_flips"] == 1


def test_snapshot_grid_mismatch_is_rejected() -> None:
    grid = build_frozen_dynamic_support_probe(_tiny_corpus())
    snapshot = _snapshot(grid, favor_noop=False)
    bad = PolicyProbeSnapshot(
        grid_sha256="wrong",
        context_ids=snapshot.context_ids,
        context_probabilities=snapshot.context_probabilities,
        context_actions=snapshot.context_actions,
        context_engage_probabilities=snapshot.context_engage_probabilities,
        context_entropies=snapshot.context_entropies,
        state_ids=snapshot.state_ids,
        joint_actions=snapshot.joint_actions,
    )
    with pytest.raises(ValueError, match="frozen grid"):
        compute_dynamic_support_update_metrics(
            grid,
            snapshot,
            bad,
            update_id=1,
            approx_kl=0.0,
            clip_fraction=0.0,
            entropy=0.0,
        )
