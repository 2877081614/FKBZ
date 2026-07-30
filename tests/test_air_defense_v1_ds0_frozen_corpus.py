from __future__ import annotations

import numpy as np

from rein_learning.common import PolicyProbeCorpus
from scripts.rebuild_air_defense_v1_ds0_frozen_corpus import (
    PROBE_DIR,
    branch_high_threat_unassigned,
    branch_prefix_denied,
    deterministic_completion,
    engagement_extreme_direction,
    independent_mask_from_observation,
)


def test_independent_observation_mask_matches_frozen_probe_masks() -> None:
    corpus = PolicyProbeCorpus.load(PROBE_DIR, verify_hash=True)

    for index in range(corpus.size):
        expected = corpus.action_masks[index].reshape(3, 6)
        observed = independent_mask_from_observation(corpus.observations[index])
        assert np.array_equal(observed, expected)


def test_deterministic_completion_respects_forced_prefix_and_target_occupancy() -> None:
    base_mask = np.ones((3, 3), dtype=bool)
    logits = np.asarray(
        [
            [9.0, 1.0, 0.0],
            [8.0, 7.0, 0.0],
            [6.0, 5.0, 0.0],
        ]
    )

    joint = deterministic_completion(
        logits,
        base_mask,
        fixed_ordered_actions=(0,),
    )

    assert joint.tolist() == [0, 1, 2]


def test_branch_outcomes_use_only_masks_threats_and_completed_actions() -> None:
    base_mask = np.ones((3, 3), dtype=bool)
    threats = np.asarray([0.9, 0.6])
    joint = np.asarray([2, 2, 2])

    assert branch_high_threat_unassigned(
        joint_action=joint,
        base_mask=base_mask,
        threats=threats,
        current_position=0,
    )
    assert branch_prefix_denied(
        action=0,
        base_mask=base_mask,
        current_position=0,
    )
    assert not branch_prefix_denied(
        action=2,
        base_mask=base_mask,
        current_position=0,
    )


def test_engagement_extreme_direction_requires_an_extreme_boundary() -> None:
    assert engagement_extreme_direction(1, 2, 3) == 0
    assert engagement_extreme_direction(1, 3, 3) == 1
    assert engagement_extreme_direction(3, 2, 3) == -1
    assert engagement_extreme_direction(0, 2, 3) == 1
    assert engagement_extreme_direction(2, 0, 3) == -1
    assert engagement_extreme_direction(0, 0, 3) == 0

