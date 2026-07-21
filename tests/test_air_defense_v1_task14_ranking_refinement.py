import numpy as np
import pytest
import torch

from rein_learning.common import (
    action_group_ids,
    build_pairwise_training_data,
    center_by_group,
    integer_group_codes,
    q_critic_training_loss,
    validation_difference_score,
)
from scripts.run_air_defense_v1_task14_ranking_refinement import _combine_datasets


def _minimal_dataset(
    state_ids: list[str], splits: list[str], rollout_count: int
) -> dict[str, np.ndarray]:
    count = len(state_ids)
    return {
        "observations": np.zeros((count, 2), dtype=np.float32),
        "unit_indices": np.zeros(count, dtype=np.int64),
        "candidate_actions": np.zeros(count, dtype=np.int64),
        "prefix_occupancy": np.zeros((count, 1), dtype=np.float32),
        "legal_action_masks": np.ones((count, 2), dtype=np.float32),
        "q_labels": np.zeros(count, dtype=np.float32),
        "q_standard_errors": np.ones(count, dtype=np.float32),
        "one_step_rewards": np.zeros(count, dtype=np.float32),
        "frozen_values": np.zeros(count, dtype=np.float32),
        "conditional_target_probabilities": np.zeros(count, dtype=np.float32),
        "state_ids": np.asarray(state_ids),
        "scenarios": np.full(count, "medium"),
        "source_seeds": np.full(count, 8, dtype=np.int64),
        "splits": np.asarray(splits),
        "return_samples": np.zeros((count, rollout_count), dtype=np.float32),
        "generation_seconds": np.asarray(1.0),
    }


def test_pairwise_training_data_never_crosses_action_groups() -> None:
    labels = np.asarray([0.0, 2.0, 1.0, 4.0])
    groups = np.asarray(["a", "a", "b", "b"])
    samples = np.asarray(
        [
            [0.0, 1.0, 0.0, 1.0],
            [2.0, 3.0, 2.0, 3.0],
            [1.0, 2.0, 1.0, 2.0],
            [4.0, 5.0, 4.0, 5.0],
        ]
    )

    pairs = build_pairwise_training_data(labels, groups, samples)

    assert pairs["left"].tolist() == [0, 2]
    assert pairs["right"].tolist() == [1, 3]
    assert np.all(pairs["reliability"] == pytest.approx(4.0))


def test_pairwise_training_data_ignores_padded_rollouts() -> None:
    samples = np.asarray(
        [[0.0, 1.0, np.nan], [2.0, 3.0, np.nan]], dtype=np.float64
    )
    pairs = build_pairwise_training_data([0.5, 2.5], ["a", "a"], samples)

    assert len(pairs["left"]) == 1
    assert pairs["standard_error"][0] == pytest.approx(0.0)


def test_group_center_and_difference_loss_are_zero_for_exact_prediction() -> None:
    labels = torch.tensor([1.0, 3.0, 2.0, 5.0])
    groups = torch.tensor([0, 0, 1, 1])
    centered = center_by_group(labels, groups)
    loss, components = q_critic_training_loss(
        labels.clone(),
        labels,
        groups,
        pair_left=torch.tensor([0, 2]),
        pair_right=torch.tensor([1, 3]),
        pair_weights=torch.ones(2),
        centered_weight=1.0,
        pairwise_weight=0.5,
    )

    assert centered.tolist() == pytest.approx([-1.0, 1.0, -1.5, 1.5])
    assert loss.item() == pytest.approx(0.0)
    assert all(component.item() == pytest.approx(0.0) for component in components.values())


def test_validation_score_combines_absolute_and_centered_mae() -> None:
    score = validation_difference_score(
        labels=[0.0, 2.0, 10.0, 12.0],
        predictions=[1.0, 3.0, 8.0, 10.0],
        group_ids=["a", "a", "b", "b"],
        scale=2.0,
    )

    assert score["absolute_mae"] == pytest.approx(0.75)
    assert score["centered_mae"] == pytest.approx(0.0)
    assert score["score"] == pytest.approx(0.75)


def test_action_group_ids_and_codes_are_stable() -> None:
    groups = action_group_ids(["s0", "s0", "s1"], [0, 1, 0])
    codes = integer_group_codes(groups)

    assert groups.tolist() == ["s0/unit0", "s0/unit1", "s1/unit0"]
    assert len(np.unique(codes)) == 3


def test_combined_dataset_excludes_old_test_and_pads_training_returns() -> None:
    base = _minimal_dataset(
        ["old-train", "old-validation", "old-test"],
        ["train", "validation", "test"],
        2,
    )
    fresh = _minimal_dataset(["task14r/new-test"], ["test"], 4)

    combined, audit = _combine_datasets(base, fresh)

    assert combined["state_ids"].tolist() == [
        "old-train",
        "old-validation",
        "task14r/new-test",
    ]
    assert combined["return_samples"].shape == (3, 4)
    assert np.all(np.isnan(combined["return_samples"][0, 2:]))
    assert audit["old_test_rows_excluded"] == 1
    assert audit["old_test_rows_in_combined"] == 0
    assert audit["fresh_state_id_overlap"] == 0


def test_combined_dataset_rejects_state_id_overlap() -> None:
    base = _minimal_dataset(["same"], ["train"], 2)
    fresh = _minimal_dataset(["same"], ["test"], 2)

    with pytest.raises(RuntimeError, match="overlap"):
        _combine_datasets(base, fresh)
