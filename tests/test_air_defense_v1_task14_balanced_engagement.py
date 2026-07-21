from __future__ import annotations

import numpy as np
import pytest
import torch

from rein_learning.common import (
    balanced_engagement_loss,
    engagement_criticality_features,
    select_diverse_critical_snapshots,
)
from rein_learning.envs import AirDefenseResourceAssignmentEnvV1


def test_criticality_uses_only_legal_unit_target_relations() -> None:
    env = AirDefenseResourceAssignmentEnvV1()
    legal_pair = None
    for seed in range(20):
        env.reset(seed=seed)
        actual_masks = env.action_masks().reshape(
            env.num_defense_units, env.num_targets + 1
        )
        pairs = np.argwhere(actual_masks[:, : env.num_targets])
        if len(pairs):
            legal_pair = tuple(int(value) for value in pairs[0])
            break
    assert legal_pair is not None
    masks = np.zeros((env.num_defense_units, env.num_targets + 1), dtype=bool)
    masks[:, env.num_targets] = True
    empty = engagement_criticality_features(env, masks)
    assert empty["criticality_score"] == 0.0
    assert empty["legal_relation_count"] == 0

    masks[legal_pair] = True
    selected = engagement_criticality_features(env, masks)
    assert selected["criticality_score"] > 0.0
    assert selected["legal_relation_count"] == 1
    assert selected["max_threat"] == pytest.approx(env.targets[legal_pair[1]].threat)
    env.close()


def test_critical_snapshot_selection_is_deterministic_and_diverse() -> None:
    records = [
        {
            "state_id": f"state-{index}",
            "criticality_score": float(20 - index),
            "episode_index": index // 5,
            "step_index": (index % 5) * 3,
        }
        for index in range(20)
    ]
    first = select_diverse_critical_snapshots(records, 10, seed=9)
    second = select_diverse_critical_snapshots(records, 10, seed=9)
    assert [row["state_id"] for row in first] == [row["state_id"] for row in second]
    assert len({row["state_id"] for row in first}) == 10
    assert "state-0" in {row["state_id"] for row in first}
    assert sum(float(row["criticality_score"]) >= 10.0 for row in first) >= 8


def test_balanced_loss_ignores_ambiguous_and_balances_classes() -> None:
    predictions = torch.zeros((6, 2), requires_grad=True)
    labels = torch.as_tensor([1, 0, 0, 0, 0, -1])
    loss, parts = balanced_engagement_loss(predictions, labels)
    assert float(parts["positive_count"]) == 1.0
    assert float(parts["negative_count"]) == 4.0
    assert float(loss.detach()) == pytest.approx(np.log(2.0), rel=1e-6)
    loss.backward()
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


def test_margin_rewards_correctly_separated_logits() -> None:
    labels = torch.as_tensor([1, 0])
    wrong = torch.as_tensor([[0.0, -1.0], [-1.0, 0.0]])
    correct = torch.as_tensor([[0.0, 2.0], [2.0, 0.0]])
    wrong_loss, _ = balanced_engagement_loss(
        wrong, labels, margin_weight=0.5
    )
    correct_loss, _ = balanced_engagement_loss(
        correct, labels, margin_weight=0.5
    )
    assert correct_loss < wrong_loss


def test_balanced_loss_requires_both_reliable_classes() -> None:
    with pytest.raises(ValueError, match="Both engagement classes"):
        balanced_engagement_loss(
            torch.zeros((3, 2)), torch.as_tensor([0, 0, -1])
        )
