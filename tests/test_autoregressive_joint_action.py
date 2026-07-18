import numpy as np
import pytest
import torch
from torch.distributions import Categorical

pytest.importorskip("stable_baselines3")
pytest.importorskip("sb3_contrib")

from rein_learning.models import AutoregressiveMaskedMultiCategorical


ACTION_DIMS = (4, 4, 4)


def make_all_legal_masks(batch_size: int = 1) -> np.ndarray:
    return np.ones((batch_size, sum(ACTION_DIMS)), dtype=bool)


def test_deterministic_actions_are_conditionally_conflict_free() -> None:
    logits = torch.zeros((2, sum(ACTION_DIMS)))
    distribution = AutoregressiveMaskedMultiCategorical(
        logits,
        ACTION_DIMS,
        make_all_legal_masks(batch_size=2),
    )

    evaluation = distribution.sample(deterministic=True)

    assert evaluation.actions.tolist() == [[0, 1, 2], [0, 1, 2]]
    assert torch.isfinite(evaluation.log_prob).all()
    assert torch.isfinite(evaluation.entropy).all()


def test_generation_order_is_independent_from_environment_action_index() -> None:
    distribution = AutoregressiveMaskedMultiCategorical(
        torch.zeros((1, sum(ACTION_DIMS))),
        ACTION_DIMS,
        make_all_legal_masks(),
        unit_order=(2, 1, 0),
    )

    evaluation = distribution.sample(deterministic=True)
    masks = distribution.conditional_masks(evaluation.actions)

    assert evaluation.actions.tolist() == [[2, 1, 0]]
    assert masks[0, 2].tolist() == [True, True, True, True]
    assert masks[0, 1].tolist() == [False, True, True, True]
    assert masks[0, 0].tolist() == [False, False, True, True]


def test_unit_order_must_be_a_complete_permutation() -> None:
    with pytest.raises(ValueError, match="permutation"):
        AutoregressiveMaskedMultiCategorical(
            torch.zeros((1, sum(ACTION_DIMS))),
            ACTION_DIMS,
            make_all_legal_masks(),
            unit_order=(0, 0, 2),
        )


def test_conditional_masks_remove_targets_selected_by_prefix() -> None:
    distribution = AutoregressiveMaskedMultiCategorical(
        torch.zeros((1, sum(ACTION_DIMS))),
        ACTION_DIMS,
        make_all_legal_masks(),
    )
    actions = torch.tensor([[1, 2, 3]])

    masks = distribution.conditional_masks(actions)

    assert masks[0, 0].tolist() == [True, True, True, True]
    assert masks[0, 1].tolist() == [True, False, True, True]
    assert masks[0, 2].tolist() == [True, False, False, True]


def test_base_masks_and_prefix_masks_are_combined() -> None:
    base_masks = np.asarray(
        [[
            True, False, True, True,
            True, True, True, True,
            False, True, True, True,
        ]],
        dtype=bool,
    )
    distribution = AutoregressiveMaskedMultiCategorical(
        torch.zeros((1, sum(ACTION_DIMS))),
        ACTION_DIMS,
        base_masks,
    )

    masks = distribution.conditional_masks(torch.tensor([[2, 0, 3]]))

    assert masks[0, 0].tolist() == [True, False, True, True]
    assert masks[0, 1].tolist() == [True, True, False, True]
    assert masks[0, 2].tolist() == [False, True, False, True]


def test_joint_log_prob_matches_manual_conditional_product() -> None:
    logits = torch.tensor(
        [[
            0.1, 0.2, 0.3, 0.4,
            1.0, 0.0, -1.0, 0.5,
            -0.2, 0.7, 0.4, 0.1,
        ]],
        dtype=torch.float32,
    )
    actions = torch.tensor([[2, 0, 1]])
    distribution = AutoregressiveMaskedMultiCategorical(
        logits,
        ACTION_DIMS,
        make_all_legal_masks(),
    )

    evaluation = distribution.evaluate(actions)

    unit_0 = Categorical(logits=logits[:, 0:4])
    unit_1_logits = logits[:, 4:8].clone()
    unit_1_logits[:, 2] = -1e8
    unit_1 = Categorical(logits=unit_1_logits)
    unit_2_logits = logits[:, 8:12].clone()
    unit_2_logits[:, 0] = -1e8
    unit_2_logits[:, 2] = -1e8
    unit_2 = Categorical(logits=unit_2_logits)
    expected = (
        unit_0.log_prob(actions[:, 0])
        + unit_1.log_prob(actions[:, 1])
        + unit_2.log_prob(actions[:, 2])
    )

    assert evaluation.log_prob == pytest.approx(expected)
    assert torch.exp(evaluation.log_prob) == pytest.approx(torch.exp(expected))


def test_duplicate_or_base_illegal_action_is_rejected() -> None:
    distribution = AutoregressiveMaskedMultiCategorical(
        torch.zeros((1, sum(ACTION_DIMS))),
        ACTION_DIMS,
        make_all_legal_masks(),
    )

    with pytest.raises(ValueError, match="autoregressive mask"):
        distribution.evaluate(torch.tensor([[0, 0, 3]]))


def test_log_prob_and_entropy_propagate_finite_gradients() -> None:
    logits = torch.randn((3, sum(ACTION_DIMS)), requires_grad=True)
    distribution = AutoregressiveMaskedMultiCategorical(
        logits,
        ACTION_DIMS,
        make_all_legal_masks(batch_size=3),
    )
    evaluation = distribution.sample(deterministic=False)

    loss = -(evaluation.log_prob + 0.01 * evaluation.entropy).mean()
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0


def test_sampled_action_log_prob_is_reproduced_from_saved_prefix() -> None:
    logits = torch.randn((5, sum(ACTION_DIMS)))
    distribution = AutoregressiveMaskedMultiCategorical(
        logits,
        ACTION_DIMS,
        make_all_legal_masks(batch_size=5),
    )

    sampled = distribution.sample(deterministic=False)
    replayed = distribution.evaluate(sampled.actions)

    assert torch.equal(replayed.actions, sampled.actions)
    assert torch.allclose(replayed.log_prob, sampled.log_prob)
    assert torch.allclose(replayed.entropy, sampled.entropy)
