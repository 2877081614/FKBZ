from __future__ import annotations

import pytest
import torch

from rein_learning.common import (
    ResourceCreditComponents,
    compose_component_auxiliary_loss,
    scalar_label_is_semantically_ambiguous,
)


def test_zero_substitution_preserves_direct_cost() -> None:
    components = ResourceCreditComponents(direct_cost=2.0)
    assert components.total_substitution == 0.0
    assert components.episode_cost_delta == 2.0
    assert not scalar_label_is_semantically_ambiguous(components)


def test_same_step_and_future_substitution_are_kept_separate() -> None:
    components = ResourceCreditComponents(
        direct_cost=2.0,
        same_step_other_substitution=0.25,
        future_probe_substitution=0.5,
        future_other_substitution=0.75,
    )
    assert components.as_vector() == (2.0, 0.25, 0.5, 0.75)
    assert components.total_substitution == 1.5
    assert components.episode_cost_delta == 0.5


def test_substitution_can_mask_positive_direct_cost() -> None:
    components = ResourceCreditComponents(
        direct_cost=0.5,
        same_step_other_substitution=0.1,
        future_probe_substitution=0.2,
        future_other_substitution=0.2,
    )
    assert components.episode_cost_delta == pytest.approx(0.0)
    assert scalar_label_is_semantically_ambiguous(components)


def test_future_substitution_can_reverse_episode_delta_sign() -> None:
    components = ResourceCreditComponents(
        direct_cost=0.5,
        future_other_substitution=0.75,
    )
    assert components.episode_cost_delta == -0.25
    assert scalar_label_is_semantically_ambiguous(components)


def test_negative_direct_cost_is_rejected() -> None:
    with pytest.raises(ValueError, match="direct_cost"):
        ResourceCreditComponents(direct_cost=-0.1)


def test_zero_auxiliary_coefficient_preserves_loss_and_gradient() -> None:
    parameter = torch.tensor(0.7, requires_grad=True)
    joint_loss = (parameter - 1.0).square()
    auxiliary_loss = (parameter + 3.0).square()
    composed = compose_component_auxiliary_loss(
        joint_ppo_loss=joint_loss,
        component_auxiliary_loss=auxiliary_loss,
        coefficient=0.0,
    )
    assert composed is joint_loss
    expected_gradient = torch.autograd.grad(joint_loss, parameter, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(composed, parameter)[0]
    torch.testing.assert_close(
        actual_gradient,
        expected_gradient,
        rtol=0.0,
        atol=0.0,
    )


def test_negative_auxiliary_coefficient_is_rejected() -> None:
    with pytest.raises(ValueError, match="coefficient"):
        compose_component_auxiliary_loss(
            joint_ppo_loss=torch.tensor(1.0),
            component_auxiliary_loss=torch.tensor(2.0),
            coefficient=-0.1,
        )
