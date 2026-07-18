import numpy as np
import pytest
import torch

from rein_learning.algorithms.policy_gradient import (
    FactorizedEngagementActorCriticPolicy,
    FactorizedEngagementMaskablePPO,
    RoleConditionedAutoregressiveActorCriticPolicy,
)
from rein_learning.envs import AirDefenseResourceAssignmentEnvV1
from rein_learning.models import FactorizedEngagementAutoregressiveDistribution
from rein_learning.trainers.air_defense_v1_ppo import (
    AirDefenseV1PPOConfig,
    evaluate_air_defense_v1_model,
    train_factorized_engagement_autoregressive_ppo,
)


def test_factorized_probabilities_log_prob_and_entropy_match_manual_formula() -> None:
    target_logits = torch.tensor([[[np.log(2.0), 0.0], [0.0, 0.0]]])
    engage_logits = torch.tensor([[-np.log(3.0), 0.0]])
    masks = torch.ones((1, 2, 3), dtype=torch.bool)
    distribution = FactorizedEngagementAutoregressiveDistribution(
        target_logits, engage_logits, (3, 3), masks
    )

    evaluation = distribution.evaluate(torch.tensor([[0, 1]]))

    first_probabilities = torch.tensor([1.0 / 6.0, 1.0 / 12.0, 3.0 / 4.0])
    second_probabilities = torch.tensor([0.0, 1.0 / 2.0, 1.0 / 2.0])
    expected_log_prob = torch.log(first_probabilities[0]) + torch.log(
        second_probabilities[1]
    )
    expected_entropy = -torch.sum(
        first_probabilities * torch.log(first_probabilities)
    ) - torch.sum(
        second_probabilities[1:] * torch.log(second_probabilities[1:])
    )
    assert evaluation.log_prob.item() == pytest.approx(expected_log_prob.item())
    assert evaluation.entropy.item() == pytest.approx(expected_entropy.item())


def test_factorized_distribution_forces_noop_without_legal_target() -> None:
    distribution = FactorizedEngagementAutoregressiveDistribution(
        torch.zeros((2, 1, 2)),
        torch.full((2, 1), 20.0),
        (3,),
        torch.tensor([[False, False, True], [False, False, True]]),
    )

    deterministic = distribution.sample(deterministic=True)
    stochastic = distribution.sample(deterministic=False)

    assert torch.equal(deterministic.actions, torch.tensor([[2], [2]]))
    assert torch.equal(stochastic.actions, torch.tensor([[2], [2]]))
    assert torch.equal(deterministic.log_prob, torch.zeros(2))
    assert torch.equal(deterministic.entropy, torch.zeros(2))


def test_factorized_deterministic_action_uses_engagement_then_target_argmax() -> None:
    distribution = FactorizedEngagementAutoregressiveDistribution(
        torch.zeros((1, 1, 2)),
        torch.tensor([[np.log(1.5)]]),
        (3,),
        torch.ones((1, 1, 3), dtype=torch.bool),
    )

    evaluation = distribution.sample(deterministic=True)

    # Final probabilities are [0.3, 0.3, 0.4], but aggregate engagement is 0.6.
    assert evaluation.actions.item() == 0


def test_factorized_distribution_reconstructs_prefix_and_rejects_duplicate() -> None:
    distribution = FactorizedEngagementAutoregressiveDistribution(
        torch.zeros((1, 2, 2)),
        torch.full((1, 2), 10.0),
        (3, 3),
        torch.ones((1, 2, 3), dtype=torch.bool),
    )

    valid = distribution.evaluate(torch.tensor([[0, 1]]))

    assert torch.isfinite(valid.log_prob).all()
    with pytest.raises(ValueError, match="illegal"):
        distribution.evaluate(torch.tensor([[0, 0]]))


def test_factorized_actor_matches_role_capacity_and_critic() -> None:
    env = AirDefenseResourceAssignmentEnvV1()
    schedule = lambda _: 3e-4
    baseline = RoleConditionedAutoregressiveActorCriticPolicy(
        env.observation_space,
        env.action_space,
        schedule,
        net_arch=[128, 128],
    )
    candidate = FactorizedEngagementActorCriticPolicy(
        env.observation_space,
        env.action_space,
        schedule,
        net_arch=[128, 128],
    )

    assert candidate.parameter_counts()["actor_parameters"] == baseline.parameter_counts()[
        "actor_parameters"
    ]
    assert candidate.parameter_counts()["critic_parameters"] == baseline.parameter_counts()[
        "critic_parameters"
    ]
    assert torch.equal(
        candidate.action_net.engage_output.bias,
        torch.zeros_like(candidate.action_net.engage_output.bias),
    )
    env.close()


def test_factorized_training_save_load_and_evaluate(tmp_path) -> None:
    training = AirDefenseV1PPOConfig(
        total_timesteps=8,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        verbose=0,
    )
    save_path = tmp_path / "factorized"
    model = train_factorized_engagement_autoregressive_ppo(
        train_config=training,
        save_path=save_path,
        unit_order=(0, 1, 2),
    )
    env = AirDefenseResourceAssignmentEnvV1()
    loaded = FactorizedEngagementMaskablePPO.load(save_path, env=env)

    assert loaded.action_generator_signature == model.action_generator_signature
    assert loaded.action_generator_signature["probability_schema"]["entropy"] == (
        "exact_final_discrete_distribution"
    )
    metrics = evaluate_air_defense_v1_model(
        loaded, episodes=1, seed=123, use_action_masks=True
    )
    assert metrics["avg_invalid_actions"] == 0.0
    assert metrics["assignment_conflict_rate"] == 0.0
    assert 0.0 <= metrics["actionable_engagement_rate"] <= 1.0
    env.close()
