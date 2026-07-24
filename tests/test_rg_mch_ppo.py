from pathlib import Path

import torch

from rein_learning.algorithms.policy_gradient import ReliabilityGatedMCHPPO
from rein_learning.algorithms.policy_gradient.mch_ppo import (
    CounterfactualAdvantageBatch,
)
from rein_learning.envs import AirDefenseResourceAssignmentEnvV1
from rein_learning.models import (
    AirDefenseV1ObservationLayout,
    HierarchicalMaskedQCritic,
)
from rein_learning.trainers.air_defense_v1_ppo import (
    AirDefenseV1PPOConfig,
    evaluate_air_defense_v1_model,
    train_rg_mch_ppo,
)


def _write_q_checkpoint(path: Path) -> None:
    env = AirDefenseResourceAssignmentEnvV1()
    layout = AirDefenseV1ObservationLayout.infer(
        env.observation_space, env.action_space
    )
    critic = HierarchicalMaskedQCritic(layout)
    torch.save(
        {
            "state_dict": critic.state_dict(),
            "signature": critic.signature(),
            "normalization": {
                "engagement_mean": 0.0,
                "engagement_std": 1.0,
                "target_mean": 0.0,
                "target_std": 1.0,
            },
            "train_seed": 0,
        },
        path,
    )
    env.close()


def _algorithm_shell() -> ReliabilityGatedMCHPPO:
    algorithm = object.__new__(ReliabilityGatedMCHPPO)
    algorithm.engagement_residual_coef = 0.5
    algorithm.target_residual_coef = 0.5
    algorithm.residual_clip = 0.5
    algorithm.reliability_threshold = 0.5
    return algorithm


def test_ensemble_reliability_is_scale_free_and_detects_disagreement() -> None:
    agreeing = torch.tensor([[2.0, -3.0], [1.0, -1.0], [4.0, -2.0]])
    cancelling = torch.tensor([[2.0, -3.0], [-2.0, 3.0], [0.0, 0.0]])

    agreeing_score = ReliabilityGatedMCHPPO._ensemble_reliability(agreeing)
    cancelling_score = ReliabilityGatedMCHPPO._ensemble_reliability(cancelling)

    assert torch.all((0.0 <= agreeing_score) & (agreeing_score <= 1.0))
    assert torch.allclose(
        agreeing_score,
        ReliabilityGatedMCHPPO._ensemble_reliability(agreeing * 10.0),
    )
    assert torch.all(agreeing_score > cancelling_score)
    assert torch.allclose(cancelling_score, torch.zeros_like(cancelling_score))


def test_zero_reliability_degrades_to_hierarchical_gae() -> None:
    algorithm = _algorithm_shell()
    valid = torch.tensor([[True, True], [True, False], [True, True]])
    engaged = torch.tensor([[True, False], [True, False], [False, True]])
    counterfactual = CounterfactualAdvantageBatch(
        engagement=torch.tensor([[2.0, -2.0], [1.0, 0.0], [-1.0, 1.0]]),
        target=torch.tensor([[2.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]),
        engagement_reliability=torch.zeros((3, 2)),
        target_reliability=torch.zeros((3, 2)),
        engagement_support=torch.ones((3, 2)),
        target_support=engaged.float(),
        actionable=valid,
        engaged=engaged,
    )
    rollout_advantages = torch.tensor([1.0, -1.0, 0.5])

    engagement, target, diagnostics = algorithm._actor_advantages(
        rollout_advantages, counterfactual
    )
    base = algorithm._normalize_vector(rollout_advantages)[:, None].expand(3, 2)

    assert torch.allclose(
        engagement, algorithm._normalize_valid(base, valid)
    )
    assert torch.allclose(target, algorithm._normalize_valid(base, engaged))
    assert diagnostics["engagement_residual_abs"] == 0.0
    assert diagnostics["target_residual_abs"] == 0.0


def test_reliable_counterfactual_residual_is_bounded() -> None:
    algorithm = _algorithm_shell()
    valid = torch.ones((3, 2), dtype=torch.bool)
    counterfactual = CounterfactualAdvantageBatch(
        engagement=torch.full((3, 2), 100.0),
        target=torch.full((3, 2), -100.0),
        engagement_reliability=torch.ones((3, 2)),
        target_reliability=torch.ones((3, 2)),
        engagement_support=torch.ones((3, 2)),
        target_support=torch.ones((3, 2)),
        actionable=valid,
        engaged=valid,
    )

    _, _, diagnostics = algorithm._actor_advantages(
        torch.tensor([-1.0, 0.0, 1.0]), counterfactual
    )

    assert diagnostics["engagement_residual_abs"] <= 0.5
    assert diagnostics["target_residual_abs"] <= 0.5
    assert diagnostics["engagement_gate_active_rate"] == 1.0
    assert diagnostics["target_gate_active_rate"] == 1.0


def test_rg_mch_trains_saves_loads_and_evaluates(tmp_path) -> None:
    critic_path = tmp_path / "critic.pt"
    _write_q_checkpoint(critic_path)
    training = AirDefenseV1PPOConfig(
        total_timesteps=8,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        verbose=0,
        mch_q_critic_paths=(str(critic_path),),
    )
    save_path = tmp_path / "rg_mch"
    model = train_rg_mch_ppo(
        train_config=training,
        save_path=save_path,
        unit_order=(0, 1, 2),
    )

    assert model.action_generator_signature["optimizer"]["type"] == (
        "reliability_gated_mch_ppo"
    )
    assert all(
        not parameter.requires_grad
        for critic in model._q_critics
        for parameter in critic.parameters()
    )

    env = AirDefenseResourceAssignmentEnvV1()
    loaded = ReliabilityGatedMCHPPO.load(save_path, env=env)
    metrics = evaluate_air_defense_v1_model(
        loaded, episodes=1, seed=123, use_action_masks=True
    )
    assert metrics["avg_invalid_actions"] == 0.0
    assert metrics["assignment_conflict_rate"] == 0.0
    env.close()
