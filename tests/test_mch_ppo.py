from pathlib import Path

import torch

from rein_learning.algorithms.policy_gradient import (
    MaskedCounterfactualHierarchicalPPO,
)
from rein_learning.envs import AirDefenseResourceAssignmentEnvV1
from rein_learning.models import (
    AirDefenseV1ObservationLayout,
    HierarchicalMaskedQCritic,
)
from rein_learning.trainers.air_defense_v1_ppo import (
    AirDefenseV1PPOConfig,
    evaluate_air_defense_v1_model,
    train_mch_ppo,
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


def test_mch_ppo_trains_with_frozen_critic_and_loads_for_inference(tmp_path) -> None:
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
    save_path = tmp_path / "mch_ppo"
    model = train_mch_ppo(
        train_config=training,
        save_path=save_path,
        unit_order=(0, 1, 2),
    )

    assert all(
        not parameter.requires_grad
        for critic in model._q_critics
        for parameter in critic.parameters()
    )
    assert model.action_generator_signature["optimizer"]["type"] == (
        "masked_counterfactual_hierarchical_ppo"
    )

    env = AirDefenseResourceAssignmentEnvV1()
    loaded = MaskedCounterfactualHierarchicalPPO.load(save_path, env=env)
    metrics = evaluate_air_defense_v1_model(
        loaded,
        episodes=1,
        seed=123,
        use_action_masks=True,
    )
    assert metrics["avg_invalid_actions"] == 0.0
    assert metrics["assignment_conflict_rate"] == 0.0
    env.close()


def test_mch_valid_advantage_normalization_ignores_masked_entries() -> None:
    values = torch.tensor([[1.0, 9.0], [3.0, 9.0]])
    valid = torch.tensor([[True, False], [True, False]])

    normalized = MaskedCounterfactualHierarchicalPPO._normalize_valid(
        values, valid
    )

    assert torch.equal(normalized[:, 1], torch.zeros(2))
    assert torch.allclose(normalized[:, 0], torch.tensor([-1.0, 1.0]))
