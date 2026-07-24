from pathlib import Path

import numpy as np
import torch

from rein_learning.algorithms.policy_gradient import SupportAnchoredRGMCHPPO
from rein_learning.algorithms.policy_gradient.mch_ppo import (
    CounterfactualAdvantageBatch,
)
from rein_learning.common import MaskedContextSupportIndex
from rein_learning.envs import AirDefenseResourceAssignmentEnvV1
from rein_learning.models import (
    AirDefenseV1ObservationLayout,
    HierarchicalMaskedQCritic,
)
from rein_learning.trainers.air_defense_v1_ppo import (
    AirDefenseV1PPOConfig,
    evaluate_air_defense_v1_model,
    train_sa_rg_mch_ppo,
)


def _write_small_support_dataset(path: Path) -> None:
    np.savez(
        path,
        observations=np.asarray(
            [[0.0, 0.0], [0.2, 0.1], [0.4, 0.2], [100.0, 100.0]],
            dtype=np.float32,
        ),
        unit_indices=np.asarray([0, 1, 0, 1], dtype=np.int64),
        candidate_actions=np.asarray([0, 1, 0, 1], dtype=np.int64),
        prefix_occupancy=np.asarray(
            [[0.0], [0.0], [1.0], [1.0]], dtype=np.float32
        ),
        legal_action_masks=np.ones((4, 2), dtype=np.float32),
        splits=np.asarray(["train", "train", "train", "test"]),
    )


def _write_air_defense_support_dataset(path: Path) -> None:
    env = AirDefenseResourceAssignmentEnvV1()
    observations = []
    unit_indices = []
    candidate_actions = []
    prefix_occupancy = []
    legal_action_masks = []
    for seed in range(12):
        observation, _ = env.reset(seed=seed)
        unit_index = seed % env.num_defense_units
        mask = np.asarray(env.action_masks(), dtype=bool).reshape(
            env.num_defense_units, -1
        )[unit_index]
        legal_targets = np.flatnonzero(mask[: env.num_targets])
        candidate = (
            int(legal_targets[0])
            if legal_targets.size
            else int(env.num_targets)
        )
        observations.append(observation)
        unit_indices.append(unit_index)
        candidate_actions.append(candidate)
        prefix_occupancy.append(np.zeros(env.num_targets, dtype=np.float32))
        legal_action_masks.append(mask.astype(np.float32))
    np.savez(
        path,
        observations=np.asarray(observations, dtype=np.float32),
        unit_indices=np.asarray(unit_indices, dtype=np.int64),
        candidate_actions=np.asarray(candidate_actions, dtype=np.int64),
        prefix_occupancy=np.asarray(prefix_occupancy, dtype=np.float32),
        legal_action_masks=np.asarray(legal_action_masks, dtype=np.float32),
        splits=np.asarray(["train"] * len(observations)),
    )
    env.close()


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


def _algorithm_shell() -> SupportAnchoredRGMCHPPO:
    algorithm = object.__new__(SupportAnchoredRGMCHPPO)
    algorithm.engagement_residual_coef = 0.5
    algorithm.target_residual_coef = 0.5
    algorithm.residual_clip = 0.5
    algorithm.reliability_threshold = 0.5
    algorithm.anchor_kl_budget = 0.10
    algorithm.anchor_kl_coef = 1.0
    return algorithm


def test_context_support_uses_train_split_and_decays_outside_support(
    tmp_path,
) -> None:
    dataset = tmp_path / "support.npz"
    _write_small_support_dataset(dataset)
    index = MaskedContextSupportIndex.from_npz(
        dataset, num_units=2, device="cpu"
    )
    exact = index.engagement_scores(
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([0]),
        torch.tensor([[0.0]]),
        torch.tensor([[1.0, 1.0]]),
    )
    outside = index.engagement_scores(
        torch.tensor([[100.0, 100.0]]),
        torch.tensor([1]),
        torch.tensor([[1.0]]),
        torch.tensor([[1.0, 1.0]]),
    )

    assert index.train_row_count == 3
    assert exact.item() > 0.999
    assert 0.0 <= outside.item() < exact.item()


def test_support_multiplies_ensemble_reliability_and_reduces_residual() -> None:
    algorithm = _algorithm_shell()
    valid = torch.ones((3, 2), dtype=torch.bool)
    counterfactual = CounterfactualAdvantageBatch(
        engagement=torch.tensor([[2.0, -2.0], [1.0, -1.0], [-1.0, 1.0]]),
        target=torch.tensor([[1.0, -1.0], [2.0, -2.0], [-1.0, 1.0]]),
        engagement_reliability=torch.ones((3, 2)),
        target_reliability=torch.ones((3, 2)),
        engagement_support=torch.zeros((3, 2)),
        target_support=torch.zeros((3, 2)),
        actionable=valid,
        engaged=valid,
    )

    _, _, diagnostics = algorithm._actor_advantages(
        torch.tensor([-1.0, 0.0, 1.0]), counterfactual
    )

    assert diagnostics["engagement_reliability"] == 0.0
    assert diagnostics["target_reliability"] == 0.0
    assert diagnostics["engagement_residual_abs"] == 0.0
    assert diagnostics["target_residual_abs"] == 0.0


def test_anchor_penalty_is_zero_within_budget_and_positive_beyond_it() -> None:
    algorithm = _algorithm_shell()
    actionable = torch.ones((2, 2), dtype=torch.bool)
    anchor = {"engage_probability": torch.full((2, 2), 0.5)}
    within = {"engage_probability": torch.full((2, 2), 0.5)}
    outside = {"engage_probability": torch.full((2, 2), 0.99)}

    within_loss, within_diagnostics = algorithm._policy_regularization(
        within, within, anchor, actionable
    )
    outside_loss, outside_diagnostics = algorithm._policy_regularization(
        outside, outside, anchor, actionable
    )

    assert within_loss.item() == 0.0
    assert within_diagnostics["anchor_excess_rate"] == 0.0
    assert outside_loss.item() > 0.0
    assert outside_diagnostics["anchor_excess_rate"] == 1.0


def test_sa_rg_mch_trains_saves_loads_and_evaluates(tmp_path) -> None:
    critic_path = tmp_path / "critic.pt"
    support_path = tmp_path / "support.npz"
    _write_q_checkpoint(critic_path)
    _write_air_defense_support_dataset(support_path)
    training = AirDefenseV1PPOConfig(
        total_timesteps=8,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        verbose=0,
        mch_q_critic_paths=(str(critic_path),),
        sa_rg_mch_support_dataset_path=str(support_path),
    )
    save_path = tmp_path / "sa_rg_mch"
    model = train_sa_rg_mch_ppo(
        train_config=training,
        save_path=save_path,
        unit_order=(0, 1, 2),
    )

    assert model.action_generator_signature["optimizer"]["type"] == (
        "support_anchored_reliability_gated_mch_ppo"
    )
    assert all(
        not parameter.requires_grad
        for parameter in model._engagement_anchor_policy.parameters()
    )
    assert 0.0 <= model.last_mch_training_diagnostics[
        "mch_engagement_support"
    ] <= 1.0

    env = AirDefenseResourceAssignmentEnvV1()
    loaded = SupportAnchoredRGMCHPPO.load(save_path, env=env)
    metrics = evaluate_air_defense_v1_model(
        loaded, episodes=1, seed=123, use_action_masks=True
    )
    assert metrics["avg_invalid_actions"] == 0.0
    assert metrics["assignment_conflict_rate"] == 0.0
    env.close()
