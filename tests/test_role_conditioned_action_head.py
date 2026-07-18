import pytest
import torch

from rein_learning.algorithms.policy_gradient import (
    AutoregressiveMaskableActorCriticPolicy,
    RoleConditionedAutoregressiveActorCriticPolicy,
    RoleConditionedAutoregressiveMaskablePPO,
    policy_parameter_counts,
)
from rein_learning.common import (
    aggregate_collapsed_unit_counts,
    aggregate_decision_rows,
)
from rein_learning.envs import (
    AirDefenseResourceAssignmentEnvV1,
    get_air_defense_v1_scenario_profile,
)
from rein_learning.models import (
    AirDefenseV1ObservationLayout,
    RoleConditionedAirDefenseActionHead,
    StructuredAirDefenseV1Observation,
)
from rein_learning.trainers.air_defense_v1_ppo import (
    AirDefenseV1PPOConfig,
    evaluate_air_defense_v1_model,
    train_role_conditioned_autoregressive_ppo,
)


def _default_observation_and_masks() -> tuple[
    AirDefenseResourceAssignmentEnvV1,
    torch.Tensor,
    torch.Tensor,
]:
    env = AirDefenseResourceAssignmentEnvV1()
    observation, _ = env.reset(seed=5)
    return (
        env,
        torch.as_tensor(observation).unsqueeze(0),
        torch.as_tensor(env.action_masks()).unsqueeze(0),
    )


def test_observation_layout_round_trip_matches_flat_environment_observation() -> None:
    env, observation, _ = _default_observation_and_masks()
    layout = AirDefenseV1ObservationLayout.infer(
        env.observation_space,
        env.action_space,
    )

    structured = layout.split(observation)

    assert structured.zones.shape == (1, 2, 7)
    assert structured.targets.shape == (1, 5, 15)
    assert structured.units.shape == (1, 3, 15)
    assert structured.global_features.shape == (1, 8)
    assert torch.equal(structured.flatten(), observation)
    env.close()


def test_role_head_is_equivariant_to_unit_permutation() -> None:
    torch.manual_seed(1)
    env, observation, masks = _default_observation_and_masks()
    layout = AirDefenseV1ObservationLayout.infer(
        env.observation_space,
        env.action_space,
    )
    head = RoleConditionedAirDefenseActionHead(layout)
    structured = layout.split(observation)
    permutation = (2, 0, 1)
    permuted_observation = StructuredAirDefenseV1Observation(
        zones=structured.zones,
        targets=structured.targets,
        units=structured.units[:, permutation, :],
        global_features=structured.global_features,
    ).flatten()
    mask_blocks = masks.reshape(1, layout.num_units, layout.num_targets + 1)
    permuted_masks = mask_blocks[:, permutation, :].reshape(1, -1)

    logits = head(observation, masks).reshape(
        1, layout.num_units, layout.num_targets + 1
    )
    permuted_logits = head(permuted_observation, permuted_masks).reshape_as(logits)

    assert torch.allclose(permuted_logits, logits[:, permutation, :], atol=1e-6)
    env.close()


def test_role_head_is_equivariant_to_target_permutation() -> None:
    torch.manual_seed(2)
    env, observation, masks = _default_observation_and_masks()
    layout = AirDefenseV1ObservationLayout.infer(
        env.observation_space,
        env.action_space,
    )
    head = RoleConditionedAirDefenseActionHead(layout)
    structured = layout.split(observation)
    permutation = (3, 0, 4, 1, 2)
    permuted_observation = StructuredAirDefenseV1Observation(
        zones=structured.zones,
        targets=structured.targets[:, permutation, :],
        units=structured.units,
        global_features=structured.global_features,
    ).flatten()
    mask_blocks = masks.reshape(1, layout.num_units, layout.num_targets + 1)
    permuted_masks = torch.cat(
        (
            mask_blocks[:, :, permutation],
            mask_blocks[:, :, -1:],
        ),
        dim=2,
    ).reshape(1, -1)

    logits = head(observation, masks).reshape(
        1, layout.num_units, layout.num_targets + 1
    )
    permuted_logits = head(permuted_observation, permuted_masks).reshape_as(logits)
    expected = torch.cat(
        (
            logits[:, :, permutation],
            logits[:, :, -1:],
        ),
        dim=2,
    )

    assert torch.allclose(permuted_logits, expected, atol=1e-6)
    env.close()


def test_resource_role_feature_changes_relation_logits() -> None:
    torch.manual_seed(3)
    env, observation, masks = _default_observation_and_masks()
    layout = AirDefenseV1ObservationLayout.infer(
        env.observation_space,
        env.action_space,
    )
    head = RoleConditionedAirDefenseActionHead(layout)
    structured = layout.split(observation)
    changed_units = structured.units.clone()
    changed_units[:, 0, 2] = 1.0 - changed_units[:, 0, 2]
    changed_observation = StructuredAirDefenseV1Observation(
        zones=structured.zones,
        targets=structured.targets,
        units=changed_units,
        global_features=structured.global_features,
    ).flatten()

    original = head(observation, masks)
    changed = head(changed_observation, masks)

    assert not torch.allclose(original, changed)
    env.close()


def test_role_actor_parameter_count_is_capacity_matched() -> None:
    env = AirDefenseResourceAssignmentEnvV1()
    schedule = lambda _: 3e-4
    baseline = AutoregressiveMaskableActorCriticPolicy(
        env.observation_space,
        env.action_space,
        schedule,
        net_arch=[128, 128],
    )
    candidate = RoleConditionedAutoregressiveActorCriticPolicy(
        env.observation_space,
        env.action_space,
        schedule,
        net_arch=[128, 128],
    )

    baseline_counts = policy_parameter_counts(baseline)
    candidate_counts = candidate.parameter_counts()
    actor_ratio = (
        candidate_counts["actor_parameters"]
        / baseline_counts["actor_parameters"]
    )

    assert actor_ratio == pytest.approx(0.94098, rel=1e-3)
    assert candidate_counts["critic_parameters"] == baseline_counts[
        "critic_parameters"
    ]
    assert candidate.action_net.pair_output.weight.requires_grad
    env.close()


def test_role_conditioned_training_save_load_preserves_order_and_signature(
    tmp_path,
) -> None:
    env_config = get_air_defense_v1_scenario_profile("medium").config
    train_config = AirDefenseV1PPOConfig(
        total_timesteps=8,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        net_arch=(128, 128),
        seed=0,
        verbose=0,
    )
    save_path = tmp_path / "role_conditioned_order_201"
    model = train_role_conditioned_autoregressive_ppo(
        env_config=env_config,
        train_config=train_config,
        save_path=save_path,
        unit_order=(2, 0, 1),
    )
    env = AirDefenseResourceAssignmentEnvV1(config=env_config)
    loaded = RoleConditionedAutoregressiveMaskablePPO.load(save_path, env=env)

    assert loaded.policy.unit_order == (2, 0, 1)
    assert loaded.action_generator_signature == model.action_generator_signature
    assert loaded.action_generator_signature["unit_order"] == [2, 0, 1]
    assert loaded.action_generator_signature["actor_head"][
        "unit_index_embedding"
    ] is False
    metrics = evaluate_air_defense_v1_model(
        loaded,
        env_config=env_config,
        episodes=1,
        seed=90,
        use_action_masks=True,
    )
    assert metrics["avg_invalid_actions"] == 0.0
    assert metrics["assignment_conflict_rate"] == 0.0
    assert metrics["overkill_rate"] == 0.0
    env.close()


def test_collapsed_unit_requires_enough_actionable_decisions() -> None:
    rows = []
    for index in range(100):
        rows.append(
            {
                "group": "collapsed",
                "selected_noop": True,
                "num_conditional_legal_targets": 1,
                "matching_efficiency": None,
                "expected_damage_reduction": None,
                "target_threat": None,
                "num_conditional_high_threat_targets": 1,
                "prefix_denied_target_count": 0,
                "num_base_legal_targets": 1,
                "avoidable_noop": True,
                "selected_high_threat": False,
            }
        )
    summary = aggregate_decision_rows(rows, group_keys=("group",))
    counts = aggregate_collapsed_unit_counts(summary, group_keys=("group",))

    assert summary[0]["collapsed_unit"] is True
    assert counts == [
        {
            "group": "collapsed",
            "collapsed_unit_count": 1,
            "evaluated_unit_count": 1,
        }
    ]

    insufficient = aggregate_decision_rows(rows[:99], group_keys=("group",))
    assert insufficient[0]["collapsed_unit"] is False
