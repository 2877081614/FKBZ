from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from rein_learning.algorithms.policy_gradient import (
    FactorizedEngagementActorCriticPolicy,
    FactorizedEngagementMaskablePPO,
)
from rein_learning.common import (
    ActionSubstitutionConfirmationConfig,
    audit_confirmation_context,
    collect_confirmation_contexts,
    summarize_confirmation,
    validate_confirmation_contexts,
)
from rein_learning.common.action_substitution_confirmation import (
    _interval,
    grouped_summary_rows,
)
from rein_learning.envs import (
    AirDefenseResourceAssignmentEnvV1,
    get_air_defense_v1_scenario,
)


def _small_model() -> tuple[
    AirDefenseResourceAssignmentEnvV1,
    FactorizedEngagementMaskablePPO,
]:
    env = AirDefenseResourceAssignmentEnvV1(
        config=get_air_defense_v1_scenario("time_pressure")
    )
    model = FactorizedEngagementMaskablePPO(
        FactorizedEngagementActorCriticPolicy,
        env,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        seed=37,
        device="cpu",
        verbose=0,
        policy_kwargs={"net_arch": [32], "unit_order": (0, 1, 2)},
    )
    return env, model


def test_confirmation_contexts_enforce_type_quota_and_identity() -> None:
    env, model = _small_model()
    config = ActionSubstitutionConfirmationConfig(pool_episodes=6, repeats=2)
    contexts = collect_confirmation_contexts(
        policy=model.policy,
        env_config=env.config,
        scenario="time_pressure",
        policy_seed=37,
        excluded_observation_hashes=set(),
        config=config,
    )
    assert len(contexts) == 12
    resource = [context for context in contexts if context.slot == "resource"]
    types = [
        context.snapshot.defense_units[context.unit_index].resource_type
        for context in resource
    ]
    assert types.count("missile") == 3
    assert types.count("laser") == 3
    identity = validate_confirmation_contexts(
        contexts,
        policy=model.policy,
        excluded_observation_hashes=set(),
        probability_tolerance=config.probability_tolerance,
    )
    assert all(row["matched"] for row in identity)
    excluded = {contexts[0].observation_hash}
    invalid = validate_confirmation_contexts(
        contexts,
        policy=model.policy,
        excluded_observation_hashes=excluded,
        probability_tolerance=config.probability_tolerance,
    )
    assert invalid[0]["old_hash_overlap"]
    assert not invalid[0]["matched"]
    env.close()


def test_confirmation_audit_is_read_only_and_keeps_cost_ledgers() -> None:
    env, model = _small_model()
    config = ActionSubstitutionConfirmationConfig(pool_episodes=6, repeats=2)
    contexts = collect_confirmation_contexts(
        policy=model.policy,
        env_config=env.config,
        scenario="time_pressure",
        policy_seed=37,
        excluded_observation_hashes=set(),
        config=config,
    )
    before = deepcopy(model.policy.state_dict())
    aggregate, repeats, targets = audit_confirmation_context(
        policy=model.policy,
        env_config=env.config,
        context=contexts[6],
        config=config,
    )
    assert aggregate["repeat_count"] == 2
    assert len(repeats) == 2
    assert len(targets) == 2 * len(contexts[6].legal_targets)
    assert all(float(row["direct_cost"]) > 0.0 for row in targets)
    assert all(
        abs(float(row["protocol_residual"])) <= 1e-9 for row in targets
    )
    assert all(
        abs(float(row["extended_residual"])) <= 1e-9 for row in targets
    )
    assert all(
        abs(float(row["sub_cost_decomposition_residual"])) <= 1e-9
        for row in targets
    )
    for name, value in model.policy.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)
    env.close()


def test_confirmation_summary_applies_independent_and_type_gates() -> None:
    config = ActionSubstitutionConfirmationConfig(repeats=2)
    contexts: list[dict[str, object]] = []
    identities: list[dict[str, object]] = []
    repeats: list[dict[str, object]] = []
    targets: list[dict[str, object]] = []
    for scenario in ("medium", "time_pressure", "heterogeneity_pressure"):
        for seed in (17, 18, 19):
            for index in range(12):
                slot = "safety" if index < 6 else "resource"
                resource_type = (
                    "missile"
                    if slot == "safety" or index < 9
                    else "laser"
                )
                context_id = f"{scenario}_{seed}_{index}"
                contexts.append(
                    {
                        "context_id": context_id,
                        "scenario": scenario,
                        "policy_seed": seed,
                        "slot": slot,
                        "resource_type": resource_type,
                        "repeat_count": 2,
                        "actual_extra_transitions": 10,
                        "sub_shot_mean": 1.0,
                        "sub_shot_lower": 0.5,
                        "sub_cost_mean": 2.0,
                        "episode_cost_delta_mean": -0.5,
                        "cost_sign_masked_rate": 0.75,
                    }
                )
                identities.append(
                    {
                        "context_id": context_id,
                        "matched": True,
                        "old_hash_overlap": False,
                        "maximum_probability_difference": 0.0,
                    }
                )
                repeats.append({"context_id": context_id})
                targets.append(
                    {
                        "context_id": context_id,
                        "direct_cost": 2.0,
                        "protocol_residual": 0.0,
                        "future_only_residual": 0.0,
                        "extended_residual": 0.0,
                        "sub_cost_decomposition_residual": 0.0,
                    }
                )
    summary = summarize_confirmation(
        contexts,
        repeats,
        targets,
        identities,
        source_model_count=9,
        config=config,
        maximum_actor_parameter_difference=0.0,
        software_tests_passed=True,
    )
    assert all(summary["integrity_gates"].values())
    assert all(summary["mechanism_gates"].values())
    assert summary["stage_passed"]
    assert (
        summary["decision"]
        == "freeze_cross_resource_measurement_distortion_claim"
    )


def test_confirmation_config_rejects_nonbalanced_resource_quota() -> None:
    with pytest.raises(ValueError, match="resource type quotas"):
        ActionSubstitutionConfirmationConfig(
            contexts_per_slot=6,
            resource_contexts_per_type=2,
        )


def test_single_context_smoke_group_has_degenerate_interval() -> None:
    rows = grouped_summary_rows(
        [
            {
                "scenario": "time_pressure",
                "slot": "resource",
                "resource_type": "missile",
                "sub_shot_mean": 1.0,
                "sub_cost_mean": 2.0,
                "rho_sub_mean": 1.0,
                "episode_cost_delta_mean": 0.0,
                "cost_sign_masked_rate": 1.0,
            }
        ],
        group_fields=("scenario", "slot", "resource_type"),
        config=ActionSubstitutionConfirmationConfig(repeats=2),
    )
    assert rows[0]["sub_shot_mean_lower"] == 1.0
    assert rows[0]["sub_shot_mean_upper"] == 1.0
    assert _interval(
        [], ActionSubstitutionConfirmationConfig(repeats=2)
    )["mean"] != _interval(
        [], ActionSubstitutionConfirmationConfig(repeats=2)
    )["mean"]
