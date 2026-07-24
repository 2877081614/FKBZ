from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from rein_learning.algorithms.policy_gradient import (
    FactorizedEngagementActorCriticPolicy,
    FactorizedEngagementMaskablePPO,
)
from rein_learning.common import (
    ActionSubstitutionOpportunityCostConfig,
    BPCELabelSemanticsConfig,
    audit_action_substitution_context,
    collect_bpce_audit_contexts,
    intervention_is_unique,
    reliable_positive_opportunity,
    restore_probed_ammo,
    summarize_action_substitution_audit,
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
        seed=31,
        device="cpu",
        verbose=0,
        policy_kwargs={"net_arch": [32], "unit_order": (0, 1, 2)},
    )
    return env, model


def _interval(lower: float) -> dict[str, float]:
    return {
        "mean": lower + 0.1,
        "standard_error": 0.0,
        "lower": lower,
        "upper": lower + 0.2,
    }


def test_restore_probed_ammo_changes_only_one_inventory() -> None:
    env = AirDefenseResourceAssignmentEnvV1(
        config=get_air_defense_v1_scenario("time_pressure")
    )
    env.reset(seed=17)
    before = env.snapshot_state()
    unit_index = 0
    env.defense_units[unit_index].ammo -= 1
    engage = env.snapshot_state()
    observation, mask, restored = restore_probed_ammo(
        env,
        unit_index=unit_index,
        pre_action_ammo=before.defense_units[unit_index].ammo,
    )
    restored_snapshot = env.snapshot_state()
    assert restored == 1
    assert intervention_is_unique(
        engage,
        restored_snapshot,
        unit_index=unit_index,
        expected_ammo_gain=1,
    )
    assert observation.shape == env.observation_space.shape
    assert mask.shape == (env.num_defense_units * env.num_unit_actions,)
    env.close()


def test_reliable_opportunity_requires_safety_and_future_option() -> None:
    config = ActionSubstitutionOpportunityCostConfig(repeats=2)
    assert reliable_positive_opportunity(
        _interval(0.06),
        _interval(-0.05),
        mean_reuse_probe=0.1,
        mean_option_edge=0.0,
        config=config,
    )
    assert not reliable_positive_opportunity(
        _interval(0.06),
        _interval(-0.05),
        mean_reuse_probe=0.0,
        mean_option_edge=0.0,
        config=config,
    )
    assert not reliable_positive_opportunity(
        _interval(0.01),
        _interval(-0.05),
        mean_reuse_probe=1.0,
        mean_option_edge=1.0,
        config=config,
    )


def test_context_audit_is_read_only_and_cost_decomposition_is_exact() -> None:
    env, model = _small_model()
    contexts = collect_bpce_audit_contexts(
        policy=model.policy,
        env_config=env.config,
        scenario="time_pressure",
        policy_seed=31,
        config=BPCELabelSemanticsConfig(
            contexts_per_slot=1,
            pool_episodes=1,
            repeats=2,
        ),
    )
    before = deepcopy(model.policy.state_dict())
    aggregate, repeats, target_rows, integrity = (
        audit_action_substitution_context(
            policy=model.policy,
            env_config=env.config,
            context=contexts[0],
            config=ActionSubstitutionOpportunityCostConfig(repeats=2),
        )
    )
    assert aggregate["repeat_count"] == 2
    assert len(repeats) == 2
    assert len(target_rows) == 2 * len(contexts[0].legal_targets)
    assert len(integrity) == len(target_rows)
    assert all(row["current_step_identity"] for row in integrity)
    assert all(row["intervention_unique"] for row in integrity)
    assert all(
        abs(float(row["decomposition_residual"])) <= 1e-9
        for row in repeats
    )
    assert all(np.isfinite(row["immediate_cost_difference"]) for row in repeats)
    for name, value in model.policy.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)
    env.close()


def test_summary_rejects_missing_cross_scenario_opportunity_value() -> None:
    config = ActionSubstitutionOpportunityCostConfig(repeats=2)
    context_rows: list[dict[str, object]] = []
    identity_rows: list[dict[str, object]] = []
    for scenario in ("time_pressure", "heterogeneity_pressure"):
        for seed in (8, 9, 10):
            for index in range(12):
                slot = "safety" if index < 6 else "resource"
                context_id = f"{scenario}_{seed}_{index}"
                reliable = (
                    slot == "resource"
                    and scenario == "heterogeneity_pressure"
                )
                context_rows.append(
                    {
                        "context_id": context_id,
                        "scenario": scenario,
                        "policy_seed": seed,
                        "slot": slot,
                        "unit_type": "missile" if index % 2 else "laser",
                        "repeat_count": 2,
                        "sub_shot_mean": 1.0,
                        "sub_shot_lower": 0.5,
                        "sub_cost_mean": 1.0,
                        "future_cost_composition_advantage_mean": 1.0,
                        "total_cost_difference_mean": -0.5,
                        "reuse_probe_mean": 1.0 if reliable else 0.0,
                        "option_edge_mean": 1.0 if reliable else 0.0,
                        "reliable_opportunity": reliable,
                        "maximum_decomposition_residual": 0.0,
                        "actual_extra_transitions": 10,
                    }
                )
                identity_rows.append(
                    {
                        "context_id": context_id,
                        "matched": True,
                        "maximum_probability_difference": 0.0,
                    }
                )
    integrity_rows = [
        {
            "current_step_identity": True,
            "intervention_unique": True,
        }
    ]
    summary = summarize_action_substitution_audit(
        context_rows,
        identity_rows,
        integrity_rows,
        config=config,
        maximum_actor_parameter_difference=0.0,
        software_tests_passed=True,
    )
    assert summary["mechanism_gates"]["P-R1_action_substitution"]
    assert not summary["mechanism_gates"]["P-R2_opportunity_value"]
    assert not summary["stage_passed"]
    assert (
        summary["decision"]
        == "conditional_heterogeneity_only_opportunity_value"
    )
