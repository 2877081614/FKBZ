from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from rein_learning.algorithms.policy_gradient import (
    FactorizedEngagementActorCriticPolicy,
    FactorizedEngagementMaskablePPO,
)
from rein_learning.common import (
    BPCELabelSemanticsConfig,
    audit_bpce_context,
    collect_bpce_audit_contexts,
    estimate_effect,
    sample_fixed_actions_with_uniforms,
    summarize_bpce_semantics,
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
        seed=17,
        device="cpu",
        verbose=0,
        policy_kwargs={"net_arch": [32], "unit_order": (0, 1, 2)},
    )
    return env, model


def test_effect_estimate_requires_effect_and_nonzero_interval() -> None:
    reliable = estimate_effect(
        [2.0, 2.2, 1.8, 2.1],
        minimum_return_effect=1.0,
        confidence_z=1.96,
    )
    weak = estimate_effect(
        [-2.0, 2.0, -2.0, 2.0],
        minimum_return_effect=1.0,
        confidence_z=1.96,
    )
    assert reliable["sign"] == 1
    assert reliable["reliable"] is True
    assert weak["reliable"] is False


def test_explicit_uniform_sampling_is_reproducible_and_legal() -> None:
    env, model = _small_model()
    observation, _ = env.reset(seed=31)
    observation_tensor = torch.as_tensor(observation[None, :])
    mask = torch.as_tensor(env.action_masks()[None, :])
    distribution = model.policy.get_distribution(
        observation_tensor, action_masks=mask
    )
    fixed = torch.full((1, env.num_defense_units), -1, dtype=torch.long)
    uniforms = np.asarray([[0.13, 0.57, 0.91]])
    first = sample_fixed_actions_with_uniforms(
        distribution, fixed, uniforms
    )
    second = sample_fixed_actions_with_uniforms(
        distribution, fixed, uniforms
    )
    np.testing.assert_array_equal(
        first.detach().cpu().numpy(),
        second.detach().cpu().numpy(),
    )
    probabilities, action_masks = distribution.conditional_probabilities(
        first
    )
    del probabilities
    for unit, action in enumerate(first[0].tolist()):
        assert bool(action_masks[0, unit, action])
    env.close()


def test_context_audit_computes_three_labels_without_actor_update() -> None:
    env, model = _small_model()
    env_config = get_air_defense_v1_scenario("time_pressure")
    config = BPCELabelSemanticsConfig(
        contexts_per_slot=1,
        pool_episodes=1,
        repeats=2,
    )
    before = deepcopy(model.policy.state_dict())
    contexts = collect_bpce_audit_contexts(
        policy=model.policy,
        env_config=env_config,
        scenario="time_pressure",
        policy_seed=17,
        config=config,
    )
    assert len(contexts) == 2
    aggregate, repeats, targets = audit_bpce_context(
        policy=model.policy,
        env_config=env_config,
        context=contexts[0],
        config=config,
    )
    assert len(repeats) == 2
    assert len(targets) == aggregate["legal_target_count"]
    for label in ("a", "b", "c"):
        assert f"{label}_mean" in aggregate
        assert f"{label}_standard_error" in aggregate
        assert f"{label}_reliable" in aggregate
    assert sum(contexts[0].target_probabilities) == pytest.approx(1.0)
    for name, value in model.policy.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)
    env.close()


def test_semantics_summary_applies_frozen_stage_a_gates() -> None:
    rows: list[dict[str, object]] = []
    for scenario in ("time_pressure", "heterogeneity_pressure"):
        for seed in (8, 9, 10):
            for index in range(12):
                sign = 1 if index < 6 else -1
                rows.append(
                    {
                        "scenario": scenario,
                        "policy_seed": seed,
                        "a_reliable": True,
                        "b_reliable": True,
                        "c_reliable": True,
                        "a_sign": sign,
                        "b_sign": sign,
                        "c_sign": sign,
                        "c_zone_damage_mean": -0.1 if sign > 0 else 0.1,
                        "c_high_threat_leaks_mean": (
                            -0.1 if sign > 0 else 0.1
                        ),
                    }
                )
    summary = summarize_bpce_semantics(
        rows,
        config=BPCELabelSemanticsConfig(),
    )
    assert summary["stage_a_passed"] is True
    assert (
        summary["label_decision"]
        == "argmax_det_is_acceptable_low_cost_approximation"
    )


def test_sign_gate_does_not_hide_unreliable_context_conflicts() -> None:
    rows: list[dict[str, object]] = []
    for index in range(72):
        scenario = (
            "time_pressure" if index < 36 else "heterogeneity_pressure"
        )
        local_index = index % 36
        sign = 1 if local_index < 18 else -1
        conflicts = local_index < 8
        rows.append(
            {
                "scenario": scenario,
                "policy_seed": 8 + (local_index // 12),
                "a_reliable": not conflicts,
                "b_reliable": not conflicts,
                "c_reliable": not conflicts,
                "a_sign": sign,
                "b_sign": sign,
                "c_sign": -sign if conflicts else sign,
                "c_zone_damage_mean": -0.1,
                "c_high_threat_leaks_mean": -0.1,
            }
        )
    summary = summarize_bpce_semantics(
        rows,
        config=BPCELabelSemanticsConfig(),
    )
    assert summary["reliable_overlap_agreement"]["b_c"]["rate"] == 1.0
    assert summary["b_c_agreement"]["rate"] == pytest.approx(56 / 72)
    assert summary["gates"]["b_c_sign_agreement"] is False
    assert summary["label_decision"] == "deterministic_continuation_rejected"
