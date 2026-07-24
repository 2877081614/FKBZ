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
    BPCEShortHorizonConfig,
    BranchTrajectory,
    audit_short_horizon_context,
    classify_component_label,
    collect_bpce_audit_contexts,
    summarize_short_horizon_audit,
    target_event_horizon,
    validate_context_identity,
)
from rein_learning.common.bpce_label_semantics import BranchOutcome
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
        seed=29,
        device="cpu",
        verbose=0,
        policy_kwargs={"net_arch": [32], "unit_order": (0, 1, 2)},
    )
    return env, model


def _interval(lower: float, upper: float) -> dict[str, float]:
    return {
        "mean": (lower + upper) / 2.0,
        "standard_error": 0.0,
        "lower": lower,
        "upper": upper,
    }


@pytest.mark.parametrize(
    ("damage", "leaks", "cost", "expected"),
    [
        ((-0.20, -0.06), (-0.05, 0.05), (0.1, 0.2), "ENGAGE"),
        ((-0.05, 0.02), (-0.10, 0.05), (0.1, 0.2), "STOP"),
        ((-0.06, 0.02), (-0.11, 0.05), (-0.1, 0.2), "AMBIGUOUS"),
    ],
)
def test_three_state_component_boundaries(
    damage: tuple[float, float],
    leaks: tuple[float, float],
    cost: tuple[float, float],
    expected: str,
) -> None:
    label = classify_component_label(
        damage=_interval(*damage),
        high_threat_leaks=_interval(*leaks),
        resource_cost=_interval(*cost),
        config=BPCEShortHorizonConfig(),
    )
    assert label == expected


def test_branch_trajectory_uses_same_requested_horizon_after_termination() -> None:
    outcomes = tuple(
        BranchOutcome(float(i), float(i), 0.0, float(i), 1)
        for i in range(1, 4)
    )
    trajectory = BranchTrajectory(outcomes)
    assert trajectory.at_horizon(1) == outcomes[0]
    assert trajectory.at_horizon(3) == outcomes[2]
    assert trajectory.at_horizon(10) == outcomes[2]


def test_target_horizon_follows_frozen_tti_formula() -> None:
    env = AirDefenseResourceAssignmentEnvV1(
        config=get_air_defense_v1_scenario("time_pressure")
    )
    env.reset(seed=19)
    snapshot = env.snapshot_state()
    target = next(
        index for index, state in enumerate(snapshot.targets) if state.alive
    )
    expected = min(
        env.config.max_steps - snapshot.current_step,
        int(np.ceil(snapshot.targets[target].time_to_impact)) + 1,
    )
    assert target_event_horizon(
        snapshot, target, max_steps=env.config.max_steps
    ) == expected
    env.close()


def test_context_identity_detects_probability_mismatch() -> None:
    env, model = _small_model()
    contexts = collect_bpce_audit_contexts(
        policy=model.policy,
        env_config=env.config,
        scenario="time_pressure",
        policy_seed=29,
        config=BPCELabelSemanticsConfig(
            contexts_per_slot=1,
            pool_episodes=1,
            repeats=2,
        ),
    )
    context = contexts[0]
    reference = {
        "context_id": context.context_id,
        "scenario": context.scenario,
        "policy_seed": context.policy_seed,
        "environment_seed": context.environment_seed,
        "environment_step": context.environment_step,
        "unit_index": context.unit_index,
        "observation_hash": context.observation_hash,
        "legal_targets": ",".join(map(str, context.legal_targets)),
        "target_probabilities": ",".join(
            str(value + (1e-3 if index == 0 else 0.0))
            for index, value in enumerate(context.target_probabilities)
        ),
    }
    rows = validate_context_identity(
        (context,), (reference,), probability_tolerance=1e-9
    )
    assert rows[0]["matched"] is False
    assert rows[0]["mismatch_fields"] == "target_probabilities"
    env.close()


def test_short_horizon_audit_is_read_only_and_keeps_full_control() -> None:
    env, model = _small_model()
    context_config = BPCELabelSemanticsConfig(
        contexts_per_slot=1,
        pool_episodes=1,
        repeats=2,
    )
    contexts = collect_bpce_audit_contexts(
        policy=model.policy,
        env_config=env.config,
        scenario="time_pressure",
        policy_seed=29,
        config=context_config,
    )
    before = deepcopy(model.policy.state_dict())
    aggregate, repeats, horizons = audit_short_horizon_context(
        policy=model.policy,
        env_config=env.config,
        context=contexts[0],
        config=BPCEShortHorizonConfig(repeats=2),
    )
    assert len(repeats) == 2
    assert len(horizons) == len(contexts[0].legal_targets)
    assert aggregate["repeat_count"] == 2
    assert aggregate["short_label"] in {"ENGAGE", "STOP", "AMBIGUOUS"}
    assert aggregate["full_label"] in {"ENGAGE", "STOP", "AMBIGUOUS"}
    assert aggregate["projected_window_transitions"] <= aggregate[
        "extra_transitions"
    ]
    for name, value in model.policy.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)
    env.close()


def test_stage_a2_summary_applies_cross_seed_bidirectional_gate() -> None:
    config = BPCEShortHorizonConfig(repeats=2)
    rows: list[dict[str, object]] = []
    identity_rows: list[dict[str, object]] = []
    for scenario in ("time_pressure", "heterogeneity_pressure"):
        for seed in (8, 9, 10):
            for index in range(12):
                label = "ENGAGE" if index < 6 else "STOP"
                context_id = f"{scenario}_{seed}_{index}"
                rows.append(
                    {
                        "context_id": context_id,
                        "scenario": scenario,
                        "policy_seed": seed,
                        "slot": "safety" if index < 6 else "resource",
                        "short_label": label,
                        "full_label": label,
                        "label_changed": False,
                        "short_zone_damage_upper": (
                            -0.1 if label == "ENGAGE" else 0.0
                        ),
                        "short_high_threat_leaks_upper": 0.0,
                        "extra_transitions": 10,
                        "projected_window_transitions": 5,
                        "repeat_count": 2,
                    }
                )
                identity_rows.append(
                    {"context_id": context_id, "matched": True}
                )
    summary = summarize_short_horizon_audit(
        rows,
        identity_rows,
        config=config,
        maximum_actor_parameter_difference=0.0,
        software_tests_passed=True,
    )
    assert summary["stage_a2_passed"] is True
    assert summary["decision"] == "proceed_to_auxiliary_dose_audit"
