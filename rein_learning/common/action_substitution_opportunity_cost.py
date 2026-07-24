from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..envs.air_defense_v1 import (
    AirDefenseResourceAssignmentEnvV1,
    AirDefenseV1EnvConfig,
    AirDefenseV1StateSnapshot,
)
from .bpce_label_semantics import (
    BPCEAuditContext,
    make_bpce_fixed_prefix,
    make_bpce_paired_random_tapes,
    sample_fixed_actions_with_uniforms,
)
from .bpce_short_horizon_labels import component_interval


@dataclass(frozen=True)
class ActionSubstitutionOpportunityCostConfig:
    repeats: int = 32
    confidence_z: float = 1.96
    high_threat_threshold: float = 0.8
    damage_effect: float = 0.05
    leak_effect: float = 0.10
    identity_probability_tolerance: float = 1e-12
    decomposition_tolerance: float = 1e-6
    branch_base_seed: int = 983_000
    maximum_extra_transitions: int = 266_198

    def __post_init__(self) -> None:
        if self.repeats <= 1:
            raise ValueError("repeats must be greater than one")
        if self.confidence_z <= 0.0:
            raise ValueError("confidence_z must be positive")
        if not 0.0 <= self.high_threat_threshold <= 1.0:
            raise ValueError("high_threat_threshold must be in [0, 1]")
        if self.damage_effect < 0.0 or self.leak_effect < 0.0:
            raise ValueError("minimum effects must be non-negative")
        if self.maximum_extra_transitions <= 0:
            raise ValueError("maximum_extra_transitions must be positive")


@dataclass(frozen=True)
class OpportunityBranchTrace:
    current_reward: float
    current_resource_cost: float
    current_observation: np.ndarray
    current_action_mask: np.ndarray
    current_info: Mapping[str, Any]
    current_snapshot: AirDefenseV1StateSnapshot
    continuation_snapshot: AirDefenseV1StateSnapshot
    future_total_shots: float
    future_shots_by_unit: tuple[float, ...]
    future_cost_by_unit: tuple[float, ...]
    future_resource_cost: float
    legal_edge_sum: float
    high_threat_coverable_sum: float
    future_cumulative_shots: tuple[float, ...]
    final_zone_damage: float
    final_high_threat_leaks: float
    total_resource_cost: float
    transitions: int
    opportunity_not_observable: bool
    restored_ammo: int


def reliable_positive_opportunity(
    damage_interval: Mapping[str, float],
    leak_interval: Mapping[str, float],
    *,
    mean_reuse_probe: float,
    mean_option_edge: float,
    config: ActionSubstitutionOpportunityCostConfig,
) -> bool:
    safety_gate = (
        float(damage_interval["lower"]) > config.damage_effect
        and float(leak_interval["lower"]) >= -config.leak_effect
    ) or (
        float(leak_interval["lower"]) > config.leak_effect
        and float(damage_interval["lower"]) >= -config.damage_effect
    )
    opportunity_gate = mean_reuse_probe > 0.0 or mean_option_edge > 0.0
    return bool(safety_gate and opportunity_gate)


def restore_probed_ammo(
    env: AirDefenseResourceAssignmentEnvV1,
    *,
    unit_index: int,
    pre_action_ammo: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    unit = env.defense_units[unit_index]
    restored = min(int(pre_action_ammo), int(unit.ammo) + 1) - int(unit.ammo)
    unit.ammo += restored
    observation = env._get_observation()
    action_mask = env.action_masks().copy()
    return observation, action_mask, restored


def intervention_is_unique(
    engage_snapshot: AirDefenseV1StateSnapshot,
    restored_snapshot: AirDefenseV1StateSnapshot,
    *,
    unit_index: int,
    expected_ammo_gain: int,
) -> bool:
    if (
        restored_snapshot.defense_units[unit_index].ammo
        - engage_snapshot.defense_units[unit_index].ammo
        != expected_ammo_gain
    ):
        return False
    normalized = deepcopy(restored_snapshot)
    normalized.defense_units[unit_index].ammo = (
        engage_snapshot.defense_units[unit_index].ammo
    )
    return _values_equal(engage_snapshot, normalized)


@torch.no_grad()
def audit_action_substitution_context(
    *,
    policy: Any,
    env_config: AirDefenseV1EnvConfig,
    context: BPCEAuditContext,
    config: ActionSubstitutionOpportunityCostConfig,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    device = next(policy.parameters()).device
    observation_tensor = torch.as_tensor(
        context.observation[None, :], device=device, dtype=torch.float32
    )
    mask_tensor = torch.as_tensor(context.action_mask[None, :], device=device)
    original_action = torch.as_tensor(
        context.original_action, device=device, dtype=torch.long
    )[None, :]
    distribution = policy.get_distribution(
        observation_tensor, action_masks=mask_tensor
    )
    probabilities = np.asarray(
        context.target_probabilities, dtype=np.float64
    )
    probed_unit = context.snapshot.defense_units[context.unit_index]
    probe_env = AirDefenseResourceAssignmentEnvV1(config=env_config)
    probe_env.reset(seed=config.branch_base_seed)

    repeat_rows: list[dict[str, Any]] = []
    target_repeat_rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    total_transitions = 0
    for repeat in range(config.repeats):
        environment_tape, policy_tape = make_bpce_paired_random_tapes(
            context=context,
            repeat=repeat,
            env_config=env_config,
            branch_base_seed=config.branch_base_seed,
        )
        noop_action = _sample_first_action(
            distribution,
            original_action,
            unit_index=context.unit_index,
            selected_action=distribution.noop_action,
            uniforms=policy_tape[context.environment_step],
        )
        noop = _rollout_branch(
            env=probe_env,
            snapshot=context.snapshot,
            environment_tape=environment_tape,
            first_action=noop_action,
            policy=policy,
            policy_tape=policy_tape,
            high_threat_threshold=config.high_threat_threshold,
            restore_unit_index=None,
        )
        total_transitions += noop.transitions

        weighted = {
            "future_shots_e": 0.0,
            "future_shots_er": 0.0,
            "future_probe_shots_e": 0.0,
            "future_probe_shots_er": 0.0,
            "future_other_shots_e": 0.0,
            "future_cost_e": 0.0,
            "future_cost_er": 0.0,
            "future_composition_e": 0.0,
            "immediate_cost_e": 0.0,
            "total_cost_e": 0.0,
            "option_edges_e": 0.0,
            "option_edges_er": 0.0,
            "option_threat_e": 0.0,
            "option_threat_er": 0.0,
            "damage_e": 0.0,
            "damage_er": 0.0,
            "leaks_e": 0.0,
            "leaks_er": 0.0,
            "transitions_e": 0.0,
            "transitions_er": 0.0,
        }
        substitution_times: list[tuple[float, int | None]] = []
        observable_probability = 0.0
        for probability, target in zip(
            probabilities, context.legal_targets
        ):
            engage_action = _sample_first_action(
                distribution,
                original_action,
                unit_index=context.unit_index,
                selected_action=target,
                uniforms=policy_tape[context.environment_step],
            )
            engage = _rollout_branch(
                env=probe_env,
                snapshot=context.snapshot,
                environment_tape=environment_tape,
                first_action=engage_action,
                policy=policy,
                policy_tape=policy_tape,
                high_threat_threshold=config.high_threat_threshold,
                restore_unit_index=None,
            )
            restored = _rollout_branch(
                env=probe_env,
                snapshot=context.snapshot,
                environment_tape=environment_tape,
                first_action=engage_action,
                policy=policy,
                policy_tape=policy_tape,
                high_threat_threshold=config.high_threat_threshold,
                restore_unit_index=context.unit_index,
            )
            total_transitions += engage.transitions + restored.transitions
            current_identity = _current_steps_equal(engage, restored)
            expected_gain = 0 if restored.opportunity_not_observable else 1
            unique = intervention_is_unique(
                engage.current_snapshot,
                restored.continuation_snapshot,
                unit_index=context.unit_index,
                expected_ammo_gain=expected_gain,
            )
            if not restored.opportunity_not_observable:
                observable_probability += float(probability)
            integrity_rows.append(
                {
                    "context_id": context.context_id,
                    "scenario": context.scenario,
                    "policy_seed": context.policy_seed,
                    "slot": context.slot,
                    "repeat": repeat,
                    "target_index": target,
                    "target_probability": probability,
                    "current_step_identity": current_identity,
                    "intervention_unique": unique,
                    "opportunity_not_observable": (
                        restored.opportunity_not_observable
                    ),
                    "restored_ammo": restored.restored_ammo,
                    "expected_ammo_gain": expected_gain,
                }
            )

            first_substitution = _first_substitution_time(
                noop.future_cumulative_shots,
                engage.future_cumulative_shots,
            )
            substitution_times.append((float(probability), first_substitution))
            target_row = {
                "context_id": context.context_id,
                "scenario": context.scenario,
                "policy_seed": context.policy_seed,
                "slot": context.slot,
                "repeat": repeat,
                "target_index": target,
                "target_probability": probability,
                "sub_shot": (
                    noop.future_total_shots - engage.future_total_shots
                ),
                "sub_cost": (
                    noop.future_resource_cost
                    - engage.future_resource_cost
                ),
                "reuse_probe": (
                    restored.future_shots_by_unit[context.unit_index]
                    - engage.future_shots_by_unit[context.unit_index]
                ),
                "reuse_total": (
                    restored.future_total_shots
                    - engage.future_total_shots
                ),
                "option_edge": (
                    restored.legal_edge_sum - engage.legal_edge_sum
                ),
                "option_threat": (
                    restored.high_threat_coverable_sum
                    - engage.high_threat_coverable_sum
                ),
                "ammo_gain_damage": (
                    engage.final_zone_damage
                    - restored.final_zone_damage
                ),
                "ammo_gain_leaks": (
                    engage.final_high_threat_leaks
                    - restored.final_high_threat_leaks
                ),
                "first_substitution_time": (
                    "" if first_substitution is None else first_substitution
                ),
                "opportunity_not_observable": (
                    restored.opportunity_not_observable
                ),
                "transitions": engage.transitions + restored.transitions,
            }
            target_repeat_rows.append(target_row)

            probe = context.unit_index
            for key, value in {
                "future_shots_e": engage.future_total_shots,
                "future_shots_er": restored.future_total_shots,
                "future_probe_shots_e": engage.future_shots_by_unit[probe],
                "future_probe_shots_er": restored.future_shots_by_unit[probe],
                "future_other_shots_e": (
                    engage.future_total_shots
                    - engage.future_shots_by_unit[probe]
                ),
                "future_cost_e": engage.future_resource_cost,
                "future_cost_er": restored.future_resource_cost,
                "future_composition_e": sum(engage.future_cost_by_unit),
                "immediate_cost_e": engage.current_resource_cost,
                "total_cost_e": engage.total_resource_cost,
                "option_edges_e": engage.legal_edge_sum,
                "option_edges_er": restored.legal_edge_sum,
                "option_threat_e": engage.high_threat_coverable_sum,
                "option_threat_er": restored.high_threat_coverable_sum,
                "damage_e": engage.final_zone_damage,
                "damage_er": restored.final_zone_damage,
                "leaks_e": engage.final_high_threat_leaks,
                "leaks_er": restored.final_high_threat_leaks,
                "transitions_e": engage.transitions,
                "transitions_er": restored.transitions,
            }.items():
                weighted[key] += float(probability) * float(value)

        sub_cost = noop.future_resource_cost - weighted["future_cost_e"]
        immediate_cost_difference = (
            weighted["immediate_cost_e"] - noop.current_resource_cost
        )
        total_cost_difference = (
            weighted["total_cost_e"] - noop.total_resource_cost
        )
        decomposition_residual = total_cost_difference - (
            immediate_cost_difference - sub_cost
        )
        first_times = [
            probability * float(value)
            for probability, value in substitution_times
            if value is not None
        ]
        repeat_rows.append(
            {
                "context_id": context.context_id,
                "scenario": context.scenario,
                "policy_seed": context.policy_seed,
                "slot": context.slot,
                "repeat": repeat,
                "unit_index": context.unit_index,
                "unit_type": probed_unit.resource_type,
                "unit_cost": probed_unit.cost,
                "observable_target_probability": observable_probability,
                "future_total_shots_n": noop.future_total_shots,
                "future_total_shots_e": weighted["future_shots_e"],
                "future_probed_shots_n": (
                    noop.future_shots_by_unit[context.unit_index]
                ),
                "future_probed_shots_e": weighted[
                    "future_probe_shots_e"
                ],
                "future_other_shots_n": (
                    noop.future_total_shots
                    - noop.future_shots_by_unit[context.unit_index]
                ),
                "future_other_shots_e": weighted[
                    "future_other_shots_e"
                ],
                "sub_shot": (
                    noop.future_total_shots - weighted["future_shots_e"]
                ),
                "sub_cost": sub_cost,
                "future_cost_composition_advantage": (
                    noop.future_resource_cost
                    - weighted["future_composition_e"]
                ),
                "immediate_cost_difference": immediate_cost_difference,
                "total_cost_difference": total_cost_difference,
                "decomposition_residual": decomposition_residual,
                "reuse_probe": (
                    weighted["future_probe_shots_er"]
                    - weighted["future_probe_shots_e"]
                ),
                "reuse_total": (
                    weighted["future_shots_er"]
                    - weighted["future_shots_e"]
                ),
                "option_edge": (
                    weighted["option_edges_er"]
                    - weighted["option_edges_e"]
                ),
                "option_threat": (
                    weighted["option_threat_er"]
                    - weighted["option_threat_e"]
                ),
                "ammo_gain_damage": (
                    weighted["damage_e"] - weighted["damage_er"]
                ),
                "ammo_gain_leaks": (
                    weighted["leaks_e"] - weighted["leaks_er"]
                ),
                "first_substitution_time_expected": (
                    sum(first_times)
                    if len(first_times) == len(substitution_times)
                    else ""
                ),
                "extra_transitions": (
                    noop.transitions
                    + weighted["transitions_e"]
                    + weighted["transitions_er"]
                ),
                "actual_extra_transitions": (
                    noop.transitions
                    + sum(
                        int(row["transitions"])
                        for row in target_repeat_rows
                        if row["context_id"] == context.context_id
                        and int(row["repeat"]) == repeat
                    )
                ),
            }
        )

    probe_env.close()
    aggregate = _aggregate_context(
        context=context,
        repeat_rows=repeat_rows,
        total_transitions=total_transitions,
        config=config,
    )
    return aggregate, repeat_rows, target_repeat_rows, integrity_rows


def aggregate_target_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    confidence_z: float,
) -> list[dict[str, Any]]:
    keys = sorted(
        {
            (str(row["context_id"]), int(row["target_index"]))
            for row in rows
        }
    )
    output: list[dict[str, Any]] = []
    metrics = (
        "sub_shot",
        "sub_cost",
        "reuse_probe",
        "reuse_total",
        "option_edge",
        "option_threat",
        "ammo_gain_damage",
        "ammo_gain_leaks",
    )
    for context_id, target_index in keys:
        selected = [
            row
            for row in rows
            if str(row["context_id"]) == context_id
            and int(row["target_index"]) == target_index
        ]
        first = selected[0]
        result: dict[str, Any] = {
            "context_id": context_id,
            "scenario": first["scenario"],
            "policy_seed": first["policy_seed"],
            "slot": first["slot"],
            "target_index": target_index,
            "target_probability": first["target_probability"],
            "repeat_count": len(selected),
        }
        for metric in metrics:
            interval = component_interval(
                [float(row[metric]) for row in selected],
                confidence_z=confidence_z,
            )
            for field, value in interval.items():
                result[f"{metric}_{field}"] = value
        output.append(result)
    return output


def summarize_action_substitution_audit(
    context_rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
    integrity_rows: Sequence[Mapping[str, Any]],
    *,
    config: ActionSubstitutionOpportunityCostConfig,
    maximum_actor_parameter_difference: float,
    software_tests_passed: bool,
) -> dict[str, Any]:
    scenarios = ("time_pressure", "heterogeneity_pressure")
    seeds = (8, 9, 10)
    actual_transitions = sum(
        int(row["actual_extra_transitions"]) for row in context_rows
    )
    identity_passed = (
        len(identity_rows) == 72
        and all(_as_bool(row["matched"]) for row in identity_rows)
        and max(
            (
                float(row.get("maximum_probability_difference", 0.0))
                for row in identity_rows
            ),
            default=float("inf"),
        )
        <= config.identity_probability_tolerance
    )
    intervention_passed = bool(integrity_rows) and all(
        _as_bool(row["current_step_identity"])
        and _as_bool(row["intervention_unique"])
        for row in integrity_rows
    )
    integrity_gates = {
        "context_count": len(context_rows) == 72,
        "context_identity": identity_passed,
        "repeat_count": all(
            int(row["repeat_count"]) == config.repeats
            for row in context_rows
        ),
        "intervention_integrity": intervention_passed,
        "actor_frozen": maximum_actor_parameter_difference == 0.0,
        "transition_budget": (
            actual_transitions <= config.maximum_extra_transitions
        ),
        "software_regression": software_tests_passed,
    }

    time_resource = _select(
        context_rows, scenario="time_pressure", slot="resource"
    )
    nonpositive_cost = [
        row
        for row in time_resource
        if float(row["total_cost_difference_mean"]) <= 0.0
    ]
    explained = [
        row
        for row in nonpositive_cost
        if float(row["sub_cost_mean"]) > 0.0
        or float(row["future_cost_composition_advantage_mean"]) > 0.0
    ]
    pr1_details = {
        "time_resource_contexts": len(time_resource),
        "positive_mean_sub_shot": sum(
            float(row["sub_shot_mean"]) > 0.0 for row in time_resource
        ),
        "positive_lower_sub_shot": sum(
            float(row["sub_shot_lower"]) > 0.0 for row in time_resource
        ),
        "nonpositive_total_cost_contexts": len(nonpositive_cost),
        "explained_nonpositive_cost_contexts": len(explained),
        "explained_fraction": (
            len(explained) / len(nonpositive_cost)
            if nonpositive_cost
            else 1.0
        ),
        "maximum_decomposition_error": max(
            (
                abs(float(row["maximum_decomposition_residual"]))
                for row in context_rows
            ),
            default=float("inf"),
        ),
    }
    pr1 = (
        len(time_resource) == 18
        and pr1_details["positive_mean_sub_shot"] >= 12
        and pr1_details["positive_lower_sub_shot"] >= 6
        and pr1_details["explained_fraction"] >= 0.80
        and pr1_details["maximum_decomposition_error"]
        <= config.decomposition_tolerance
    )

    resource_counts: dict[str, int] = {}
    safety_counts: dict[str, int] = {}
    block_resource_counts: dict[str, int] = {}
    attribution_fractions: dict[str, float] = {}
    for scenario in scenarios:
        resource = _select(context_rows, scenario=scenario, slot="resource")
        safety = _select(context_rows, scenario=scenario, slot="safety")
        resource_reliable = [
            row for row in resource if _as_bool(row["reliable_opportunity"])
        ]
        resource_counts[scenario] = len(resource_reliable)
        safety_counts[scenario] = sum(
            _as_bool(row["reliable_opportunity"]) for row in safety
        )
        attribution_fractions[scenario] = (
            float(
                np.mean(
                    [
                        float(row["reuse_probe_mean"]) > 0.0
                        or float(row["option_edge_mean"]) > 0.0
                        for row in resource_reliable
                    ]
                )
            )
            if resource_reliable
            else 0.0
        )
        for seed in seeds:
            key = f"{scenario}/seed{seed}"
            block_resource_counts[key] = sum(
                _as_bool(row["reliable_opportunity"])
                for row in resource
                if int(row["policy_seed"]) == seed
            )
    pr2 = (
        all(resource_counts[scenario] >= 6 for scenario in scenarios)
        and all(value >= 2 for value in block_resource_counts.values())
        and all(
            attribution_fractions[scenario] >= 0.80
            for scenario in scenarios
        )
    )

    reliable_resource_types = sorted(
        {
            str(row["unit_type"])
            for row in context_rows
            if str(row["slot"]) == "resource"
            and _as_bool(row["reliable_opportunity"])
        }
    )
    transition_efficiency: dict[str, dict[str, float]] = {}
    for scenario in scenarios:
        transition_efficiency[scenario] = {}
        for slot in ("safety", "resource"):
            rows = _select(context_rows, scenario=scenario, slot=slot)
            transition_efficiency[scenario][slot] = (
                sum(_as_bool(row["reliable_opportunity"]) for row in rows)
                / max(
                    1,
                    sum(int(row["actual_extra_transitions"]) for row in rows),
                )
            )
    pr3 = (
        all(
            resource_counts[scenario] > safety_counts[scenario]
            for scenario in scenarios
        )
        and all(
            transition_efficiency[scenario]["resource"]
            >= transition_efficiency[scenario]["safety"]
            for scenario in scenarios
        )
        and min(block_resource_counts.values(), default=0) > 0
        and len(reliable_resource_types) >= 2
    )
    mechanism_gates = {
        "P-R1_action_substitution": pr1,
        "P-R2_opportunity_value": pr2,
        "P-R3_resource_criticality": pr3,
    }
    all_integrity = all(integrity_gates.values())
    all_mechanism = all(mechanism_gates.values())
    if not all_integrity:
        decision = "invalidate_results_intervention_or_integrity_failure"
    elif all_mechanism:
        decision = "allow_independent_opportunity_oracle_predictability_audit"
    elif not pr1:
        decision = "retract_or_narrow_action_substitution_explanation"
    elif (
        resource_counts["heterogeneity_pressure"] >= 6
        and resource_counts["time_pressure"] < 6
    ):
        decision = "conditional_heterogeneity_only_opportunity_value"
    else:
        decision = "retain_substitution_stop_general_opportunity_route"
    return {
        "context_count": len(context_rows),
        "integrity_rows": len(integrity_rows),
        "actual_extra_transitions": actual_transitions,
        "maximum_actor_parameter_difference": (
            maximum_actor_parameter_difference
        ),
        "integrity_gates": integrity_gates,
        "P-R1": pr1_details,
        "reliable_resource_contexts": resource_counts,
        "reliable_safety_contexts": safety_counts,
        "resource_block_counts": block_resource_counts,
        "resource_attribution_fractions": attribution_fractions,
        "transition_efficiency": transition_efficiency,
        "reliable_resource_unit_types": reliable_resource_types,
        "mechanism_gates": mechanism_gates,
        "stage_passed": all_integrity and all_mechanism,
        "decision": decision,
    }


def _aggregate_context(
    *,
    context: BPCEAuditContext,
    repeat_rows: Sequence[Mapping[str, Any]],
    total_transitions: int,
    config: ActionSubstitutionOpportunityCostConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "context_id": context.context_id,
        "scenario": context.scenario,
        "policy_seed": context.policy_seed,
        "slot": context.slot,
        "environment_seed": context.environment_seed,
        "environment_step": context.environment_step,
        "unit_index": context.unit_index,
        "unit_type": context.snapshot.defense_units[
            context.unit_index
        ].resource_type,
        "unit_cost": context.snapshot.defense_units[
            context.unit_index
        ].cost,
        "legal_targets": ",".join(map(str, context.legal_targets)),
        "target_probabilities": ",".join(
            f"{value:.12g}" for value in context.target_probabilities
        ),
        "repeat_count": len(repeat_rows),
        "actual_extra_transitions": total_transitions,
    }
    metrics = (
        "sub_shot",
        "sub_cost",
        "future_cost_composition_advantage",
        "immediate_cost_difference",
        "total_cost_difference",
        "reuse_probe",
        "reuse_total",
        "option_edge",
        "option_threat",
        "ammo_gain_damage",
        "ammo_gain_leaks",
    )
    intervals: dict[str, dict[str, float]] = {}
    for metric in metrics:
        interval = component_interval(
            [float(row[metric]) for row in repeat_rows],
            confidence_z=config.confidence_z,
        )
        intervals[metric] = interval
        for field, value in interval.items():
            result[f"{metric}_{field}"] = value
    result["maximum_decomposition_residual"] = max(
        abs(float(row["decomposition_residual"])) for row in repeat_rows
    )
    result["reliable_opportunity"] = reliable_positive_opportunity(
        intervals["ammo_gain_damage"],
        intervals["ammo_gain_leaks"],
        mean_reuse_probe=intervals["reuse_probe"]["mean"],
        mean_option_edge=intervals["option_edge"]["mean"],
        config=config,
    )
    return result


@torch.no_grad()
def _rollout_branch(
    *,
    env: AirDefenseResourceAssignmentEnvV1,
    snapshot: AirDefenseV1StateSnapshot,
    environment_tape: np.ndarray,
    first_action: np.ndarray,
    policy: Any,
    policy_tape: np.ndarray,
    high_threat_threshold: float,
    restore_unit_index: int | None,
) -> OpportunityBranchTrace:
    env.restore_state(snapshot)
    env.set_hit_random_tape(environment_tape)
    pre_action_ammo = (
        snapshot.defense_units[restore_unit_index].ammo
        if restore_unit_index is not None
        else 0
    )
    observation, reward, terminated, truncated, info = env.step(first_action)
    current_observation = observation.copy()
    current_action_mask = env.action_masks().copy()
    current_snapshot = env.snapshot_state()
    current_cost = -float(info["reward_breakdown"]["cost"])
    opportunity_not_observable = bool(terminated or truncated)
    restored_ammo = 0
    if restore_unit_index is not None and not opportunity_not_observable:
        observation, _, restored_ammo = restore_probed_ammo(
            env,
            unit_index=restore_unit_index,
            pre_action_ammo=pre_action_ammo,
        )
    continuation_snapshot = env.snapshot_state()

    unit_shots = np.zeros(env.num_defense_units, dtype=np.float64)
    unit_costs = np.zeros(env.num_defense_units, dtype=np.float64)
    cumulative_shots: list[float] = []
    future_resource_cost = 0.0
    legal_edge_sum = 0.0
    threat_coverable_sum = 0.0
    transitions = 1
    device = next(policy.parameters()).device
    while not (terminated or truncated):
        action_mask = env.action_mask()
        legal_edge_sum += float(action_mask[:, : env.num_targets].sum())
        coverable = np.any(
            action_mask[:, : env.num_targets].astype(bool), axis=0
        )
        threat_coverable_sum += float(
            sum(
                target.alive
                and target.threat >= high_threat_threshold
                and bool(coverable[index])
                for index, target in enumerate(env.targets)
            )
        )
        observation_tensor = torch.as_tensor(
            observation[None, :], device=device, dtype=torch.float32
        )
        mask_tensor = torch.as_tensor(
            action_mask.reshape(1, -1), device=device
        )
        distribution = policy.get_distribution(
            observation_tensor, action_masks=mask_tensor
        )
        fixed = torch.full(
            (1, distribution.num_units),
            -1,
            device=device,
            dtype=torch.long,
        )
        action = sample_fixed_actions_with_uniforms(
            distribution,
            fixed,
            policy_tape[env.current_step][None, :],
        )[0].detach().cpu().numpy()
        observation, _, terminated, truncated, future_info = env.step(action)
        transitions += 1
        step_cost = -float(future_info["reward_breakdown"]["cost"])
        future_resource_cost += step_cost
        for unit_result in future_info["unit_results"]:
            if (
                unit_result["action_type"] == "engage"
                and bool(unit_result["legal"])
            ):
                unit_index = int(unit_result["unit_index"])
                unit_shots[unit_index] += 1.0
                unit_costs[unit_index] += env.defense_units[unit_index].cost
        cumulative_shots.append(float(unit_shots.sum()))
    return OpportunityBranchTrace(
        current_reward=float(reward),
        current_resource_cost=current_cost,
        current_observation=current_observation,
        current_action_mask=current_action_mask,
        current_info=deepcopy(info),
        current_snapshot=current_snapshot,
        continuation_snapshot=continuation_snapshot,
        future_total_shots=float(unit_shots.sum()),
        future_shots_by_unit=tuple(unit_shots.tolist()),
        future_cost_by_unit=tuple(unit_costs.tolist()),
        future_resource_cost=future_resource_cost,
        legal_edge_sum=legal_edge_sum,
        high_threat_coverable_sum=threat_coverable_sum,
        future_cumulative_shots=tuple(cumulative_shots),
        final_zone_damage=env.total_damage,
        final_high_threat_leaks=float(
            sum(
                target.status == "leaked"
                and target.threat >= high_threat_threshold
                for target in env.targets
            )
        ),
        total_resource_cost=current_cost + future_resource_cost,
        transitions=transitions,
        opportunity_not_observable=opportunity_not_observable,
        restored_ammo=restored_ammo,
    )


def _sample_first_action(
    distribution: Any,
    original_action: torch.Tensor,
    *,
    unit_index: int,
    selected_action: int,
    uniforms: np.ndarray,
) -> np.ndarray:
    fixed = make_bpce_fixed_prefix(
        distribution,
        original_action,
        unit_index=unit_index,
        selected_action=selected_action,
    )
    return (
        sample_fixed_actions_with_uniforms(
            distribution, fixed, uniforms[None, :]
        )[0]
        .detach()
        .cpu()
        .numpy()
    )


def _current_steps_equal(
    first: OpportunityBranchTrace, second: OpportunityBranchTrace
) -> bool:
    return bool(
        first.current_reward == second.current_reward
        and first.current_resource_cost == second.current_resource_cost
        and np.array_equal(
            first.current_observation, second.current_observation
        )
        and np.array_equal(
            first.current_action_mask, second.current_action_mask
        )
        and _values_equal(first.current_info, second.current_info)
        and _values_equal(first.current_snapshot, second.current_snapshot)
    )


def _values_equal(first: Any, second: Any) -> bool:
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return bool(np.array_equal(np.asarray(first), np.asarray(second)))
    if hasattr(first, "__dataclass_fields__") and hasattr(
        second, "__dataclass_fields__"
    ):
        return all(
            _values_equal(getattr(first, name), getattr(second, name))
            for name in first.__dataclass_fields__
        )
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        return first.keys() == second.keys() and all(
            _values_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (tuple, list)) and isinstance(second, (tuple, list)):
        return len(first) == len(second) and all(
            _values_equal(left, right)
            for left, right in zip(first, second)
        )
    return bool(first == second)


def _first_substitution_time(
    noop_cumulative: Sequence[float], engage_cumulative: Sequence[float]
) -> int | None:
    length = max(len(noop_cumulative), len(engage_cumulative))
    if length == 0:
        return None
    for index in range(length):
        noop = (
            noop_cumulative[index]
            if index < len(noop_cumulative)
            else (noop_cumulative[-1] if noop_cumulative else 0.0)
        )
        engage = (
            engage_cumulative[index]
            if index < len(engage_cumulative)
            else (engage_cumulative[-1] if engage_cumulative else 0.0)
        )
        if noop > engage:
            return index + 1
    return None


def _select(
    rows: Sequence[Mapping[str, Any]], *, scenario: str, slot: str
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if str(row["scenario"]) == scenario and str(row["slot"]) == slot
    ]


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)
