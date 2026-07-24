from __future__ import annotations

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
    BranchOutcome,
    make_bpce_fixed_prefix,
    make_bpce_paired_random_tapes,
    sample_fixed_actions_with_uniforms,
)


@dataclass(frozen=True)
class BPCEShortHorizonConfig:
    repeats: int = 32
    confidence_z: float = 1.96
    damage_equivalence: float = 0.05
    high_threat_equivalence: float = 0.10
    high_threat_threshold: float = 0.8
    branch_base_seed: int = 983_000
    identity_probability_tolerance: float = 1e-9
    maximum_extra_transitions: int = 266_198

    def __post_init__(self) -> None:
        if self.repeats <= 1:
            raise ValueError("repeats must be greater than one")
        if self.confidence_z <= 0.0:
            raise ValueError("confidence_z must be positive")
        if self.damage_equivalence < 0.0:
            raise ValueError("damage_equivalence must be non-negative")
        if self.high_threat_equivalence < 0.0:
            raise ValueError("high_threat_equivalence must be non-negative")
        if not 0.0 <= self.high_threat_threshold <= 1.0:
            raise ValueError("high_threat_threshold must be in [0, 1]")
        if self.identity_probability_tolerance < 0.0:
            raise ValueError(
                "identity_probability_tolerance must be non-negative"
            )
        if self.maximum_extra_transitions <= 0:
            raise ValueError("maximum_extra_transitions must be positive")


@dataclass(frozen=True)
class BranchTrajectory:
    outcomes: tuple[BranchOutcome, ...]

    def __post_init__(self) -> None:
        if not self.outcomes:
            raise ValueError("A branch trajectory must contain an outcome")

    @property
    def transitions(self) -> int:
        return len(self.outcomes)

    @property
    def final(self) -> BranchOutcome:
        return self.outcomes[-1]

    def at_horizon(self, horizon: int) -> BranchOutcome:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        return self.outcomes[min(horizon, len(self.outcomes)) - 1]


def component_interval(
    values: Sequence[float],
    *,
    confidence_z: float,
) -> dict[str, float]:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or samples.size <= 1:
        raise ValueError("values must contain at least two scalar samples")
    mean = float(samples.mean())
    standard_error = float(samples.std(ddof=1) / np.sqrt(samples.size))
    return {
        "mean": mean,
        "standard_error": standard_error,
        "lower": mean - confidence_z * standard_error,
        "upper": mean + confidence_z * standard_error,
    }


def classify_component_label(
    *,
    damage: Mapping[str, float],
    high_threat_leaks: Mapping[str, float],
    resource_cost: Mapping[str, float],
    config: BPCEShortHorizonConfig,
) -> str:
    engage = (
        damage["upper"] < -config.damage_equivalence
        and high_threat_leaks["upper"]
        <= config.high_threat_equivalence
    ) or (
        high_threat_leaks["upper"]
        < -config.high_threat_equivalence
        and damage["upper"] <= config.damage_equivalence
    )
    stop = (
        damage["lower"] >= -config.damage_equivalence
        and high_threat_leaks["lower"]
        >= -config.high_threat_equivalence
        and resource_cost["lower"] > 0.0
    )
    if engage:
        return "ENGAGE"
    if stop:
        return "STOP"
    return "AMBIGUOUS"


def target_event_horizon(
    snapshot: AirDefenseV1StateSnapshot,
    target_index: int,
    *,
    max_steps: int,
) -> int:
    remaining_steps = max(1, max_steps - snapshot.current_step)
    time_to_impact = snapshot.targets[target_index].time_to_impact
    return int(
        min(
            remaining_steps,
            max(1, int(np.ceil(time_to_impact)) + 1),
        )
    )


def validate_context_identity(
    contexts: Sequence[BPCEAuditContext],
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    probability_tolerance: float,
) -> list[dict[str, Any]]:
    references = {str(row["context_id"]): row for row in reference_rows}
    rebuilt = {context.context_id: context for context in contexts}
    all_ids = sorted(set(references) | set(rebuilt))
    results: list[dict[str, Any]] = []
    for context_id in all_ids:
        reference = references.get(context_id)
        context = rebuilt.get(context_id)
        row: dict[str, Any] = {
            "context_id": context_id,
            "present_in_reference": reference is not None,
            "present_in_rebuild": context is not None,
        }
        if reference is None or context is None:
            row["matched"] = False
            row["mismatch_fields"] = "missing_context"
            results.append(row)
            continue

        expected_targets = tuple(
            int(value)
            for value in str(reference["legal_targets"]).split(",")
            if value != ""
        )
        expected_probabilities = np.asarray(
            [
                float(value)
                for value in str(reference["target_probabilities"]).split(",")
                if value != ""
            ],
            dtype=np.float64,
        )
        actual_probabilities = np.asarray(
            context.target_probabilities, dtype=np.float64
        )
        probability_difference = (
            float(
                np.max(
                    np.abs(
                        expected_probabilities - actual_probabilities
                    )
                )
            )
            if expected_probabilities.shape == actual_probabilities.shape
            else float("inf")
        )
        checks = {
            "scenario": context.scenario == str(reference["scenario"]),
            "policy_seed": context.policy_seed
            == int(reference["policy_seed"]),
            "environment_seed": context.environment_seed
            == int(reference["environment_seed"]),
            "environment_step": context.environment_step
            == int(reference["environment_step"]),
            "unit_index": context.unit_index
            == int(reference["unit_index"]),
            "observation_hash": context.observation_hash
            == str(reference["observation_hash"]),
            "legal_targets": context.legal_targets == expected_targets,
            "target_probabilities": probability_difference
            <= probability_tolerance,
        }
        mismatches = [name for name, passed in checks.items() if not passed]
        row.update(
            {
                "scenario": context.scenario,
                "policy_seed": context.policy_seed,
                "slot": context.slot,
                "maximum_probability_difference": probability_difference,
                "matched": not mismatches,
                "mismatch_fields": ",".join(mismatches),
            }
        )
        results.append(row)
    return results


@torch.no_grad()
def audit_short_horizon_context(
    *,
    policy: Any,
    env_config: AirDefenseV1EnvConfig,
    context: BPCEAuditContext,
    config: BPCEShortHorizonConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
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
    horizons = {
        target: target_event_horizon(
            context.snapshot,
            target,
            max_steps=env_config.max_steps,
        )
        for target in context.legal_targets
    }
    target_probabilities = np.asarray(
        context.target_probabilities, dtype=np.float64
    )
    probe_env = AirDefenseResourceAssignmentEnvV1(config=env_config)
    probe_env.reset(seed=config.branch_base_seed)

    repeat_rows: list[dict[str, Any]] = []
    total_extra_transitions = 0
    projected_window_transitions = 0
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
        noop_trajectory = _rollout_stochastic_trajectory(
            env=probe_env,
            snapshot=context.snapshot,
            environment_tape=environment_tape,
            first_action=noop_action,
            policy=policy,
            policy_tape=policy_tape,
            config=config,
        )
        engage_trajectories: list[BranchTrajectory] = []
        for target in context.legal_targets:
            engage_action = _sample_first_action(
                distribution,
                original_action,
                unit_index=context.unit_index,
                selected_action=target,
                uniforms=policy_tape[context.environment_step],
            )
            engage_trajectories.append(
                _rollout_stochastic_trajectory(
                    env=probe_env,
                    snapshot=context.snapshot,
                    environment_tape=environment_tape,
                    first_action=engage_action,
                    policy=policy,
                    policy_tape=policy_tape,
                    config=config,
                )
            )

        short_deltas = {
            component: 0.0
            for component in (
                "zone_damage",
                "high_threat_leaks",
                "resource_cost",
            )
        }
        full_deltas = dict(short_deltas)
        for probability, target, engage_trajectory in zip(
            target_probabilities,
            context.legal_targets,
            engage_trajectories,
        ):
            horizon = horizons[target]
            engage_short = engage_trajectory.at_horizon(horizon)
            noop_short = noop_trajectory.at_horizon(horizon)
            for component in short_deltas:
                short_deltas[component] += probability * (
                    float(getattr(engage_short, component))
                    - float(getattr(noop_short, component))
                )
                full_deltas[component] += probability * (
                    float(getattr(engage_trajectory.final, component))
                    - float(getattr(noop_trajectory.final, component))
                )

        repeat_rows.append(
            {
                "context_id": context.context_id,
                "scenario": context.scenario,
                "policy_seed": context.policy_seed,
                "slot": context.slot,
                "repeat": repeat,
                **{
                    f"short_{component}_delta": value
                    for component, value in short_deltas.items()
                },
                **{
                    f"full_{component}_delta": value
                    for component, value in full_deltas.items()
                },
            }
        )
        total_extra_transitions += noop_trajectory.transitions + sum(
            trajectory.transitions for trajectory in engage_trajectories
        )
        maximum_horizon = max(horizons.values())
        projected_window_transitions += min(
            maximum_horizon, noop_trajectory.transitions
        ) + sum(
            min(horizons[target], trajectory.transitions)
            for target, trajectory in zip(
                context.legal_targets, engage_trajectories
            )
        )

    probe_env.close()
    aggregate: dict[str, Any] = {
        "context_id": context.context_id,
        "scenario": context.scenario,
        "policy_seed": context.policy_seed,
        "slot": context.slot,
        "environment_seed": context.environment_seed,
        "environment_step": context.environment_step,
        "unit_index": context.unit_index,
        "observation_hash": context.observation_hash,
        "legal_targets": ",".join(map(str, context.legal_targets)),
        "target_probabilities": ",".join(
            f"{value:.12g}" for value in context.target_probabilities
        ),
        "minimum_horizon": min(horizons.values()),
        "maximum_horizon": max(horizons.values()),
        "mean_horizon": float(np.mean(tuple(horizons.values()))),
        "extra_transitions": total_extra_transitions,
        "projected_window_transitions": projected_window_transitions,
        "projected_saved_transitions": (
            total_extra_transitions - projected_window_transitions
        ),
        "repeat_count": len(repeat_rows),
    }
    for scope in ("short", "full"):
        intervals: dict[str, dict[str, float]] = {}
        for component in (
            "zone_damage",
            "high_threat_leaks",
            "resource_cost",
        ):
            interval = component_interval(
                [
                    float(row[f"{scope}_{component}_delta"])
                    for row in repeat_rows
                ],
                confidence_z=config.confidence_z,
            )
            intervals[component] = interval
            for field, value in interval.items():
                aggregate[f"{scope}_{component}_{field}"] = value
        aggregate[f"{scope}_label"] = classify_component_label(
            damage=intervals["zone_damage"],
            high_threat_leaks=intervals["high_threat_leaks"],
            resource_cost=intervals["resource_cost"],
            config=config,
        )
    aggregate["label_changed"] = (
        aggregate["short_label"] != aggregate["full_label"]
    )
    horizon_rows = [
        {
            "context_id": context.context_id,
            "scenario": context.scenario,
            "policy_seed": context.policy_seed,
            "slot": context.slot,
            "target_index": target,
            "target_probability": probability,
            "time_to_impact": context.snapshot.targets[
                target
            ].time_to_impact,
            "event_horizon": horizons[target],
        }
        for target, probability in zip(
            context.legal_targets, context.target_probabilities
        )
    ]
    return aggregate, repeat_rows, horizon_rows


def summarize_short_horizon_audit(
    context_rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
    *,
    config: BPCEShortHorizonConfig,
    maximum_actor_parameter_difference: float,
    software_tests_passed: bool,
) -> dict[str, Any]:
    if not context_rows:
        raise ValueError("context_rows must not be empty")
    scenarios = sorted({str(row["scenario"]) for row in context_rows})
    blocks = sorted(
        {
            (str(row["scenario"]), int(row["policy_seed"]))
            for row in context_rows
        }
    )

    def counts(rows: Sequence[Mapping[str, Any]], scope: str) -> dict[str, int]:
        return {
            label: sum(str(row[f"{scope}_label"]) == label for row in rows)
            for label in ("ENGAGE", "STOP", "AMBIGUOUS")
        }

    block_counts: dict[str, dict[str, int]] = {}
    for scenario, seed in blocks:
        selected = [
            row
            for row in context_rows
            if str(row["scenario"]) == scenario
            and int(row["policy_seed"]) == seed
        ]
        block_counts[f"{scenario}/seed{seed}"] = counts(selected, "short")
    scenario_counts = {
        scenario: counts(
            [
                row
                for row in context_rows
                if str(row["scenario"]) == scenario
            ],
            "short",
        )
        for scenario in scenarios
    }
    slot_counts = {
        f"{scenario}/{slot}": counts(
            [
                row
                for row in context_rows
                if str(row["scenario"]) == scenario
                and str(row["slot"]) == slot
            ],
            "short",
        )
        for scenario in scenarios
        for slot in ("safety", "resource")
    }
    total_counts = counts(context_rows, "short")
    full_counts = counts(context_rows, "full")
    actionable = total_counts["ENGAGE"] + total_counts["STOP"]
    total_transitions = sum(
        int(row["extra_transitions"]) for row in context_rows
    )
    projected_transitions = sum(
        int(row["projected_window_transitions"]) for row in context_rows
    )
    engage_rows = [
        row for row in context_rows if str(row["short_label"]) == "ENGAGE"
    ]
    engage_consistent = all(
        (
            float(row["short_zone_damage_upper"])
            <= config.damage_equivalence
            and float(row["short_high_threat_leaks_upper"])
            <= config.high_threat_equivalence
        )
        for row in engage_rows
    )
    identity_passed = (
        len(identity_rows) == 72
        and all(_as_bool(row["matched"]) for row in identity_rows)
    )
    confusion = {
        short_label: {
            full_label: sum(
                str(row["short_label"]) == short_label
                and str(row["full_label"]) == full_label
                for row in context_rows
            )
            for full_label in ("ENGAGE", "STOP", "AMBIGUOUS")
        }
        for short_label in ("ENGAGE", "STOP", "AMBIGUOUS")
    }
    gates = {
        "data_integrity": (
            len(context_rows) == 72
            and all(
                int(row["repeat_count"]) == config.repeats
                for row in context_rows
            )
        ),
        "context_identity": identity_passed,
        "overall_actionable_labels": actionable >= 48,
        "block_power": all(
            value["ENGAGE"] + value["STOP"] >= 6
            for value in block_counts.values()
        ),
        "scenario_bidirectional_coverage": all(
            value["ENGAGE"] >= 6 and value["STOP"] >= 6
            for value in scenario_counts.values()
        ),
        "cross_seed_bidirectional_coverage": all(
            value["ENGAGE"] >= 2 and value["STOP"] >= 2
            for value in block_counts.values()
        ),
        "engage_safety_consistency": engage_consistent,
        "actor_frozen": maximum_actor_parameter_difference == 0.0,
        "transition_budget": (
            total_transitions <= config.maximum_extra_transitions
        ),
        "software_regression": software_tests_passed,
    }
    return {
        "context_count": len(context_rows),
        "short_label_counts": total_counts,
        "full_label_counts": full_counts,
        "block_counts": block_counts,
        "scenario_counts": scenario_counts,
        "slot_counts": slot_counts,
        "short_full_confusion": confusion,
        "label_changed_count": sum(
            _as_bool(row["label_changed"]) for row in context_rows
        ),
        "extra_transitions": total_transitions,
        "projected_window_transitions": projected_transitions,
        "projected_saved_transitions": (
            total_transitions - projected_transitions
        ),
        "actionable_labels_per_1000_transitions": (
            1000.0 * actionable / max(1, total_transitions)
        ),
        "safety_slot_engage_rate": _slot_rate(
            context_rows, slot="safety", label="ENGAGE"
        ),
        "resource_slot_stop_rate": _slot_rate(
            context_rows, slot="resource", label="STOP"
        ),
        "maximum_actor_parameter_difference": (
            maximum_actor_parameter_difference
        ),
        "gates": gates,
        "stage_a2_passed": all(gates.values()),
        "decision": (
            "proceed_to_auxiliary_dose_audit"
            if all(gates.values())
            else "pause_bpce_online_auxiliary"
        ),
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _slot_rate(
    rows: Sequence[Mapping[str, Any]],
    *,
    slot: str,
    label: str,
) -> float:
    selected = [row for row in rows if str(row["slot"]) == slot]
    if not selected:
        return 0.0
    return float(
        np.mean([str(row["short_label"]) == label for row in selected])
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
    sampled = sample_fixed_actions_with_uniforms(
        distribution, fixed, uniforms[None, :]
    )
    return sampled[0].detach().cpu().numpy()


@torch.no_grad()
def _rollout_stochastic_trajectory(
    *,
    env: AirDefenseResourceAssignmentEnvV1,
    snapshot: AirDefenseV1StateSnapshot,
    environment_tape: np.ndarray,
    first_action: np.ndarray,
    policy: Any,
    policy_tape: np.ndarray,
    config: BPCEShortHorizonConfig,
) -> BranchTrajectory:
    env.restore_state(snapshot)
    env.set_hit_random_tape(environment_tape)
    observation, reward, terminated, truncated, info = env.step(first_action)
    total_return = float(reward)
    resource_cost = -float(info["reward_breakdown"]["cost"])
    outcomes = [
        _current_outcome(
            env,
            total_return=total_return,
            resource_cost=resource_cost,
            high_threat_threshold=config.high_threat_threshold,
        )
    ]
    device = next(policy.parameters()).device
    while not (terminated or truncated):
        observation_tensor = torch.as_tensor(
            observation[None, :], device=device, dtype=torch.float32
        )
        action_mask = torch.as_tensor(
            env.action_masks()[None, :], device=device
        )
        distribution = policy.get_distribution(
            observation_tensor, action_masks=action_mask
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
        )[0]
        observation, reward, terminated, truncated, info = env.step(
            action.detach().cpu().numpy()
        )
        total_return += float(reward)
        resource_cost -= float(info["reward_breakdown"]["cost"])
        outcomes.append(
            _current_outcome(
                env,
                total_return=total_return,
                resource_cost=resource_cost,
                high_threat_threshold=config.high_threat_threshold,
            )
        )
    return BranchTrajectory(tuple(outcomes))


def _current_outcome(
    env: AirDefenseResourceAssignmentEnvV1,
    *,
    total_return: float,
    resource_cost: float,
    high_threat_threshold: float,
) -> BranchOutcome:
    high_threat_leaks = sum(
        target.status == "leaked"
        and target.threat >= high_threat_threshold
        for target in env.targets
    )
    return BranchOutcome(
        total_return=total_return,
        zone_damage=env.total_damage,
        high_threat_leaks=float(high_threat_leaks),
        resource_cost=resource_cost,
        transitions=1,
    )
