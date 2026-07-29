from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

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
from .future_coverability import future_coverability_externality


@dataclass(frozen=True)
class FCRCPredictiveValidationConfig:
    repeats: int = 64
    branch_base_seed: int = 1_493_000
    confidence_z: float = 1.96

    def __post_init__(self) -> None:
        if self.repeats <= 1:
            raise ValueError("repeats must be greater than one")
        if self.branch_base_seed < 0:
            raise ValueError("branch_base_seed must be non-negative")
        if self.confidence_z <= 0.0:
            raise ValueError("confidence_z must be positive")


@dataclass(frozen=True)
class FCRCCandidate:
    target_index: int
    externality: float
    coverability_before: float
    coverability_after: float


@dataclass(frozen=True)
class FCRCBranchTrace:
    target_statuses: tuple[str, ...]
    leaked_damage: tuple[float, ...]
    transitions: int


def select_fcrc_candidate_pair(
    context: BPCEAuditContext,
) -> tuple[FCRCCandidate, FCRCCandidate]:
    """Return deterministic low/high FCRC targets for one frozen context."""

    if len(context.legal_targets) < 2:
        raise ValueError("a paired FCRC context needs at least two legal targets")
    candidates: list[FCRCCandidate] = []
    for target_index in context.legal_targets:
        certificate = future_coverability_externality(
            context.snapshot,
            unit_index=context.unit_index,
            target_index=target_index,
        )
        candidates.append(
            FCRCCandidate(
                target_index=target_index,
                externality=certificate.externality,
                coverability_before=(
                    certificate.other_threat_coverability_before
                ),
                coverability_after=certificate.other_threat_coverability_after,
            )
        )
    low = min(candidates, key=lambda item: (item.externality, item.target_index))
    high = min(candidates, key=lambda item: (-item.externality, item.target_index))
    if high.externality <= low.externality:
        raise ValueError("the paired FCRC context has no positive spread")
    return low, high


def other_threat_outcomes(
    trace: FCRCBranchTrace,
    snapshot: AirDefenseV1StateSnapshot,
    *,
    excluded_target_index: int,
) -> tuple[float, float]:
    """Return intercepted threat weight and leaked damage outside one target."""

    if len(trace.target_statuses) != len(snapshot.targets):
        raise ValueError("trace and snapshot target counts must match")
    intercepted_weight = 0.0
    leaked_damage = 0.0
    for target_index, (target, status, damage) in enumerate(
        zip(snapshot.targets, trace.target_statuses, trace.leaked_damage)
    ):
        if target_index == excluded_target_index or not target.alive:
            continue
        zone = snapshot.protected_zones[target.target_zone]
        if status == "intercepted":
            intercepted_weight += float(
                target.payload * target.threat * zone.value
            )
        leaked_damage += float(damage)
    return intercepted_weight, leaked_damage


def candidate_harm(
    *,
    engage: FCRCBranchTrace,
    noop: FCRCBranchTrace,
    snapshot: AirDefenseV1StateSnapshot,
    target_index: int,
) -> tuple[float, float]:
    """Measure the causal harm of engaging one target to all other targets."""

    engage_intercepted, engage_damage = other_threat_outcomes(
        engage,
        snapshot,
        excluded_target_index=target_index,
    )
    noop_intercepted, noop_damage = other_threat_outcomes(
        noop,
        snapshot,
        excluded_target_index=target_index,
    )
    return (
        noop_intercepted - engage_intercepted,
        engage_damage - noop_damage,
    )


def mean_interval(
    values: Sequence[float],
    *,
    confidence_z: float = 1.96,
) -> dict[str, float]:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or samples.size <= 1:
        raise ValueError("values must contain at least two scalar samples")
    mean = float(samples.mean())
    standard_error = float(samples.std(ddof=1) / np.sqrt(samples.size))
    return {
        "mean": mean,
        "standard_error": standard_error,
        "ci_lower": mean - confidence_z * standard_error,
        "ci_upper": mean + confidence_z * standard_error,
    }


@torch.no_grad()
def audit_fcrc_predictive_context(
    *,
    policy: Any,
    env_config: AirDefenseV1EnvConfig,
    context: BPCEAuditContext,
    config: FCRCPredictiveValidationConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the frozen no-op/high/low common-random-number comparison."""

    low, high = select_fcrc_candidate_pair(context)
    env = AirDefenseResourceAssignmentEnvV1(config=env_config)
    device = next(policy.parameters()).device
    observation = torch.as_tensor(
        context.observation[None, :],
        device=device,
        dtype=torch.float32,
    )
    action_mask = torch.as_tensor(
        context.action_mask[None, :],
        device=device,
    )
    distribution = policy.get_distribution(observation, action_masks=action_mask)
    original_action = torch.as_tensor(
        context.original_action,
        device=device,
        dtype=torch.long,
    )[None, :]

    repeat_rows: list[dict[str, Any]] = []
    harms: dict[str, dict[str, list[float]]] = {
        "low": {"intercept": [], "damage": []},
        "high": {"intercept": [], "damage": []},
    }
    total_transitions = 0
    for repeat in range(config.repeats):
        environment_tape, policy_tape = make_bpce_paired_random_tapes(
            context=context,
            repeat=repeat,
            env_config=env_config,
            branch_base_seed=config.branch_base_seed,
        )
        uniforms = policy_tape[context.environment_step][None, :]
        first_actions = {
            "noop": _first_action(
                distribution=distribution,
                original_action=original_action,
                unit_index=context.unit_index,
                selected_action=distribution.num_targets,
                uniforms=uniforms,
            ),
            "low": _first_action(
                distribution=distribution,
                original_action=original_action,
                unit_index=context.unit_index,
                selected_action=low.target_index,
                uniforms=uniforms,
            ),
            "high": _first_action(
                distribution=distribution,
                original_action=original_action,
                unit_index=context.unit_index,
                selected_action=high.target_index,
                uniforms=uniforms,
            ),
        }
        traces = {
            branch: _rollout_branch(
                env=env,
                snapshot=context.snapshot,
                environment_tape=environment_tape,
                first_action=first_action,
                policy=policy,
                policy_tape=policy_tape,
            )
            for branch, first_action in first_actions.items()
        }
        total_transitions += sum(trace.transitions for trace in traces.values())
        effects: dict[str, tuple[float, float]] = {}
        for label, candidate in (("low", low), ("high", high)):
            effects[label] = candidate_harm(
                engage=traces[label],
                noop=traces["noop"],
                snapshot=context.snapshot,
                target_index=candidate.target_index,
            )
            harms[label]["intercept"].append(effects[label][0])
            harms[label]["damage"].append(effects[label][1])
        repeat_rows.append(
            {
                "context_id": context.context_id,
                "scenario": context.scenario,
                "policy_seed": context.policy_seed,
                "repeat": repeat,
                "low_target_index": low.target_index,
                "high_target_index": high.target_index,
                "low_intercept_harm": effects["low"][0],
                "high_intercept_harm": effects["high"][0],
                "delta_intercept_harm": effects["high"][0] - effects["low"][0],
                "low_damage_harm": effects["low"][1],
                "high_damage_harm": effects["high"][1],
                "delta_damage_harm": effects["high"][1] - effects["low"][1],
                "noop_transitions": traces["noop"].transitions,
                "low_transitions": traces["low"].transitions,
                "high_transitions": traces["high"].transitions,
            }
        )
    env.close()

    candidate_rows: list[dict[str, Any]] = []
    for label, candidate in (("low", low), ("high", high)):
        intercept = mean_interval(
            harms[label]["intercept"],
            confidence_z=config.confidence_z,
        )
        damage = mean_interval(
            harms[label]["damage"],
            confidence_z=config.confidence_z,
        )
        candidate_rows.append(
            {
                "context_id": context.context_id,
                "scenario": context.scenario,
                "policy_seed": context.policy_seed,
                "candidate_label": label,
                "target_index": candidate.target_index,
                "fcrc": candidate.externality,
                "coverability_before": candidate.coverability_before,
                "coverability_after": candidate.coverability_after,
                "intercept_harm_mean": intercept["mean"],
                "intercept_harm_standard_error": intercept["standard_error"],
                "intercept_harm_ci_lower": intercept["ci_lower"],
                "intercept_harm_ci_upper": intercept["ci_upper"],
                "damage_harm_mean": damage["mean"],
                "damage_harm_standard_error": damage["standard_error"],
                "damage_harm_ci_lower": damage["ci_lower"],
                "damage_harm_ci_upper": damage["ci_upper"],
            }
        )

    delta_intercept = [
        row["delta_intercept_harm"] for row in repeat_rows
    ]
    delta_damage = [row["delta_damage_harm"] for row in repeat_rows]
    intercept_summary = mean_interval(
        delta_intercept,
        confidence_z=config.confidence_z,
    )
    damage_summary = mean_interval(
        delta_damage,
        confidence_z=config.confidence_z,
    )
    aggregate = {
        "context_id": context.context_id,
        "scenario": context.scenario,
        "policy_seed": context.policy_seed,
        "slot": context.slot,
        "environment_step": context.environment_step,
        "observation_hash": context.observation_hash,
        "unit_index": context.unit_index,
        "resource_type": context.snapshot.defense_units[
            context.unit_index
        ].resource_type,
        "legal_target_count": len(context.legal_targets),
        "low_target_index": low.target_index,
        "high_target_index": high.target_index,
        "low_fcrc": low.externality,
        "high_fcrc": high.externality,
        "fcrc_spread": high.externality - low.externality,
        "delta_intercept_harm_mean": intercept_summary["mean"],
        "delta_intercept_harm_standard_error": intercept_summary[
            "standard_error"
        ],
        "delta_intercept_harm_ci_lower": intercept_summary["ci_lower"],
        "delta_intercept_harm_ci_upper": intercept_summary["ci_upper"],
        "delta_damage_harm_mean": damage_summary["mean"],
        "delta_damage_harm_standard_error": damage_summary["standard_error"],
        "delta_damage_harm_ci_lower": damage_summary["ci_lower"],
        "delta_damage_harm_ci_upper": damage_summary["ci_upper"],
        "repeats": config.repeats,
        "transitions": total_transitions,
    }
    return aggregate, repeat_rows, candidate_rows


def _first_action(
    *,
    distribution: Any,
    original_action: torch.Tensor,
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
    sampled = sample_fixed_actions_with_uniforms(distribution, fixed, uniforms)
    return sampled[0].detach().cpu().numpy()


@torch.no_grad()
def _rollout_branch(
    *,
    env: AirDefenseResourceAssignmentEnvV1,
    snapshot: AirDefenseV1StateSnapshot,
    environment_tape: np.ndarray,
    first_action: np.ndarray,
    policy: Any,
    policy_tape: np.ndarray,
) -> FCRCBranchTrace:
    env.restore_state(snapshot)
    env.set_hit_random_tape(environment_tape)
    observation, _, terminated, truncated, _ = env.step(first_action)
    transitions = 1
    device = next(policy.parameters()).device
    while not (terminated or truncated):
        observation_tensor = torch.as_tensor(
            observation[None, :],
            device=device,
            dtype=torch.float32,
        )
        action_mask = torch.as_tensor(
            env.action_masks()[None, :],
            device=device,
        )
        distribution = policy.get_distribution(
            observation_tensor,
            action_masks=action_mask,
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
        observation, _, terminated, truncated, _ = env.step(
            action.detach().cpu().numpy()
        )
        transitions += 1
    return FCRCBranchTrace(
        target_statuses=tuple(target.status for target in env.targets),
        leaked_damage=tuple(float(target.leaked_damage) for target in env.targets),
        transitions=transitions,
    )
