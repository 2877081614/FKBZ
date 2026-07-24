from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Sequence

import numpy as np
import torch

from ..envs.air_defense_v1 import (
    AirDefenseResourceAssignmentEnvV1,
    AirDefenseV1EnvConfig,
    AirDefenseV1StateSnapshot,
)


@dataclass(frozen=True)
class BPCELabelSemanticsConfig:
    contexts_per_slot: int = 6
    pool_episodes: int = 12
    repeats: int = 32
    engagement_threshold: float = 0.5
    minimum_return_effect: float = 1.0
    confidence_z: float = 1.96
    high_threat_threshold: float = 0.8
    component_damage_tolerance: float = 0.05
    component_high_threat_tolerance: float = 0.10
    minimum_component_consistency: float = 0.80
    context_base_seed: int = 973_000
    branch_base_seed: int = 983_000

    def __post_init__(self) -> None:
        if self.contexts_per_slot <= 0:
            raise ValueError("contexts_per_slot must be positive")
        if self.pool_episodes <= 0:
            raise ValueError("pool_episodes must be positive")
        if self.repeats <= 1:
            raise ValueError("repeats must be greater than one")
        if not 0.0 <= self.engagement_threshold <= 1.0:
            raise ValueError("engagement_threshold must be in [0, 1]")
        if self.minimum_return_effect < 0.0:
            raise ValueError("minimum_return_effect must be non-negative")
        if self.confidence_z <= 0.0:
            raise ValueError("confidence_z must be positive")
        if not 0.0 <= self.high_threat_threshold <= 1.0:
            raise ValueError("high_threat_threshold must be in [0, 1]")


@dataclass(frozen=True)
class BPCEAuditContext:
    context_id: str
    scenario: str
    policy_seed: int
    slot: str
    episode_index: int
    environment_seed: int
    environment_step: int
    unit_index: int
    prefix_actions: tuple[int, ...]
    original_action: tuple[int, ...]
    observation_hash: str
    observation: np.ndarray
    action_mask: np.ndarray
    snapshot: AirDefenseV1StateSnapshot
    engage_probability: float
    engagement_margin: float
    legal_targets: tuple[int, ...]
    target_probabilities: tuple[float, ...]
    argmax_target: int
    safety_score: float
    resource_score: float


@dataclass(frozen=True)
class BranchOutcome:
    total_return: float
    zone_damage: float
    high_threat_leaks: float
    resource_cost: float
    transitions: int

    def minus(self, other: BranchOutcome) -> BranchOutcome:
        return BranchOutcome(
            total_return=self.total_return - other.total_return,
            zone_damage=self.zone_damage - other.zone_damage,
            high_threat_leaks=self.high_threat_leaks - other.high_threat_leaks,
            resource_cost=self.resource_cost - other.resource_cost,
            transitions=self.transitions + other.transitions,
        )


def estimate_effect(
    values: Sequence[float],
    *,
    minimum_return_effect: float,
    confidence_z: float,
) -> dict[str, float | int | bool]:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or samples.size <= 1:
        raise ValueError("values must contain at least two scalar samples")
    mean = float(samples.mean())
    standard_error = float(samples.std(ddof=1) / np.sqrt(samples.size))
    lower = mean - confidence_z * standard_error
    upper = mean + confidence_z * standard_error
    direction = int(np.sign(mean))
    reliable = bool(
        direction != 0
        and abs(mean) >= minimum_return_effect
        and (lower > 0.0 or upper < 0.0)
    )
    return {
        "mean": mean,
        "standard_error": standard_error,
        "ci_lower": lower,
        "ci_upper": upper,
        "sign": direction,
        "reliable": reliable,
    }


def sample_fixed_actions_with_uniforms(
    distribution: Any,
    fixed_actions: torch.Tensor,
    uniforms: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Complete a partially fixed autoregressive action with explicit uniforms."""

    fixed = fixed_actions.long().reshape(-1, distribution.num_units)
    if fixed.shape[0] != distribution.target_logits.shape[0]:
        raise ValueError("Fixed-action and distribution batches must match")
    uniform_tensor = torch.as_tensor(
        uniforms,
        device=fixed.device,
        dtype=distribution.target_logits.dtype,
    ).reshape(-1, distribution.num_units)
    if uniform_tensor.shape[0] != fixed.shape[0]:
        raise ValueError("Uniform and fixed-action batches must match")
    if bool(torch.any((uniform_tensor < 0.0) | (uniform_tensor >= 1.0))):
        raise ValueError("Uniform values must be in [0, 1)")

    actions = torch.empty_like(fixed)
    used_targets = torch.zeros(
        (fixed.shape[0], distribution.num_targets),
        device=fixed.device,
        dtype=torch.bool,
    )
    batch_indices = torch.arange(fixed.shape[0], device=fixed.device)
    for unit_index in distribution.unit_order:
        probabilities, mask = distribution._unit_probabilities(
            unit_index, used_targets
        )
        sampled = torch.sum(
            probabilities.cumsum(dim=1)
            <= uniform_tensor[:, unit_index, None],
            dim=1,
        ).clamp_max(distribution.num_actions - 1)
        requested = fixed[:, unit_index]
        action = torch.where(requested >= 0, requested, sampled)
        if not bool(torch.all(mask[batch_indices, action])):
            raise ValueError("A fixed action is illegal under its prefix mask")
        actions[:, unit_index] = action
        used_targets = distribution._add_selected_target(used_targets, action)
    return actions


@torch.no_grad()
def collect_bpce_audit_contexts(
    *,
    policy: Any,
    env_config: AirDefenseV1EnvConfig,
    scenario: str,
    policy_seed: int,
    config: BPCELabelSemanticsConfig,
) -> tuple[BPCEAuditContext, ...]:
    """Collect frozen-policy contexts without reading counterfactual outcomes."""

    env = AirDefenseResourceAssignmentEnvV1(config=env_config)
    device = next(policy.parameters()).device
    scenario_offset = int.from_bytes(
        sha256(scenario.encode("utf-8")).digest()[:4], "little"
    )
    candidates: list[dict[str, Any]] = []

    for episode_index in range(config.pool_episodes):
        environment_seed = (
            config.context_base_seed
            + scenario_offset
            + 10_000 * policy_seed
            + episode_index
        ) % (2**31 - 1)
        observation, _ = env.reset(seed=environment_seed)
        terminated = truncated = False
        while not (terminated or truncated):
            snapshot = env.snapshot_state()
            action_mask = env.action_masks().copy()
            observation_tensor = torch.as_tensor(
                observation[None, :], device=device, dtype=torch.float32
            )
            mask_tensor = torch.as_tensor(action_mask[None, :], device=device)
            distribution = policy.get_distribution(
                observation_tensor, action_masks=mask_tensor
            )
            original = distribution.sample_with_engagement_threshold(
                config.engagement_threshold
            ).actions
            probabilities, masks = distribution.conditional_probabilities(
                original
            )
            action_values = original[0].detach().cpu().numpy().astype(np.int64)
            probability_values = probabilities[0].detach().cpu().numpy()
            mask_values = masks[0].detach().cpu().numpy()
            prefix: list[int] = []
            max_cost = max(unit.cost for unit in snapshot.defense_units)
            base_mask = env.action_mask()[:, : env.num_targets].astype(bool)

            for unit_index in distribution.unit_order:
                legal_targets = np.flatnonzero(
                    mask_values[unit_index, : distribution.num_targets]
                )
                if legal_targets.size:
                    target_mass = probability_values[
                        unit_index, : distribution.num_targets
                    ]
                    engage_probability = float(target_mass.sum())
                    conditional = target_mass[legal_targets] / engage_probability
                    clipped = np.clip(engage_probability, 1e-8, 1.0 - 1e-8)
                    margin = float(np.log(clipped) - np.log1p(-clipped))
                    safety_score = max(
                        _target_safety_score(snapshot, int(target))
                        for target in legal_targets
                    )
                    unit = snapshot.defense_units[unit_index]
                    alternative_fraction = float(
                        np.mean(
                            [
                                bool(
                                    np.any(
                                        np.delete(
                                            base_mask[:, int(target)],
                                            unit_index,
                                        )
                                    )
                                )
                                for target in legal_targets
                            ]
                        )
                    )
                    resource_score = (
                        1.0 - unit.ammo / max(1, unit.max_ammo)
                        + unit.cost / max(1e-8, max_cost)
                        + alternative_fraction
                    )
                    observation_hash = sha256(
                        np.asarray(observation, dtype=np.float32).tobytes()
                    ).hexdigest()
                    candidates.append(
                        {
                            "scenario": scenario,
                            "policy_seed": policy_seed,
                            "episode_index": episode_index,
                            "environment_seed": environment_seed,
                            "environment_step": snapshot.current_step,
                            "unit_index": int(unit_index),
                            "prefix_actions": tuple(prefix),
                            "original_action": tuple(int(v) for v in action_values),
                            "observation_hash": observation_hash,
                            "observation": np.asarray(
                                observation, dtype=np.float32
                            ).copy(),
                            "action_mask": action_mask,
                            "snapshot": snapshot,
                            "engage_probability": engage_probability,
                            "engagement_margin": margin,
                            "legal_targets": tuple(
                                int(target) for target in legal_targets
                            ),
                            "target_probabilities": tuple(
                                float(value) for value in conditional
                            ),
                            "argmax_target": int(
                                legal_targets[int(np.argmax(conditional))]
                            ),
                            "safety_score": safety_score,
                            "resource_score": resource_score,
                        }
                    )
                prefix.append(int(action_values[unit_index]))

            observation, _, terminated, truncated, _ = env.step(action_values)

    env.close()
    required = 2 * config.contexts_per_slot
    if len(candidates) < required:
        raise RuntimeError(
            f"Only {len(candidates)} actionable contexts available; need {required}"
        )
    identity = lambda row: (
        row["episode_index"],
        row["environment_step"],
        row["unit_index"],
    )
    safety = sorted(
        candidates,
        key=lambda row: (
            -row["safety_score"],
            abs(row["engagement_margin"]),
            identity(row),
        ),
    )[: config.contexts_per_slot]
    selected_ids = {identity(row) for row in safety}
    resource_pool = [
        row for row in candidates if identity(row) not in selected_ids
    ]
    resource = sorted(
        resource_pool,
        key=lambda row: (
            -row["resource_score"],
            abs(row["engagement_margin"]),
            identity(row),
        ),
    )[: config.contexts_per_slot]
    if len(resource) != config.contexts_per_slot:
        raise RuntimeError("Not enough distinct resource-slot contexts")

    selected: list[BPCEAuditContext] = []
    for slot, rows in (("safety", safety), ("resource", resource)):
        for slot_index, row in enumerate(rows):
            context_id = (
                f"{scenario}_seed{policy_seed}_{slot}{slot_index:02d}_"
                f"e{row['episode_index']:02d}_t{row['environment_step']:02d}_"
                f"u{row['unit_index']}"
            )
            selected.append(
                BPCEAuditContext(
                    context_id=context_id,
                    slot=slot,
                    **row,
                )
            )
    return tuple(selected)


@torch.no_grad()
def audit_bpce_context(
    *,
    policy: Any,
    env_config: AirDefenseV1EnvConfig,
    context: BPCEAuditContext,
    config: BPCELabelSemanticsConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute labels A/B/C for one frozen context without parameter updates."""

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

    deterministic_actions = {
        target: _complete_first_action(
            distribution,
            original_action,
            unit_index=context.unit_index,
            selected_action=target,
            threshold=config.engagement_threshold,
        )
        for target in context.legal_targets
    }
    deterministic_noop = _complete_first_action(
        distribution,
        original_action,
        unit_index=context.unit_index,
        selected_action=distribution.noop_action,
        threshold=config.engagement_threshold,
    )
    target_probabilities = np.asarray(
        context.target_probabilities, dtype=np.float64
    )
    probe_env = AirDefenseResourceAssignmentEnvV1(config=env_config)
    probe_env.reset(seed=config.branch_base_seed)

    repeat_rows: list[dict[str, Any]] = []
    target_accumulator: dict[int, dict[str, list[float]]] = {
        target: {
            "det_total_return": [],
            "stochastic_total_return": [],
            "det_delta": [],
            "stochastic_delta": [],
        }
        for target in context.legal_targets
    }
    total_extra_transitions = 0
    for repeat in range(config.repeats):
        environment_tape, policy_tape = make_bpce_paired_random_tapes(
            context=context,
            repeat=repeat,
            env_config=env_config,
            branch_base_seed=config.branch_base_seed,
        )
        deterministic_noop_outcome = _rollout_branch(
            env=probe_env,
            snapshot=context.snapshot,
            environment_tape=environment_tape,
            first_action=deterministic_noop,
            policy=policy,
            deterministic=True,
            policy_tape=policy_tape,
            config=config,
        )
        stochastic_noop_action = _sample_first_action(
            distribution,
            original_action,
            unit_index=context.unit_index,
            selected_action=distribution.noop_action,
            uniforms=policy_tape[context.environment_step],
        )
        stochastic_noop_outcome = _rollout_branch(
            env=probe_env,
            snapshot=context.snapshot,
            environment_tape=environment_tape,
            first_action=stochastic_noop_action,
            policy=policy,
            deterministic=False,
            policy_tape=policy_tape,
            config=config,
        )

        deterministic_targets: list[BranchOutcome] = []
        stochastic_targets: list[BranchOutcome] = []
        for target in context.legal_targets:
            deterministic_outcome = _rollout_branch(
                env=probe_env,
                snapshot=context.snapshot,
                environment_tape=environment_tape,
                first_action=deterministic_actions[target],
                policy=policy,
                deterministic=True,
                policy_tape=policy_tape,
                config=config,
            )
            stochastic_action = _sample_first_action(
                distribution,
                original_action,
                unit_index=context.unit_index,
                selected_action=target,
                uniforms=policy_tape[context.environment_step],
            )
            stochastic_outcome = _rollout_branch(
                env=probe_env,
                snapshot=context.snapshot,
                environment_tape=environment_tape,
                first_action=stochastic_action,
                policy=policy,
                deterministic=False,
                policy_tape=policy_tape,
                config=config,
            )
            deterministic_targets.append(deterministic_outcome)
            stochastic_targets.append(stochastic_outcome)
            target_accumulator[target]["det_total_return"].append(
                deterministic_outcome.total_return
            )
            target_accumulator[target]["stochastic_total_return"].append(
                stochastic_outcome.total_return
            )
            target_accumulator[target]["det_delta"].append(
                deterministic_outcome.total_return
                - deterministic_noop_outcome.total_return
            )
            target_accumulator[target]["stochastic_delta"].append(
                stochastic_outcome.total_return
                - stochastic_noop_outcome.total_return
            )

        deterministic_marginal = _weighted_outcome(
            deterministic_targets, target_probabilities
        )
        stochastic_marginal = _weighted_outcome(
            stochastic_targets, target_probabilities
        )
        argmax_index = context.legal_targets.index(context.argmax_target)
        label_outcomes = {
            "a": deterministic_targets[argmax_index].minus(
                deterministic_noop_outcome
            ),
            "b": deterministic_marginal.minus(deterministic_noop_outcome),
            "c": stochastic_marginal.minus(stochastic_noop_outcome),
        }
        total_extra_transitions += (
            deterministic_noop_outcome.transitions
            + stochastic_noop_outcome.transitions
            + sum(item.transitions for item in deterministic_targets)
            + sum(item.transitions for item in stochastic_targets)
        )
        row: dict[str, Any] = {
            "context_id": context.context_id,
            "scenario": context.scenario,
            "policy_seed": context.policy_seed,
            "repeat": repeat,
        }
        for label, outcome in label_outcomes.items():
            for field in (
                "total_return",
                "zone_damage",
                "high_threat_leaks",
                "resource_cost",
            ):
                row[f"{label}_{field}_delta"] = float(
                    getattr(outcome, field)
                )
        repeat_rows.append(row)

    probe_env.close()
    aggregate: dict[str, Any] = {
        "context_id": context.context_id,
        "scenario": context.scenario,
        "policy_seed": context.policy_seed,
        "slot": context.slot,
        "episode_index": context.episode_index,
        "environment_seed": context.environment_seed,
        "environment_step": context.environment_step,
        "unit_index": context.unit_index,
        "prefix_actions": ",".join(map(str, context.prefix_actions)),
        "original_action": ",".join(map(str, context.original_action)),
        "observation_hash": context.observation_hash,
        "engage_probability": context.engage_probability,
        "engagement_margin": context.engagement_margin,
        "legal_target_count": len(context.legal_targets),
        "legal_targets": ",".join(map(str, context.legal_targets)),
        "target_probabilities": ",".join(
            f"{value:.12g}" for value in context.target_probabilities
        ),
        "argmax_target": context.argmax_target,
        "safety_score": context.safety_score,
        "resource_score": context.resource_score,
        "extra_transitions": total_extra_transitions,
    }
    for label in ("a", "b", "c"):
        return_values = [
            row[f"{label}_total_return_delta"] for row in repeat_rows
        ]
        estimate = estimate_effect(
            return_values,
            minimum_return_effect=config.minimum_return_effect,
            confidence_z=config.confidence_z,
        )
        for name, value in estimate.items():
            aggregate[f"{label}_{name}"] = value
        for component in (
            "zone_damage",
            "high_threat_leaks",
            "resource_cost",
        ):
            aggregate[f"{label}_{component}_mean"] = float(
                np.mean(
                    [
                        row[f"{label}_{component}_delta"]
                        for row in repeat_rows
                    ]
                )
            )

    target_rows: list[dict[str, Any]] = []
    deterministic_means: dict[int, float] = {}
    for target, probability in zip(
        context.legal_targets, context.target_probabilities
    ):
        values = target_accumulator[target]
        deterministic_means[target] = float(
            np.mean(values["det_total_return"])
        )
        target_rows.append(
            {
                "context_id": context.context_id,
                "scenario": context.scenario,
                "policy_seed": context.policy_seed,
                "slot": context.slot,
                "unit_index": context.unit_index,
                "target_index": target,
                "conditional_probability": probability,
                "is_argmax_target": target == context.argmax_target,
                "det_total_return_mean": deterministic_means[target],
                "stochastic_total_return_mean": float(
                    np.mean(values["stochastic_total_return"])
                ),
                "det_delta_mean": float(np.mean(values["det_delta"])),
                "stochastic_delta_mean": float(
                    np.mean(values["stochastic_delta"])
                ),
            }
        )
    aggregate["argmax_target_regret"] = (
        max(deterministic_means.values())
        - deterministic_means[context.argmax_target]
    )
    aggregate["a_b_sign_consistent"] = (
        aggregate["a_sign"] == aggregate["b_sign"]
    )
    aggregate["b_c_sign_consistent"] = (
        aggregate["b_sign"] == aggregate["c_sign"]
    )
    aggregate["target_selection_sign_reversal"] = bool(
        aggregate["a_reliable"]
        and aggregate["b_reliable"]
        and aggregate["a_sign"] != aggregate["b_sign"]
    )
    return aggregate, repeat_rows, target_rows


def summarize_bpce_semantics(
    rows: Sequence[dict[str, Any]],
    *,
    config: BPCELabelSemanticsConfig,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    scenarios = sorted({str(row["scenario"]) for row in rows})
    blocks = sorted(
        {
            (str(row["scenario"]), int(row["policy_seed"]))
            for row in rows
        }
    )

    def reliable_count(label: str, selected: Sequence[dict[str, Any]]) -> int:
        return sum(_as_bool(row[f"{label}_reliable"]) for row in selected)

    def agreement(
        left: str,
        right: str,
        selected: Sequence[dict[str, Any]],
        *,
        reliable_only: bool = False,
    ) -> tuple[float, int]:
        eligible = [
            row
            for row in selected
            if int(row[f"{left}_sign"]) != 0
            and int(row[f"{right}_sign"]) != 0
            and (
                not reliable_only
                or (
                    _as_bool(row[f"{left}_reliable"])
                    and _as_bool(row[f"{right}_reliable"])
                )
            )
        ]
        if not eligible:
            return 0.0, 0
        return (
            float(
                np.mean(
                    [
                        int(row[f"{left}_sign"])
                        == int(row[f"{right}_sign"])
                        for row in eligible
                    ]
                )
            ),
            len(eligible),
        )

    block_reliable = {
        f"{scenario}/seed{seed}": reliable_count(
            "c",
            [
                row
                for row in rows
                if row["scenario"] == scenario
                and int(row["policy_seed"]) == seed
            ],
        )
        for scenario, seed in blocks
    }
    a_b_overall, a_b_n = agreement("a", "b", rows)
    b_c_overall, b_c_n = agreement("b", "c", rows)
    reliable_a_b, reliable_a_b_n = agreement(
        "a", "b", rows, reliable_only=True
    )
    reliable_b_c, reliable_b_c_n = agreement(
        "b", "c", rows, reliable_only=True
    )
    scenario_agreements: dict[str, dict[str, float | int]] = {}
    bidirectional: dict[str, dict[str, int | bool]] = {}
    for scenario in scenarios:
        scenario_rows = [
            row for row in rows if row["scenario"] == scenario
        ]
        a_b_rate, a_b_count = agreement("a", "b", scenario_rows)
        b_c_rate, b_c_count = agreement("b", "c", scenario_rows)
        positive = sum(
            _as_bool(row["c_reliable"]) and int(row["c_sign"]) > 0
            for row in scenario_rows
        )
        negative = sum(
            _as_bool(row["c_reliable"]) and int(row["c_sign"]) < 0
            for row in scenario_rows
        )
        scenario_agreements[scenario] = {
            "a_b_rate": a_b_rate,
            "a_b_count": a_b_count,
            "b_c_rate": b_c_rate,
            "b_c_count": b_c_count,
        }
        bidirectional[scenario] = {
            "positive": positive,
            "negative": negative,
            "passed": positive >= 6 and negative >= 6,
        }

    target_eligible = [
        row
        for row in rows
        if _as_bool(row["a_reliable"]) and _as_bool(row["b_reliable"])
    ]
    target_reversal_rate = (
        float(
            np.mean(
                [
                    int(row["a_sign"]) != int(row["b_sign"])
                    for row in target_eligible
                ]
            )
        )
        if target_eligible
        else 1.0
    )
    positive_c = [
        row
        for row in rows
        if _as_bool(row["c_reliable"]) and int(row["c_sign"]) > 0
    ]
    component_consistency = (
        float(
            np.mean(
                [
                    float(row["c_zone_damage_mean"])
                    <= config.component_damage_tolerance
                    and float(row["c_high_threat_leaks_mean"])
                    <= config.component_high_threat_tolerance
                    for row in positive_c
                ]
            )
        )
        if positive_c
        else 0.0
    )

    gates = {
        "exact_context_count": len(rows) == 72,
        "overall_label_power": reliable_count("c", rows) >= 48,
        "block_label_power": min(block_reliable.values()) >= 6,
        "a_b_sign_agreement": (
            a_b_overall >= 0.80
            and min(
                float(value["a_b_rate"])
                for value in scenario_agreements.values()
            )
            >= 0.70
        ),
        "b_c_sign_agreement": (
            b_c_overall >= 0.80
            and min(
                float(value["b_c_rate"])
                for value in scenario_agreements.values()
            )
            >= 0.70
        ),
        "target_selection_confounding": target_reversal_rate <= 0.20,
        "bidirectional_coverage": all(
            bool(value["passed"]) for value in bidirectional.values()
        ),
        "component_consistency": (
            component_consistency >= config.minimum_component_consistency
        ),
    }
    if gates["a_b_sign_agreement"] and gates["b_c_sign_agreement"]:
        label_decision = "argmax_det_is_acceptable_low_cost_approximation"
    elif (not gates["a_b_sign_agreement"]) and gates["b_c_sign_agreement"]:
        label_decision = "use_target_marginalized_label"
    elif not gates["b_c_sign_agreement"]:
        label_decision = "deterministic_continuation_rejected"
    else:
        label_decision = "pause_bpce_label_semantics_unstable"
    return {
        "context_count": len(rows),
        "reliable_counts": {
            label: reliable_count(label, rows) for label in ("a", "b", "c")
        },
        "block_reliable_c": block_reliable,
        "a_b_agreement": {"rate": a_b_overall, "count": a_b_n},
        "b_c_agreement": {"rate": b_c_overall, "count": b_c_n},
        "reliable_overlap_agreement": {
            "a_b": {"rate": reliable_a_b, "count": reliable_a_b_n},
            "b_c": {"rate": reliable_b_c, "count": reliable_b_c_n},
        },
        "scenario_agreements": scenario_agreements,
        "target_selection_reversal_rate": target_reversal_rate,
        "target_selection_eligible_count": len(target_eligible),
        "bidirectional_coverage": bidirectional,
        "component_consistency_rate": component_consistency,
        "component_positive_count": len(positive_c),
        "gates": gates,
        "stage_a_passed": all(gates.values()),
        "label_decision": label_decision,
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _target_safety_score(
    snapshot: AirDefenseV1StateSnapshot, target_index: int
) -> float:
    target = snapshot.targets[target_index]
    zone = snapshot.protected_zones[target.target_zone]
    return float(
        target.threat
        * target.payload
        * zone.value
        / max(0.25, target.time_to_impact)
    )


def make_bpce_fixed_prefix(
    distribution: Any,
    original_action: torch.Tensor,
    *,
    unit_index: int,
    selected_action: int,
) -> torch.Tensor:
    fixed = torch.full_like(original_action, -1)
    order_position = tuple(distribution.unit_order).index(unit_index)
    for prefix_unit in distribution.unit_order[:order_position]:
        fixed[:, prefix_unit] = original_action[:, prefix_unit]
    fixed[:, unit_index] = selected_action
    return fixed


def _complete_first_action(
    distribution: Any,
    original_action: torch.Tensor,
    *,
    unit_index: int,
    selected_action: int,
    threshold: float,
) -> np.ndarray:
    fixed = make_bpce_fixed_prefix(
        distribution,
        original_action,
        unit_index=unit_index,
        selected_action=selected_action,
    )
    completed = distribution.complete_fixed_actions_with_engagement_threshold(
        fixed, threshold=threshold
    ).actions
    return completed[0].detach().cpu().numpy()


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


def make_bpce_paired_random_tapes(
    *,
    context: BPCEAuditContext,
    repeat: int,
    env_config: AirDefenseV1EnvConfig,
    branch_base_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    context_code = int.from_bytes(
        sha256(context.context_id.encode("utf-8")).digest()[:8], "little"
    )
    seed = np.random.SeedSequence(
        [branch_base_seed, context_code, repeat]
    )
    rng = np.random.default_rng(seed)
    num_targets = len(env_config.targets) or env_config.num_random_targets
    num_units = len(env_config.defense_units)
    return (
        rng.random((env_config.max_steps, num_targets), dtype=np.float64),
        rng.random((env_config.max_steps, num_units), dtype=np.float64),
    )


@torch.no_grad()
def _rollout_branch(
    *,
    env: AirDefenseResourceAssignmentEnvV1,
    snapshot: AirDefenseV1StateSnapshot,
    environment_tape: np.ndarray,
    first_action: np.ndarray,
    policy: Any,
    deterministic: bool,
    policy_tape: np.ndarray,
    config: BPCELabelSemanticsConfig,
) -> BranchOutcome:
    env.restore_state(snapshot)
    env.set_hit_random_tape(environment_tape)
    observation, reward, terminated, truncated, info = env.step(first_action)
    total_return = float(reward)
    resource_cost = -float(info["reward_breakdown"]["cost"])
    transitions = 1
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
        if deterministic:
            action = distribution.sample_with_engagement_threshold(
                config.engagement_threshold
            ).actions[0]
        else:
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
        transitions += 1
    high_threat_leaks = sum(
        target.status == "leaked"
        and target.threat >= config.high_threat_threshold
        for target in env.targets
    )
    return BranchOutcome(
        total_return=total_return,
        zone_damage=env.total_damage,
        high_threat_leaks=float(high_threat_leaks),
        resource_cost=resource_cost,
        transitions=transitions,
    )


def _weighted_outcome(
    outcomes: Sequence[BranchOutcome], weights: np.ndarray
) -> BranchOutcome:
    if len(outcomes) != len(weights) or not outcomes:
        raise ValueError("Outcome and weight counts must match and be non-empty")
    normalized = np.asarray(weights, dtype=np.float64)
    normalized = normalized / normalized.sum()
    return BranchOutcome(
        total_return=float(
            sum(weight * item.total_return for weight, item in zip(normalized, outcomes))
        ),
        zone_damage=float(
            sum(weight * item.zone_damage for weight, item in zip(normalized, outcomes))
        ),
        high_threat_leaks=float(
            sum(
                weight * item.high_threat_leaks
                for weight, item in zip(normalized, outcomes)
            )
        ),
        resource_cost=float(
            sum(weight * item.resource_cost for weight, item in zip(normalized, outcomes))
        ),
        transitions=int(sum(item.transitions for item in outcomes)),
    )
