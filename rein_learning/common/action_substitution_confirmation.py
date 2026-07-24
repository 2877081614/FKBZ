from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
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
class ActionSubstitutionConfirmationConfig:
    contexts_per_slot: int = 6
    resource_contexts_per_type: int = 3
    pool_episodes: int = 24
    repeats: int = 32
    engagement_threshold: float = 0.5
    confidence_z: float = 1.96
    context_base_seed: int = 1_283_000
    branch_base_seed: int = 1_293_000
    probability_tolerance: float = 1e-12
    decomposition_tolerance: float = 1e-6
    maximum_extra_transitions: int = 266_198

    def __post_init__(self) -> None:
        if self.contexts_per_slot != 6:
            raise ValueError("contexts_per_slot is frozen at six")
        if self.resource_contexts_per_type * 2 != self.contexts_per_slot:
            raise ValueError("resource type quotas must fill the resource slot")
        if self.pool_episodes <= 0:
            raise ValueError("pool_episodes must be positive")
        if self.repeats <= 1:
            raise ValueError("repeats must be greater than one")


@dataclass(frozen=True)
class CostBranchTrace:
    current_cost_by_unit: tuple[float, ...]
    future_cost_by_unit: tuple[float, ...]
    future_shots_by_unit: tuple[float, ...]
    future_cumulative_shots: tuple[float, ...]
    total_cost: float
    transitions: int

    @property
    def current_total_cost(self) -> float:
        return float(sum(self.current_cost_by_unit))

    @property
    def future_total_cost(self) -> float:
        return float(sum(self.future_cost_by_unit))

    @property
    def future_total_shots(self) -> float:
        return float(sum(self.future_shots_by_unit))


@torch.no_grad()
def collect_confirmation_contexts(
    *,
    policy: Any,
    env_config: AirDefenseV1EnvConfig,
    scenario: str,
    policy_seed: int,
    excluded_observation_hashes: set[str],
    config: ActionSubstitutionConfirmationConfig,
) -> tuple[BPCEAuditContext, ...]:
    env = AirDefenseResourceAssignmentEnvV1(config=env_config)
    device = next(policy.parameters()).device
    scenario_offset = int.from_bytes(
        sha256(scenario.encode("utf-8")).digest()[:4], "little"
    )
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
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
            observation_hash = sha256(
                np.asarray(observation, dtype=np.float32).tobytes()
            ).hexdigest()
            prefix: list[int] = []
            max_cost = max(unit.cost for unit in snapshot.defense_units)
            base_mask = env.action_mask()[:, : env.num_targets].astype(bool)
            for unit_index in distribution.unit_order:
                legal_targets = np.flatnonzero(
                    mask_values[unit_index, : distribution.num_targets]
                )
                identity = (
                    observation_hash,
                    int(unit_index),
                    snapshot.current_step,
                    tuple(int(value) for value in legal_targets),
                )
                if (
                    legal_targets.size
                    and observation_hash not in excluded_observation_hashes
                    and identity not in seen
                ):
                    target_mass = probability_values[
                        unit_index, : distribution.num_targets
                    ]
                    engage_probability = float(target_mass.sum())
                    conditional = target_mass[legal_targets] / engage_probability
                    clipped = np.clip(engage_probability, 1e-8, 1.0 - 1e-8)
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
                    candidates.append(
                        {
                            "scenario": scenario,
                            "policy_seed": policy_seed,
                            "episode_index": episode_index,
                            "environment_seed": environment_seed,
                            "environment_step": snapshot.current_step,
                            "unit_index": int(unit_index),
                            "prefix_actions": tuple(prefix),
                            "original_action": tuple(
                                int(value) for value in action_values
                            ),
                            "observation_hash": observation_hash,
                            "observation": np.asarray(
                                observation, dtype=np.float32
                            ).copy(),
                            "action_mask": action_mask,
                            "snapshot": snapshot,
                            "engage_probability": engage_probability,
                            "engagement_margin": float(
                                np.log(clipped) - np.log1p(-clipped)
                            ),
                            "legal_targets": tuple(
                                int(target) for target in legal_targets
                            ),
                            "target_probabilities": tuple(
                                float(value) for value in conditional
                            ),
                            "argmax_target": int(
                                legal_targets[int(np.argmax(conditional))]
                            ),
                            "safety_score": max(
                                _target_safety_score(snapshot, int(target))
                                for target in legal_targets
                            ),
                            "resource_score": (
                                1.0 - unit.ammo / max(1, unit.max_ammo)
                                + unit.cost / max(1e-8, max_cost)
                                + alternative_fraction
                            ),
                            "resource_type": unit.resource_type,
                        }
                    )
                    seen.add(identity)
                prefix.append(int(action_values[unit_index]))
            observation, _, terminated, truncated, _ = env.step(action_values)
    env.close()

    key = lambda row: (
        row["observation_hash"],
        row["unit_index"],
        row["environment_step"],
        row["legal_targets"],
    )
    safety = sorted(
        candidates,
        key=lambda row: (
            -row["safety_score"],
            abs(row["engagement_margin"]),
            key(row),
        ),
    )[: config.contexts_per_slot]
    used = {key(row) for row in safety}
    resource: list[dict[str, Any]] = []
    for resource_type in ("missile", "laser"):
        typed = sorted(
            (
                row
                for row in candidates
                if row["resource_type"] == resource_type
                and key(row) not in used
            ),
            key=lambda row: (
                -row["resource_score"],
                abs(row["engagement_margin"]),
                key(row),
            ),
        )
        selected = typed[: config.resource_contexts_per_type]
        if len(selected) != config.resource_contexts_per_type:
            raise RuntimeError(
                f"type_quota_unavailable:{scenario}/seed{policy_seed}/"
                f"{resource_type}:{len(selected)}"
            )
        resource.extend(selected)
        used.update(key(row) for row in selected)
    if len(safety) != config.contexts_per_slot:
        raise RuntimeError(
            f"not_enough_safety_contexts:{scenario}/seed{policy_seed}"
        )

    contexts: list[BPCEAuditContext] = []
    for slot, rows in (("safety", safety), ("resource", resource)):
        for slot_index, row in enumerate(rows):
            payload = dict(row)
            payload.pop("resource_type")
            contexts.append(
                BPCEAuditContext(
                    context_id=(
                        f"confirm_{scenario}_seed{policy_seed}_"
                        f"{slot}{slot_index:02d}_e{row['episode_index']:02d}_"
                        f"t{row['environment_step']:02d}_u{row['unit_index']}"
                    ),
                    slot=slot,
                    **payload,
                )
            )
    return tuple(contexts)


@torch.no_grad()
def validate_confirmation_contexts(
    contexts: Sequence[BPCEAuditContext],
    *,
    policy: Any,
    excluded_observation_hashes: set[str],
    probability_tolerance: float,
) -> list[dict[str, Any]]:
    device = next(policy.parameters()).device
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for context in contexts:
        observation = torch.as_tensor(
            context.observation[None, :], device=device, dtype=torch.float32
        )
        mask = torch.as_tensor(context.action_mask[None, :], device=device)
        original = torch.as_tensor(
            context.original_action, device=device, dtype=torch.long
        )[None, :]
        distribution = policy.get_distribution(
            observation, action_masks=mask
        )
        probabilities, masks = distribution.conditional_probabilities(original)
        legal = np.flatnonzero(
            masks[0, context.unit_index, : distribution.num_targets]
            .detach()
            .cpu()
            .numpy()
        )
        mass = (
            probabilities[0, context.unit_index, legal]
            .detach()
            .cpu()
            .numpy()
        )
        rebuilt = mass / mass.sum()
        expected = np.asarray(context.target_probabilities)
        maximum_error = (
            float(np.max(np.abs(rebuilt - expected)))
            if rebuilt.shape == expected.shape
            else float("inf")
        )
        identity = (
            context.observation_hash,
            context.unit_index,
            context.environment_step,
            context.legal_targets,
        )
        duplicate = identity in seen
        seen.add(identity)
        rows.append(
            {
                "context_id": context.context_id,
                "scenario": context.scenario,
                "policy_seed": context.policy_seed,
                "slot": context.slot,
                "unit_index": context.unit_index,
                "resource_type": context.snapshot.defense_units[
                    context.unit_index
                ].resource_type,
                "observation_hash": context.observation_hash,
                "old_hash_overlap": (
                    context.observation_hash in excluded_observation_hashes
                ),
                "duplicate_context": duplicate,
                "maximum_probability_difference": maximum_error,
                "matched": (
                    tuple(int(value) for value in legal)
                    == context.legal_targets
                    and maximum_error <= probability_tolerance
                    and context.observation_hash
                    not in excluded_observation_hashes
                    and not duplicate
                ),
            }
        )
    return rows


@torch.no_grad()
def audit_confirmation_context(
    *,
    policy: Any,
    env_config: AirDefenseV1EnvConfig,
    context: BPCEAuditContext,
    config: ActionSubstitutionConfirmationConfig,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    device = next(policy.parameters()).device
    distribution = policy.get_distribution(
        torch.as_tensor(
            context.observation[None, :],
            device=device,
            dtype=torch.float32,
        ),
        action_masks=torch.as_tensor(
            context.action_mask[None, :], device=device
        ),
    )
    original = torch.as_tensor(
        context.original_action, device=device, dtype=torch.long
    )[None, :]
    probabilities = np.asarray(context.target_probabilities, dtype=np.float64)
    env = AirDefenseResourceAssignmentEnvV1(config=env_config)
    env.reset(seed=config.branch_base_seed)
    target_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    actual_transitions = 0
    for repeat in range(config.repeats):
        environment_tape, policy_tape = make_bpce_paired_random_tapes(
            context=context,
            repeat=repeat,
            env_config=env_config,
            branch_base_seed=config.branch_base_seed,
        )
        noop_action = _sample_first_action(
            distribution,
            original,
            unit_index=context.unit_index,
            selected_action=distribution.noop_action,
            uniforms=policy_tape[context.environment_step],
        )
        noop = _rollout_cost_branch(
            env=env,
            snapshot=context.snapshot,
            environment_tape=environment_tape,
            first_action=noop_action,
            policy=policy,
            policy_tape=policy_tape,
        )
        actual_transitions += noop.transitions
        weighted = {
            "direct_cost": 0.0,
            "current_other_delta": 0.0,
            "same_step_other_sub_cost": 0.0,
            "future_sub_cost_probe": 0.0,
            "future_sub_cost_other": 0.0,
            "future_sub_cost": 0.0,
            "sub_cost_probe": 0.0,
            "sub_cost_other": 0.0,
            "sub_cost": 0.0,
            "sub_shot_probe": 0.0,
            "sub_shot_other": 0.0,
            "sub_shot": 0.0,
            "episode_cost_delta": 0.0,
            "rho_sub": 0.0,
            "cost_sign_masked": 0.0,
            "engage_transitions": 0.0,
        }
        substitution_times: list[tuple[float, int | None]] = []
        for probability, target in zip(
            probabilities, context.legal_targets
        ):
            engage_action = _sample_first_action(
                distribution,
                original,
                unit_index=context.unit_index,
                selected_action=target,
                uniforms=policy_tape[context.environment_step],
            )
            engage = _rollout_cost_branch(
                env=env,
                snapshot=context.snapshot,
                environment_tape=environment_tape,
                first_action=engage_action,
                policy=policy,
                policy_tape=policy_tape,
            )
            actual_transitions += engage.transitions
            ledger = _cost_ledger(
                noop=noop,
                engage=engage,
                unit_index=context.unit_index,
            )
            first_substitution = _first_substitution_time(
                noop.future_cumulative_shots,
                engage.future_cumulative_shots,
            )
            target_rows.append(
                {
                    "context_id": context.context_id,
                    "scenario": context.scenario,
                    "policy_seed": context.policy_seed,
                    "slot": context.slot,
                    "repeat": repeat,
                    "target_index": target,
                    "target_probability": probability,
                    "unit_index": context.unit_index,
                    "resource_type": context.snapshot.defense_units[
                        context.unit_index
                    ].resource_type,
                    "unit_cost": context.snapshot.defense_units[
                        context.unit_index
                    ].cost,
                    "current_probe_cost_n": noop.current_cost_by_unit[
                        context.unit_index
                    ],
                    "current_probe_cost_e": engage.current_cost_by_unit[
                        context.unit_index
                    ],
                    "current_other_cost_n": (
                        noop.current_total_cost
                        - noop.current_cost_by_unit[context.unit_index]
                    ),
                    "current_other_cost_e": (
                        engage.current_total_cost
                        - engage.current_cost_by_unit[context.unit_index]
                    ),
                    "future_probe_cost_n": noop.future_cost_by_unit[
                        context.unit_index
                    ],
                    "future_probe_cost_e": engage.future_cost_by_unit[
                        context.unit_index
                    ],
                    "future_other_cost_n": (
                        noop.future_total_cost
                        - noop.future_cost_by_unit[context.unit_index]
                    ),
                    "future_other_cost_e": (
                        engage.future_total_cost
                        - engage.future_cost_by_unit[context.unit_index]
                    ),
                    "future_probe_shots_n": noop.future_shots_by_unit[
                        context.unit_index
                    ],
                    "future_probe_shots_e": engage.future_shots_by_unit[
                        context.unit_index
                    ],
                    "future_other_shots_n": (
                        noop.future_total_shots
                        - noop.future_shots_by_unit[context.unit_index]
                    ),
                    "future_other_shots_e": (
                        engage.future_total_shots
                        - engage.future_shots_by_unit[context.unit_index]
                    ),
                    "first_future_shot_n": _first_future_shot(
                        noop.future_cumulative_shots
                    ),
                    "first_future_shot_e": _first_future_shot(
                        engage.future_cumulative_shots
                    ),
                    "first_substitution_step": (
                        "" if first_substitution is None else first_substitution
                    ),
                    **ledger,
                }
            )
            substitution_times.append((float(probability), first_substitution))
            for key in weighted:
                if key == "engage_transitions":
                    weighted[key] += float(probability) * engage.transitions
                else:
                    weighted[key] += float(probability) * float(ledger[key])
        first_values = [
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
                "resource_type": context.snapshot.defense_units[
                    context.unit_index
                ].resource_type,
                **weighted,
                "first_substitution_step_expected": (
                    sum(first_values)
                    if len(first_values) == len(substitution_times)
                    else ""
                ),
            }
        )
    env.close()
    aggregate = _aggregate_context(
        context=context,
        repeat_rows=repeat_rows,
        actual_transitions=actual_transitions,
        config=config,
    )
    return aggregate, repeat_rows, target_rows


def summarize_confirmation(
    context_rows: Sequence[Mapping[str, Any]],
    repeat_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
    *,
    source_model_count: int,
    config: ActionSubstitutionConfirmationConfig,
    maximum_actor_parameter_difference: float,
    software_tests_passed: bool,
) -> dict[str, Any]:
    scenarios = ("medium", "time_pressure", "heterogeneity_pressure")
    seeds = (17, 18, 19)
    transitions = sum(
        int(row["actual_extra_transitions"]) for row in context_rows
    )
    resource_quotas = all(
        sum(
            str(row["scenario"]) == scenario
            and int(row["policy_seed"]) == seed
            and str(row["slot"]) == "resource"
            and str(row["resource_type"]) == resource_type
            for row in context_rows
        )
        == 3
        for scenario in scenarios
        for seed in seeds
        for resource_type in ("missile", "laser")
    )
    integrity = {
        "source_models": source_model_count == 9,
        "context_count": len(context_rows) == 108,
        "old_hash_overlap_zero": all(
            not _as_bool(row["old_hash_overlap"]) for row in identity_rows
        ),
        "context_identity": (
            len(identity_rows) == 108
            and all(_as_bool(row["matched"]) for row in identity_rows)
            and max(
                float(row["maximum_probability_difference"])
                for row in identity_rows
            )
            <= config.probability_tolerance
        ),
        "resource_type_quotas": resource_quotas,
        "repeat_count": all(
            int(row["repeat_count"]) == config.repeats
            for row in context_rows
        ),
        "actor_frozen": maximum_actor_parameter_difference == 0.0,
        "transition_budget": transitions <= config.maximum_extra_transitions,
        "software_regression": software_tests_passed,
    }
    maximum_protocol_error = max(
        abs(float(row["protocol_residual"])) for row in target_rows
    )
    maximum_extended_error = max(
        abs(float(row["extended_residual"])) for row in target_rows
    )
    maximum_future_only_error = max(
        abs(float(row["future_only_residual"])) for row in target_rows
    )
    maximum_sub_error = max(
        abs(float(row["sub_cost_decomposition_residual"]))
        for row in target_rows
    )
    pc1 = (
        maximum_protocol_error <= config.decomposition_tolerance
        and maximum_sub_error <= config.decomposition_tolerance
        and all(float(row["direct_cost"]) > 0.0 for row in target_rows)
    )
    time_resource = _select(
        context_rows, scenario="time_pressure", slot="resource"
    )
    time_blocks = {
        seed: _interval(
            [
                float(row["sub_shot_mean"])
                for row in time_resource
                if int(row["policy_seed"]) == seed
            ],
            config,
        )
        for seed in seeds
    }
    seed_masked = {
        seed: (
            float(np.mean(values))
            if (
                values := [
                    float(row["cost_sign_masked_rate"])
                    for row in time_resource
                    if int(row["policy_seed"]) == seed
                ]
            )
            else float("nan")
        )
        for seed in seeds
    }
    nonpositive = [
        row
        for row in time_resource
        if float(row["episode_cost_delta_mean"]) <= 0.0
    ]
    pc2_details = {
        "positive_mean_sub_shot": sum(
            float(row["sub_shot_mean"]) > 0.0 for row in time_resource
        ),
        "positive_lower_sub_shot": sum(
            float(row["sub_shot_lower"]) > 0.0 for row in time_resource
        ),
        "positive_block_lower_seeds": sum(
            interval["lower"] > 0.0 for interval in time_blocks.values()
        ),
        "masked_rate_at_least_half_seeds": sum(
            value >= 0.5 for value in seed_masked.values()
        ),
        "nonpositive_contexts": len(nonpositive),
        "nonpositive_with_positive_sub_cost": sum(
            float(row["sub_cost_mean"]) > 0.0 for row in nonpositive
        ),
        "nonpositive_explained_fraction": (
            sum(float(row["sub_cost_mean"]) > 0.0 for row in nonpositive)
            / len(nonpositive)
            if nonpositive
            else 1.0
        ),
        "seed_block_intervals": time_blocks,
        "seed_masked_rates": seed_masked,
    }
    pc2 = (
        len(time_resource) == 18
        and pc2_details["positive_mean_sub_shot"] >= 12
        and pc2_details["positive_lower_sub_shot"] >= 6
        and pc2_details["positive_block_lower_seeds"] >= 2
        and pc2_details["masked_rate_at_least_half_seeds"] >= 2
        and pc2_details["nonpositive_explained_fraction"] >= 0.8
    )

    type_details: dict[str, Any] = {}
    type_passes: list[bool] = []
    for resource_type in ("missile", "laser"):
        selected = [
            row
            for row in time_resource
            if str(row["resource_type"]) == resource_type
        ]
        interval = _interval(
            [float(row["sub_shot_mean"]) for row in selected], config
        )
        positive_seed_blocks = int(
            sum(
                bool(values) and float(np.mean(values)) > 0.0
                for seed in seeds
                if (
                    values := [
                    float(row["sub_shot_mean"])
                    for row in selected
                    if int(row["policy_seed"]) == seed
                    ]
                )
            )
        )
        masked_contexts = int(
            sum(
                float(row["episode_cost_delta_mean"]) <= 0.0
                for row in selected
            )
        )
        type_details[resource_type] = {
            "contexts": len(selected),
            "sub_shot_interval": interval,
            "positive_seed_blocks": positive_seed_blocks,
            "masked_contexts": masked_contexts,
        }
        type_passes.append(
            interval["lower"] > 0.0
            and positive_seed_blocks >= 2
            and masked_contexts >= 3
        )
    pc3 = all(type_passes)
    mechanism = {
        "P-C1_cost_decomposition": pc1,
        "P-C2_independent_substitution": pc2,
        "P-C3_cross_resource_type": pc3,
    }
    all_integrity = all(integrity.values())
    if not all_integrity:
        decision = "independent_confirmation_invalid_data_integrity"
    elif not pc1:
        decision = "invalid_cost_ledger_fix_only"
    elif pc2 and pc3:
        decision = "freeze_cross_resource_measurement_distortion_claim"
    elif pc2:
        decision = "freeze_resource_type_conditional_claim"
    else:
        decision = "downgrade_r1_to_old_seed_conditional_finding"
    return {
        "context_count": len(context_rows),
        "repeat_rows": len(repeat_rows),
        "target_ledger_rows": len(target_rows),
        "actual_extra_transitions": transitions,
        "maximum_actor_parameter_difference": maximum_actor_parameter_difference,
        "maximum_protocol_decomposition_error": maximum_protocol_error,
        "maximum_extended_decomposition_error": maximum_extended_error,
        "maximum_future_only_decomposition_error": maximum_future_only_error,
        "maximum_sub_cost_decomposition_error": maximum_sub_error,
        "integrity_gates": integrity,
        "P-C2": pc2_details,
        "P-C3": type_details,
        "mechanism_gates": mechanism,
        "stage_passed": all_integrity and all(mechanism.values()),
        "decision": decision,
    }


def grouped_summary_rows(
    context_rows: Sequence[Mapping[str, Any]],
    *,
    group_fields: Sequence[str],
    config: ActionSubstitutionConfirmationConfig,
) -> list[dict[str, Any]]:
    keys = sorted(
        {
            tuple(str(row[field]) for field in group_fields)
            for row in context_rows
        }
    )
    output: list[dict[str, Any]] = []
    for key in keys:
        selected = [
            row
            for row in context_rows
            if tuple(str(row[field]) for field in group_fields) == key
        ]
        result: dict[str, Any] = {
            field: value for field, value in zip(group_fields, key)
        }
        result["contexts"] = len(selected)
        for metric in (
            "sub_shot_mean",
            "sub_cost_mean",
            "rho_sub_mean",
            "episode_cost_delta_mean",
            "cost_sign_masked_rate",
        ):
            values = [float(row[metric]) for row in selected]
            interval = _interval(values, config)
            result[f"{metric}_aggregate"] = interval["mean"]
            result[f"{metric}_lower"] = interval["lower"]
            result[f"{metric}_upper"] = interval["upper"]
            result[f"{metric}_median"] = float(np.median(values))
            result[f"{metric}_q25"] = float(np.quantile(values, 0.25))
            result[f"{metric}_q75"] = float(np.quantile(values, 0.75))
        output.append(result)
    return output


def _cost_ledger(
    *,
    noop: CostBranchTrace,
    engage: CostBranchTrace,
    unit_index: int,
) -> dict[str, float | bool]:
    direct = (
        engage.current_cost_by_unit[unit_index]
        - noop.current_cost_by_unit[unit_index]
    )
    current_other_delta = (
        engage.current_total_cost
        - engage.current_cost_by_unit[unit_index]
        - noop.current_total_cost
        + noop.current_cost_by_unit[unit_index]
    )
    future_sub_probe = (
        noop.future_cost_by_unit[unit_index]
        - engage.future_cost_by_unit[unit_index]
    )
    future_sub_other = (
        noop.future_total_cost
        - noop.future_cost_by_unit[unit_index]
        - engage.future_total_cost
        + engage.future_cost_by_unit[unit_index]
    )
    future_sub_cost = noop.future_total_cost - engage.future_total_cost
    same_step_other_sub_cost = -current_other_delta
    sub_probe = future_sub_probe
    sub_other = future_sub_other + same_step_other_sub_cost
    sub_cost = future_sub_cost + same_step_other_sub_cost
    sub_shot_probe = (
        noop.future_shots_by_unit[unit_index]
        - engage.future_shots_by_unit[unit_index]
    )
    sub_shot_other = (
        noop.future_total_shots
        - noop.future_shots_by_unit[unit_index]
        - engage.future_total_shots
        + engage.future_shots_by_unit[unit_index]
    )
    sub_shot = noop.future_total_shots - engage.future_total_shots
    episode_delta = engage.total_cost - noop.total_cost
    return {
        "direct_cost": direct,
        "current_other_delta": current_other_delta,
        "same_step_other_sub_cost": same_step_other_sub_cost,
        "future_sub_cost_probe": future_sub_probe,
        "future_sub_cost_other": future_sub_other,
        "future_sub_cost": future_sub_cost,
        "sub_cost_probe": sub_probe,
        "sub_cost_other": sub_other,
        "sub_cost": sub_cost,
        "sub_shot_probe": sub_shot_probe,
        "sub_shot_other": sub_shot_other,
        "sub_shot": sub_shot,
        "episode_cost_delta": episode_delta,
        "rho_sub": sub_cost / direct if direct > 0.0 else float("nan"),
        "cost_sign_masked": bool(direct > 0.0 and episode_delta <= 0.0),
        "protocol_residual": episode_delta - (direct - sub_cost),
        "future_only_residual": (
            episode_delta - (direct - future_sub_cost)
        ),
        "extended_residual": (
            episode_delta
            - (direct + current_other_delta - future_sub_cost)
        ),
        "sub_cost_decomposition_residual": (
            sub_cost - sub_probe - sub_other
        ),
    }


def _aggregate_context(
    *,
    context: BPCEAuditContext,
    repeat_rows: Sequence[Mapping[str, Any]],
    actual_transitions: int,
    config: ActionSubstitutionConfirmationConfig,
) -> dict[str, Any]:
    unit = context.snapshot.defense_units[context.unit_index]
    result: dict[str, Any] = {
        "context_id": context.context_id,
        "scenario": context.scenario,
        "policy_seed": context.policy_seed,
        "slot": context.slot,
        "environment_seed": context.environment_seed,
        "environment_step": context.environment_step,
        "observation_hash": context.observation_hash,
        "unit_index": context.unit_index,
        "resource_type": unit.resource_type,
        "unit_cost": unit.cost,
        "repeat_count": len(repeat_rows),
        "actual_extra_transitions": actual_transitions,
    }
    for metric in (
        "direct_cost",
        "current_other_delta",
        "same_step_other_sub_cost",
        "future_sub_cost_probe",
        "future_sub_cost_other",
        "future_sub_cost",
        "sub_cost_probe",
        "sub_cost_other",
        "sub_cost",
        "sub_shot_probe",
        "sub_shot_other",
        "sub_shot",
        "episode_cost_delta",
        "rho_sub",
    ):
        interval = component_interval(
            [float(row[metric]) for row in repeat_rows],
            confidence_z=config.confidence_z,
        )
        for field, value in interval.items():
            result[f"{metric}_{field}"] = value
    result["rho_sub_median"] = float(
        np.median([float(row["rho_sub"]) for row in repeat_rows])
    )
    result["rho_sub_q25"] = float(
        np.quantile([float(row["rho_sub"]) for row in repeat_rows], 0.25)
    )
    result["rho_sub_q75"] = float(
        np.quantile([float(row["rho_sub"]) for row in repeat_rows], 0.75)
    )
    result["cost_sign_masked_rate"] = float(
        np.mean([_as_bool(row["cost_sign_masked"]) for row in repeat_rows])
    )
    observed_times = [
        float(row["first_substitution_step_expected"])
        for row in repeat_rows
        if row["first_substitution_step_expected"] != ""
    ]
    result["first_substitution_step_mean"] = (
        float(np.mean(observed_times)) if observed_times else ""
    )
    return result


@torch.no_grad()
def _rollout_cost_branch(
    *,
    env: AirDefenseResourceAssignmentEnvV1,
    snapshot: AirDefenseV1StateSnapshot,
    environment_tape: np.ndarray,
    first_action: np.ndarray,
    policy: Any,
    policy_tape: np.ndarray,
) -> CostBranchTrace:
    env.restore_state(snapshot)
    env.set_hit_random_tape(environment_tape)
    observation, _, terminated, truncated, info = env.step(first_action)
    current_costs = _step_costs(env, info)
    future_costs = np.zeros(env.num_defense_units, dtype=np.float64)
    future_shots = np.zeros(env.num_defense_units, dtype=np.float64)
    cumulative: list[float] = []
    transitions = 1
    device = next(policy.parameters()).device
    while not (terminated or truncated):
        action_mask = env.action_masks()
        distribution = policy.get_distribution(
            torch.as_tensor(
                observation[None, :], device=device, dtype=torch.float32
            ),
            action_masks=torch.as_tensor(
                action_mask[None, :], device=device
            ),
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
        observation, _, terminated, truncated, info = env.step(action)
        step_costs = _step_costs(env, info)
        future_costs += step_costs
        future_shots += step_costs > 0.0
        cumulative.append(float(future_shots.sum()))
        transitions += 1
    return CostBranchTrace(
        current_cost_by_unit=tuple(current_costs.tolist()),
        future_cost_by_unit=tuple(future_costs.tolist()),
        future_shots_by_unit=tuple(future_shots.tolist()),
        future_cumulative_shots=tuple(cumulative),
        total_cost=float(current_costs.sum() + future_costs.sum()),
        transitions=transitions,
    )


def _step_costs(
    env: AirDefenseResourceAssignmentEnvV1, info: Mapping[str, Any]
) -> np.ndarray:
    costs = np.zeros(env.num_defense_units, dtype=np.float64)
    for result in info["unit_results"]:
        if result["action_type"] == "engage" and bool(result["legal"]):
            unit_index = int(result["unit_index"])
            costs[unit_index] = env.defense_units[unit_index].cost
    return costs


def _sample_first_action(
    distribution: Any,
    original: torch.Tensor,
    *,
    unit_index: int,
    selected_action: int,
    uniforms: np.ndarray,
) -> np.ndarray:
    fixed = make_bpce_fixed_prefix(
        distribution,
        original,
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


def _first_future_shot(cumulative: Sequence[float]) -> int | str:
    for index, value in enumerate(cumulative):
        if value > 0.0:
            return index + 1
    return ""


def _first_substitution_time(
    noop: Sequence[float], engage: Sequence[float]
) -> int | None:
    for index in range(max(len(noop), len(engage))):
        left = noop[index] if index < len(noop) else (noop[-1] if noop else 0.0)
        right = (
            engage[index]
            if index < len(engage)
            else (engage[-1] if engage else 0.0)
        )
        if left > right:
            return index + 1
    return None


def _interval(
    values: Sequence[float],
    config: ActionSubstitutionConfirmationConfig,
) -> dict[str, float]:
    if not values:
        return {
            "mean": float("nan"),
            "standard_error": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
        }
    if len(values) == 1:
        value = float(values[0])
        return {
            "mean": value,
            "standard_error": 0.0,
            "lower": value,
            "upper": value,
        }
    return component_interval(values, confidence_z=config.confidence_z)


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
