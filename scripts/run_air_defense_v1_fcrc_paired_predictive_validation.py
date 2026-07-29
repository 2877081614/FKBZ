from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.algorithms.policy_gradient import (  # noqa: E402
    FactorizedEngagementMaskablePPO,
)
from rein_learning.common import (  # noqa: E402
    ActionSubstitutionConfirmationConfig,
    FCRCPredictiveValidationConfig,
    audit_fcrc_predictive_context,
    collect_confirmation_contexts,
    select_fcrc_candidate_pair,
)
from rein_learning.envs import (  # noqa: E402
    AirDefenseResourceAssignmentEnvV1,
    get_air_defense_v1_scenario,
)
from rein_learning.simulators import compute_hit_probability  # noqa: E402


SCENARIOS = ("medium", "time_pressure", "heterogeneity_pressure")
POLICY_SEEDS = (17, 18, 19)
METHOD = "factorized_engagement_ar_ppo_order_012"
R2_ROOT = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "action_substitution_confirmation"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "fcrc_paired_predictive_validation"
)
PREREGISTRATION_PATH = (
    PROJECT_ROOT
    / "configs"
    / "air_defense_v1"
    / "n3_fcrc_paired_predictive_preregistration.json"
)
STAGE_GATE_PATH = (
    PROJECT_ROOT / "configs" / "air_defense_v1" / "n3_stage_gate.json"
)
PRIOR_CONTEXT_PATHS = (
    R2_ROOT / "context_selection.csv",
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "bpce_label_semantics_audit"
    / "context_labels.csv",
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "bpce_short_horizon_label_audit"
    / "context_component_labels.csv",
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "action_substitution_opportunity_cost_audit"
    / "context_opportunity_estimates.csv",
)
OUTCOME_FILES = (
    "repeat_paired_outcomes.csv",
    "candidate_effects.csv",
    "context_effects.csv",
    "gate_summary.json",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered N3 FCRC paired predictive validation."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--software-tests-passed", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one block, one selected context, and two repeats.",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _model_path(scenario: str, seed: int) -> Path:
    return (
        R2_ROOT
        / "source_models"
        / scenario
        / f"{METHOD}_seed{seed}.zip"
    )


def _prior_hashes() -> set[str]:
    values: set[str] = set()
    for path in PRIOR_CONTEXT_PATHS:
        if not path.is_file():
            continue
        for row in _read_csv(path):
            value = row.get("observation_hash")
            if value:
                values.add(value)
    return values


def _policy_hash(policy: Any) -> str:
    digest = sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x = _rankdata(left)
    y = _rankdata(right)
    if np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _sign_flip_p(
    values: Sequence[float],
    *,
    permutations: int,
    seed: int,
) -> float:
    samples = np.asarray(values, dtype=np.float64)
    observed = float(samples.mean())
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=samples.size)
        exceedances += float(np.mean(samples * signs)) >= observed
    return float((exceedances + 1) / (permutations + 1))


def _linear_predictions(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    standardized_train = (train_x - mean) / scale
    standardized_test = (test_x - mean) / scale
    design = np.column_stack(
        [np.ones(standardized_train.shape[0]), standardized_train]
    )
    coefficients = np.linalg.lstsq(design, train_y, rcond=None)[0]
    return np.column_stack(
        [np.ones(standardized_test.shape[0]), standardized_test]
    ) @ coefficients


def _leave_block_out_mae(
    rows: Sequence[Mapping[str, Any]],
    features: Sequence[str],
) -> float:
    groups = np.asarray(
        [f"{row['scenario']}|{row['policy_seed']}" for row in rows]
    )
    x = np.asarray(
        [[float(row[name]) for name in features] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray(
        [float(row["intercept_harm_mean"]) for row in rows],
        dtype=np.float64,
    )
    predictions = np.empty_like(y)
    for group in sorted(set(groups.tolist())):
        test = groups == group
        train = ~test
        if train.sum() <= len(features):
            raise RuntimeError(f"insufficient training rows for block {group}")
        predictions[test] = _linear_predictions(
            x[train],
            y[train],
            x[test],
        )
    return float(np.mean(np.abs(y - predictions)))


def _candidate_features(
    context: Any,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    target_index = int(row["target_index"])
    snapshot = context.snapshot
    unit = snapshot.defense_units[context.unit_index]
    target = snapshot.targets[target_index]
    zone = snapshot.protected_zones[target.target_zone]
    return {
        "unit_resource_cost": float(unit.cost),
        "target_threat_weight": float(
            target.payload * target.threat * zone.value
        ),
        "current_hit_probability": compute_hit_probability(
            defense_position=unit.position,
            target_position=target.position,
            max_range=unit.max_range,
            base_hit_probability=unit.base_hit_probability,
            target_evasion=target.evasion,
        ),
        "legal_target_count": len(context.legal_targets),
    }


def _selection_row(context: Any, low: Any, high: Any) -> dict[str, Any]:
    return {
        "context_id": context.context_id,
        "scenario": context.scenario,
        "policy_seed": context.policy_seed,
        "slot": context.slot,
        "episode_index": context.episode_index,
        "environment_seed": context.environment_seed,
        "environment_step": context.environment_step,
        "observation_hash": context.observation_hash,
        "unit_index": context.unit_index,
        "resource_type": context.snapshot.defense_units[
            context.unit_index
        ].resource_type,
        "legal_targets": ",".join(map(str, context.legal_targets)),
        "low_target_index": low.target_index,
        "high_target_index": high.target_index,
        "low_fcrc": low.externality,
        "high_fcrc": high.externality,
        "fcrc_spread": high.externality - low.externality,
        "selection_uses_outcomes": False,
    }


def main() -> None:
    args = _parse_args()
    preregistration = json.loads(
        PREREGISTRATION_PATH.read_text(encoding="utf-8")
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.smoke and any((output_dir / name).exists() for name in OUTCOME_FILES):
        raise RuntimeError(
            "formal N3 outcome files already exist; refusing an unregistered rerun"
        )

    scenarios = SCENARIOS[:1] if args.smoke else SCENARIOS
    policy_seeds = POLICY_SEEDS[:1] if args.smoke else POLICY_SEEDS
    maximum_per_block = (
        1 if args.smoke
        else int(preregistration["selected_contexts_per_block_maximum"])
    )
    repeats = 2 if args.smoke else int(preregistration["paired_repeats"])
    collection_config = replace(
        ActionSubstitutionConfirmationConfig(),
        context_base_seed=int(preregistration["context_base_seed"]),
        branch_base_seed=int(preregistration["branch_base_seed"]),
    )
    audit_config = FCRCPredictiveValidationConfig(
        repeats=repeats,
        branch_base_seed=int(preregistration["branch_base_seed"]),
    )
    prior_hashes = _prior_hashes()
    selected_contexts: list[Any] = []
    selection_rows: list[dict[str, Any]] = []
    policies: dict[tuple[str, int], Any] = {}
    policy_hashes_before: dict[str, str] = {}
    block_candidate_counts: dict[str, int] = {}

    for scenario in scenarios:
        env_config = get_air_defense_v1_scenario(scenario)
        for policy_seed in policy_seeds:
            model_path = _model_path(scenario, policy_seed)
            if not model_path.is_file():
                raise FileNotFoundError(model_path)
            load_env = AirDefenseResourceAssignmentEnvV1(config=env_config)
            model = FactorizedEngagementMaskablePPO.load(
                model_path,
                env=load_env,
                device=args.device,
            )
            model.policy.set_training_mode(False)
            key = (scenario, policy_seed)
            policies[key] = model.policy
            policy_hashes_before[f"{scenario}|{policy_seed}"] = _policy_hash(
                model.policy
            )
            contexts = collect_confirmation_contexts(
                policy=model.policy,
                env_config=env_config,
                scenario=scenario,
                policy_seed=policy_seed,
                excluded_observation_hashes=prior_hashes,
                config=collection_config,
            )
            load_env.close()
            eligible: list[tuple[Any, Any, Any]] = []
            for original_context in contexts:
                context = replace(
                    original_context,
                    context_id=f"n3_{original_context.context_id}",
                )
                try:
                    low, high = select_fcrc_candidate_pair(context)
                except ValueError:
                    continue
                spread = high.externality - low.externality
                if spread > float(preregistration["minimum_fcrc_spread"]):
                    eligible.append((context, low, high))
            eligible.sort(
                key=lambda item: (
                    -(item[2].externality - item[1].externality),
                    item[0].context_id,
                )
            )
            block = f"{scenario}|{policy_seed}"
            block_candidate_counts[block] = len(eligible)
            for context, low, high in eligible[:maximum_per_block]:
                selected_contexts.append(context)
                selection_rows.append(_selection_row(context, low, high))
            print(
                f"[select] {block}: eligible={len(eligible)}, "
                f"selected={min(len(eligible), maximum_per_block)}"
            )

    if not selected_contexts:
        raise RuntimeError("no positive-spread FCRC context was selected")
    selection_path = output_dir / "context_selection.csv"
    _write_csv(selection_path, selection_rows)
    selection_freeze = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_file": str(selection_path),
        "selection_file_sha256": _file_hash(selection_path),
        "outcome_files_present_when_frozen": False,
        "prior_hash_count": len(prior_hashes),
        "selected_context_count": len(selected_contexts),
        "block_candidate_counts": block_candidate_counts,
        "selection_uses_outcomes": False,
        "preregistration_sha256": _file_hash(PREREGISTRATION_PATH),
    }
    _write_json(output_dir / "selection_freeze.json", selection_freeze)

    context_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for index, context in enumerate(selected_contexts, start=1):
        key = (context.scenario, context.policy_seed)
        aggregate, repeats_local, candidates_local = (
            audit_fcrc_predictive_context(
                policy=policies[key],
                env_config=get_air_defense_v1_scenario(context.scenario),
                context=context,
                config=audit_config,
            )
        )
        context_rows.append(aggregate)
        repeat_rows.extend(repeats_local)
        for row in candidates_local:
            enriched = dict(row)
            enriched.update(_candidate_features(context, row))
            candidate_rows.append(enriched)
        print(
            f"[audit {index:02d}/{len(selected_contexts):02d}] "
            f"{context.context_id}: "
            f"delta_I={aggregate['delta_intercept_harm_mean']:.6f}"
        )

    _write_csv(output_dir / "repeat_paired_outcomes.csv", repeat_rows)
    _write_csv(output_dir / "candidate_effects.csv", candidate_rows)
    _write_csv(output_dir / "context_effects.csv", context_rows)

    actor_unchanged = True
    policy_hashes_after: dict[str, str] = {}
    for (scenario, policy_seed), policy in policies.items():
        block = f"{scenario}|{policy_seed}"
        policy_hashes_after[block] = _policy_hash(policy)
        actor_unchanged &= (
            policy_hashes_after[block] == policy_hashes_before[block]
        )
    _write_json(
        output_dir / "actor_integrity.json",
        {
            "before": policy_hashes_before,
            "after": policy_hashes_after,
            "actor_unchanged": actor_unchanged,
            "actor_updates": 0,
        },
    )

    delta_intercept = [
        float(row["delta_intercept_harm_mean"]) for row in context_rows
    ]
    delta_damage = [
        float(row["delta_damage_harm_mean"]) for row in context_rows
    ]
    mean_delta_intercept = float(np.mean(delta_intercept))
    mean_delta_damage = float(np.mean(delta_damage))
    sign_flip_p = _sign_flip_p(
        delta_intercept,
        permutations=int(
            preregistration["statistics"]["sign_flip_permutations"]
        ),
        seed=int(preregistration["statistics"]["sign_flip_seed"]),
    )
    candidate_spearman = _spearman(
        [float(row["fcrc"]) for row in candidate_rows],
        [float(row["intercept_harm_mean"]) for row in candidate_rows],
    )
    baseline_features = preregistration["statistics"]["baseline_features"]
    full_features = [
        *baseline_features,
        *preregistration["statistics"]["full_model_extra_features"],
    ]
    if args.smoke:
        baseline_mae = 0.0
        full_mae = 0.0
        mae_reduction = 0.0
    else:
        baseline_mae = _leave_block_out_mae(candidate_rows, baseline_features)
        full_mae = _leave_block_out_mae(candidate_rows, full_features)
        mae_reduction = (
            (baseline_mae - full_mae) / baseline_mae
            if baseline_mae > 0.0
            else -1.0
        )
    scenario_means = {
        scenario: float(
            np.mean(
                [
                    float(row["delta_intercept_harm_mean"])
                    for row in context_rows
                    if row["scenario"] == scenario
                ]
            )
        )
        for scenario in scenarios
    }
    positive_scenarios = sum(value > 0.0 for value in scenario_means.values())
    selected_hashes = {row["observation_hash"] for row in selection_rows}
    prior_overlap = len(selected_hashes & prior_hashes)
    block_selected_counts = {
        f"{scenario}|{seed}": sum(
            row["scenario"] == scenario and int(row["policy_seed"]) == seed
            for row in selection_rows
        )
        for scenario in scenarios
        for seed in policy_seeds
    }
    transitions = sum(int(row["transitions"]) for row in context_rows)
    thresholds = preregistration["thresholds"]
    formal = not args.smoke
    propositions = {
        "N3-P1_independence_and_integrity": bool(
            formal
            and prior_overlap
            <= int(thresholds["maximum_prior_hash_overlap"])
            and len(context_rows)
            >= int(preregistration["minimum_selected_contexts_total"])
            and all(
                count
                >= int(preregistration["minimum_selected_contexts_per_block"])
                for count in block_selected_counts.values()
            )
            and selection_freeze["selection_uses_outcomes"] is False
            and actor_unchanged
        ),
        "N3-P2_causal_direction": bool(
            mean_delta_intercept
            > float(thresholds["minimum_mean_delta_intercept_harm"])
            and sign_flip_p
            < float(thresholds["maximum_one_sided_sign_flip_p"])
            and candidate_spearman
            > float(thresholds["minimum_candidate_spearman"])
        ),
        "N3-P3_incremental_prediction": bool(
            mae_reduction
            >= float(thresholds["minimum_cv_mae_reduction_fraction"])
        ),
        "N3-P4_scenario_and_safety_consistency": bool(
            formal
            and positive_scenarios
            >= int(thresholds["minimum_positive_scenarios"])
            and mean_delta_damage
            >= float(thresholds["minimum_mean_delta_damage_harm"])
        ),
        "N3-P5_execution_integrity": bool(
            formal
            and all(int(row["repeats"]) == repeats for row in context_rows)
            and repeats == int(preregistration["paired_repeats"])
            and transitions <= int(preregistration["maximum_new_transitions"])
            and actor_unchanged
        ),
    }
    if not propositions["N3-P1_independence_and_integrity"] or not propositions[
        "N3-P5_execution_integrity"
    ]:
        decision = "N3-E4_invalid_experiment"
    elif not propositions["N3-P2_causal_direction"]:
        decision = "N3-E3_reject_predictive_proposition"
    elif not propositions["N3-P3_incremental_prediction"] or not propositions[
        "N3-P4_scenario_and_safety_consistency"
    ]:
        decision = "N3-E2_diagnostic_only"
    else:
        decision = "N3-E1_enter_constraint_interface_design"

    gate = {
        "schema_version": "1.0",
        "task": "N3_fcrc_frozen_paired_predictive_validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_run": formal,
        "decision": decision if formal else "smoke_only_no_stage_decision",
        "online_training_authorized": False,
        "actor_updates": 0,
        "selection": {
            "contexts": len(context_rows),
            "block_selected_counts": block_selected_counts,
            "prior_hash_overlap": prior_overlap,
            "outcome_blind": True,
            "selection_file_sha256": selection_freeze[
                "selection_file_sha256"
            ],
        },
        "observed": {
            "mean_delta_intercept_harm": mean_delta_intercept,
            "one_sided_sign_flip_p": sign_flip_p,
            "candidate_fcrc_intercept_harm_spearman": candidate_spearman,
            "baseline_leave_block_out_mae": baseline_mae,
            "full_leave_block_out_mae": full_mae,
            "cv_mae_reduction_fraction": mae_reduction,
            "scenario_mean_delta_intercept_harm": scenario_means,
            "positive_scenarios": positive_scenarios,
            "mean_delta_damage_harm": mean_delta_damage,
            "transitions": transitions,
            "actor_unchanged": actor_unchanged,
            "software_tests_passed": bool(args.software_tests_passed),
        },
        "propositions": propositions,
        "prohibited_actions_remain": preregistration["prohibited_actions"],
    }
    _write_json(output_dir / "gate_summary.json", gate)
    if formal:
        _write_json(STAGE_GATE_PATH, gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
