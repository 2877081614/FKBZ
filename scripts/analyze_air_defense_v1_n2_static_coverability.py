from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
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
    collect_confirmation_contexts,
    future_coverability_externality,
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
    / "n2_static_coverability_audit"
)
OLD_CONTEXT_PATHS = (
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Development-only N2 future-coverability static audit."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--software-tests-passed", action="store_true")
    parser.add_argument("--literature-gate-passed", action="store_true")
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
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _model_path(scenario: str, seed: int) -> Path:
    return (
        R2_ROOT
        / "source_models"
        / scenario
        / f"{METHOD}_seed{seed}.zip"
    )


def _old_hashes() -> set[str]:
    hashes: set[str] = set()
    for path in OLD_CONTEXT_PATHS:
        for row in _read_csv(path):
            if row.get("observation_hash"):
                hashes.add(row["observation_hash"])
    return hashes


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


def _expected_context_rows() -> dict[str, dict[str, str]]:
    rows = _read_csv(R2_ROOT / "context_selection.csv")
    return {row["context_id"]: row for row in rows}


def _substitution_by_context() -> dict[str, float]:
    return {
        row["context_id"]: float(row["sub_cost_mean"])
        for row in _read_csv(
            R2_ROOT / "context_substitution_estimates.csv"
        )
    }


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = _expected_context_rows()
    substitution = _substitution_by_context()
    excluded_hashes = _old_hashes()
    config = ActionSubstitutionConfirmationConfig()

    action_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []

    for scenario in SCENARIOS:
        env_config = get_air_defense_v1_scenario(scenario)
        for policy_seed in POLICY_SEEDS:
            model_path = _model_path(scenario, policy_seed)
            load_env = AirDefenseResourceAssignmentEnvV1(config=env_config)
            model = FactorizedEngagementMaskablePPO.load(
                model_path,
                env=load_env,
                device=args.device,
            )
            model.policy.set_training_mode(False)
            contexts = collect_confirmation_contexts(
                policy=model.policy,
                env_config=env_config,
                scenario=scenario,
                policy_seed=policy_seed,
                excluded_observation_hashes=excluded_hashes,
                config=config,
            )
            load_env.close()

            for context in contexts:
                expected_row = expected.get(context.context_id)
                matched = bool(
                    expected_row
                    and expected_row["observation_hash"]
                    == context.observation_hash
                    and int(expected_row["unit_index"]) == context.unit_index
                    and expected_row["legal_targets"]
                    == ",".join(map(str, context.legal_targets))
                )
                identity_rows.append(
                    {
                        "context_id": context.context_id,
                        "scenario": scenario,
                        "policy_seed": policy_seed,
                        "observation_hash": context.observation_hash,
                        "expected_context_present": expected_row is not None,
                        "matched": matched,
                    }
                )
                if not matched:
                    raise RuntimeError(
                        f"R2 context replay mismatch: {context.context_id}"
                    )

                snapshot = context.snapshot
                unit = snapshot.defense_units[context.unit_index]
                started = perf_counter()
                local_rows: list[dict[str, Any]] = []
                for target_index in context.legal_targets:
                    target = snapshot.targets[target_index]
                    zone = snapshot.protected_zones[target.target_zone]
                    certificate = future_coverability_externality(
                        snapshot,
                        unit_index=context.unit_index,
                        target_index=target_index,
                    )
                    target_weight = float(
                        target.payload * target.threat * zone.value
                    )
                    hit_probability = compute_hit_probability(
                        defense_position=unit.position,
                        target_position=target.position,
                        max_range=unit.max_range,
                        base_hit_probability=unit.base_hit_probability,
                        target_evasion=target.evasion,
                    )
                    local_rows.append(
                        {
                            "context_id": context.context_id,
                            "scenario": scenario,
                            "policy_seed": policy_seed,
                            "slot": context.slot,
                            "environment_step": context.environment_step,
                            "observation_hash": context.observation_hash,
                            "unit_index": context.unit_index,
                            "resource_type": unit.resource_type,
                            "unit_cost": float(unit.cost),
                            "unit_ammo": int(unit.ammo),
                            "legal_target_count": len(context.legal_targets),
                            "target_index": target_index,
                            "target_weight": target_weight,
                            "target_tti": float(target.time_to_impact),
                            "hit_probability": hit_probability,
                            "coverability_before": (
                                certificate.other_threat_coverability_before
                            ),
                            "coverability_after": (
                                certificate.other_threat_coverability_after
                            ),
                            "fcrc_externality": certificate.externality,
                            "n1_total_substitution_mean": substitution[
                                context.context_id
                            ],
                        }
                    )
                elapsed_ms = (perf_counter() - started) * 1000.0
                action_rows.extend(local_rows)
                externalities = [
                    row["fcrc_externality"] for row in local_rows
                ]
                context_rows.append(
                    {
                        "context_id": context.context_id,
                        "scenario": scenario,
                        "policy_seed": policy_seed,
                        "slot": context.slot,
                        "resource_type": unit.resource_type,
                        "legal_actions": len(local_rows),
                        "positive_externality_actions": sum(
                            value > 1e-9 for value in externalities
                        ),
                        "externality_min": min(externalities),
                        "externality_max": max(externalities),
                        "externality_span": (
                            max(externalities) - min(externalities)
                        ),
                        "audit_elapsed_ms": elapsed_ms,
                    }
                )
            print(
                f"[replay] {scenario}/seed{policy_seed}: {len(contexts)} contexts",
                flush=True,
            )

    if len(context_rows) != 108 or not all(row["matched"] for row in identity_rows):
        raise RuntimeError("N2 audit did not reproduce all 108 frozen R2 contexts")

    positive_actions = sum(
        row["fcrc_externality"] > 1e-9 for row in action_rows
    )
    spread_contexts = sum(
        row["externality_span"] > 1e-9 for row in context_rows
    )
    cost_correlation = _spearman(
        [row["fcrc_externality"] for row in action_rows],
        [row["unit_cost"] for row in action_rows],
    )
    threat_correlation = _spearman(
        [row["fcrc_externality"] for row in action_rows],
        [row["target_weight"] for row in action_rows],
    )
    substitution_correlation = _spearman(
        [row["fcrc_externality"] for row in action_rows],
        [row["n1_total_substitution_mean"] for row in action_rows],
    )
    mean_context_ms = float(
        np.mean([row["audit_elapsed_ms"] for row in context_rows])
    )
    maximum_context_ms = float(
        np.max([row["audit_elapsed_ms"] for row in context_rows])
    )
    positive_rate = positive_actions / len(action_rows)

    propositions = {
        "N2-P1_formal_implementation_consistency": bool(
            args.software_tests_passed
        ),
        "N2-P2_non_degenerate_signal": bool(
            spread_contexts >= 30 and positive_rate >= 0.15
        ),
        "N2-P3_not_cost_or_threat_rename": bool(
            abs(cost_correlation) < 0.90
            and abs(threat_correlation) < 0.90
            and spread_contexts >= 20
        ),
        "N2-P4_static_compute_budget": bool(
            mean_context_ms <= 5.0 and maximum_context_ms <= 25.0
        ),
        "N2-P5_provisional_literature_distance": bool(
            args.literature_gate_passed
        ),
    }
    stage_passed = all(propositions.values())
    decision = (
        "N2-E1_enter_frozen_paired_predictive_validation"
        if stage_passed
        else (
            "N2-E2_diagnostic_component_only"
            if all(
                propositions[key]
                for key in (
                    "N2-P1_formal_implementation_consistency",
                    "N2-P2_non_degenerate_signal",
                    "N2-P3_not_cost_or_threat_rename",
                    "N2-P4_static_compute_budget",
                )
            )
            else "N2-E3_reject_fcrc_signal"
        )
    )

    summary = {
        "context_count": len(context_rows),
        "legal_action_rows": len(action_rows),
        "positive_externality_actions": positive_actions,
        "positive_externality_rate": positive_rate,
        "contexts_with_within_unit_target_spread": spread_contexts,
        "spearman_externality_vs_unit_cost": cost_correlation,
        "spearman_externality_vs_target_weight": threat_correlation,
        "spearman_externality_vs_n1_substitution": substitution_correlation,
        "mean_context_audit_ms": mean_context_ms,
        "maximum_context_audit_ms": maximum_context_ms,
        "N2_propositions": propositions,
        "stage_passed": stage_passed,
        "decision": decision,
        "online_training_authorized": False,
        "paired_predictive_validation_authorized": stage_passed,
    }
    experiment = {
        "schema_version": 1,
        "task": "N2_future_coverability_static_audit",
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(R2_ROOT.resolve()),
        "source_contexts_are_development_only": True,
        "context_replay_only": True,
        "new_context_selection": False,
        "new_counterfactual_rollouts": 0,
        "actor_updates": 0,
        "scenarios": list(SCENARIOS),
        "policy_seeds": list(POLICY_SEEDS),
        "thresholds": {
            "minimum_spread_contexts": 30,
            "minimum_positive_action_rate": 0.15,
            "maximum_absolute_cost_or_threat_spearman": 0.90,
            "minimum_same_unit_target_difference_contexts": 20,
            "maximum_mean_context_ms": 5.0,
            "maximum_single_context_ms": 25.0,
        },
    }

    _write_csv(output_dir / "context_identity_check.csv", identity_rows)
    _write_csv(output_dir / "action_certificates.csv", action_rows)
    _write_csv(output_dir / "context_summary.csv", context_rows)
    _write_json(output_dir / "experiment_config.json", experiment)
    _write_json(output_dir / "gate_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
