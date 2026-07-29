from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.common import ResourceCreditComponents


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "action_substitution_confirmation"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "n1_offline_semantic_audit"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _float(row: dict[str, str], field: str) -> float:
    return float(row[field])


def _support_row(
    name: str,
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    tolerance = 1e-12
    count = len(rows)
    return {
        "group": name,
        "contexts": count,
        "mean_direct_cost": _mean(_float(row, "direct_cost_mean") for row in rows),
        "same_step_nonzero_rate": _mean(
            abs(_float(row, "same_step_other_sub_cost_mean")) > tolerance
            for row in rows
        ),
        "same_step_reliable_positive_rate": _mean(
            _float(row, "same_step_other_sub_cost_lower") > 0.0 for row in rows
        ),
        "future_probe_reliable_positive_rate": _mean(
            _float(row, "future_sub_cost_probe_lower") > 0.0 for row in rows
        ),
        "future_other_reliable_positive_rate": _mean(
            _float(row, "future_sub_cost_other_lower") > 0.0 for row in rows
        ),
        "total_substitution_reliable_positive_rate": _mean(
            _float(row, "sub_cost_lower") > 0.0 for row in rows
        ),
        "negative_or_zero_episode_delta_rate": _mean(
            _float(row, "episode_cost_delta_mean") <= tolerance for row in rows
        ),
        "mean_cost_sign_masked_rate": _mean(
            _float(row, "cost_sign_masked_rate") for row in rows
        ),
        "contexts_majority_masked_rate": _mean(
            _float(row, "cost_sign_masked_rate") >= 0.5 for row in rows
        ),
        "mean_substitution_ratio": _mean(
            _float(row, "rho_sub_mean") for row in rows
        ),
    }


def _support_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    groups["overall"].extend(rows)
    for row in rows:
        groups[f"scenario={row['scenario']}"].append(row)
        groups[f"resource_type={row['resource_type']}"].append(row)
        groups[f"slot={row['slot']}"].append(row)
        groups[
            f"scenario={row['scenario']}|resource_type={row['resource_type']}"
        ].append(row)
    return [_support_row(name, groups[name]) for name in sorted(groups)]


def _candidate_rows() -> list[dict[str, Any]]:
    return [
        {
            "candidate": "component_preserving_constrained_credit",
            "problem_fit": "strong",
            "label_semantics": "development_pass",
            "literature_distance": "adjacent_not_yet_sufficient",
            "policy_objective_risk": "high",
            "fallback_contract": "design_pass",
            "decision": "method_component_only",
            "reason": (
                "The ledger channels are identifiable, but forcing direct cost "
                "to remain uncompensated may change the global episode-cost "
                "objective; ICML 2025 already decomposes action effects through "
                "subsequent agents and state transitions."
            ),
        },
        {
            "candidate": "global_cmdp_constraint",
            "problem_fit": "strong_for_global_budget",
            "label_semantics": "pass",
            "literature_distance": "directly_covered",
            "policy_objective_risk": "low",
            "fallback_contract": "available_in_prior_work",
            "decision": "strong_baseline",
            "reason": (
                "CPO, constrained multi-objective RL, and scalable safe MARL "
                "already treat cumulative cost as an explicit global constraint."
            ),
        },
        {
            "candidate": "controlled_continuation_difference_reward",
            "problem_fit": "partial",
            "label_semantics": "fails_without_new_causal_assumptions",
            "literature_distance": "directly_adjacent",
            "policy_objective_risk": "high",
            "fallback_contract": "not_established",
            "decision": "reject_as_primary",
            "reason": (
                "CCA, COCOA, DAE, and causal effect-decomposition work already "
                "cover related interventions; fixing continuation can introduce "
                "policy bias or leave the factual policy distribution."
            ),
        },
    ]


def run_audit(
    *,
    input_dir: Path,
    output_dir: Path,
    software_tests_passed: bool,
) -> dict[str, Any]:
    contexts = _read_csv(input_dir / "context_substitution_estimates.csv")
    ledger = _read_csv(input_dir / "repeat_cost_ledger.csv")
    source_gate = json.loads(
        (input_dir / "gate_summary.json").read_text(encoding="utf-8")
    )

    maximum_identity_error = 0.0
    maximum_direct_cost_error = 0.0
    ambiguous_rows = 0
    for row in ledger:
        components = ResourceCreditComponents(
            direct_cost=_float(row, "direct_cost"),
            same_step_other_substitution=_float(
                row, "same_step_other_sub_cost"
            ),
            future_probe_substitution=_float(row, "future_sub_cost_probe"),
            future_other_substitution=_float(row, "future_sub_cost_other"),
        )
        maximum_identity_error = max(
            maximum_identity_error,
            abs(
                components.episode_cost_delta
                - _float(row, "episode_cost_delta")
            ),
        )
        maximum_direct_cost_error = max(
            maximum_direct_cost_error,
            abs(components.direct_cost - _float(row, "unit_cost")),
        )
        ambiguous_rows += int(
            components.direct_cost > 1e-12
            and components.episode_cost_delta <= 1e-12
        )

    support_rows = _support_rows(contexts)
    candidate_rows = _candidate_rows()
    _write_csv(output_dir / "support_summary.csv", support_rows)
    _write_csv(output_dir / "candidate_comparison.csv", candidate_rows)

    _write_json(
        output_dir / "experiment_config.json",
        {
            "task": "N1_offline_semantic_audit",
            "data_role": "development_only",
            "input_directory": str(input_dir.relative_to(PROJECT_ROOT)),
            "output_directory": str(output_dir.relative_to(PROJECT_ROOT)),
            "decomposition_tolerance": 1e-6,
            "software_tests_passed": software_tests_passed,
            "online_training_authorized": False,
        },
    )
    _write_json(
        output_dir / "label_dictionary.json",
        {
            "direction": "N_minus_E substitution; E_minus_N episode cost",
            "direct_cost": "known immediate cost of the probed engage action",
            "same_step_other_substitution": (
                "other-unit cost in N minus E at the same autoregressive step"
            ),
            "future_probe_substitution": (
                "future probed-unit cost in N minus E"
            ),
            "future_other_substitution": (
                "future other-unit cost in N minus E"
            ),
            "episode_cost_delta": (
                "direct_cost - all substitution components"
            ),
            "prohibited_label_rule": (
                "episode_cost_delta sign must not be converted directly into "
                "a local STOP/ENGAGE target"
            ),
        },
    )
    _write_json(
        output_dir / "seed_usage_audit.json",
        {
            "development_policy_seeds": [17, 18, 19],
            "development_scenarios": [
                "medium",
                "time_pressure",
                "heterogeneity_pressure",
            ],
            "contexts": len(contexts),
            "target_ledger_rows": len(ledger),
            "independent_confirmation_claim_for_new_algorithm": False,
            "future_rule": (
                "select consecutive unused seeds before observing any online "
                "candidate outcome; never replace a preregistered seed"
            ),
        },
    )

    p1 = (
        maximum_identity_error <= 1e-6
        and maximum_direct_cost_error <= 1e-6
        and source_gate["maximum_sub_cost_decomposition_error"] <= 1e-6
    )
    p2 = False
    p3 = software_tests_passed
    p4 = True
    p5 = True
    summary = {
        "context_count": len(contexts),
        "target_ledger_rows": len(ledger),
        "maximum_reconstructed_identity_error": maximum_identity_error,
        "maximum_direct_cost_error": maximum_direct_cost_error,
        "ambiguous_target_ledger_rows": ambiguous_rows,
        "ambiguous_target_ledger_rate": ambiguous_rows / len(ledger),
        "N1_propositions": {
            "N1-P1_label_semantics": p1,
            "N1-P2_defensible_algorithmic_difference": p2,
            "N1-P3_strict_fallback_contract": p3,
            "N1-P4_development_data_boundary": p4,
            "N1-P5_online_stop_rules_frozen": p5,
        },
        "stage_passed": all((p1, p2, p3, p4, p5)),
        "decision": "N1-E4_redefine_mainline_algorithm_problem",
        "online_training_authorized": False,
        "decision_reason": (
            "The decomposition is semantically exact, but no candidate both "
            "preserves the intended global resource objective and establishes "
            "sufficient algorithmic distance from existing causal credit and "
            "constrained-RL methods."
        ),
    }
    _write_json(output_dir / "gate_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the N1 development-only resource-credit semantic audit."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--software-tests-passed",
        action="store_true",
        help="Record that the focused N1 software tests passed in this run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_audit(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        software_tests_passed=args.software_tests_passed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
