from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.algorithms.policy_gradient import (  # noqa: E402
    FactorizedEngagementMaskablePPO,
)
from rein_learning.common import (  # noqa: E402
    ActionSubstitutionOpportunityCostConfig,
    BPCELabelSemanticsConfig,
    aggregate_target_rows,
    audit_action_substitution_context,
    collect_bpce_audit_contexts,
    summarize_action_substitution_audit,
    validate_context_identity,
)
from rein_learning.envs import (  # noqa: E402
    AirDefenseResourceAssignmentEnvV1,
    get_air_defense_v1_scenario,
)


SCENARIOS = ("time_pressure", "heterogeneity_pressure")
POLICY_SEEDS = (8, 9, 10)
METHOD = "factorized_engagement_ar_ppo_order_012"
MODEL_ROOT = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "mch_ppo_mechanism_stress_test"
    / "models"
)
REFERENCE_PATH = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "bpce_label_semantics_audit"
    / "context_labels.csv"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "action_substitution_opportunity_cost_audit"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit action substitution and frozen-policy ammunition "
            "opportunity cost."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument(
        "--reference-contexts", type=Path, default=REFERENCE_PATH
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true")
    parser.add_argument("--software-tests-passed", action="store_true")
    return parser.parse_args()


def _model_path(model_root: Path, scenario: str, seed: int) -> Path:
    return model_root / scenario / f"{METHOD}_seed{seed}.zip"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _parameter_snapshot(policy: Any) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
    }


def _maximum_parameter_difference(
    before: Mapping[str, torch.Tensor], policy: Any
) -> float:
    return max(
        float(
            torch.max(
                torch.abs(before[name] - value.detach().cpu())
            ).item()
        )
        for name, value in policy.state_dict().items()
    )


def _block_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for seed in POLICY_SEEDS:
            for slot in ("safety", "resource"):
                selected = [
                    row
                    for row in rows
                    if str(row["scenario"]) == scenario
                    and int(row["policy_seed"]) == seed
                    and str(row["slot"]) == slot
                ]
                if not selected:
                    continue
                output.append(
                    {
                        "scenario": scenario,
                        "policy_seed": seed,
                        "slot": slot,
                        "contexts": len(selected),
                        "reliable_opportunity": sum(
                            _as_bool(row["reliable_opportunity"])
                            for row in selected
                        ),
                        "mean_sub_shot": sum(
                            float(row["sub_shot_mean"])
                            for row in selected
                        )
                        / len(selected),
                        "mean_sub_cost": sum(
                            float(row["sub_cost_mean"])
                            for row in selected
                        )
                        / len(selected),
                        "mean_reuse_probe": sum(
                            float(row["reuse_probe_mean"])
                            for row in selected
                        )
                        / len(selected),
                        "mean_option_edge": sum(
                            float(row["option_edge_mean"])
                            for row in selected
                        )
                        / len(selected),
                        "mean_ammo_gain_damage": sum(
                            float(row["ammo_gain_damage_mean"])
                            for row in selected
                        )
                        / len(selected),
                        "mean_ammo_gain_leaks": sum(
                            float(row["ammo_gain_leaks_mean"])
                            for row in selected
                        )
                        / len(selected),
                        "actual_extra_transitions": sum(
                            int(row["actual_extra_transitions"])
                            for row in selected
                        ),
                    }
                )
    return output


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _summarize_existing(
    output_dir: Path,
    *,
    config: ActionSubstitutionOpportunityCostConfig,
    software_tests_passed: bool,
) -> None:
    context_rows = _read_csv(output_dir / "context_opportunity_estimates.csv")
    identity_rows = _read_csv(output_dir / "context_identity_check.csv")
    integrity_rows = _read_csv(output_dir / "intervention_integrity.csv")
    existing_path = output_dir / "gate_summary.json"
    existing = (
        json.loads(existing_path.read_text(encoding="utf-8"))
        if existing_path.is_file()
        else {}
    )
    maximum_difference = float(
        existing.get("maximum_actor_parameter_difference", 0.0)
    )
    summary = summarize_action_substitution_audit(
        context_rows,
        identity_rows,
        integrity_rows,
        config=config,
        maximum_actor_parameter_difference=maximum_difference,
        software_tests_passed=software_tests_passed,
    )
    _write_json(existing_path, summary)
    _write_csv(output_dir / "block_summary.csv", _block_summary(context_rows))
    experiment_path = output_dir / "experiment_config.json"
    if experiment_path.is_file():
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        experiment["software_tests_passed"] = software_tests_passed
        experiment["stage_passed"] = summary["stage_passed"]
        experiment["decision"] = summary["decision"]
        experiment["summary_recomputed_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        _write_json(experiment_path, experiment)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = _parse_args()
    started_at = datetime.now(timezone.utc)
    output_dir = args.output_dir.resolve()
    if args.smoke and args.output_dir == OUTPUT_ROOT:
        output_dir = OUTPUT_ROOT.with_name(
            "action_substitution_opportunity_cost_audit_smoke"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = (
        ActionSubstitutionOpportunityCostConfig(repeats=2)
        if args.smoke
        else ActionSubstitutionOpportunityCostConfig()
    )
    if args.summarize_existing:
        _summarize_existing(
            output_dir,
            config=config,
            software_tests_passed=args.software_tests_passed,
        )
        return

    experiment_path = output_dir / "experiment_config.json"
    if not args.smoke and experiment_path.is_file():
        existing = json.loads(experiment_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            raise FileExistsError(
                "Formal audit already completed; use --summarize-existing"
            )
    reference_path = args.reference_contexts.resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    reference_rows = _read_csv(reference_path)

    scenarios = ("time_pressure",) if args.smoke else SCENARIOS
    seeds = (8,) if args.smoke else POLICY_SEEDS
    model_paths = {
        f"{scenario}/seed{seed}": str(
            _model_path(args.model_root.resolve(), scenario, seed)
        )
        for scenario in scenarios
        for seed in seeds
    }
    missing = [path for path in model_paths.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing frozen models:\n" + "\n".join(missing)
        )
    experiment: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": started_at.isoformat(),
        "task": "action_substitution_resource_opportunity_cost_audit",
        "actor_updates": False,
        "smoke": bool(args.smoke),
        "scenarios": list(scenarios),
        "policy_seeds": list(seeds),
        "model_paths": model_paths,
        "reference_contexts": str(reference_path),
        "device": args.device,
        "audit_config": asdict(config),
        "frozen_branch_semantics": {
            "N": "current probed unit forced noop",
            "E": "current probed unit forced legal engage",
            "E-R": (
                "same current E step, then restore exactly one probed-unit "
                "ammo before the next policy observation"
            ),
            "future_metrics_exclude_current_step": True,
            "target_handling": "exact conditional probability marginalization",
        },
        "frozen_gates": {
            "contexts": 72,
            "repeats_per_context": 32,
            "P-R1_positive_mean_sub_shot": 12,
            "P-R1_positive_lower_sub_shot": 6,
            "P-R1_explained_fraction": 0.80,
            "P-R2_resource_contexts_per_scenario": 6,
            "P-R2_resource_contexts_per_block": 2,
            "P-R3_multiple_unit_types": 2,
            "maximum_extra_transitions": (
                config.maximum_extra_transitions
            ),
        },
    }
    _write_json(experiment_path, experiment)

    context_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    target_repeat_rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    maximum_parameter_difference = 0.0
    context_config = BPCELabelSemanticsConfig()

    for scenario in scenarios:
        env_config = get_air_defense_v1_scenario(scenario)
        for seed in seeds:
            load_env = AirDefenseResourceAssignmentEnvV1(config=env_config)
            model = FactorizedEngagementMaskablePPO.load(
                Path(model_paths[f"{scenario}/seed{seed}"]),
                env=load_env,
                device=args.device,
            )
            model.policy.set_training_mode(False)
            before = _parameter_snapshot(model.policy)
            contexts = collect_bpce_audit_contexts(
                policy=model.policy,
                env_config=env_config,
                scenario=scenario,
                policy_seed=seed,
                config=context_config,
            )
            block_references = [
                row
                for row in reference_rows
                if row["scenario"] == scenario
                and int(row["policy_seed"]) == seed
            ]
            block_identity = validate_context_identity(
                contexts,
                block_references,
                probability_tolerance=(
                    config.identity_probability_tolerance
                ),
            )
            identity_rows.extend(block_identity)
            _write_csv(
                output_dir / "context_identity_check.csv", identity_rows
            )
            if not all(row["matched"] for row in block_identity):
                raise RuntimeError(
                    f"Context identity mismatch in {scenario}/seed{seed}"
                )
            selected = (
                (contexts[0], contexts[6]) if args.smoke else contexts
            )
            print(
                f"[{scenario}/seed{seed}] identity matched "
                f"{len(contexts)}/{len(contexts)}",
                flush=True,
            )
            for index, context in enumerate(selected, start=1):
                aggregate, repeats, target_repeats, integrity = (
                    audit_action_substitution_context(
                        policy=model.policy,
                        env_config=env_config,
                        context=context,
                        config=config,
                    )
                )
                context_rows.append(aggregate)
                repeat_rows.extend(repeats)
                target_repeat_rows.extend(target_repeats)
                integrity_rows.extend(integrity)
                print(
                    f"[{scenario}/seed{seed}] {index}/{len(selected)} "
                    f"{context.context_id}",
                    flush=True,
                )
            maximum_parameter_difference = max(
                maximum_parameter_difference,
                _maximum_parameter_difference(before, model.policy),
            )
            load_env.close()
            _write_csv(
                output_dir / "context_opportunity_estimates.csv",
                context_rows,
            )
            _write_csv(
                output_dir / "repeat_branch_metrics.csv", repeat_rows
            )
            _write_csv(
                output_dir / "intervention_integrity.csv", integrity_rows
            )
            _write_csv(
                output_dir / "target_opportunity_estimates.csv",
                aggregate_target_rows(
                    target_repeat_rows,
                    confidence_z=config.confidence_z,
                ),
            )
            _write_csv(
                output_dir / "block_summary.csv",
                _block_summary(context_rows),
            )

    summary = summarize_action_substitution_audit(
        context_rows,
        identity_rows,
        integrity_rows,
        config=config,
        maximum_actor_parameter_difference=(
            maximum_parameter_difference
        ),
        software_tests_passed=args.software_tests_passed,
    )
    _write_json(output_dir / "gate_summary.json", summary)
    experiment.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "software_tests_passed": args.software_tests_passed,
            "result_counts": {
                "identity_rows": len(identity_rows),
                "contexts": len(context_rows),
                "repeat_rows": len(repeat_rows),
                "target_repeat_rows": len(target_repeat_rows),
                "intervention_integrity_rows": len(integrity_rows),
                "actual_extra_transitions": summary[
                    "actual_extra_transitions"
                ],
            },
            "stage_passed": summary["stage_passed"],
            "decision": summary["decision"],
            "maximum_actor_parameter_difference": (
                maximum_parameter_difference
            ),
        }
    )
    _write_json(experiment_path, experiment)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
