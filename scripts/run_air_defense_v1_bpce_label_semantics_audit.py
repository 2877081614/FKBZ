from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.algorithms.policy_gradient import (  # noqa: E402
    FactorizedEngagementMaskablePPO,
)
from rein_learning.common import (  # noqa: E402
    BPCELabelSemanticsConfig,
    audit_bpce_context,
    collect_bpce_audit_contexts,
    summarize_bpce_semantics,
)
from rein_learning.envs import (  # noqa: E402
    AirDefenseResourceAssignmentEnvV1,
    get_air_defense_v1_scenario,
)


SCENARIOS = ("time_pressure", "heterogeneity_pressure")
POLICY_SEEDS = (8, 9, 10)
MODEL_ROOT = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "mch_ppo_mechanism_stress_test"
    / "models"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "bpce_label_semantics_audit"
)
METHOD = "factorized_engagement_ar_ppo_order_012"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen BPCE engagement-label semantics audit."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Recompute summaries from an existing context_labels.csv.",
    )
    return parser.parse_args()


def _model_path(model_root: Path, scenario: str, seed: int) -> Path:
    return (
        model_root
        / scenario
        / f"{METHOD}_seed{seed}.zip"
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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
    before: dict[str, torch.Tensor], policy: Any
) -> float:
    return max(
        float(
            torch.max(
                torch.abs(before[name] - value.detach().cpu())
            ).item()
        )
        for name, value in policy.state_dict().items()
    )


def _block_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks = sorted(
        {
            (str(row["scenario"]), int(row["policy_seed"]))
            for row in rows
        }
    )
    summaries: list[dict[str, Any]] = []
    for scenario, seed in blocks:
        selected = [
            row
            for row in rows
            if row["scenario"] == scenario
            and int(row["policy_seed"]) == seed
        ]
        summary: dict[str, Any] = {
            "scenario": scenario,
            "policy_seed": seed,
            "contexts": len(selected),
        }
        for label in ("a", "b", "c"):
            reliable = [
                row
                for row in selected
                if _as_bool(row[f"{label}_reliable"])
            ]
            summary[f"{label}_reliable"] = len(reliable)
            summary[f"{label}_positive"] = sum(
                int(row[f"{label}_sign"]) > 0 for row in reliable
            )
            summary[f"{label}_negative"] = sum(
                int(row[f"{label}_sign"]) < 0 for row in reliable
            )
        for left, right in (("a", "b"), ("b", "c")):
            eligible = [
                row
                for row in selected
                if _as_bool(row[f"{left}_reliable"])
                and _as_bool(row[f"{right}_reliable"])
            ]
            summary[f"{left}_{right}_agreement"] = (
                float(
                    np.mean(
                        [
                            int(row[f"{left}_sign"])
                            == int(row[f"{right}_sign"])
                            for row in eligible
                        ]
                    )
                )
                if eligible
                else 0.0
            )
            summary[f"{left}_{right}_eligible"] = len(eligible)
        summary["target_sign_reversal_count"] = sum(
            _as_bool(row["target_selection_sign_reversal"])
            for row in selected
        )
        summary["mean_argmax_target_regret"] = float(
            np.mean([float(row["argmax_target_regret"]) for row in selected])
        )
        summary["extra_transitions"] = sum(
            int(row["extra_transitions"]) for row in selected
        )
        summaries.append(summary)
    return summaries


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _read_context_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["policy_seed"] = int(row["policy_seed"])
        for label in ("a", "b", "c"):
            row[f"{label}_sign"] = int(row[f"{label}_sign"])
            row[f"{label}_reliable"] = _as_bool(
                row[f"{label}_reliable"]
            )
        row["target_selection_sign_reversal"] = _as_bool(
            row["target_selection_sign_reversal"]
        )
    return rows


def main() -> None:
    args = _parse_args()
    started_at = datetime.now(timezone.utc)
    scenarios = SCENARIOS
    policy_seeds = POLICY_SEEDS
    config = BPCELabelSemanticsConfig()
    output_dir = args.output_dir.resolve()
    if args.smoke:
        scenarios = ("time_pressure",)
        policy_seeds = (8,)
        config = BPCELabelSemanticsConfig(
            contexts_per_slot=1,
            pool_episodes=2,
            repeats=2,
        )
        if args.output_dir == OUTPUT_ROOT:
            output_dir = OUTPUT_ROOT.with_name(
                "bpce_label_semantics_audit_smoke"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize_existing:
        context_path = output_dir / "context_labels.csv"
        if not context_path.is_file():
            raise FileNotFoundError(context_path)
        context_rows = _read_context_rows(context_path)
        gate_summary = summarize_bpce_semantics(
            context_rows, config=config
        )
        existing_gate_path = output_dir / "gate_summary.json"
        if existing_gate_path.is_file():
            existing_gate = json.loads(
                existing_gate_path.read_text(encoding="utf-8")
            )
            gate_summary["maximum_actor_parameter_difference"] = (
                existing_gate.get("maximum_actor_parameter_difference", 0.0)
            )
            gate_summary["actor_unchanged"] = existing_gate.get(
                "actor_unchanged", True
            )
        _write_json(existing_gate_path, gate_summary)
        _write_csv(
            output_dir / "block_summary.csv",
            _block_summary(context_rows),
        )
        config_path = output_dir / "experiment_config.json"
        if config_path.is_file():
            existing_config = json.loads(
                config_path.read_text(encoding="utf-8")
            )
            existing_config["stage_a_passed"] = gate_summary[
                "stage_a_passed"
            ]
            existing_config["label_decision"] = gate_summary[
                "label_decision"
            ]
            existing_config["summary_recomputed_at_utc"] = datetime.now(
                timezone.utc
            ).isoformat()
            _write_json(config_path, existing_config)
        print(json.dumps(gate_summary, ensure_ascii=False, indent=2))
        return

    model_paths = {
        f"{scenario}/seed{seed}": str(
            _model_path(args.model_root.resolve(), scenario, seed)
        )
        for scenario in scenarios
        for seed in policy_seeds
    }
    missing = [path for path in model_paths.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing frozen factorized PPO models:\n" + "\n".join(missing)
        )

    experiment_config: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": started_at.isoformat(),
        "task": "bpce_label_semantics_audit_stage_a",
        "actor_updates": False,
        "smoke": bool(args.smoke),
        "scenarios": list(scenarios),
        "policy_seeds": list(policy_seeds),
        "model_paths": model_paths,
        "device": args.device,
        "audit_config": asdict(config),
        "frozen_gates": {
            "total_contexts": 72,
            "minimum_reliable_contexts": 48,
            "minimum_reliable_per_block": 6,
            "minimum_overall_a_b_sign_agreement": 0.80,
            "minimum_worst_scenario_a_b_sign_agreement": 0.70,
            "minimum_overall_b_c_sign_agreement": 0.80,
            "minimum_worst_scenario_b_c_sign_agreement": 0.70,
            "maximum_target_selection_sign_reversal": 0.20,
            "minimum_positive_and_negative_per_scenario": 6,
            "minimum_component_consistency": (
                config.minimum_component_consistency
            ),
        },
    }
    _write_json(output_dir / "experiment_config.json", experiment_config)

    context_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    maximum_parameter_difference = 0.0
    for scenario in scenarios:
        env_config = get_air_defense_v1_scenario(scenario)
        for seed in policy_seeds:
            model_path = Path(model_paths[f"{scenario}/seed{seed}"])
            load_env = AirDefenseResourceAssignmentEnvV1(config=env_config)
            model = FactorizedEngagementMaskablePPO.load(
                model_path,
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
                config=config,
            )
            print(
                f"[{scenario}/seed{seed}] collected {len(contexts)} contexts",
                flush=True,
            )
            for context_index, context in enumerate(contexts, start=1):
                aggregate, repeats, targets = audit_bpce_context(
                    policy=model.policy,
                    env_config=env_config,
                    context=context,
                    config=config,
                )
                context_rows.append(aggregate)
                repeat_rows.extend(repeats)
                target_rows.extend(targets)
                print(
                    f"[{scenario}/seed{seed}] "
                    f"{context_index}/{len(contexts)} {context.context_id}",
                    flush=True,
                )
            maximum_parameter_difference = max(
                maximum_parameter_difference,
                _maximum_parameter_difference(before, model.policy),
            )
            load_env.close()
            _write_csv(output_dir / "context_labels.csv", context_rows)
            _write_csv(output_dir / "repeat_deltas.csv", repeat_rows)
            _write_csv(output_dir / "target_outcomes.csv", target_rows)
            _write_csv(
                output_dir / "block_summary.csv",
                _block_summary(context_rows),
            )

    gate_summary = summarize_bpce_semantics(context_rows, config=config)
    gate_summary["maximum_actor_parameter_difference"] = (
        maximum_parameter_difference
    )
    gate_summary["actor_unchanged"] = maximum_parameter_difference == 0.0
    _write_json(output_dir / "gate_summary.json", gate_summary)

    experiment_config.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "result_counts": {
                "contexts": len(context_rows),
                "repeat_rows": len(repeat_rows),
                "target_rows": len(target_rows),
                "extra_transitions": sum(
                    int(row["extra_transitions"]) for row in context_rows
                ),
            },
            "stage_a_passed": gate_summary["stage_a_passed"],
            "label_decision": gate_summary["label_decision"],
            "maximum_actor_parameter_difference": (
                maximum_parameter_difference
            ),
        }
    )
    _write_json(output_dir / "experiment_config.json", experiment_config)
    print(json.dumps(gate_summary, ensure_ascii=False, indent=2))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
