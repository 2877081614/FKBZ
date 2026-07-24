from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
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
    ActionSubstitutionConfirmationConfig,
    audit_confirmation_context,
    collect_confirmation_contexts,
    grouped_summary_rows,
    summarize_confirmation,
    validate_confirmation_contexts,
)
from rein_learning.envs import (  # noqa: E402
    AirDefenseResourceAssignmentEnvV1,
    get_air_defense_v1_scenario,
)
from rein_learning.trainers.air_defense_v1_ppo import (  # noqa: E402
    AirDefenseV1PPOConfig,
    train_factorized_engagement_autoregressive_ppo,
)


SCENARIOS = ("medium", "time_pressure", "heterogeneity_pressure")
POLICY_SEEDS = (17, 18, 19)
METHOD = "factorized_engagement_ar_ppo_order_012"
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "action_substitution_confirmation"
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
        description="Independent confirmation of action-substitution cost distortion."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare-models-only", action="store_true")
    parser.add_argument(
        "--rerun-ledger-correction",
        action="store_true",
        help="Archive the invalid first ledger and perform the one allowed rerun.",
    )
    parser.add_argument("--software-tests-passed", action="store_true")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
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
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )


def _archive_pre_correction(output_dir: Path) -> None:
    archive = output_dir / "pre_ledger_correction"
    archive.mkdir(parents=True, exist_ok=True)
    for name in (
        "experiment_config.json",
        "seed_usage_audit.json",
        "source_model_manifest.json",
        "source_model_training_log.csv",
        "context_identity_check.csv",
        "context_selection.csv",
        "repeat_cost_ledger.csv",
        "repeat_marginal_metrics.csv",
        "context_substitution_estimates.csv",
        "block_summary.csv",
        "resource_type_summary.csv",
        "scenario_boundary_summary.csv",
        "gate_summary.json",
    ):
        source = output_dir / name
        if source.is_file():
            shutil.copy2(source, archive / name)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _model_path(output_dir: Path, scenario: str, seed: int) -> Path:
    return (
        output_dir
        / "source_models"
        / scenario
        / f"{METHOD}_seed{seed}.zip"
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_usage_audit(output_dir: Path) -> dict[str, Any]:
    factorized_hits = [
        str(path.relative_to(PROJECT_ROOT))
        for path in (
            PROJECT_ROOT / "results" / "air_defense_v1"
        ).rglob("*.zip")
        if METHOD in path.name
        and any(f"seed{seed}" in path.name for seed in POLICY_SEEDS)
    ]
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_seeds": list(POLICY_SEEDS),
        "replacement_required": False,
        "factorized_source_models_found_before_task": factorized_hits,
        "action_substitution_or_cost_label_design_use": [],
        "incidental_non_policy_uses": {
            "unit_tests": [17, 19],
            "task14_predictor_training": [17, 18, 19],
        },
        "decision": (
            "retain_17_18_19_no_prior_action_substitution_design_use"
        ),
        "audit_scope": (
            "Existing result/model names and project text were checked before "
            "source-model training; incidental test or predictor seeds do not "
            "constitute action-substitution policy selection."
        ),
    }
    _write_json(output_dir / "seed_usage_audit.json", payload)
    return payload


def _training_config(seed: int, device: str, timesteps: int) -> AirDefenseV1PPOConfig:
    return AirDefenseV1PPOConfig(
        total_timesteps=timesteps,
        n_steps=min(256, timesteps),
        batch_size=64,
        n_epochs=2,
        seed=seed,
        device=device,
        verbose=0,
    )


def _prepare_models(
    output_dir: Path,
    *,
    scenarios: Sequence[str],
    seeds: Sequence[int],
    device: str,
    timesteps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest: list[dict[str, Any]] = []
    training_log: list[dict[str, Any]] = []
    for scenario in scenarios:
        env_config = get_air_defense_v1_scenario(scenario)
        for seed in seeds:
            path = _model_path(output_dir, scenario, seed)
            started = time.perf_counter()
            loaded_existing = path.is_file()
            if not loaded_existing:
                train_factorized_engagement_autoregressive_ppo(
                    env_config=env_config,
                    train_config=_training_config(seed, device, timesteps),
                    save_path=path,
                    unit_order=(0, 1, 2),
                )
            elapsed = time.perf_counter() - started
            if not path.is_file():
                raise FileNotFoundError(path)
            manifest.append(
                {
                    "scenario": scenario,
                    "policy_seed": seed,
                    "method": METHOD,
                    "requested_timesteps": timesteps,
                    "n_epochs": 2,
                    "model_path": str(path.resolve()),
                    "sha256": _file_sha256(path),
                    "loaded_existing": loaded_existing,
                    "selected_by_behavior": False,
                }
            )
            training_log.append(
                {
                    "scenario": scenario,
                    "policy_seed": seed,
                    "status": "loaded" if loaded_existing else "trained",
                    "elapsed_seconds": elapsed,
                    "model_path": str(path.resolve()),
                }
            )
            print(
                f"[source] {scenario}/seed{seed} "
                f"{'loaded' if loaded_existing else 'trained'}",
                flush=True,
            )
            _write_json(
                output_dir / "source_model_manifest.json",
                {"models": manifest},
            )
            _write_csv(
                output_dir / "source_model_training_log.csv", training_log
            )
    return manifest, training_log


def _old_hashes() -> tuple[set[str], list[dict[str, Any]]]:
    hashes: set[str] = set()
    sources: list[dict[str, Any]] = []
    for path in OLD_CONTEXT_PATHS:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = _read_csv(path)
        available = {
            row["observation_hash"]
            for row in rows
            if row.get("observation_hash")
        }
        hashes.update(available)
        sources.append(
            {
                "path": str(path.resolve()),
                "rows": len(rows),
                "hashes_available": len(available),
            }
        )
    return hashes, sources


def _context_selection_rows(contexts: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
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
            "unit_cost": context.snapshot.defense_units[
                context.unit_index
            ].cost,
            "legal_targets": ",".join(map(str, context.legal_targets)),
            "target_probabilities": ",".join(
                f"{value:.12g}" for value in context.target_probabilities
            ),
            "safety_score": context.safety_score,
            "resource_score": context.resource_score,
        }
        for context in contexts
    ]


def _parameter_snapshot(policy: Any) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
    }


def _parameter_difference(
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


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    if args.smoke and args.output_dir == OUTPUT_ROOT:
        output_dir = OUTPUT_ROOT.with_name(
            "action_substitution_confirmation_smoke"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    formal_config = ActionSubstitutionConfirmationConfig()
    config = (
        replace(formal_config, repeats=2)
        if args.smoke
        else formal_config
    )
    scenarios = ("time_pressure",) if args.smoke else SCENARIOS
    seeds = (17,) if args.smoke else POLICY_SEEDS
    timesteps = 256 if args.smoke else 10_000
    experiment_path = output_dir / "experiment_config.json"
    if not args.smoke and experiment_path.is_file():
        existing = json.loads(experiment_path.read_text(encoding="utf-8"))
        if args.rerun_ledger_correction:
            if existing.get("decision") != "invalid_cost_ledger_fix_only":
                raise RuntimeError(
                    "Ledger correction rerun requires the frozen P-C1 failure"
                )
            _archive_pre_correction(output_dir)
        elif existing.get("status") == "completed":
            raise FileExistsError("Formal independent confirmation is complete")

    seed_audit = _seed_usage_audit(output_dir)
    old_hashes, old_hash_sources = _old_hashes()
    experiment: dict[str, Any] = {
        "schema_version": 1,
        "status": "preparing_models",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "action_substitution_independent_confirmation_r2",
        "scenarios": list(scenarios),
        "policy_seeds": list(seeds),
        "method": METHOD,
        "source_training": {
            "requested_timesteps": timesteps,
            "n_steps": min(256, timesteps),
            "batch_size": 64,
            "n_epochs": 2,
            "device": args.device,
            "behavior_selection": False,
        },
        "confirmation_config": asdict(config),
        "old_hash_sources": old_hash_sources,
        "old_hash_count": len(old_hashes),
        "seed_usage_decision": seed_audit["decision"],
        "E-R_enabled": False,
        "actor_updates_during_confirmation": False,
        "ledger_correction_rerun": bool(args.rerun_ledger_correction),
    }
    _write_json(experiment_path, experiment)
    manifest, _ = _prepare_models(
        output_dir,
        scenarios=scenarios,
        seeds=seeds,
        device=args.device,
        timesteps=timesteps,
    )
    if args.prepare_models_only:
        experiment["status"] = "models_prepared"
        experiment["source_model_count"] = len(manifest)
        _write_json(experiment_path, experiment)
        return

    experiment["status"] = "running_confirmation"
    _write_json(experiment_path, experiment)
    identity_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    maximum_parameter_difference = 0.0
    for scenario in scenarios:
        env_config = get_air_defense_v1_scenario(scenario)
        for seed in seeds:
            load_env = AirDefenseResourceAssignmentEnvV1(config=env_config)
            model = FactorizedEngagementMaskablePPO.load(
                _model_path(output_dir, scenario, seed),
                env=load_env,
                device=args.device,
            )
            model.policy.set_training_mode(False)
            before = _parameter_snapshot(model.policy)
            contexts = collect_confirmation_contexts(
                policy=model.policy,
                env_config=env_config,
                scenario=scenario,
                policy_seed=seed,
                excluded_observation_hashes=old_hashes,
                config=config,
            )
            block_identity = validate_confirmation_contexts(
                contexts,
                policy=model.policy,
                excluded_observation_hashes=old_hashes,
                probability_tolerance=config.probability_tolerance,
            )
            identity_rows.extend(block_identity)
            selection_rows.extend(_context_selection_rows(contexts))
            _write_csv(
                output_dir / "context_identity_check.csv", identity_rows
            )
            _write_csv(
                output_dir / "context_selection.csv", selection_rows
            )
            if not all(row["matched"] for row in block_identity):
                raise RuntimeError(
                    f"Context integrity failure in {scenario}/seed{seed}"
                )
            selected_contexts = (
                tuple(
                    next(
                        context
                        for context in contexts
                        if context.slot == "resource"
                        and context.snapshot.defense_units[
                            context.unit_index
                        ].resource_type
                        == resource_type
                    )
                    for resource_type in ("missile", "laser")
                )
                if args.smoke
                else contexts
            )
            print(
                f"[{scenario}/seed{seed}] selected {len(contexts)} contexts",
                flush=True,
            )
            for index, context in enumerate(selected_contexts, start=1):
                aggregate, repeats, targets = audit_confirmation_context(
                    policy=model.policy,
                    env_config=env_config,
                    context=context,
                    config=config,
                )
                context_rows.append(aggregate)
                repeat_rows.extend(repeats)
                target_rows.extend(targets)
                print(
                    f"[{scenario}/seed{seed}] {index}/"
                    f"{len(selected_contexts)} {context.context_id}",
                    flush=True,
                )
            maximum_parameter_difference = max(
                maximum_parameter_difference,
                _parameter_difference(before, model.policy),
            )
            load_env.close()
            _write_csv(
                output_dir / "context_substitution_estimates.csv",
                context_rows,
            )
            _write_csv(
                output_dir / "repeat_marginal_metrics.csv", repeat_rows
            )
            _write_csv(
                output_dir / "repeat_cost_ledger.csv", target_rows
            )

    _write_csv(
        output_dir / "block_summary.csv",
        grouped_summary_rows(
            context_rows,
            group_fields=("scenario", "policy_seed", "slot"),
            config=config,
        ),
    )
    _write_csv(
        output_dir / "resource_type_summary.csv",
        grouped_summary_rows(
            context_rows,
            group_fields=("scenario", "slot", "resource_type"),
            config=config,
        ),
    )
    _write_csv(
        output_dir / "scenario_boundary_summary.csv",
        grouped_summary_rows(
            context_rows,
            group_fields=("scenario", "slot"),
            config=config,
        ),
    )
    summary = summarize_confirmation(
        context_rows,
        repeat_rows,
        target_rows,
        identity_rows,
        source_model_count=len(manifest),
        config=config,
        maximum_actor_parameter_difference=maximum_parameter_difference,
        software_tests_passed=args.software_tests_passed,
    )
    _write_json(output_dir / "gate_summary.json", summary)
    experiment.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_model_count": len(manifest),
            "software_tests_passed": args.software_tests_passed,
            "result_counts": {
                "identity_rows": len(identity_rows),
                "selected_contexts": len(context_rows),
                "repeat_rows": len(repeat_rows),
                "target_ledger_rows": len(target_rows),
                "actual_extra_transitions": summary[
                    "actual_extra_transitions"
                ],
            },
            "maximum_actor_parameter_difference": (
                maximum_parameter_difference
            ),
            "stage_passed": summary["stage_passed"],
            "decision": summary["decision"],
        }
    )
    _write_json(experiment_path, experiment)
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    )
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
