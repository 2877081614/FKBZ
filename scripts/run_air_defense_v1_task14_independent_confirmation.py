from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.common import (
    ParetoRecallConstraints,
    confirmation_power,
    constrained_value_metrics,
    engagement_delta_targets,
    evaluate_frozen_threshold,
    frozen_thresholds_from_rows,
    safety_resource_oracle,
)
from rein_learning.envs import AirDefenseResourceAssignmentEnvV1
from rein_learning.models import (
    AirDefenseV1ObservationLayout,
    StateConditionedEngagementValue,
    StateConditionedEngagementValueConfig,
)
from scripts.run_air_defense_v1_task14_engagement_calibration import (
    DEFAULT_CRITIC_DIR,
    _load_critic,
    _load_npz,
    _predict,
)
from scripts.run_air_defense_v1_task14_engagement_utility import (
    _components,
    _observation_overlap_count,
)
from scripts.run_air_defense_v1_task14_q_critic import (
    DEFAULT_MODEL_DIR,
    SCENARIOS,
    _write_csv,
)
from scripts.run_air_defense_v1_task14_state_conditioned_value import (
    _generate_test_dataset,
    _predict_value,
)


DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task14_multibatch_leave_one_out"
)
DEFAULT_CALIBRATION_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task14_oob_pareto_audit"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task14_independent_confirmation"
)
FROZEN_OBJECTIVE = "scenario_robust_reliable_cost"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confirm frozen OOB thresholds on one independent batch."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR
    )
    parser.add_argument("--source-model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--critic-dir", type=Path, default=DEFAULT_CRITIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-seeds", nargs="+", type=int, default=(8, 10))
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--episodes-per-stratum", type=int, default=30)
    parser.add_argument("--states-per-stratum", type=int, default=12)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--eval-seed", type=int, default=887_000)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=(20, 21, 22))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-test-dataset", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _generation_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model_dir=args.source_model_dir,
        output_dir=args.output_dir,
        source_seeds=tuple(args.source_seeds),
        scenarios=tuple(args.scenarios),
        episodes_per_stratum=args.episodes_per_stratum,
        states_per_stratum=args.states_per_stratum,
        rollouts=args.rollouts,
        gamma=args.gamma,
        eval_seed=args.eval_seed,
        device=args.device,
    )


def _load_value_model(
    path: Path,
    layout: AirDefenseV1ObservationLayout,
    device: str,
) -> tuple[StateConditionedEngagementValue, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    signature = dict(checkpoint["model_signature"])
    config = StateConditionedEngagementValueConfig(
        hidden_dims=tuple(int(value) for value in signature["hidden_dims"]),
        budget_mode=str(signature["budget_mode"]),
        max_budget_multiplier=float(signature["max_budget_multiplier"]),
    )
    model = StateConditionedEngagementValue(layout, config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if model.signature() != signature:
        raise ValueError(f"Checkpoint signature mismatch: {path}")
    return model, checkpoint


def _historical_dataset_paths(output_dir: Path, source_dir: Path) -> list[Path]:
    root = PROJECT_ROOT / "results" / "air_defense_v1"
    excluded = (output_dir / "test_dataset.npz").resolve()
    paths = {
        path.resolve()
        for path in root.rglob("test_dataset.npz")
        if path.resolve() != excluded
    }
    training_path = (source_dir / "training_dataset.npz").resolve()
    if training_path.exists():
        paths.add(training_path)
    return sorted(paths, key=str)


def _flatten_point(point: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in point.items() if key != "checks"
    } | {
        f"check_{key}": value for key, value in point["checks"].items()
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = args.calibration_dir / "seed_summary.csv"
    calibration_gate_path = args.calibration_dir / "gate_summary.json"
    source_config_path = args.source_dir / "experiment_config.json"
    with calibration_gate_path.open(encoding="utf-8") as handle:
        calibration_gate = json.load(handle)
    with source_config_path.open(encoding="utf-8") as handle:
        source_config = json.load(handle)

    thresholds = frozen_thresholds_from_rows(
        _read_csv(threshold_path),
        objective=FROZEN_OBJECTIVE,
        model_seeds=args.model_seeds,
    )
    history_paths = _historical_dataset_paths(args.output_dir, args.source_dir)
    test_path = args.output_dir / "test_dataset.npz"
    dataset = (
        _load_npz(test_path)
        if args.reuse_test_dataset and test_path.exists()
        else _generate_test_dataset(_generation_args(args))
    )

    indices = np.arange(len(dataset["group_ids"]))
    targets = engagement_delta_targets(_components(dataset, indices))
    oracle = safety_resource_oracle(_components(dataset, indices))["labels"]
    power = confirmation_power(oracle, dataset["scenarios"])
    valid = oracle >= 0

    environment = AirDefenseResourceAssignmentEnvV1()
    layout = AirDefenseV1ObservationLayout.infer(
        environment.observation_space, environment.action_space
    )
    environment.close()

    constraints = ParetoRecallConstraints()
    per_seed: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    checkpoint_objectives: dict[str, bool] = {}
    passed_seed_count = 0
    for model_seed in args.model_seeds:
        checkpoint_path = (
            args.source_dir
            / "models"
            / f"{FROZEN_OBJECTIVE}_seed{model_seed}.pt"
        )
        model, checkpoint = _load_value_model(
            checkpoint_path, layout, args.device
        )
        checkpoint_hashes[str(model_seed)] = _sha256(checkpoint_path)
        checkpoint_objectives[str(model_seed)] = (
            checkpoint["selected_objective"] == FROZEN_OBJECTIVE
        )
        margin_model, _ = _load_critic(
            args.critic_dir / f"balanced_bce_margin_seed{model_seed}.pt",
            layout,
            args.device,
        )
        margin_values = _predict(
            margin_model, dataset, indices, args.device
        )
        margin_logits = margin_values[:, 1] - margin_values[:, 0]
        started = perf_counter()
        prediction = _predict_value(
            model,
            dataset,
            indices,
            args.device,
            margin_logits,
            checkpoint["scales"],
            float(checkpoint["margin_scale"]),
        )
        inference_seconds = perf_counter() - started
        value_metrics = constrained_value_metrics(
            targets["safety_gain"][valid],
            targets["cost_delta"][valid],
            prediction["safety_gain"][valid],
            prediction["cost_delta"][valid],
        )
        safety_sign = float(value_metrics["safety_sign_accuracy"])
        calibrated = evaluate_frozen_threshold(
            prediction["score"],
            oracle,
            dataset["scenarios"],
            threshold=thresholds[model_seed],
            safety_sign_accuracy=safety_sign,
            constraints=constraints,
        )
        zero = evaluate_frozen_threshold(
            prediction["score"],
            oracle,
            dataset["scenarios"],
            threshold=0.0,
            safety_sign_accuracy=safety_sign,
            constraints=constraints,
        )
        inference_passed = inference_seconds < float(dataset["generation_seconds"])
        passed = bool(calibrated["feasible"]) and inference_passed
        passed_seed_count += int(passed)
        per_seed[str(model_seed)] = {
            "frozen_threshold": thresholds[model_seed],
            "calibrated": calibrated,
            "zero_threshold": zero,
            "value_metrics": value_metrics,
            "inference_seconds": inference_seconds,
            "inference_faster_than_rollouts": inference_passed,
            "passed": passed,
        }
        for method, point in (("frozen_oob", calibrated), ("zero", zero)):
            metric_rows.append(
                {
                    "model_seed": model_seed,
                    "method": method,
                    **_flatten_point(point),
                    "inference_seconds": inference_seconds,
                }
            )
        predicted = prediction["score"] > thresholds[model_seed]
        for index in indices:
            diagnostics.append(
                {
                    "model_seed": model_seed,
                    "group_id": str(dataset["group_ids"][index]),
                    "scenario": str(dataset["scenarios"][index]),
                    "oracle_label": int(oracle[index]),
                    "frozen_threshold": thresholds[model_seed],
                    "predicted_engage": int(predicted[index]),
                    "score": float(prediction["score"][index]),
                    "safety_target": float(targets["safety_gain"][index]),
                    "safety_prediction": float(
                        prediction["safety_gain"][index]
                    ),
                    "cost_target": float(targets["cost_delta"][index]),
                    "cost_prediction": float(prediction["cost_delta"][index]),
                    "budget_multiplier": float(
                        prediction["budget_multiplier"][index]
                    ),
                }
            )

    overlaps = {
        str(path): _observation_overlap_count(
            dataset["observations"], _load_npz(path)["observations"]
        )
        for path in history_paths
    }
    reconstructed = (
        dataset["operational_return_samples"]
        - dataset["resource_cost_samples"]
        - 30.0 * dataset["damage_samples"]
    )
    reconstruction_error = float(
        np.max(np.abs(dataset["total_return_samples"] - reconstructed))
    )
    expected_states = (
        len(args.source_seeds) * len(args.scenarios) * args.states_per_stratum
    )
    source_checks = {
        "oob_gate_passed": bool(
            calibration_gate["task14_oob_pareto_feasibility_passed"]
        ),
        "objective_frozen": source_config["selected_objective"]
        == FROZEN_OBJECTIVE,
        "checkpoint_objectives": all(checkpoint_objectives.values()),
        "model_seeds_frozen": set(args.model_seeds)
        == set(source_config["model_seeds"]),
    }
    data_checks = {
        "state_count": int(dataset["state_count"]) == expected_states,
        "rollout_count": int(dataset["total_return_samples"].shape[2])
        == args.rollouts,
        "historical_overlap_zero": all(count == 0 for count in overlaps.values()),
        "return_reconstruction": reconstruction_error <= 1e-4,
        "single_confirmation_batch": True,
        "thresholds_not_refit": True,
        "models_not_retrained": True,
    }
    source_integrity = all(source_checks.values())
    data_integrity = all(data_checks.values())
    stage_checks = {
        "source_integrity": source_integrity,
        "data_integrity": data_integrity,
        "power_sufficient": bool(power["passed"]),
        "two_of_three_seeds_passed": passed_seed_count >= 2,
    }
    stage_passed = all(stage_checks.values())
    summary = {
        "schema_version": 1,
        "protocol": {
            "objective": FROZEN_OBJECTIVE,
            "eval_seed": args.eval_seed,
            "source_seeds": list(args.source_seeds),
            "scenarios": list(args.scenarios),
            "states_per_stratum": args.states_per_stratum,
            "episodes_per_stratum": args.episodes_per_stratum,
            "rollouts": args.rollouts,
            "gamma": args.gamma,
            "model_seeds": list(args.model_seeds),
            "frozen_thresholds": thresholds,
            "constraints": constraints.__dict__,
            "threshold_rule": "score > frozen_threshold",
            "thresholds_refit_on_confirmation": False,
            "models_retrained": False,
            "confirmation_batch_count": 1,
        },
        "source": {
            "threshold_file": str(threshold_path),
            "threshold_file_sha256": _sha256(threshold_path),
            "calibration_gate_sha256": _sha256(calibration_gate_path),
            "source_config_sha256": _sha256(source_config_path),
            "confirmation_dataset_sha256": _sha256(test_path),
            "checkpoint_sha256": checkpoint_hashes,
            "checks": source_checks,
            "passed": source_integrity,
        },
        "data_audit": {
            "checks": data_checks,
            "passed": data_integrity,
            "historical_dataset_count": len(history_paths),
            "historical_observation_overlaps": overlaps,
            "return_reconstruction_max_error": reconstruction_error,
        },
        "dataset": {
            "states": int(dataset["state_count"]),
            "groups": int(len(dataset["group_ids"])),
            "rollouts": int(dataset["total_return_samples"].shape[2]),
            "generation_seconds": float(dataset["generation_seconds"]),
        },
        "power": power,
        "model_gate": {
            "per_seed": per_seed,
            "passed_seed_count": passed_seed_count,
            "required_passed_seed_count": 2,
        },
        "stage_gate": {"checks": stage_checks, "passed": stage_passed},
        "task14_independent_confirmation_passed": stage_passed,
        "allow_mch_ppo_screening": stage_passed,
        "resume_mch_ppo": stage_passed,
        "enter_gnn": False,
        "next_action": (
            "freeze_mch_ppo_method_and_run_30k_three_seed_screening"
            if stage_passed
            else "keep_mch_ppo_frozen_and_revise_calibration_semantics"
        ),
    }
    _write_csv(args.output_dir / "model_metrics.csv", metric_rows)
    _write_csv(args.output_dir / "confirmation_diagnostics.csv", diagnostics)
    with (args.output_dir / "gate_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with (args.output_dir / "experiment_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary["protocol"], handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
