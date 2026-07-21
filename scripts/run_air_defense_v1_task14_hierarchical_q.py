from __future__ import annotations

import argparse
import csv
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
    action_group_ids,
    build_hierarchical_q_data,
    build_pairwise_training_data,
    hierarchical_q_metrics,
    integer_group_codes,
    q_critic_training_loss,
    validation_difference_score,
)
from rein_learning.envs import AirDefenseResourceAssignmentEnvV1
from rein_learning.models import (
    AirDefenseV1ObservationLayout,
    HierarchicalMaskedQCritic,
    HierarchicalMaskedQCriticConfig,
    MaskedActionQCritic,
)
from scripts.run_air_defense_v1_task14_q_critic import (
    SCENARIOS,
    _generate_dataset,
    _load_dataset,
    _model_inputs,
    _seed_everything,
    _write_csv,
)
from scripts.run_air_defense_v1_task14_ranking_refinement import _combine_datasets


DEFAULT_BASE_DATASET = (
    PROJECT_ROOT / "results" / "air_defense_v1" / "task14_q_critic" / "dataset.npz"
)
DEFAULT_POLICY_MODEL_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task12_factorized_screening_30k_3seeds"
)
DEFAULT_BASELINE_MODEL_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task14_q_critic_ranking_refinement"
    / "models"
)
DEFAULT_PREVIOUS_TEST = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task14_q_critic_ranking_refinement"
    / "test_dataset.npz"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "air_defense_v1" / "task14_hierarchical_q"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Task 14 explicit engagement/target Q gate."
    )
    parser.add_argument("--base-dataset", type=Path, default=DEFAULT_BASE_DATASET)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_POLICY_MODEL_DIR)
    parser.add_argument(
        "--baseline-model-dir", type=Path, default=DEFAULT_BASELINE_MODEL_DIR
    )
    parser.add_argument("--previous-test", type=Path, default=DEFAULT_PREVIOUS_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-seeds", nargs="+", type=int, default=(8, 10))
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--test-episodes-per-stratum", type=int, default=36)
    parser.add_argument("--test-states-per-stratum", type=int, default=18)
    parser.add_argument("--test-rollouts", type=int, default=32)
    parser.add_argument("--test-eval-seed", type=int, default=291_000)
    parser.add_argument("--split-seed", type=int, default=14)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=(14, 15, 16))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-test-dataset", action="store_true")
    return parser.parse_args()


def _generate_fresh_test(args: argparse.Namespace) -> dict[str, np.ndarray]:
    fresh_dir = args.output_dir / "fresh_test"
    fresh_dir.mkdir(parents=True, exist_ok=True)
    generation_args = argparse.Namespace(
        model_dir=args.model_dir,
        output_dir=fresh_dir,
        source_seeds=args.source_seeds,
        scenarios=args.scenarios,
        episodes_per_stratum=args.test_episodes_per_stratum,
        states_per_stratum=args.test_states_per_stratum,
        rollouts=args.test_rollouts,
        gamma=args.gamma,
        eval_seed=args.test_eval_seed,
        split_seed=args.split_seed,
        device=args.device,
    )
    dataset = _generate_dataset(generation_args)
    dataset["state_ids"] = np.asarray(
        [f"task14h/{state_id}" for state_id in dataset["state_ids"]]
    )
    dataset["splits"] = np.full(len(dataset["q_labels"]), "test")
    np.savez_compressed(args.output_dir / "test_dataset.npz", **dataset)
    with (fresh_dir / "dataset_samples.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["state_id"] = f"task14h/{row['state_id']}"
        row["sample_id"] = f"task14h/{row['sample_id']}"
        row["split"] = "test"
    _write_csv(args.output_dir / "test_dataset_samples.csv", rows)
    return dataset


def _observation_overlap_count(
    left: np.ndarray, right: np.ndarray
) -> int:
    right_values = {row.tobytes() for row in np.asarray(right)}
    return len({row.tobytes() for row in np.asarray(left)}.intersection(right_values))


def _engagement_inputs(
    dataset: dict[str, np.ndarray], context_indices: np.ndarray, device: str
) -> tuple[torch.Tensor, ...]:
    return (
        torch.as_tensor(dataset["observations"][context_indices], device=device),
        torch.as_tensor(dataset["unit_indices"][context_indices], device=device),
        torch.as_tensor(dataset["prefix_occupancy"][context_indices], device=device),
        torch.as_tensor(dataset["legal_action_masks"][context_indices], device=device),
    )


def _validation_score(
    *,
    hierarchy: dict[str, np.ndarray],
    dataset: dict[str, np.ndarray],
    engagement_predictions: np.ndarray,
    target_predictions: np.ndarray,
    engagement_scale: float,
    target_scale: float,
) -> dict[str, float]:
    engagement_groups = np.repeat(hierarchy["group_ids"], 2)
    engagement = validation_difference_score(
        hierarchy["engagement_labels"].reshape(-1),
        engagement_predictions.reshape(-1),
        engagement_groups,
        scale=engagement_scale,
    )
    target = validation_difference_score(
        dataset["q_labels"][hierarchy["target_indices"]],
        target_predictions,
        hierarchy["target_group_ids"],
        scale=target_scale,
    )
    return {
        "engagement_absolute_mae": engagement["absolute_mae"],
        "engagement_centered_mae": engagement["centered_mae"],
        "target_absolute_mae": target["absolute_mae"],
        "target_centered_mae": target["centered_mae"],
        "score": engagement["score"] + target["score"],
    }


def _train_hierarchical_model(
    *,
    dataset: dict[str, np.ndarray],
    layout: AirDefenseV1ObservationLayout,
    train_seed: int,
    args: argparse.Namespace,
) -> tuple[
    HierarchicalMaskedQCritic,
    dict[str, np.ndarray],
    list[dict[str, Any]],
    dict[str, Any],
]:
    _seed_everything(train_seed)
    model = HierarchicalMaskedQCritic(layout).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_hierarchy = build_hierarchical_q_data(
        dataset,
        np.flatnonzero(dataset["splits"] == "train"),
        noop_action=layout.num_targets,
    )
    validation_hierarchy = build_hierarchical_q_data(
        dataset,
        np.flatnonzero(dataset["splits"] == "validation"),
        noop_action=layout.num_targets,
    )
    test_hierarchy = build_hierarchical_q_data(
        dataset,
        np.flatnonzero(dataset["splits"] == "test"),
        noop_action=layout.num_targets,
    )
    engagement_mean = float(np.mean(train_hierarchy["engagement_labels"]))
    engagement_std = max(float(np.std(train_hierarchy["engagement_labels"])), 1e-6)
    target_train_labels = dataset["q_labels"][train_hierarchy["target_indices"]]
    target_mean = float(np.mean(target_train_labels))
    target_std = max(float(np.std(target_train_labels)), 1e-6)
    engagement_labels = torch.as_tensor(
        (train_hierarchy["engagement_labels"] - engagement_mean) / engagement_std,
        device=args.device,
        dtype=torch.float32,
    )
    target_labels = torch.as_tensor(
        (target_train_labels - target_mean) / target_std,
        device=args.device,
        dtype=torch.float32,
    )
    engagement_codes = torch.arange(
        len(train_hierarchy["group_ids"]), device=args.device
    ).repeat_interleave(2)
    target_codes = torch.as_tensor(
        integer_group_codes(train_hierarchy["target_group_ids"]), device=args.device
    )
    target_pairs = build_pairwise_training_data(
        target_train_labels,
        train_hierarchy["target_group_ids"],
        dataset["return_samples"][train_hierarchy["target_indices"]],
    )
    pair_left = torch.as_tensor(target_pairs["left"], device=args.device)
    pair_right = torch.as_tensor(target_pairs["right"], device=args.device)
    pair_weights = torch.as_tensor(target_pairs["reliability"], device=args.device)

    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    curves: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        engagement_prediction = model.forward_engagement(
            *_engagement_inputs(
                dataset, train_hierarchy["context_indices"], args.device
            )
        )
        target_prediction = model.forward_target(
            *_model_inputs(dataset, train_hierarchy["target_indices"], args.device)
        )
        engagement_loss, engagement_components = q_critic_training_loss(
            engagement_prediction.reshape(-1),
            engagement_labels.reshape(-1),
            engagement_codes,
            centered_weight=1.0,
        )
        target_loss, target_components = q_critic_training_loss(
            target_prediction,
            target_labels,
            target_codes,
            pair_left=pair_left,
            pair_right=pair_right,
            pair_weights=pair_weights,
            centered_weight=1.0,
            pairwise_weight=0.5,
        )
        loss = engagement_loss + target_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_engagement = (
                model.forward_engagement(
                    *_engagement_inputs(
                        dataset, validation_hierarchy["context_indices"], args.device
                    )
                )
                .cpu()
                .numpy()
                * engagement_std
                + engagement_mean
            )
            validation_target = (
                model.forward_target(
                    *_model_inputs(
                        dataset, validation_hierarchy["target_indices"], args.device
                    )
                )
                .cpu()
                .numpy()
                * target_std
                + target_mean
            )
        validation = _validation_score(
            hierarchy=validation_hierarchy,
            dataset=dataset,
            engagement_predictions=validation_engagement,
            target_predictions=validation_target,
            engagement_scale=engagement_std,
            target_scale=target_std,
        )
        curves.append(
            {
                "train_seed": train_seed,
                "epoch": epoch,
                "total_loss": float(loss.detach().cpu()),
                "engagement_absolute_loss": float(
                    engagement_components["absolute"].detach().cpu()
                ),
                "engagement_centered_loss": float(
                    engagement_components["centered"].detach().cpu()
                ),
                "target_absolute_loss": float(
                    target_components["absolute"].detach().cpu()
                ),
                "target_centered_loss": float(
                    target_components["centered"].detach().cpu()
                ),
                "target_pairwise_loss": float(
                    target_components["pairwise"].detach().cpu()
                ),
                **{f"validation_{key}": value for key, value in validation.items()},
            }
        )
        if validation["score"] < best_score - 1e-6:
            best_score = validation["score"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("Hierarchical Q training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    inference_started = perf_counter()
    with torch.no_grad():
        test_engagement = (
            model.forward_engagement(
                *_engagement_inputs(dataset, test_hierarchy["context_indices"], args.device)
            )
            .cpu()
            .numpy()
            * engagement_std
            + engagement_mean
        )
        test_target = (
            model.forward_target(
                *_model_inputs(dataset, test_hierarchy["target_indices"], args.device)
            )
            .cpu()
            .numpy()
            * target_std
            + target_mean
        )
    inference_seconds = perf_counter() - inference_started
    predictions = {
        "engagement": test_engagement,
        "target": test_target,
    }
    record = {
        "train_seed": train_seed,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "engagement_mean": engagement_mean,
        "engagement_std": engagement_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "parameter_count": model.parameter_count(),
        "target_training_pairs": int(len(target_pairs["left"])),
        "test_inference_seconds": inference_seconds,
    }
    return model, predictions, curves, record


def _load_baseline_predictions(
    *,
    dataset: dict[str, np.ndarray],
    hierarchy: dict[str, np.ndarray],
    layout: AirDefenseV1ObservationLayout,
    train_seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, float]:
    path = args.baseline_model_dir / f"difference_aware_seed{train_seed}.pt"
    payload = torch.load(path, map_location=args.device, weights_only=False)
    model = MaskedActionQCritic(layout).to(args.device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    test_indices = np.flatnonzero(dataset["splits"] == "test")
    started = perf_counter()
    with torch.no_grad():
        action_predictions = (
            model(*_model_inputs(dataset, test_indices, args.device)).cpu().numpy()
            * float(payload["q_std"])
            + float(payload["q_mean"])
        )
    seconds = perf_counter() - started
    full_predictions = np.full(len(dataset["q_labels"]), np.nan, dtype=np.float32)
    full_predictions[test_indices] = action_predictions
    prediction_dataset = {**dataset, "q_labels": full_predictions}
    prediction_hierarchy = build_hierarchical_q_data(
        prediction_dataset, test_indices, noop_action=layout.num_targets
    )
    return (
        prediction_hierarchy["engagement_labels"],
        full_predictions[hierarchy["target_indices"]],
        seconds,
    )


def _gate_record(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
    *,
    inference_seconds: float,
    dataset: dict[str, np.ndarray],
) -> dict[str, Any]:
    test_states = int(dataset["fresh_test_state_count"])
    mc_per_state = float(dataset["generation_seconds"]) / test_states
    inference_per_state = inference_seconds / test_states
    gates = {
        "engagement_overall": metrics["engagement_sign_count"] >= 30
        and metrics["engagement_sign_accuracy"] >= 0.70,
        "engagement_scenarios": all(
            metrics[f"scenario_{scenario}_engagement_count"] >= 10
            and metrics[f"scenario_{scenario}_engagement_accuracy"] >= 0.60
            for scenario in SCENARIOS
        ),
        "target_overall": metrics["target_ranking_count"] >= 30
        and metrics["target_ranking_accuracy"] >= 0.65,
        "target_top": metrics["target_top_count"] >= 30
        and metrics["target_top_accuracy"] >= 0.50,
        "target_scenarios": all(
            metrics[f"scenario_{scenario}_target_count"] >= 10
            and metrics[f"scenario_{scenario}_target_accuracy"] >= 0.60
            for scenario in SCENARIOS
        ),
        "engagement_mae_noninferiority": metrics["engagement_mae"]
        <= 1.10 * baseline["engagement_mae"],
        "target_mae_noninferiority": metrics["target_mae"]
        <= 1.10 * baseline["target_mae"],
        "efficiency": inference_per_state < mc_per_state,
    }
    return {
        "gates": gates,
        "passed_gate_count": sum(gates.values()),
        "total_gate_count": len(gates),
        "passed": all(gates.values()),
        "inference_seconds_per_test_state": inference_per_state,
        "mc_seconds_per_test_state": mc_per_state,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.test_states_per_stratum < 3:
        raise ValueError("test-states-per-stratum must be at least three")
    test_path = args.output_dir / "test_dataset.npz"
    fresh_test = (
        _load_dataset(test_path)
        if args.reuse_test_dataset
        else _generate_fresh_test(args)
    )
    base = _load_dataset(args.base_dataset)
    dataset, isolation = _combine_datasets(base, fresh_test)
    previous_test = _load_dataset(args.previous_test)
    isolation["previous_test_observation_overlap"] = _observation_overlap_count(
        fresh_test["observations"], previous_test["observations"]
    )
    if isolation["previous_test_observation_overlap"] != 0:
        raise RuntimeError("Fresh test observations overlap the previous formal test")
    np.savez_compressed(args.output_dir / "analysis_dataset.npz", **dataset)

    env = AirDefenseResourceAssignmentEnvV1()
    layout = AirDefenseV1ObservationLayout.infer(
        env.observation_space, env.action_space
    )
    env.close()
    test_hierarchy = build_hierarchical_q_data(
        dataset,
        np.flatnonzero(dataset["splits"] == "test"),
        noop_action=layout.num_targets,
    )
    models_dir = args.output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    curves: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    baseline_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []

    for train_seed in args.train_seeds:
        baseline_engagement, baseline_target, baseline_seconds = _load_baseline_predictions(
            dataset=dataset,
            hierarchy=test_hierarchy,
            layout=layout,
            train_seed=train_seed,
            args=args,
        )
        baseline_metrics = hierarchical_q_metrics(
            hierarchy=test_hierarchy,
            dataset=dataset,
            engagement_predictions=baseline_engagement,
            target_predictions=baseline_target,
        )
        baseline_records.append(
            {
                "train_seed": train_seed,
                "metrics": baseline_metrics,
                "test_inference_seconds": baseline_seconds,
            }
        )
        model, predictions, model_curves, record = _train_hierarchical_model(
            dataset=dataset,
            layout=layout,
            train_seed=train_seed,
            args=args,
        )
        curves.extend(model_curves)
        records.append(record)
        candidate_metrics = hierarchical_q_metrics(
            hierarchy=test_hierarchy,
            dataset=dataset,
            engagement_predictions=predictions["engagement"],
            target_predictions=predictions["target"],
        )
        gate = _gate_record(
            candidate_metrics,
            baseline_metrics,
            inference_seconds=record["test_inference_seconds"],
            dataset=dataset,
        )
        candidate_records.append(
            {"train_seed": train_seed, "metrics": candidate_metrics, **gate}
        )
        for method, values in (
            ("monolithic_difference_aware", baseline_metrics),
            ("hierarchical", candidate_metrics),
        ):
            metric_rows.extend(
                {
                    "method": method,
                    "train_seed": train_seed,
                    "metric": key,
                    "value": value,
                }
                for key, value in values.items()
            )
        for group_index, group_id in enumerate(test_hierarchy["group_ids"]):
            prediction_rows.append(
                {
                    "level": "engagement",
                    "train_seed": train_seed,
                    "group_id": str(group_id),
                    "scenario": str(test_hierarchy["scenarios"][group_index]),
                    "true_noop_q": float(
                        test_hierarchy["engagement_labels"][group_index, 0]
                    ),
                    "true_engage_q": float(
                        test_hierarchy["engagement_labels"][group_index, 1]
                    ),
                    "baseline_noop_q": float(baseline_engagement[group_index, 0]),
                    "baseline_engage_q": float(baseline_engagement[group_index, 1]),
                    "hierarchical_noop_q": float(predictions["engagement"][group_index, 0]),
                    "hierarchical_engage_q": float(predictions["engagement"][group_index, 1]),
                }
            )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "signature": model.signature(),
                "normalization": {
                    key: record[key]
                    for key in (
                        "engagement_mean",
                        "engagement_std",
                        "target_mean",
                        "target_std",
                    )
                },
                "train_seed": train_seed,
            },
            models_dir / f"hierarchical_seed{train_seed}.pt",
        )
        print(
            f"train seed={train_seed} engage="
            f"{candidate_metrics['engagement_sign_accuracy']:.3f}/"
            f"{candidate_metrics['engagement_sign_count']} target="
            f"{candidate_metrics['target_ranking_accuracy']:.3f}/"
            f"{candidate_metrics['target_ranking_count']}",
            flush=True,
        )

    baseline_by_seed = {row["train_seed"]: row for row in baseline_records}
    candidate_by_seed = {row["train_seed"]: row for row in candidate_records}
    seeds = sorted(candidate_by_seed)
    engage_improvements = [
        candidate_by_seed[seed]["metrics"]["engagement_sign_accuracy"]
        - baseline_by_seed[seed]["metrics"]["engagement_sign_accuracy"]
        for seed in seeds
    ]
    target_changes = [
        candidate_by_seed[seed]["metrics"]["target_ranking_accuracy"]
        - baseline_by_seed[seed]["metrics"]["target_ranking_accuracy"]
        for seed in seeds
    ]
    additional_gates = {
        "mean_engagement_improvement": float(np.mean(engage_improvements)) >= 0.10,
        "mean_target_noninferiority": float(np.mean(target_changes)) >= -0.05,
    }
    passed_seeds = sum(record["passed"] for record in candidate_records)
    first_metrics = candidate_records[0]["metrics"]
    power_sufficient = (
        first_metrics["engagement_sign_count"] >= 30
        and first_metrics["target_ranking_count"] >= 30
        and first_metrics["target_top_count"] >= 30
    )
    task_passed = power_sufficient and passed_seeds >= 2 and all(additional_gates.values())
    report = {
        "schema_version": 1,
        "isolation_audit": isolation,
        "dataset": {
            "rows": int(len(dataset["q_labels"])),
            "train_rows": int(np.sum(dataset["splits"] == "train")),
            "validation_rows": int(np.sum(dataset["splits"] == "validation")),
            "test_rows": int(np.sum(dataset["splits"] == "test")),
            "fresh_test_states": int(dataset["fresh_test_state_count"]),
            "fresh_test_rollouts": int(fresh_test["return_samples"].shape[1]),
            "engagement_groups": int(len(test_hierarchy["group_ids"])),
            "target_rows": int(len(test_hierarchy["target_indices"])),
            "power_sufficient": power_sufficient,
        },
        "baseline": baseline_records,
        "hierarchical": candidate_records,
        "comparison": {
            "engagement_improvement_by_seed": dict(
                zip((str(seed) for seed in seeds), engage_improvements)
            ),
            "mean_engagement_improvement": float(np.mean(engage_improvements)),
            "target_change_by_seed": dict(
                zip((str(seed) for seed in seeds), target_changes)
            ),
            "mean_target_change": float(np.mean(target_changes)),
            "additional_gates": additional_gates,
        },
        "passed_seed_count": passed_seeds,
        "required_passed_seed_count": 2,
        "task14_hierarchical_passed": task_passed,
        "resume_mch_ppo": task_passed,
        "enter_gnn": False,
    }
    _write_csv(args.output_dir / "training_curves.csv", curves)
    _write_csv(args.output_dir / "training_records.csv", records)
    _write_csv(args.output_dir / "metrics.csv", metric_rows)
    _write_csv(args.output_dir / "engagement_predictions.csv", prediction_rows)
    (args.output_dir / "gate_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config = {
        "schema_version": 1,
        "base_dataset": str(args.base_dataset.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "baseline_model_dir": str(args.baseline_model_dir.resolve()),
        "previous_test": str(args.previous_test.resolve()),
        "source_seeds": list(args.source_seeds),
        "scenarios": list(args.scenarios),
        "test_episodes_per_stratum": args.test_episodes_per_stratum,
        "test_states_per_stratum": args.test_states_per_stratum,
        "test_rollouts": args.test_rollouts,
        "test_eval_seed": args.test_eval_seed,
        "gamma": args.gamma,
        "train_seeds": list(args.train_seeds),
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "model": HierarchicalMaskedQCriticConfig().signature(),
    }
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"output_dir={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
