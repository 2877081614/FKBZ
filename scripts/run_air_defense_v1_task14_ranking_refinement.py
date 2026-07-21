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
    build_pairwise_training_data,
    integer_group_codes,
    q_critic_training_loss,
    validation_difference_score,
)
from rein_learning.envs import AirDefenseResourceAssignmentEnvV1
from rein_learning.models import (
    AirDefenseV1ObservationLayout,
    MaskedActionQCritic,
    MaskedActionQCriticConfig,
)
from scripts.run_air_defense_v1_task14_q_critic import (
    SCENARIOS,
    _generate_dataset,
    _load_dataset,
    _metric_rows,
    _model_inputs,
    _seed_everything,
    _write_csv,
)


DEFAULT_BASE_DATASET = (
    PROJECT_ROOT / "results" / "air_defense_v1" / "task14_q_critic" / "dataset.npz"
)
DEFAULT_MODEL_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task12_factorized_screening_30k_3seeds"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task14_q_critic_ranking_refinement"
)
OBJECTIVES = {
    "absolute_mse": {"centered_weight": 0.0, "pairwise_weight": 0.0},
    "difference_aware": {"centered_weight": 1.0, "pairwise_weight": 0.5},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Task 14 ranking-supervision refinement gate."
    )
    parser.add_argument("--base-dataset", type=Path, default=DEFAULT_BASE_DATASET)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-seeds", nargs="+", type=int, default=(8, 10))
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--test-episodes-per-stratum", type=int, default=18)
    parser.add_argument("--test-states-per-stratum", type=int, default=6)
    parser.add_argument("--test-rollouts", type=int, default=32)
    parser.add_argument("--test-eval-seed", type=int, default=191_000)
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
        [f"task14r/{state_id}" for state_id in dataset["state_ids"]]
    )
    dataset["splits"] = np.full(len(dataset["q_labels"]), "test")
    np.savez_compressed(args.output_dir / "test_dataset.npz", **dataset)

    source_csv = fresh_dir / "dataset_samples.csv"
    with source_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["state_id"] = f"task14r/{row['state_id']}"
        row["sample_id"] = f"task14r/{row['sample_id']}"
        row["split"] = "test"
    _write_csv(args.output_dir / "test_dataset_samples.csv", rows)
    return dataset


def _pad_return_samples(samples: np.ndarray, width: int) -> np.ndarray:
    if samples.shape[1] > width:
        raise ValueError("Cannot shrink return-sample width")
    result = np.full((samples.shape[0], width), np.nan, dtype=np.float32)
    result[:, : samples.shape[1]] = samples
    return result


def _combine_datasets(
    base: dict[str, np.ndarray], fresh_test: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    retained = np.isin(base["splits"], ("train", "validation"))
    old_test_rows = int(np.sum(base["splits"] == "test"))
    old_state_ids = set(base["state_ids"].tolist())
    fresh_state_ids = set(fresh_test["state_ids"].tolist())
    overlap = old_state_ids.intersection(fresh_state_ids)
    if overlap:
        raise RuntimeError("Fresh test state ids overlap the Task 14 dataset")
    if set(np.unique(fresh_test["splits"]).tolist()) != {"test"}:
        raise RuntimeError("Fresh dataset must be test-only")

    common_keys = (
        "observations",
        "unit_indices",
        "candidate_actions",
        "prefix_occupancy",
        "legal_action_masks",
        "q_labels",
        "q_standard_errors",
        "one_step_rewards",
        "frozen_values",
        "conditional_target_probabilities",
        "state_ids",
        "scenarios",
        "source_seeds",
        "splits",
    )
    combined = {
        key: np.concatenate((base[key][retained], fresh_test[key]))
        for key in common_keys
    }
    rollout_width = max(
        base["return_samples"].shape[1], fresh_test["return_samples"].shape[1]
    )
    combined["return_samples"] = np.concatenate(
        (
            _pad_return_samples(base["return_samples"][retained], rollout_width),
            _pad_return_samples(fresh_test["return_samples"], rollout_width),
        )
    )
    combined["generation_seconds"] = np.asarray(
        fresh_test["generation_seconds"], dtype=np.float64
    )
    combined["state_count"] = np.asarray(
        len(np.unique(combined["state_ids"])), dtype=np.int64
    )
    combined["fresh_test_state_count"] = np.asarray(
        len(fresh_state_ids), dtype=np.int64
    )
    combined["origins"] = np.concatenate(
        (
            np.full(int(np.sum(retained)), "task14_train_validation"),
            np.full(len(fresh_test["q_labels"]), "task14r_fresh_test"),
        )
    )
    audit = {
        "old_test_rows_excluded": old_test_rows,
        "old_test_rows_in_combined": int(
            np.sum(
                (combined["origins"] == "task14_train_validation")
                & (combined["splits"] == "test")
            )
        ),
        "fresh_state_id_overlap": len(overlap),
        "fresh_test_states": len(fresh_state_ids),
        "fresh_test_rows": int(len(fresh_test["q_labels"])),
        "fresh_test_rollouts": int(fresh_test["return_samples"].shape[1]),
    }
    return combined, audit


def _train_model(
    *,
    dataset: dict[str, np.ndarray],
    layout: AirDefenseV1ObservationLayout,
    objective: str,
    train_seed: int,
    args: argparse.Namespace,
) -> tuple[MaskedActionQCritic, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    _seed_everything(train_seed)
    model = MaskedActionQCritic(layout, MaskedActionQCriticConfig()).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_indices = np.flatnonzero(dataset["splits"] == "train")
    validation_indices = np.flatnonzero(dataset["splits"] == "validation")
    test_indices = np.flatnonzero(dataset["splits"] == "test")
    q_mean = float(np.mean(dataset["q_labels"][train_indices]))
    q_std = max(float(np.std(dataset["q_labels"][train_indices])), 1e-6)
    normalized_labels = torch.as_tensor(
        (dataset["q_labels"] - q_mean) / q_std,
        device=args.device,
        dtype=torch.float32,
    )

    train_group_ids = action_group_ids(
        dataset["state_ids"][train_indices], dataset["unit_indices"][train_indices]
    )
    validation_group_ids = action_group_ids(
        dataset["state_ids"][validation_indices],
        dataset["unit_indices"][validation_indices],
    )
    train_group_codes = torch.as_tensor(
        integer_group_codes(train_group_ids), device=args.device
    )
    pair_data = build_pairwise_training_data(
        dataset["q_labels"][train_indices],
        train_group_ids,
        dataset["return_samples"][train_indices],
    )
    pair_left = torch.as_tensor(pair_data["left"], device=args.device)
    pair_right = torch.as_tensor(pair_data["right"], device=args.device)
    pair_weights = torch.as_tensor(pair_data["reliability"], device=args.device)
    loss_weights = OBJECTIVES[objective]

    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    curves: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        prediction = model(*_model_inputs(dataset, train_indices, args.device))
        loss, components = q_critic_training_loss(
            prediction,
            normalized_labels[train_indices],
            train_group_codes,
            pair_left=pair_left,
            pair_right=pair_right,
            pair_weights=pair_weights,
            centered_weight=loss_weights["centered_weight"],
            pairwise_weight=loss_weights["pairwise_weight"],
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_prediction = (
                model(*_model_inputs(dataset, validation_indices, args.device))
                .cpu()
                .numpy()
                * q_std
                + q_mean
            )
        validation = validation_difference_score(
            dataset["q_labels"][validation_indices],
            validation_prediction,
            validation_group_ids,
            scale=q_std,
        )
        curves.append(
            {
                "objective": objective,
                "train_seed": train_seed,
                "epoch": epoch,
                "total_loss": float(loss.detach().cpu()),
                "absolute_loss": float(components["absolute"].detach().cpu()),
                "centered_loss": float(components["centered"].detach().cpu()),
                "pairwise_loss": float(components["pairwise"].detach().cpu()),
                "validation_absolute_mae": validation["absolute_mae"],
                "validation_centered_mae": validation["centered_mae"],
                "validation_score": validation["score"],
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
        raise RuntimeError("Q-Critic refinement did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    all_indices = np.arange(len(dataset["q_labels"]))
    with torch.no_grad():
        predictions = (
            model(*_model_inputs(dataset, all_indices, args.device)).cpu().numpy()
            * q_std
            + q_mean
        )
    inference_started = perf_counter()
    with torch.no_grad():
        model(*_model_inputs(dataset, test_indices, args.device))
    test_inference_seconds = perf_counter() - inference_started
    record = {
        "objective": objective,
        "train_seed": train_seed,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "q_mean": q_mean,
        "q_std": q_std,
        "parameter_count": model.parameter_count(),
        "training_pairs": int(len(pair_data["left"])),
        "test_inference_seconds": test_inference_seconds,
    }
    return model, predictions, curves, record


def _gate_record(
    metrics: dict[str, Any],
    *,
    test_inference_seconds: float,
    dataset: dict[str, np.ndarray],
) -> dict[str, Any]:
    test_state_count = int(dataset["fresh_test_state_count"])
    q_seconds_per_state = test_inference_seconds / max(test_state_count, 1)
    mc_seconds_per_state = float(dataset["generation_seconds"]) / max(
        test_state_count, 1
    )
    gates = {
        "mae_improvement": metrics["mae_improvement_vs_v"] >= 0.10,
        "ranking": metrics["ranking_count"] >= 30
        and metrics["ranking_accuracy"] >= 0.70,
        "engagement_sign": metrics["engagement_sign_count"] >= 30
        and metrics["engagement_sign_accuracy"] >= 0.70,
        "target_ranking": metrics["target_ranking_count"] >= 30
        and metrics["target_ranking_accuracy"] >= 0.65,
        "top_action": metrics["top_action_count"] >= 30
        and metrics["top_action_accuracy"] >= 0.50,
        "scenario_ranking": all(
            metrics[f"scenario_{scenario}_ranking_count"] >= 30
            and metrics[f"scenario_{scenario}_ranking_accuracy"] >= 0.60
            for scenario in SCENARIOS
        ),
        "efficiency": q_seconds_per_state < mc_seconds_per_state,
    }
    return {
        "gates": gates,
        "passed_gate_count": sum(gates.values()),
        "total_gate_count": len(gates),
        "passed": all(gates.values()),
        "q_seconds_per_test_state": q_seconds_per_state,
        "mc_generation_seconds_per_test_state": mc_seconds_per_state,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.test_states_per_stratum < 3:
        raise ValueError("test-states-per-stratum must be at least three")
    test_dataset_path = args.output_dir / "test_dataset.npz"
    fresh_test = (
        _load_dataset(test_dataset_path)
        if args.reuse_test_dataset
        else _generate_fresh_test(args)
    )
    base = _load_dataset(args.base_dataset)
    dataset, isolation_audit = _combine_datasets(base, fresh_test)
    if (
        isolation_audit["old_test_rows_in_combined"] != 0
        or isolation_audit["fresh_state_id_overlap"] != 0
    ):
        raise RuntimeError("Dataset isolation audit failed")
    np.savez_compressed(args.output_dir / "analysis_dataset.npz", **dataset)

    env = AirDefenseResourceAssignmentEnvV1()
    layout = AirDefenseV1ObservationLayout.infer(
        env.observation_space, env.action_space
    )
    env.close()
    models_dir = args.output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    curves: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    predictions_rows: list[dict[str, Any]] = []
    training_records: list[dict[str, Any]] = []
    gate_records: dict[str, list[dict[str, Any]]] = {
        objective: [] for objective in OBJECTIVES
    }
    for objective in OBJECTIVES:
        for train_seed in args.train_seeds:
            model, predictions, model_curves, training_record = _train_model(
                dataset=dataset,
                layout=layout,
                objective=objective,
                train_seed=train_seed,
                args=args,
            )
            curves.extend(model_curves)
            training_records.append(training_record)
            rows, values = _metric_rows(
                dataset=dataset,
                predictions=predictions,
                variant=objective,
                train_seed=train_seed,
                layout=layout,
            )
            metrics_rows.extend(rows)
            gate = _gate_record(
                values,
                test_inference_seconds=training_record["test_inference_seconds"],
                dataset=dataset,
            )
            gate_records[objective].append(
                {"train_seed": train_seed, "metrics": values, **gate}
            )
            for index in np.flatnonzero(dataset["splits"] == "test"):
                predictions_rows.append(
                    {
                        "objective": objective,
                        "train_seed": train_seed,
                        "state_id": str(dataset["state_ids"][index]),
                        "scenario": str(dataset["scenarios"][index]),
                        "source_seed": int(dataset["source_seeds"][index]),
                        "unit_index": int(dataset["unit_indices"][index]),
                        "candidate_action": int(dataset["candidate_actions"][index]),
                        "mc_q": float(dataset["q_labels"][index]),
                        "q_prediction": float(predictions[index]),
                        "frozen_v": float(dataset["frozen_values"][index]),
                    }
                )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "signature": model.signature(),
                    "q_mean": training_record["q_mean"],
                    "q_std": training_record["q_std"],
                    "objective": objective,
                    "train_seed": train_seed,
                },
                models_dir / f"{objective}_seed{train_seed}.pt",
            )
            print(
                f"train objective={objective} seed={train_seed} "
                f"mae={values['q_mae']:.3f} rank={values['ranking_accuracy']:.3f} "
                f"pairs={values['ranking_count']}",
                flush=True,
            )

    baseline_by_seed = {
        record["train_seed"]: record for record in gate_records["absolute_mse"]
    }
    candidate_by_seed = {
        record["train_seed"]: record for record in gate_records["difference_aware"]
    }
    common_seeds = sorted(set(baseline_by_seed).intersection(candidate_by_seed))
    ranking_improvements = [
        candidate_by_seed[seed]["metrics"]["ranking_accuracy"]
        - baseline_by_seed[seed]["metrics"]["ranking_accuracy"]
        for seed in common_seeds
    ]
    mae_ratios = [
        candidate_by_seed[seed]["metrics"]["q_mae"]
        / baseline_by_seed[seed]["metrics"]["q_mae"]
        for seed in common_seeds
    ]
    candidate_passed_seeds = sum(
        record["passed"] for record in gate_records["difference_aware"]
    )
    scenario_counts = {
        scenario: int(
            gate_records["difference_aware"][0]["metrics"][
                f"scenario_{scenario}_ranking_count"
            ]
        )
        for scenario in SCENARIOS
    }
    test_power_sufficient = all(count >= 30 for count in scenario_counts.values())
    additional_gates = {
        "mean_ranking_improvement": float(np.mean(ranking_improvements)) >= 0.10,
        "mean_mae_noninferiority": float(np.mean(mae_ratios)) <= 1.10,
    }
    task_passed = (
        test_power_sufficient
        and candidate_passed_seeds >= 2
        and all(additional_gates.values())
    )
    report = {
        "schema_version": 1,
        "isolation_audit": isolation_audit,
        "dataset": {
            "rows": int(len(dataset["q_labels"])),
            "train_rows": int(np.sum(dataset["splits"] == "train")),
            "validation_rows": int(np.sum(dataset["splits"] == "validation")),
            "test_rows": int(np.sum(dataset["splits"] == "test")),
            "fresh_test_states": int(dataset["fresh_test_state_count"]),
            "fresh_test_rollouts": int(fresh_test["return_samples"].shape[1]),
            "scenario_high_confidence_pair_counts": scenario_counts,
            "test_power_sufficient": test_power_sufficient,
        },
        "objectives": gate_records,
        "comparison": {
            "per_seed_ranking_improvement": dict(
                zip((str(seed) for seed in common_seeds), ranking_improvements)
            ),
            "mean_ranking_improvement": float(np.mean(ranking_improvements)),
            "per_seed_mae_ratio": dict(
                zip((str(seed) for seed in common_seeds), mae_ratios)
            ),
            "mean_mae_ratio": float(np.mean(mae_ratios)),
            "additional_gates": additional_gates,
        },
        "candidate_passed_seed_count": candidate_passed_seeds,
        "required_passed_seed_count": 2,
        "task14_refinement_passed": task_passed,
        "resume_mch_ppo": task_passed,
        "enter_gnn": False,
    }
    _write_csv(args.output_dir / "training_curves.csv", curves)
    _write_csv(args.output_dir / "metrics.csv", metrics_rows)
    _write_csv(args.output_dir / "predictions.csv", predictions_rows)
    _write_csv(args.output_dir / "training_records.csv", training_records)
    (args.output_dir / "gate_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config = {
        "schema_version": 1,
        "base_dataset": str(args.base_dataset.resolve()),
        "model_dir": str(args.model_dir.resolve()),
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
        "objectives": OBJECTIVES,
        "model": MaskedActionQCriticConfig().signature(),
    }
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"output_dir={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
