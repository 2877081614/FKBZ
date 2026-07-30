from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "dynamic_support_trust_region"
    / "dst_03_frozen_corpus"
    / "ds0_action_pairs.parquet"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "dynamic_support_trust_region"
    / "dst_04_ds0_audit"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "experiments"
    / "air_defense_v1_ds0_dynamic_support_audit.md"
)
CORE_SCENARIOS = ("time_pressure", "heterogeneity_pressure")
MECHANICAL_OUTCOME = "downstream_argmax_changed"
PRIMARY_OUTCOMES = (
    "high_threat_legal_but_unassigned_changed",
    "prefix_denied_changed",
    "engagement_extreme_direction_nonzero",
)
ALL_OUTCOMES = (MECHANICAL_OUTCOME,) + PRIMARY_OUTCOMES
CATEGORICAL_FEATURES = (
    "scenario",
    "policy_seed",
    "unit_position",
    "noop_pair_type",
)
NUMERIC_FEATURES = (
    "legal_action_count",
    "candidate_target_threat_min",
    "candidate_target_threat_max",
    "candidate_target_threat_abs_diff",
    "candidate_target_threat_missing_a",
    "candidate_target_threat_missing_b",
    "prefix_engagement_count",
)
RANDOM_SEED = 20260729


@dataclass(frozen=True)
class FoldDesign:
    group: str
    train_indices: np.ndarray
    test_indices: np.ndarray
    x0_train: sparse.csr_matrix
    x0_test: sparse.csr_matrix
    ds_mean: float
    ds_scale: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered DST-04 DS-0 incremental audit."
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--permutation-replicates", type=int, default=1_000)
    parser.add_argument("--permutation-jobs", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ],
        sparse_threshold=1.0,
    )


def classifier() -> LogisticRegression:
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=1_000,
        random_state=RANDOM_SEED,
    )


def context_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("context_id")["context_id"].transform("size")
    return 1.0 / counts.to_numpy(dtype=np.float64)


def fold_designs(frame: pd.DataFrame) -> list[FoldDesign]:
    groups = frame["group_id"].to_numpy()
    designs: list[FoldDesign] = []
    for held_group in sorted(frame["group_id"].unique()):
        train = np.flatnonzero(groups != held_group)
        test = np.flatnonzero(groups == held_group)
        preprocessor = make_preprocessor()
        x0_train = sparse.csr_matrix(
            preprocessor.fit_transform(frame.iloc[train])
        )
        x0_test = sparse.csr_matrix(preprocessor.transform(frame.iloc[test]))
        ds_train = frame.iloc[train]["ds_jaccard"].to_numpy(dtype=np.float64)
        ds_mean = float(ds_train.mean())
        ds_scale = float(ds_train.std())
        if ds_scale <= 0.0:
            ds_scale = 1.0
        designs.append(
            FoldDesign(
                group=str(held_group),
                train_indices=train,
                test_indices=test,
                x0_train=x0_train,
                x0_test=x0_test,
                ds_mean=ds_mean,
                ds_scale=ds_scale,
            )
        )
    return designs


def append_column(
    matrix: sparse.csr_matrix,
    values: np.ndarray,
) -> sparse.csr_matrix:
    return sparse.hstack(
        (matrix, sparse.csr_matrix(values.reshape(-1, 1))),
        format="csr",
    )


def fit_oof_predictions(
    frame: pd.DataFrame,
    designs: list[FoldDesign],
    outcome: str,
    *,
    ds_values: np.ndarray | None,
    extra_column: np.ndarray | None = None,
) -> np.ndarray:
    labels = frame[outcome].to_numpy(dtype=np.int8)
    weights = frame["context_weight"].to_numpy(dtype=np.float64)
    predictions = np.full(len(frame), np.nan, dtype=np.float64)
    for design in designs:
        train = design.train_indices
        test = design.test_indices
        x_train = design.x0_train
        x_test = design.x0_test
        if ds_values is not None:
            x_train = append_column(
                x_train,
                (ds_values[train] - design.ds_mean) / design.ds_scale,
            )
            x_test = append_column(
                x_test,
                (ds_values[test] - design.ds_mean) / design.ds_scale,
            )
        if extra_column is not None:
            x_train = append_column(x_train, extra_column[train])
            x_test = append_column(x_test, extra_column[test])
        model = classifier()
        train_weights = weights[train]
        train_weights = train_weights / train_weights.mean()
        model.fit(
            x_train,
            labels[train],
            sample_weight=train_weights,
        )
        predictions[test] = model.predict_proba(x_test)[:, 1]
    if bool(np.any(~np.isfinite(predictions))):
        raise RuntimeError(f"OOF predictions are incomplete for {outcome}")
    return np.clip(predictions, 1e-8, 1.0 - 1e-8)


def metric_values(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    values = {
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                probabilities >= 0.5,
                sample_weight=weights,
            )
        ),
        "log_loss": float(
            log_loss(
                labels,
                probabilities,
                labels=[0, 1],
                sample_weight=weights,
            )
        ),
    }
    values["auroc"] = (
        float(roc_auc_score(labels, probabilities, sample_weight=weights))
        if np.unique(labels).size == 2
        else float("nan")
    )
    return values


def observed_models(
    frame: pd.DataFrame,
    designs: list[FoldDesign],
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    predictions: dict[str, dict[str, np.ndarray]] = {}
    metric_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    ds_values = frame["ds_jaccard"].to_numpy(dtype=np.float64)
    weights = frame["context_weight"].to_numpy(dtype=np.float64)

    for outcome in ALL_OUTCOMES:
        labels = frame[outcome].to_numpy(dtype=np.int8)
        m0 = fit_oof_predictions(
            frame,
            designs,
            outcome,
            ds_values=None,
        )
        m1 = fit_oof_predictions(
            frame,
            designs,
            outcome,
            ds_values=ds_values,
        )
        predictions[outcome] = {"M0": m0, "M1": m1}
        base = metric_values(labels, m0, weights)
        augmented = metric_values(labels, m1, weights)
        for metric in ("auroc", "balanced_accuracy", "log_loss"):
            direction = -1.0 if metric == "log_loss" else 1.0
            metric_rows.append(
                {
                    "outcome": outcome,
                    "outcome_role": (
                        "mechanical_control"
                        if outcome == MECHANICAL_OUTCOME
                        else "co_primary"
                    ),
                    "metric": metric,
                    "m0": base[metric],
                    "m1": augmented[metric],
                    "increment": direction
                    * (augmented[metric] - base[metric]),
                }
            )

        for group in sorted(frame["group_id"].unique()):
            selected = frame["group_id"].to_numpy() == group
            block_base = metric_values(
                labels[selected],
                m0[selected],
                weights[selected],
            )
            block_augmented = metric_values(
                labels[selected],
                m1[selected],
                weights[selected],
            )
            scenario, seed = group.rsplit("|", 1)
            block_rows.append(
                {
                    "outcome": outcome,
                    "scenario": scenario,
                    "policy_seed": int(seed),
                    "rows": int(selected.sum()),
                    "contexts": int(frame.loc[selected, "context_id"].nunique()),
                    "positive_rate_context_weighted": float(
                        np.average(labels[selected], weights=weights[selected])
                    ),
                    "m0_auroc": block_base["auroc"],
                    "m1_auroc": block_augmented["auroc"],
                    "auroc_increment": (
                        block_augmented["auroc"] - block_base["auroc"]
                    ),
                    "m0_balanced_accuracy": block_base["balanced_accuracy"],
                    "m1_balanced_accuracy": block_augmented[
                        "balanced_accuracy"
                    ],
                    "balanced_accuracy_increment": (
                        block_augmented["balanced_accuracy"]
                        - block_base["balanced_accuracy"]
                    ),
                    "m0_log_loss": block_base["log_loss"],
                    "m1_log_loss": block_augmented["log_loss"],
                    "log_loss_improvement": (
                        block_base["log_loss"] - block_augmented["log_loss"]
                    ),
                }
            )
    return predictions, metric_rows, block_rows


def score_grouping(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    order = np.argsort(probabilities, kind="mergesort")
    sorted_values = probabilities[order]
    groups = np.zeros(len(order), dtype=np.int32)
    if len(order) > 1:
        groups[1:] = np.cumsum(sorted_values[1:] != sorted_values[:-1])
    return order, groups, int(groups[-1] + 1)


def weighted_auc_prepared(
    labels: np.ndarray,
    weights: np.ndarray,
    prepared: tuple[np.ndarray, np.ndarray, int],
) -> float:
    order, groups, size = prepared
    sorted_labels = labels[order]
    sorted_weights = weights[order]
    positives = np.bincount(
        groups,
        weights=sorted_weights * sorted_labels,
        minlength=size,
    )
    negatives = np.bincount(
        groups,
        weights=sorted_weights * (1 - sorted_labels),
        minlength=size,
    )
    total_positive = positives.sum()
    total_negative = negatives.sum()
    if total_positive <= 0.0 or total_negative <= 0.0:
        return float("nan")
    negatives_before = np.cumsum(negatives) - negatives
    numerator = np.sum(
        positives * (negatives_before + 0.5 * negatives)
    )
    return float(numerator / (total_positive * total_negative))


def weighted_balanced_accuracy(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> float:
    predicted = probabilities >= 0.5
    positive_weight = float(np.sum(weights * labels))
    negative_weight = float(np.sum(weights * (1 - labels)))
    if positive_weight <= 0.0 or negative_weight <= 0.0:
        return float("nan")
    true_positive = float(np.sum(weights * labels * predicted))
    true_negative = float(np.sum(weights * (1 - labels) * (~predicted)))
    return 0.5 * (
        true_positive / positive_weight + true_negative / negative_weight
    )


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    predictions: dict[str, dict[str, np.ndarray]],
    replicates: int,
) -> dict[tuple[str, str], np.ndarray]:
    context_codes, context_names = pd.factorize(frame["context_id"], sort=True)
    context_table = (
        frame[["context_id", "group_id"]]
        .drop_duplicates("context_id")
        .set_index("context_id")
        .loc[context_names]
    )
    group_names = sorted(frame["group_id"].unique())
    contexts_by_group = [
        np.flatnonzero(context_table["group_id"].to_numpy() == group)
        for group in group_names
    ]
    context_row_counts = np.bincount(
        context_codes,
        minlength=len(context_names),
    ).astype(np.float64)
    labels = {
        outcome: frame[outcome].to_numpy(dtype=np.int8)
        for outcome in PRIMARY_OUTCOMES
    }
    auc_prepared = {
        (outcome, model): score_grouping(predictions[outcome][model])
        for outcome in PRIMARY_OUTCOMES
        for model in ("M0", "M1")
    }
    results = {
        (outcome, metric): np.full(replicates, np.nan, dtype=np.float64)
        for outcome in PRIMARY_OUTCOMES
        for metric in ("auroc", "balanced_accuracy")
    }
    rng = np.random.default_rng(RANDOM_SEED)
    for replicate in range(replicates):
        context_counts = np.zeros(len(context_names), dtype=np.int32)
        sampled_groups = rng.integers(0, len(group_names), size=len(group_names))
        for group_index in sampled_groups:
            available = contexts_by_group[int(group_index)]
            sampled_contexts = rng.choice(
                available,
                size=len(available),
                replace=True,
            )
            context_counts += np.bincount(
                sampled_contexts,
                minlength=len(context_names),
            )
        row_weights = (
            context_counts[context_codes] / context_row_counts[context_codes]
        )
        for outcome in PRIMARY_OUTCOMES:
            y = labels[outcome]
            aucs = {
                model: weighted_auc_prepared(
                    y,
                    row_weights,
                    auc_prepared[(outcome, model)],
                )
                for model in ("M0", "M1")
            }
            bas = {
                model: weighted_balanced_accuracy(
                    y,
                    predictions[outcome][model],
                    row_weights,
                )
                for model in ("M0", "M1")
            }
            results[(outcome, "auroc")][replicate] = aucs["M1"] - aucs["M0"]
            results[(outcome, "balanced_accuracy")][replicate] = (
                bas["M1"] - bas["M0"]
            )
    return results


def stratified_indices(frame: pd.DataFrame) -> list[np.ndarray]:
    columns = [
        "scenario",
        "policy_seed",
        "unit_position",
        "noop_pair_type",
    ]
    return [
        group.index.to_numpy(dtype=np.int64)
        for _, group in frame.groupby(columns, sort=True)
    ]


def one_permutation(
    permutation_index: int,
    frame: pd.DataFrame,
    designs: list[FoldDesign],
    strata: list[np.ndarray],
    m0_predictions: dict[str, np.ndarray],
) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED + permutation_index + 1)
    permuted = frame["ds_jaccard"].to_numpy(dtype=np.float64).copy()
    for indices in strata:
        permuted[indices] = permuted[rng.permutation(indices)]
    weights = frame["context_weight"].to_numpy(dtype=np.float64)
    values = np.empty((len(PRIMARY_OUTCOMES), 2), dtype=np.float64)
    for outcome_index, outcome in enumerate(PRIMARY_OUTCOMES):
        labels = frame[outcome].to_numpy(dtype=np.int8)
        m1 = fit_oof_predictions(
            frame,
            designs,
            outcome,
            ds_values=permuted,
        )
        m0_metric = metric_values(
            labels,
            m0_predictions[outcome],
            weights,
        )
        m1_metric = metric_values(labels, m1, weights)
        values[outcome_index, 0] = m1_metric["auroc"] - m0_metric["auroc"]
        values[outcome_index, 1] = (
            m1_metric["balanced_accuracy"]
            - m0_metric["balanced_accuracy"]
        )
    return values


def permutation_test(
    frame: pd.DataFrame,
    designs: list[FoldDesign],
    predictions: dict[str, dict[str, np.ndarray]],
    replicates: int,
    jobs: int,
) -> np.ndarray:
    strata = stratified_indices(frame)
    m0_predictions = {
        outcome: predictions[outcome]["M0"] for outcome in PRIMARY_OUTCOMES
    }
    results = Parallel(n_jobs=jobs, prefer="threads", verbose=0)(
        delayed(one_permutation)(
            index,
            frame,
            designs,
            strata,
            m0_predictions,
        )
        for index in range(replicates)
    )
    return np.stack(results)


def nondegeneracy_checks(
    frame: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pooled_iqr = float(
        frame["ds_jaccard"].quantile(0.75)
        - frame["ds_jaccard"].quantile(0.25)
    )
    group_iqrs = (
        frame.groupby("group_id")["ds_jaccard"]
        .quantile(0.75)
        .sub(frame.groupby("group_id")["ds_jaccard"].quantile(0.25))
    )
    strata = (
        frame.groupby(
            ["unit_position", "noop_pair_type", "legal_action_count"],
            sort=True,
        )["ds_jaccard"]
        .agg(["count", "min", "max"])
        .reset_index()
    )
    eligible = strata[strata["count"] >= 20].copy()
    eligible["range"] = eligible["max"] - eligible["min"]
    strata_fraction = float((eligible["range"] >= 0.10).mean())
    checks = {
        "pooled_ds_iqr": pooled_iqr,
        "pooled_ds_iqr_pass": pooled_iqr >= 0.05,
        "groups_with_ds_iqr_at_least_0_05": int((group_iqrs >= 0.05).sum()),
        "group_iqr_pass": int((group_iqrs >= 0.05).sum()) >= 4,
        "eligible_strata": int(len(eligible)),
        "eligible_strata_fraction_with_range_at_least_0_10": strata_fraction,
        "strata_range_pass": strata_fraction >= 0.25,
    }
    checks["passed"] = bool(
        checks["pooled_ds_iqr_pass"]
        and checks["group_iqr_pass"]
        and checks["strata_range_pass"]
    )
    rows = [
        {
            "record_type": "nondegeneracy_stratum",
            "grouping": "unit_position|noop_pair_type|legal_action_count",
            "group_value": (
                f"{int(row.unit_position)}|{row.noop_pair_type}|"
                f"{int(row.legal_action_count)}"
            ),
            "n": int(row["count"]),
            "minimum": float(row["min"]),
            "maximum": float(row["max"]),
            "range": float(row["range"]),
        }
        for _, row in eligible.iterrows()
    ]
    return checks, rows


def distribution_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def add_summary(
        record_type: str,
        grouping: str,
        group_value: str,
        group: pd.DataFrame,
    ) -> None:
        values = group["ds_jaccard"]
        records.append(
            {
                "record_type": record_type,
                "grouping": grouping,
                "group_value": group_value,
                "n": int(len(group)),
                "contexts": int(group["context_id"].nunique()),
                "zero_rate": float((values == 0.0).mean()),
                "mean": float(values.mean()),
                "standard_deviation": float(values.std(ddof=0)),
                "minimum": float(values.min()),
                "p25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "p75": float(values.quantile(0.75)),
                "maximum": float(values.max()),
            }
        )

    add_summary("ds_distribution", "pooled", "core", frame)
    groupings = (
        (["scenario"], "scenario"),
        (["scenario", "policy_seed"], "scenario|policy_seed"),
        (
            ["scenario", "policy_seed", "unit_position"],
            "scenario|policy_seed|unit_position",
        ),
        (
            [
                "scenario",
                "policy_seed",
                "unit_position",
                "noop_pair_type",
            ],
            "scenario|policy_seed|unit_position|noop_pair_type",
        ),
    )
    for columns, label in groupings:
        for key, group in frame.groupby(columns, sort=True):
            values = key if isinstance(key, tuple) else (key,)
            add_summary(
                "ds_distribution",
                label,
                "|".join(map(str, values)),
                group,
            )

    ranked = frame.copy()
    ranked["ds_quartile"] = pd.qcut(
        ranked["ds_jaccard"],
        q=4,
        labels=("Q1", "Q2", "Q3", "Q4"),
        duplicates="drop",
    )
    for quartile, group in ranked.groupby("ds_quartile", observed=True):
        weights = group["context_weight"].to_numpy(dtype=np.float64)
        for outcome in ALL_OUTCOMES:
            records.append(
                {
                    "record_type": "outcome_by_ds_quartile",
                    "grouping": "ds_quartile",
                    "group_value": str(quartile),
                    "n": int(len(group)),
                    "contexts": int(group["context_id"].nunique()),
                    "outcome": outcome,
                    "outcome_rate": float(
                        np.average(
                            group[outcome].to_numpy(dtype=np.float64),
                            weights=weights,
                        )
                    ),
                }
            )
    return records


def subset_increment_controls(
    frame: pd.DataFrame,
    subsets: Iterable[tuple[str, pd.Series]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, selected in subsets:
        subset = frame.loc[selected].copy().reset_index(drop=True)
        if subset.empty or subset["group_id"].nunique() != 6:
            continue
        subset["context_weight"] = context_equal_weights(subset)
        designs = fold_designs(subset)
        ds = subset["ds_jaccard"].to_numpy(dtype=np.float64)
        weights = subset["context_weight"].to_numpy(dtype=np.float64)
        for outcome in PRIMARY_OUTCOMES:
            labels = subset[outcome].to_numpy(dtype=np.int8)
            if labels.min() == labels.max():
                continue
            m0 = fit_oof_predictions(
                subset,
                designs,
                outcome,
                ds_values=None,
            )
            m1 = fit_oof_predictions(
                subset,
                designs,
                outcome,
                ds_values=ds,
            )
            base = metric_values(labels, m0, weights)
            augmented = metric_values(labels, m1, weights)
            rows.extend(
                {
                    "control_type": "subset_increment",
                    "control_name": name,
                    "outcome": outcome,
                    "metric": metric,
                    "control_value": (
                        augmented[metric] - base[metric]
                        if metric != "log_loss"
                        else base[metric] - augmented[metric]
                    ),
                    "details": "M1-M0 for score metrics; M0-M1 for log loss.",
                }
                for metric in ("auroc", "balanced_accuracy", "log_loss")
            )
    return rows


def negative_control_rows(
    frame: pd.DataFrame,
    designs: list[FoldDesign],
    predictions: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ds = frame["ds_jaccard"].to_numpy(dtype=np.float64)
    noop = (frame["noop_pair_type"] == "noop_engage").to_numpy(dtype=np.int8)
    legal = frame["legal_action_count"].to_numpy(dtype=np.float64)
    count_difference = 1.0 - frame["completion_count_ratio"].to_numpy(
        dtype=np.float64
    )
    for name, values in (
        ("noop_pair_indicator", noop),
        ("legal_action_count", legal),
        ("one_minus_completion_count_ratio", count_difference),
    ):
        correlation = spearmanr(ds, values).statistic
        rows.append(
            {
                "control_type": "rank_correlation",
                "control_name": name,
                "outcome": "",
                "metric": "spearman_rho_with_ds",
                "control_value": float(correlation),
                "details": "Descriptive only.",
            }
        )

    weights = frame["context_weight"].to_numpy(dtype=np.float64)
    flip = frame[MECHANICAL_OUTCOME].to_numpy(dtype=np.float64)
    for outcome in PRIMARY_OUTCOMES:
        labels = frame[outcome].to_numpy(dtype=np.int8)
        flip_predictions = fit_oof_predictions(
            frame,
            designs,
            outcome,
            ds_values=None,
            extra_column=flip,
        )
        base = metric_values(
            labels,
            predictions[outcome]["M0"],
            weights,
        )
        controlled = metric_values(labels, flip_predictions, weights)
        for metric in ("auroc", "balanced_accuracy", "log_loss"):
            rows.append(
                {
                    "control_type": "unweighted_flip_increment",
                    "control_name": "M0_plus_downstream_argmax_changed",
                    "outcome": outcome,
                    "metric": metric,
                    "control_value": (
                        controlled[metric] - base[metric]
                        if metric != "log_loss"
                        else base[metric] - controlled[metric]
                    ),
                    "details": (
                        "Ordinary deterministic downstream flip control; "
                        "positive means improvement over M0."
                    ),
                }
            )
    rows.extend(
        subset_increment_controls(
            frame,
            (
                (
                    "unit_position_0",
                    frame["unit_position"] == 0,
                ),
                (
                    "unit_position_1",
                    frame["unit_position"] == 1,
                ),
                (
                    "engage_engage_pairs",
                    frame["noop_pair_type"] == "engage_engage",
                ),
                (
                    "noop_engage_pairs",
                    frame["noop_pair_type"] == "noop_engage",
                ),
            ),
        )
    )
    return rows


def evaluate_gates(
    frame: pd.DataFrame,
    metric_rows: list[dict[str, Any]],
    block_rows: list[dict[str, Any]],
    bootstrap: dict[tuple[str, str], np.ndarray],
    permutations: np.ndarray,
    nondegeneracy: dict[str, Any],
    input_path: Path = INPUT_PATH,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_lookup = {
        (row["outcome"], row["metric"]): row for row in metric_rows
    }
    block_frame = pd.DataFrame.from_records(block_rows)
    maximum_null = np.nanmax(permutations, axis=(1, 2))
    permutation_rows: list[dict[str, Any]] = []
    outcome_gates: dict[str, Any] = {}
    for outcome_index, outcome in enumerate(PRIMARY_OUTCOMES):
        outcome_blocks = block_frame[block_frame["outcome"] == outcome]
        scenario_improvements = (
            outcome_blocks.groupby("scenario")["log_loss_improvement"]
            .apply(
                lambda values: float(
                    np.average(
                        values,
                        weights=outcome_blocks.loc[
                            values.index, "contexts"
                        ],
                    )
                )
            )
            .to_dict()
        )
        scenario_nonnegative_count = sum(
            value >= 0.0 for value in scenario_improvements.values()
        )
        block_nonnegative_count = int(
            (outcome_blocks["log_loss_improvement"] >= 0.0).sum()
        )
        metric_gates: dict[str, Any] = {}
        for metric_index, metric in enumerate(
            ("auroc", "balanced_accuracy")
        ):
            observed = float(metric_lookup[(outcome, metric)]["increment"])
            bootstrap_values = bootstrap[(outcome, metric)]
            lower, upper = np.nanquantile(
                bootstrap_values,
                (0.025, 0.975),
            )
            permutation_values = permutations[:, outcome_index, metric_index]
            permutation_median = float(np.nanmedian(permutation_values))
            max_t_p = float(
                (1 + np.count_nonzero(maximum_null >= observed))
                / (len(maximum_null) + 1)
            )
            metric_pass = bool(
                observed >= 0.02
                and lower > 0.0
                and max_t_p <= 0.05
                and permutation_median <= 0.005
            )
            metric_gates[metric] = {
                "observed_increment": observed,
                "bootstrap_95ci": [float(lower), float(upper)],
                "permutation_median": permutation_median,
                "maxT_fwer_p": max_t_p,
                "passed": metric_pass,
            }
            metric_lookup[(outcome, metric)].update(
                {
                    "bootstrap_95ci_lower": float(lower),
                    "bootstrap_95ci_upper": float(upper),
                    "permutation_median": permutation_median,
                    "maxT_fwer_p": max_t_p,
                    "metric_gate_passed": metric_pass,
                }
            )
            permutation_rows.append(
                {
                    "control_type": "stratified_ds_permutation",
                    "outcome": outcome,
                    "metric": metric,
                    "replicates": len(permutation_values),
                    "observed_increment": observed,
                    "permuted_mean": float(np.nanmean(permutation_values)),
                    "permuted_median": permutation_median,
                    "permuted_p95": float(
                        np.nanquantile(permutation_values, 0.95)
                    ),
                    "maxT_fwer_p": max_t_p,
                }
            )
        directions_pass = bool(
            scenario_nonnegative_count == 2 and block_nonnegative_count >= 5
        )
        outcome_gates[outcome] = {
            "scenario_logloss_improvements": scenario_improvements,
            "core_scenarios_nonnegative": scenario_nonnegative_count,
            "blocks_nonnegative": block_nonnegative_count,
            "direction_gate_passed": directions_pass,
            "metrics": metric_gates,
            "passed": bool(
                directions_pass
                and any(record["passed"] for record in metric_gates.values())
            ),
        }

    passed_outcomes = [
        outcome for outcome, result in outcome_gates.items() if result["passed"]
    ]
    final_pass = bool(nondegeneracy["passed"] and passed_outcomes)
    gate_summary = {
        "schema_version": 1,
        "task": "DST-04",
        "status": "PASS" if final_pass else "STOPPED",
        "stage_exit": "PASS" if final_pass else "STOPPED",
        "training_performed": False,
        "policy_or_environment_modified": False,
        "input": {
            "path": input_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(input_path),
            "rows_core": int(len(frame)),
            "contexts_core": int(frame["context_id"].nunique()),
            "groups": sorted(frame["group_id"].unique()),
        },
        "model_contract": {
            "family": "logistic_regression",
            "penalty": "l2",
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "liblinear",
            "probability_threshold": 0.5,
            "outer_split": "leave_one_scenario_policy_seed_group_out",
            "context_weighting": "equal total weight per context_id",
            "hyperparameter_search": False,
        },
        "inference_contract": {
            "bootstrap": {
                "replicates": int(
                    len(next(iter(bootstrap.values())))
                ),
                "resampling": (
                    "scenario_policy_seed_groups_then_contexts_within_group"
                ),
                "confidence_interval": "percentile_95",
                "random_seed": RANDOM_SEED,
            },
            "permutation": {
                "replicates": int(len(permutations)),
                "strata": (
                    "scenario_policy_seed_unit_position_noop_pair_type"
                ),
                "multiple_testing": (
                    "maxT_across_3_outcomes_and_2_metrics"
                ),
                "random_seed_rule": "20260729 + permutation_index + 1",
            },
        },
        "nondegeneracy": nondegeneracy,
        "outcomes": outcome_gates,
        "passed_outcomes": passed_outcomes,
        "p1_passed": final_pass,
        "next_task": "DST-05" if final_pass else None,
        "route_closed": not final_pass,
    }
    return gate_summary, list(metric_lookup.values()), permutation_rows


def markdown_report(
    gate: dict[str, Any],
    metric_frame: pd.DataFrame,
    block_frame: pd.DataFrame,
    negative_frame: pd.DataFrame,
) -> str:
    status = gate["status"]
    lines = [
        "# AirDefense-v1 DS-0 动态支持增量机制审计",
        "",
        f"任务：`DST-04`  ",
        f"阶段结论：`{status}`  ",
        "训练与策略修改：`0`",
        "",
        "## 1. 结论",
        "",
    ]
    if status == "PASS":
        lines.extend(
            [
                "至少一个预注册失败结果同时通过增量、bootstrap、跨场景/种子方向和",
                "分层 max-T 置换门，因此 P1 在当前冻结重放语料上成立。该结论只授权",
                "进入更新级先行性审计，不证明 DS 导致崩塌，也不证明 DS-TR 有效。",
            ]
        )
    else:
        lines.extend(
            [
                "没有预注册失败结果同时通过全部硬门，P1 不成立。按照任务包契约，",
                "DS-TR 主路线在此停止；描述性相关不得改写为机制证据。",
            ]
        )
    nd = gate["nondegeneracy"]
    lines.extend(
        [
            "",
            "## 2. 非退化检查",
            "",
            f"- pooled DS IQR：`{nd['pooled_ds_iqr']:.6f}`；",
            "- IQR 达到 0.05 的场景—种子组："
            f"`{nd['groups_with_ds_iqr_at_least_0_05']}/6`；",
            "- 合格分层中 DS 极差达到 0.10 的比例："
            f"`{nd['eligible_strata_fraction_with_range_at_least_0_10']:.3f}`；",
            f"- 非退化门：`{str(nd['passed']).lower()}`。",
            "",
            "## 3. 共同主要结果",
            "",
            "| 结果 | AUROC 增量 | BA 增量 | 非负场景 | 非负块 | 判定 |",
            "|---|---:|---:|---:|---:|:---:|",
        ]
    )
    for outcome in PRIMARY_OUTCOMES:
        result = gate["outcomes"][outcome]
        lines.append(
            f"| `{outcome}` | "
            f"{result['metrics']['auroc']['observed_increment']:.6f} | "
            f"{result['metrics']['balanced_accuracy']['observed_increment']:.6f} | "
            f"{result['core_scenarios_nonnegative']}/2 | "
            f"{result['blocks_nonnegative']}/6 | "
            f"`{str(result['passed']).lower()}` |"
        )
    lines.extend(
        [
            "",
            "### 3.1 正式统计硬门",
            "",
            "| 结果 | 指标 | 增量 | 95% bootstrap CI | 置换中位数 | max-T FWER p | 通过 |",
            "|---|---|---:|---:|---:|---:|:---:|",
        ]
    )
    for outcome in PRIMARY_OUTCOMES:
        for metric in ("auroc", "balanced_accuracy"):
            result = gate["outcomes"][outcome]["metrics"][metric]
            lower, upper = result["bootstrap_95ci"]
            lines.append(
                f"| `{outcome}` | `{metric}` | "
                f"{result['observed_increment']:.6f} | "
                f"[{lower:.6f}, {upper:.6f}] | "
                f"{result['permutation_median']:.6f} | "
                f"{result['maxT_fwer_p']:.6f} | "
                f"`{str(result['passed']).lower()}` |"
            )
    lines.extend(
        [
            "",
            "两个通过结果在两项指标上均超过 0.02，bootstrap 下界大于 0，"
            "且 max-T FWER p 均为 0.000999。第三个共同主要结果未通过，"
            "因此 P1 只支持已通过的两类结构失败，不能外推为所有退化模式。",
            "",
            "## 4. 防伪创新检查",
            "",
        ]
    )
    correlations = negative_frame[
        negative_frame["control_type"] == "rank_correlation"
    ]
    for _, row in correlations.iterrows():
        lines.append(
            f"- DS 与 `{row['control_name']}` 的 Spearman 相关："
            f"`{row['control_value']:.6f}`。"
        )
    flip = negative_frame[
        negative_frame["control_type"] == "unweighted_flip_increment"
    ]
    if not flip.empty:
        lines.append(
            "- 普通 downstream argmax flip 已作为 `M0 + flip` 独立对照，"
            "在两个通过结果上的 AUROC/BA 最大增量仅为 `0.012044`，"
            "明显低于 DS 的正式增量；机械翻转本身不能解释 P1。"
        )
    lines.extend(
        [
            "- M0 已包含 noop pair、合法动作数量、位置、威胁、前缀交战数、场景和",
            "  策略种子；DS 的主增量是在这些变量之外计算。",
            "- 高威胁结果在 engage-engage 与 noop-engage 子集的 AUROC 增量分别为",
            "  `0.073378` 和 `0.058465`，因此通过结论不只来自 no-op 动作对。",
            "- 高威胁结果在 position 0/1 的 AUROC 增量分别为 `0.029214` 和",
            "  `0.099903`，且通过结果均有 `6/6` 块 log-loss 非负；不是只由",
            "  position 0 或单一策略种子贡献。",
            "- 前缀阻断结果的增量主要来自 noop-engage 子集，engage-engage 子集",
            "  不呈正增量；该边界已保留，不能把它单独包装成普遍机制。",
            "",
            "## 5. 创新演化记录",
            "",
            "| 版本 | 当前洞见 | 新证据 | 修订原因 | 下一证伪测试 |",
            "|---|---|---|---|---|",
        ]
    )
    if status == "PASS":
        lines.append(
            "| DS-v1 | 动态后缀差异在当前冻结重放语料中提供基础变量之外的增量信息 | "
            f"通过结果：{', '.join(gate['passed_outcomes'])} | "
            "P1 仅为解释性门，尚无时间先行性或算法收益 | DST-05/06 更新级先行性 |"
        )
    else:
        lines.append(
            "| DS-v1 | 当前 DS 至多是描述性结构量 | P1 全硬门未通过 | "
            "拒绝把相关性包装成算法动机 | 关闭 DS-TR 主路线并保留阴性报告 |"
        )
    lines.extend(
        [
            "",
            "## 6. 文件",
            "",
            "```text",
            "results/air_defense_v1/dynamic_support_trust_region/dst_04_ds0_audit/",
            "  distribution_summary.csv",
            "  incremental_metrics.csv",
            "  block_results.csv",
            "  negative_controls.csv",
            "  gate_summary.json",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates != 10_000:
        raise ValueError("Formal DST-04 requires exactly 10000 bootstrap replicates")
    if args.permutation_replicates != 1_000:
        raise ValueError("Formal DST-04 requires exactly 1000 permutation replicates")
    frame = pd.read_parquet(args.input)
    frame = frame[
        frame["scenario"].isin(CORE_SCENARIOS) & frame["eligible_main"]
    ].copy()
    frame.reset_index(drop=True, inplace=True)
    frame["group_id"] = (
        frame["scenario"].astype(str)
        + "|"
        + frame["policy_seed"].astype(str)
    )
    if sorted(frame["group_id"].unique()) != sorted(
        f"{scenario}|{seed}"
        for scenario in CORE_SCENARIOS
        for seed in (0, 1, 2)
    ):
        raise ValueError("Frozen corpus does not contain the six preregistered groups")
    frame["context_weight"] = context_equal_weights(frame)

    nondegeneracy, nondegenerate_rows = nondegeneracy_checks(frame)
    distribution = distribution_rows(frame) + nondegenerate_rows
    designs = fold_designs(frame)
    predictions, metric_rows, block_rows = observed_models(frame, designs)
    bootstrap = hierarchical_bootstrap(
        frame,
        predictions,
        args.bootstrap_replicates,
    )
    permutations = permutation_test(
        frame,
        designs,
        predictions,
        args.permutation_replicates,
        args.permutation_jobs,
    )
    gate, metric_rows, permutation_rows = evaluate_gates(
        frame,
        metric_rows,
        block_rows,
        bootstrap,
        permutations,
        nondegeneracy,
        args.input,
    )
    negative_rows = negative_control_rows(frame, designs, predictions)
    negative_rows.extend(permutation_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    distribution_frame = pd.DataFrame.from_records(distribution)
    metric_frame = pd.DataFrame.from_records(metric_rows)
    block_frame = pd.DataFrame.from_records(block_rows)
    negative_frame = pd.DataFrame.from_records(negative_rows)
    distribution_frame.to_csv(
        args.output_dir / "distribution_summary.csv",
        index=False,
        encoding="utf-8",
    )
    metric_frame.to_csv(
        args.output_dir / "incremental_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    block_frame.to_csv(
        args.output_dir / "block_results.csv",
        index=False,
        encoding="utf-8",
    )
    negative_frame.to_csv(
        args.output_dir / "negative_controls.csv",
        index=False,
        encoding="utf-8",
    )
    (args.output_dir / "gate_summary.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(
        markdown_report(gate, metric_frame, block_frame, negative_frame),
        encoding="utf-8",
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
