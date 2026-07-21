from __future__ import annotations

import argparse
import csv
import json
from math import ceil, comb, sqrt
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.common import safety_resource_oracle


DEFAULT_RESULT_DIR = (
    PROJECT_ROOT / "results" / "air_defense_v1" / "task14_engagement_utility"
)
COMPONENT_KEYS = (
    "total_return_samples",
    "operational_return_samples",
    "resource_cost_samples",
    "damage_samples",
    "high_threat_leak_samples",
    "shot_samples",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit rare engage-oracle power for Task 14 utility data."
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--required-engage", type=int, default=8)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * sqrt(
            probability * (1.0 - probability) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _probability_at_least(required: int, trials: int, probability: float) -> float:
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    below = sum(
        comb(trials, count)
        * probability**count
        * (1.0 - probability) ** (trials - count)
        for count in range(min(required, trials + 1))
    )
    return float(max(0.0, min(1.0, 1.0 - below)))


def main() -> None:
    args = parse_args()
    dataset_path = args.result_dir / "dataset.npz"
    with np.load(dataset_path, allow_pickle=False) as archive:
        dataset = {key: archive[key] for key in archive.files}
    components = {key: dataset[key] for key in COMPONENT_KEYS}
    oracle = safety_resource_oracle(components)["labels"]
    count_rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test", "all"):
        split_mask = (
            np.ones(len(oracle), dtype=bool)
            if split == "all"
            else dataset["splits"] == split
        )
        for scenario in (*np.unique(dataset["scenarios"]).tolist(), "all"):
            selected = split_mask & (
                np.ones(len(oracle), dtype=bool)
                if scenario == "all"
                else dataset["scenarios"] == scenario
            )
            labels = oracle[selected]
            count_rows.append(
                {
                    "split": split,
                    "scenario": scenario,
                    "groups": int(np.sum(selected)),
                    "valid": int(np.sum(labels >= 0)),
                    "engage": int(np.sum(labels == 1)),
                    "noop": int(np.sum(labels == 0)),
                    "ambiguous": int(np.sum(labels < 0)),
                }
            )
    _write_csv(args.result_dir / "oracle_power_counts.csv", count_rows)

    test = dataset["splits"] == "test"
    test_valid = int(np.sum(test & (oracle >= 0)))
    test_engage = int(np.sum(test & (oracle == 1)))
    point_rate = test_engage / test_valid if test_valid else 0.0
    lower_rate, upper_rate = _wilson_interval(test_engage, test_valid)
    projected_groups = [test_valid, 100, 150, 168, 200, 250, 300, 500]
    power_rows = [
        {
            "test_valid_groups": groups,
            "expected_engage_point": groups * point_rate,
            "probability_at_least_required_point": _probability_at_least(
                args.required_engage, groups, point_rate
            ),
            "expected_engage_wilson_lower": groups * lower_rate,
            "probability_at_least_required_wilson_lower": _probability_at_least(
                args.required_engage, groups, lower_rate
            ),
        }
        for groups in projected_groups
    ]
    _write_csv(args.result_dir / "engage_power_projection.csv", power_rows)

    current_states = int(dataset["state_count"])
    current_test_groups = int(np.sum(test))
    point_required_groups = (
        ceil(args.required_engage / point_rate) if point_rate > 0.0 else None
    )
    conservative_required_groups = (
        ceil(args.required_engage / lower_rate) if lower_rate > 0.0 else None
    )
    summary = {
        "test_valid_groups": test_valid,
        "test_engage_groups": test_engage,
        "observed_engage_rate": point_rate,
        "engage_rate_wilson_95": [lower_rate, upper_rate],
        "required_engage_groups": args.required_engage,
        "point_required_test_valid_groups": point_required_groups,
        "conservative_required_test_valid_groups": conservative_required_groups,
        "point_projected_total_states": (
            ceil(current_states * point_required_groups / current_test_groups)
            if point_required_groups is not None
            else None
        ),
        "conservative_projected_total_states": (
            ceil(current_states * conservative_required_groups / current_test_groups)
            if conservative_required_groups is not None
            else None
        ),
        "recommendation": (
            "Prefer targeted safety-critical state collection over uniform expansion; "
            "all confident engage cases currently occur in medium."
        ),
    }
    with (args.result_dir / "power_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
