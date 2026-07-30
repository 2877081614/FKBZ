from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.common.dynamic_support_distance import (
    dynamic_support_cost_matrix,
    enumerate_feasible_suffixes,
    jaccard_distance,
)
from rein_learning.envs import (
    AirDefenseResourceAssignmentEnvV1,
    ConflictFreeJointActionCodec,
    get_air_defense_v1_scenario,
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "dynamic_support_trust_region"
    / "dst_02_metric_validation"
)
SCENARIOS = ("medium", "time_pressure", "heterogeneity_pressure")
UNIT_ORDERS = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
SEEDS = (51001, 51002, 51003, 51004, 51005)
IMPLEMENTATION_FILES = (
    "rein_learning/common/dynamic_support_distance.py",
    "tests/test_dynamic_support_distance.py",
    "scripts/validate_air_defense_v1_dynamic_support_distance.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate exact AirDefense-v1 dynamic-support enumeration."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def codec_suffixes(
    *,
    codec: ConflictFreeJointActionCodec,
    base_mask: np.ndarray,
    prefix: tuple[int, ...],
    unit_order: tuple[int, ...],
) -> set[tuple[int, ...]]:
    expected: set[tuple[int, ...]] = set()
    for joint in codec.joint_actions:
        if not all(
            base_mask[unit_index, action]
            for unit_index, action in enumerate(joint)
        ):
            continue
        ordered = tuple(joint[unit_index] for unit_index in unit_order)
        if ordered[: len(prefix)] == prefix:
            expected.add(ordered[len(prefix) :])
    return expected


def legal_joint_actions(
    codec: ConflictFreeJointActionCodec,
    base_mask: np.ndarray,
) -> list[tuple[int, ...]]:
    return [
        joint
        for joint in codec.joint_actions
        if all(
            base_mask[unit_index, action]
            for unit_index, action in enumerate(joint)
        )
    ]


def run_validation() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    deterministic_checks = 0
    cost_matrix_checks = 0
    mask_mutation_errors = 0
    suffix_mismatch_total = 0
    states_checked = 0

    for scenario in SCENARIOS:
        for seed in SEEDS:
            env = AirDefenseResourceAssignmentEnvV1(
                config=get_air_defense_v1_scenario(scenario)
            )
            rng = np.random.default_rng(seed)
            env.reset(seed=seed)
            codec = ConflictFreeJointActionCodec(
                num_units=env.num_defense_units,
                num_targets=env.num_targets,
            )
            for state_index in range(4):
                base_mask = env.action_mask().astype(bool)
                valid_joint = legal_joint_actions(codec, base_mask)
                if not valid_joint:
                    raise RuntimeError("Official environment mask has no legal joint action")
                factual_joint = valid_joint[int(rng.integers(len(valid_joint)))]
                states_checked += 1

                for unit_order in UNIT_ORDERS:
                    ordered_joint = tuple(
                        factual_joint[unit_index] for unit_index in unit_order
                    )
                    for prefix_length in range(env.num_defense_units + 1):
                        prefix = ordered_joint[:prefix_length]
                        expected = codec_suffixes(
                            codec=codec,
                            base_mask=base_mask,
                            prefix=prefix,
                            unit_order=unit_order,
                        )
                        first = enumerate_feasible_suffixes(
                            env,
                            prefix,
                            unit_order,
                        )
                        second = enumerate_feasible_suffixes(
                            env,
                            prefix,
                            unit_order,
                        )
                        deterministic = first == second
                        deterministic_checks += 1
                        observed = set(first)
                        mismatches = len(observed.symmetric_difference(expected))
                        suffix_mismatch_total += mismatches
                        current_mask = env.action_mask().astype(bool)
                        mask_unchanged = np.array_equal(base_mask, current_mask)
                        mask_mutation_errors += int(not mask_unchanged)
                        rows.append(
                            {
                                "scenario": scenario,
                                "seed": seed,
                                "state_index": state_index,
                                "environment_step": int(env.current_step),
                                "unit_order": "".join(map(str, unit_order)),
                                "prefix_length": prefix_length,
                                "prefix": ",".join(map(str, prefix)),
                                "expected_suffix_count": len(expected),
                                "enumerated_suffix_count": len(first),
                                "duplicate_suffix_count": len(first) - len(observed),
                                "symmetric_difference_count": mismatches,
                                "deterministic_repeat": deterministic,
                                "environment_mask_unchanged": mask_unchanged,
                            }
                        )

                    for prefix_length in range(env.num_defense_units - 1):
                        matrix = dynamic_support_cost_matrix(
                            env,
                            ordered_joint[:prefix_length],
                            unit_order,
                        )
                        repeated = dynamic_support_cost_matrix(
                            env,
                            ordered_joint[:prefix_length],
                            unit_order,
                        )
                        if not np.array_equal(matrix.costs, repeated.costs):
                            raise AssertionError("Cost matrix is not deterministic")
                        if not np.array_equal(matrix.costs, matrix.costs.T):
                            raise AssertionError("Cost matrix is not symmetric")
                        if not np.allclose(np.diag(matrix.costs), 0.0):
                            raise AssertionError("Cost matrix diagonal is not zero")
                        if bool(
                            np.any(matrix.costs < 0.0)
                            or np.any(matrix.costs > 1.0)
                        ):
                            raise AssertionError("Cost matrix falls outside [0, 1]")
                        cost_matrix_checks += 1

                _, _, terminated, truncated, _ = env.step(factual_joint)
                if terminated or truncated:
                    break
            env.close()

    generic_properties = {
        "identical_sets_zero": jaccard_distance({(0,), (1,)}, {(0,), (1,)})
        == 0.0,
        "disjoint_nonempty_sets_one": jaccard_distance({(0,)}, {(1,)}) == 1.0,
        "symmetry": jaccard_distance({(0,), (1,)}, {(1,), (2,)})
        == jaccard_distance({(1,), (2,)}, {(0,), (1,)}),
    }
    passed = (
        suffix_mismatch_total == 0
        and mask_mutation_errors == 0
        and all(row["duplicate_suffix_count"] == 0 for row in rows)
        and all(row["deterministic_repeat"] for row in rows)
        and all(generic_properties.values())
    )
    summary = {
        "schema_version": 1,
        "task": "DST-02",
        "status": "PASS" if passed else "FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "environment_legality_rules_modified": False,
        "scenarios": list(SCENARIOS),
        "seeds": list(SEEDS),
        "unit_orders": ["".join(map(str, order)) for order in UNIT_ORDERS],
        "states_checked": states_checked,
        "prefix_crosscheck_rows": len(rows),
        "deterministic_repeat_checks": deterministic_checks,
        "cost_matrix_property_checks": cost_matrix_checks,
        "suffix_symmetric_difference_total": suffix_mismatch_total,
        "environment_mask_mutation_errors": mask_mutation_errors,
        "duplicate_suffix_total": sum(
            row["duplicate_suffix_count"] for row in rows
        ),
        "generic_jaccard_properties": generic_properties,
        "implementation_sha256": {
            path: hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
            for path in IMPLEMENTATION_FILES
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    return rows, summary


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    crosscheck_path = output_dir / "enumerator_crosscheck.csv"
    with crosscheck_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    rows, summary = run_validation()
    write_outputs(args.output_dir, rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
