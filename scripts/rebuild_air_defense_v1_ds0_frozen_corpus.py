from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import comb
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.algorithms.policy_gradient import (
    RoleConditionedAutoregressiveMaskablePPO,
)
from rein_learning.common import PolicyProbeCorpus
from rein_learning.common.dynamic_support_distance import (
    dynamic_support_cost_matrix,
    enumerate_feasible_suffixes,
    old_policy_structural_risk,
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "dynamic_support_trust_region"
    / "dst_03_frozen_corpus"
)
PROBE_DIR = (
    PROJECT_ROOT / "results" / "air_defense_v1" / "task12_probe_corpus"
)
REPLAY_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task12_task11_frozen_replay"
)
TASK10_FROZEN_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task10_frozen_model_diagnostics"
)
TASK10_FORMAL_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task10_order_screening_30k_3seeds"
)
TASK11_FORMAL_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task11_role_conditioned_screening_30k_3seeds"
)
MODEL_DIR = TASK11_FORMAL_DIR / "models" / "medium"
MODEL_TEMPLATE = "role_conditioned_ar_ppo_order_012_seed{seed}.zip"
POLICY_SEEDS = (0, 1, 2)
UNIT_ORDER = (0, 1, 2)
HIGH_THREAT_THRESHOLD = 0.8
SCHEMA_VERSION = "1.0.0"
CORE_SCENARIOS = ("time_pressure", "heterogeneity_pressure")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct the frozen DS-0 state-prefix action-pair corpus."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(*parts: str | bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        payload = part if isinstance(part, bytes) else part.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def canonical_json(values: Iterable[int]) -> str:
    return json.dumps([int(value) for value in values], separators=(",", ":"))


def independent_mask_from_observation(observation: np.ndarray) -> np.ndarray:
    """Recompute the official base mask from frozen observation semantics."""

    values = np.asarray(observation, dtype=np.float32).reshape(-1)
    zone_end = 2 * 7
    target_end = zone_end + 5 * 15
    unit_end = target_end + 3 * 15
    if values.size != unit_end + 8:
        raise ValueError("Expected the frozen 142-dimensional AirDefense-v1 state")
    targets = values[zone_end:target_end].reshape(5, 15)
    units = values[target_end:unit_end].reshape(3, 15)
    mask = np.zeros((3, 6), dtype=bool)
    for unit_index in range(3):
        for target_index in range(5):
            normalized_distance = float(
                np.linalg.norm(
                    units[unit_index, 0:2] - targets[target_index, 0:2]
                )
            )
            mask[unit_index, target_index] = bool(
                units[unit_index, 14] > 0.5
                and targets[target_index, 13] > 0.5
                and normalized_distance <= units[unit_index, 6] + 1e-6
            )
        mask[unit_index, 5] = True
    return mask


def observation_target_threats(observation: np.ndarray) -> np.ndarray:
    values = np.asarray(observation, dtype=np.float32).reshape(-1)
    target_start = 2 * 7
    targets = values[target_start : target_start + 5 * 15].reshape(5, 15)
    return targets[:, 6].astype(np.float64)


def conditional_mask(
    base_mask: np.ndarray,
    prefix: tuple[int, ...],
    unit_order: tuple[int, ...] = UNIT_ORDER,
) -> np.ndarray:
    position = len(prefix)
    if position >= len(unit_order):
        raise ValueError("Prefix has no current decision position")
    used_targets = {action for action in prefix if action != base_mask.shape[1] - 1}
    mask = np.asarray(base_mask[unit_order[position]], dtype=bool).copy()
    if used_targets:
        mask[list(used_targets)] = False
    return mask


def deterministic_completion(
    logits: np.ndarray,
    base_mask: np.ndarray,
    fixed_ordered_actions: tuple[int, ...],
    unit_order: tuple[int, ...] = UNIT_ORDER,
) -> np.ndarray:
    """Complete a legal fixed prefix with the policy's deterministic argmax."""

    scores = np.asarray(logits, dtype=np.float64)
    if scores.shape != base_mask.shape:
        raise ValueError("Logits and base mask must have the same shape")
    if len(fixed_ordered_actions) > len(unit_order):
        raise ValueError("Fixed prefix is longer than the unit order")
    noop_action = base_mask.shape[1] - 1
    used_targets: set[int] = set()
    joint = np.full(len(unit_order), noop_action, dtype=np.int64)
    for position, unit_index in enumerate(unit_order):
        mask = np.asarray(base_mask[unit_index], dtype=bool).copy()
        if used_targets:
            mask[list(used_targets)] = False
        if not bool(np.any(mask)):
            raise ValueError("Conditional mask is empty")
        if position < len(fixed_ordered_actions):
            action = int(fixed_ordered_actions[position])
            if not 0 <= action < mask.size or not mask[action]:
                raise ValueError("Fixed action is illegal under its prefix mask")
        else:
            action = int(np.argmax(np.where(mask, scores[unit_index], -np.inf)))
        joint[unit_index] = action
        if action != noop_action:
            used_targets.add(action)
    return joint


def softmax_on_mask(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    scores = np.where(mask, np.asarray(logits, dtype=np.float64), -np.inf)
    finite = np.isfinite(scores)
    if not bool(np.any(finite)):
        raise ValueError("Cannot normalize an empty action mask")
    shifted = scores[finite] - np.max(scores[finite])
    probabilities = np.zeros(scores.shape, dtype=np.float64)
    probabilities[finite] = np.exp(shifted)
    probabilities /= probabilities.sum()
    return probabilities


def branch_high_threat_unassigned(
    *,
    joint_action: np.ndarray,
    base_mask: np.ndarray,
    threats: np.ndarray,
    current_position: int,
    unit_order: tuple[int, ...] = UNIT_ORDER,
) -> bool:
    noop_action = base_mask.shape[1] - 1
    selected_targets = {
        int(action) for action in joint_action if int(action) != noop_action
    }
    used_targets = {
        int(joint_action[unit_order[position]])
        for position in range(current_position + 1)
        if int(joint_action[unit_order[position]]) != noop_action
    }
    legal_high_targets: set[int] = set()
    for position in range(current_position + 1, len(unit_order)):
        unit_index = unit_order[position]
        mask = np.asarray(base_mask[unit_index], dtype=bool).copy()
        if used_targets:
            mask[list(used_targets)] = False
        legal_high_targets.update(
            target_index
            for target_index in np.flatnonzero(mask[:-1])
            if threats[int(target_index)] >= HIGH_THREAT_THRESHOLD
        )
        action = int(joint_action[unit_index])
        if action != noop_action:
            used_targets.add(action)
    return bool(legal_high_targets - selected_targets)


def branch_prefix_denied(
    *,
    action: int,
    base_mask: np.ndarray,
    current_position: int,
    unit_order: tuple[int, ...] = UNIT_ORDER,
) -> bool:
    noop_action = base_mask.shape[1] - 1
    if action == noop_action:
        return False
    return any(
        bool(base_mask[unit_order[position], action])
        for position in range(current_position + 1, len(unit_order))
    )


def engagement_extreme_direction(
    count_a: int,
    count_b: int,
    max_feasible_count: int,
) -> int:
    boundaries = {0, int(max_feasible_count)}
    if count_a == count_b:
        return 0
    if count_a not in boundaries and count_b not in boundaries:
        return 0
    return 1 if count_b > count_a else -1


def raw_csv_row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def load_policy_batches(
    corpus: PolicyProbeCorpus,
) -> dict[int, dict[str, Any]]:
    observations = torch.as_tensor(corpus.observations, dtype=torch.float32)
    masks = torch.as_tensor(corpus.action_masks, dtype=torch.bool)
    batches: dict[int, dict[str, Any]] = {}
    for seed in POLICY_SEEDS:
        model_path = MODEL_DIR / MODEL_TEMPLATE.format(seed=seed)
        model = RoleConditionedAutoregressiveMaskablePPO.load(
            model_path,
            device="cpu",
        )
        model.policy.set_training_mode(False)
        with torch.no_grad():
            distribution = model.policy.get_distribution(
                observations,
                action_masks=masks,
            )
            factual = distribution.sample(deterministic=True).actions
            probabilities, conditional_masks = (
                distribution.conditional_probabilities(factual)
            )
        batches[seed] = {
            "model_path": model_path,
            "model_sha256": sha256_file(model_path),
            "logits": distribution.logits.detach().cpu().numpy(),
            "factual_actions": factual.detach().cpu().numpy().astype(np.int64),
            "probabilities": probabilities.detach().cpu().numpy(),
            "conditional_masks": conditional_masks.detach().cpu().numpy(),
        }
    return batches


def build_action_pair_rows(
    corpus: PolicyProbeCorpus,
    policy_batches: dict[int, dict[str, Any]],
    environment_config_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    counters = {
        "environment_mask_errors": 0,
        "factual_argmax_replay_errors": 0,
        "conditional_mask_errors": 0,
        "swap_symmetry_errors": 0,
        "contexts_attempted": 0,
        "contexts_with_pairs": 0,
        "contexts_without_pairs": 0,
    }
    config_sha256 = sha256_file(environment_config_path)
    source_run = project_relative(REPLAY_DIR)
    state_payload_path = PROBE_DIR / "probe_states.npz"

    for state_index in range(corpus.size):
        observation = np.asarray(corpus.observations[state_index], dtype=np.float32)
        base_mask = np.asarray(
            corpus.action_masks[state_index], dtype=bool
        ).reshape(3, 6)
        reconstructed_mask = independent_mask_from_observation(observation)
        mask_match = bool(np.array_equal(base_mask, reconstructed_mask))
        counters["environment_mask_errors"] += int(not mask_match)
        scenario = str(corpus.scenarios[state_index])
        state_hash = stable_hash(
            scenario,
            np.ascontiguousarray(observation).tobytes(),
            np.ascontiguousarray(base_mask).tobytes(),
        )
        threats = observation_target_threats(observation)

        for policy_seed, batch in policy_batches.items():
            logits = np.asarray(batch["logits"][state_index], dtype=np.float64)
            factual_joint = np.asarray(
                batch["factual_actions"][state_index],
                dtype=np.int64,
            )
            manual_factual = deterministic_completion(logits, base_mask, ())
            factual_match = bool(np.array_equal(factual_joint, manual_factual))
            counters["factual_argmax_replay_errors"] += int(not factual_match)
            ordered_factual = tuple(
                int(factual_joint[unit_index]) for unit_index in UNIT_ORDER
            )

            for unit_position in range(len(UNIT_ORDER) - 1):
                counters["contexts_attempted"] += 1
                prefix = ordered_factual[:unit_position]
                prefix_json = canonical_json(prefix)
                prefix_hash = stable_hash(prefix_json)
                unit_index = UNIT_ORDER[unit_position]
                expected_current_mask = conditional_mask(
                    base_mask,
                    prefix,
                    UNIT_ORDER,
                )
                stored_conditional_mask = np.asarray(
                    batch["conditional_masks"][state_index, unit_index],
                    dtype=bool,
                )
                conditional_match = bool(
                    np.array_equal(expected_current_mask, stored_conditional_mask)
                )
                counters["conditional_mask_errors"] += int(not conditional_match)
                matrix = dynamic_support_cost_matrix(
                    base_mask,
                    prefix,
                    UNIT_ORDER,
                )
                legal_actions = matrix.actions
                if len(legal_actions) < 2:
                    counters["contexts_without_pairs"] += 1
                    exclusions.append(
                        {
                            "ledger_scope": "context",
                            "source_path": project_relative(state_payload_path),
                            "scenario": scenario,
                            "policy_seed": policy_seed,
                            "unit_position": unit_position,
                            "excluded_count": 1,
                            "reason_code": "LEGAL_ACTION_COUNT_LT_2",
                            "details": "No unordered current-action pair exists.",
                        }
                    )
                    continue
                counters["contexts_with_pairs"] += 1

                current_probabilities = np.asarray(
                    batch["probabilities"][state_index, unit_index],
                    dtype=np.float64,
                )
                legal_probabilities = current_probabilities[list(legal_actions)]
                legal_probabilities /= legal_probabilities.sum()
                structural_risk = old_policy_structural_risk(
                    matrix.costs,
                    legal_probabilities,
                )
                supports = {
                    action: frozenset(
                        enumerate_feasible_suffixes(
                            base_mask,
                            prefix + (action,),
                            UNIT_ORDER,
                        )
                    )
                    for action in legal_actions
                }
                max_feasible_engagement = max(
                    sum(value != 5 for value in prefix + (action,) + suffix)
                    for action in legal_actions
                    for suffix in supports[action]
                )
                branches: dict[int, dict[str, Any]] = {}
                for action in legal_actions:
                    completed = deterministic_completion(
                        logits,
                        base_mask,
                        prefix + (action,),
                        UNIT_ORDER,
                    )
                    downstream = tuple(
                        int(completed[UNIT_ORDER[position]])
                        for position in range(unit_position + 1, len(UNIT_ORDER))
                    )
                    branches[action] = {
                        "joint": completed,
                        "downstream": downstream,
                        "engagement_count": int(np.count_nonzero(completed != 5)),
                        "high_threat_unassigned": branch_high_threat_unassigned(
                            joint_action=completed,
                            base_mask=base_mask,
                            threats=threats,
                            current_position=unit_position,
                            unit_order=UNIT_ORDER,
                        ),
                        "prefix_denied": branch_prefix_denied(
                            action=action,
                            base_mask=base_mask,
                            current_position=unit_position,
                            unit_order=UNIT_ORDER,
                        ),
                    }

                context_id = stable_hash(
                    source_run,
                    state_hash,
                    str(policy_seed),
                    str(batch["model_sha256"]),
                    "012",
                    str(unit_position),
                    prefix_hash,
                )
                legal_actions_json = canonical_json(legal_actions)
                for action_a_index, action_a in enumerate(legal_actions):
                    for action_b_index in range(
                        action_a_index + 1,
                        len(legal_actions),
                    ):
                        action_b = legal_actions[action_b_index]
                        suffixes_a = supports[action_a]
                        suffixes_b = supports[action_b]
                        intersection_count = len(suffixes_a & suffixes_b)
                        union_count = len(suffixes_a | suffixes_b)
                        if union_count <= 0:
                            raise RuntimeError("Encountered an empty suffix union")
                        counters["swap_symmetry_errors"] += int(
                            abs(
                                matrix.costs[action_a_index, action_b_index]
                                - matrix.costs[action_b_index, action_a_index]
                            )
                            > 1e-12
                        )
                        branch_a = branches[action_a]
                        branch_b = branches[action_b]
                        threat_a = (
                            None
                            if action_a == 5
                            else float(threats[action_a])
                        )
                        threat_b = (
                            None
                            if action_b == 5
                            else float(threats[action_b])
                        )
                        observed_threats = [
                            value for value in (threat_a, threat_b) if value is not None
                        ]
                        extreme_direction = engagement_extreme_direction(
                            branch_a["engagement_count"],
                            branch_b["engagement_count"],
                            max_feasible_engagement,
                        )
                        row = {
                            "schema_version": SCHEMA_VERSION,
                            "context_id": context_id,
                            "action_pair_id": stable_hash(
                                context_id,
                                str(action_a),
                                str(action_b),
                            ),
                            "source_priority": 2,
                            "source_kind": "replay",
                            "source_run": source_run,
                            "probe_source": str(corpus.sources[state_index]),
                            "phase": str(corpus.phases[state_index]),
                            "scenario": scenario,
                            "policy_seed": policy_seed,
                            "env_seed": int(corpus.environment_seeds[state_index]),
                            "episode_id": int(corpus.episode_indices[state_index]),
                            "step": int(corpus.step_indices[state_index]),
                            "state_hash": state_hash,
                            "state_ref": (
                                f"{project_relative(state_payload_path)}"
                                f"#row={state_index}"
                            ),
                            "model_path": project_relative(batch["model_path"]),
                            "model_sha256": batch["model_sha256"],
                            "environment_config_path": project_relative(
                                environment_config_path
                            ),
                            "environment_config_sha256": config_sha256,
                            "unit_order": "012",
                            "unit_position": unit_position,
                            "unit_id": unit_index,
                            "prefix": prefix_json,
                            "prefix_hash": prefix_hash,
                            "prefix_engagement_count": sum(
                                action != 5 for action in prefix
                            ),
                            "legal_action_ids": legal_actions_json,
                            "legal_action_count": len(legal_actions),
                            "environment_mask_match": mask_match,
                            "conditional_mask_match": conditional_match,
                            "factual_argmax_replay_match": factual_match,
                            "eligible_main": True,
                            "exclusion_reason": None,
                            "action_a": action_a,
                            "action_b": action_b,
                            "is_noop_a": action_a == 5,
                            "is_noop_b": action_b == 5,
                            "noop_pair_type": (
                                "noop_engage"
                                if action_a == 5 or action_b == 5
                                else "engage_engage"
                            ),
                            "target_id_a": None if action_a == 5 else action_a,
                            "target_id_b": None if action_b == 5 else action_b,
                            "candidate_target_threat_a": threat_a,
                            "candidate_target_threat_b": threat_b,
                            "candidate_target_threat_min": (
                                min(observed_threats) if observed_threats else None
                            ),
                            "candidate_target_threat_max": (
                                max(observed_threats) if observed_threats else None
                            ),
                            "candidate_target_threat_abs_diff": (
                                abs(threat_a - threat_b)
                                if threat_a is not None and threat_b is not None
                                else None
                            ),
                            "candidate_target_threat_missing_a": threat_a is None,
                            "candidate_target_threat_missing_b": threat_b is None,
                            "suffix_count_a": len(suffixes_a),
                            "suffix_count_b": len(suffixes_b),
                            "intersection_count": intersection_count,
                            "union_count": union_count,
                            "completion_count_ratio": min(
                                len(suffixes_a),
                                len(suffixes_b),
                            )
                            / max(len(suffixes_a), len(suffixes_b)),
                            "ds_jaccard": float(
                                matrix.costs[action_a_index, action_b_index]
                            ),
                            "old_prob_a": float(
                                legal_probabilities[action_a_index]
                            ),
                            "old_prob_b": float(
                                legal_probabilities[action_b_index]
                            ),
                            "r_old_a": float(structural_risk[action_a_index]),
                            "r_old_b": float(structural_risk[action_b_index]),
                            "downstream_argmax_a": canonical_json(
                                branch_a["downstream"]
                            ),
                            "downstream_argmax_b": canonical_json(
                                branch_b["downstream"]
                            ),
                            "downstream_argmax_changed": (
                                branch_a["downstream"] != branch_b["downstream"]
                            ),
                            "high_threat_legal_but_unassigned_a": branch_a[
                                "high_threat_unassigned"
                            ],
                            "high_threat_legal_but_unassigned_b": branch_b[
                                "high_threat_unassigned"
                            ],
                            "high_threat_legal_but_unassigned_changed": (
                                branch_a["high_threat_unassigned"]
                                != branch_b["high_threat_unassigned"]
                            ),
                            "prefix_denied_a": branch_a["prefix_denied"],
                            "prefix_denied_b": branch_b["prefix_denied"],
                            "prefix_denied_changed": (
                                branch_a["prefix_denied"]
                                != branch_b["prefix_denied"]
                            ),
                            "engagement_count_a": branch_a["engagement_count"],
                            "engagement_count_b": branch_b["engagement_count"],
                            "max_feasible_engagement_count": (
                                max_feasible_engagement
                            ),
                            "engagement_extreme_direction": extreme_direction,
                            "engagement_extreme_direction_nonzero": (
                                extreme_direction != 0
                            ),
                        }
                        rows.append(row)

            exclusions.append(
                {
                    "ledger_scope": "context",
                    "source_path": project_relative(state_payload_path),
                    "scenario": scenario,
                    "policy_seed": policy_seed,
                    "unit_position": 2,
                    "excluded_count": 1,
                    "reason_code": "LAST_POSITION",
                    "details": "DST-01 defines the final position as not_applicable.",
                }
            )
    return rows, exclusions, counters


def add_file_level_exclusions(exclusions: list[dict[str, Any]]) -> None:
    sources = (
        (
            TASK10_FORMAL_DIR / "decisions.csv",
            "STATE_UNRECOVERABLE",
            "The frozen 169887-row order diagnostic has decisions but no exact "
            "observation/state snapshot or base mask.",
        ),
        (
            TASK10_FROZEN_DIR / "decisions.csv",
            "STATE_UNRECOVERABLE",
            "Decision rows do not contain an exact observation/state snapshot.",
        ),
        (
            TASK11_FORMAL_DIR / "decisions.csv",
            "STATE_UNRECOVERABLE",
            "Decision rows do not contain an exact observation/state snapshot.",
        ),
        (
            REPLAY_DIR / "episodes.csv",
            "AGGREGATE_ONLY",
            "Episode aggregates cannot define a state-prefix DS context.",
        ),
        (
            REPLAY_DIR / "probe_diagnostics.csv",
            "AGGREGATE_ONLY",
            "Probe aggregates omit per-state prefixes and action pairs.",
        ),
        (
            REPLAY_DIR / "runs.csv",
            "AGGREGATE_ONLY",
            "Run aggregates cannot define a state-prefix DS context.",
        ),
    )
    for path, reason, details in sources:
        exclusions.append(
            {
                "ledger_scope": "source_file",
                "source_path": project_relative(path),
                "scenario": "",
                "policy_seed": "",
                "unit_position": "",
                "excluded_count": raw_csv_row_count(path),
                "reason_code": reason,
                "details": details,
            }
        )


def context_summary(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (scenario, policy_seed, unit_position), group in frame.groupby(
        ["scenario", "policy_seed", "unit_position"],
        sort=True,
    ):
        contexts = group.drop_duplicates("context_id")
        records.append(
            {
                "scenario": scenario,
                "policy_seed": int(policy_seed),
                "unit_position": int(unit_position),
                "unique_states": int(group["state_hash"].nunique()),
                "unique_contexts": int(group["context_id"].nunique()),
                "action_pair_rows": int(len(group)),
                "legal_action_count_min": int(
                    contexts["legal_action_count"].min()
                ),
                "legal_action_count_mean": float(
                    contexts["legal_action_count"].mean()
                ),
                "legal_action_count_max": int(
                    contexts["legal_action_count"].max()
                ),
                "ds_min": float(group["ds_jaccard"].min()),
                "ds_mean": float(group["ds_jaccard"].mean()),
                "ds_iqr": float(
                    group["ds_jaccard"].quantile(0.75)
                    - group["ds_jaccard"].quantile(0.25)
                ),
                "ds_max": float(group["ds_jaccard"].max()),
                "noop_engage_pair_rows": int(
                    (group["noop_pair_type"] == "noop_engage").sum()
                ),
                "engage_engage_pair_rows": int(
                    (group["noop_pair_type"] == "engage_engage").sum()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def audit_integrity(
    frame: pd.DataFrame,
    corpus: PolicyProbeCorpus,
    counters: dict[str, int],
) -> dict[str, Any]:
    pair_coverage_errors = 0
    configuration_uniqueness_errors = 0
    model_uniqueness_errors = 0
    for _, group in frame.groupby("context_id", sort=False):
        legal_action_count = int(group["legal_action_count"].iloc[0])
        if len(group) != comb(legal_action_count, 2):
            pair_coverage_errors += 1
        if group["environment_config_sha256"].nunique() != 1:
            configuration_uniqueness_errors += 1
        if group["model_sha256"].nunique() != 1:
            model_uniqueness_errors += 1

    reverse_symmetry_errors = int(
        (
            np.abs(
                frame["ds_jaccard"].to_numpy()
                - (
                    1.0
                    - frame["intersection_count"].to_numpy()
                    / frame["union_count"].to_numpy()
                )
            )
            > 1e-12
        ).sum()
    )
    duplicate_pair_ids = int(frame["action_pair_id"].duplicated().sum())
    duplicate_context_pairs = int(
        frame.duplicated(["context_id", "action_a", "action_b"]).sum()
    )
    traceable_rows = frame[
        frame["state_hash"].notna()
        & frame["model_sha256"].notna()
        & frame["state_ref"].notna()
    ]
    core_contexts = {
        scenario: int(
            frame.loc[frame["scenario"] == scenario, "context_id"].nunique()
        )
        for scenario in CORE_SCENARIOS
    }
    policy_seeds = sorted(int(value) for value in frame["policy_seed"].unique())
    source_state_hashes = {
        stable_hash(
            str(corpus.scenarios[index]),
            np.ascontiguousarray(
                corpus.observations[index], dtype=np.float32
            ).tobytes(),
            np.ascontiguousarray(
                corpus.action_masks[index].reshape(3, 6), dtype=bool
            ).tobytes(),
        )
        for index in range(corpus.size)
    }
    checks = {
        "both_core_scenarios_present": all(
            core_contexts[scenario] > 0 for scenario in CORE_SCENARIOS
        ),
        "at_least_three_policy_seeds": len(policy_seeds) >= 3,
        "all_rows_traceable": len(traceable_rows) == len(frame),
        "environment_mask_crosscheck_zero": (
            counters["environment_mask_errors"] == 0
        ),
        "conditional_mask_crosscheck_zero": (
            counters["conditional_mask_errors"] == 0
        ),
        "factual_argmax_replay_zero": (
            counters["factual_argmax_replay_errors"] == 0
        ),
        "complete_action_pair_coverage": pair_coverage_errors == 0,
        "jaccard_formula_consistent": reverse_symmetry_errors == 0,
        "swap_symmetry_zero": counters["swap_symmetry_errors"] == 0,
        "context_config_unique": configuration_uniqueness_errors == 0,
        "context_model_unique": model_uniqueness_errors == 0,
        "action_pair_ids_unique": duplicate_pair_ids == 0,
        "context_action_pairs_unique": duplicate_context_pairs == 0,
        "last_position_absent_from_main": bool(
            (frame["unit_position"] < 2).all()
        ),
        "main_rows_are_replay_labeled": set(frame["source_kind"]) == {"replay"},
        "source_states_unique": len(source_state_hashes) == corpus.size,
        "context_accounting_complete": (
            counters["contexts_attempted"]
            == counters["contexts_with_pairs"]
            + counters["contexts_without_pairs"]
        ),
        "no_bpce_or_episode_resource_outcomes": not any(
            token in column.lower()
            for column in frame.columns
            for token in ("bpce", "reward", "resource_responsibility")
        ),
    }
    return {
        "schema_version": 1,
        "task": "DST-03",
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "training_performed": False,
        "environment_legality_rules_modified": False,
        "formal_p1_gate_evaluated": False,
        "source_state_count": corpus.size,
        "source_state_unique_count": len(source_state_hashes),
        "main_unique_state_count": int(frame["state_hash"].nunique()),
        "policy_seeds": policy_seeds,
        "core_scenario_contexts": core_contexts,
        "main_action_pair_rows": int(len(frame)),
        "main_contexts": int(frame["context_id"].nunique()),
        "contexts_attempted": counters["contexts_attempted"],
        "contexts_with_pairs": counters["contexts_with_pairs"],
        "contexts_without_pairs": counters["contexts_without_pairs"],
        "pair_coverage_errors": pair_coverage_errors,
        "jaccard_formula_errors": reverse_symmetry_errors,
        "swap_symmetry_errors": counters["swap_symmetry_errors"],
        "configuration_uniqueness_errors": configuration_uniqueness_errors,
        "model_uniqueness_errors": model_uniqueness_errors,
        "duplicate_action_pair_ids": duplicate_pair_ids,
        "duplicate_context_action_pairs": duplicate_context_pairs,
        "environment_mask_errors": counters["environment_mask_errors"],
        "conditional_mask_errors": counters["conditional_mask_errors"],
        "factual_argmax_replay_errors": counters[
            "factual_argmax_replay_errors"
        ],
        "traceable_row_fraction": float(len(traceable_rows) / len(frame)),
        "checks": checks,
    }


def manifest_entry(path: Path, role: str) -> dict[str, Any]:
    item = path.resolve()
    return {
        "path": project_relative(item),
        "role": role,
        "size_bytes": item.stat().st_size,
        "modified_utc": pd.Timestamp(item.stat().st_mtime, unit="s", tz="UTC").isoformat(),
        "sha256": sha256_file(item),
    }


def source_manifest(
    *,
    output_paths: tuple[Path, ...],
) -> dict[str, Any]:
    inputs: list[tuple[Path, str]] = [
        (
            PROJECT_ROOT
            / "docs"
            / "task_guides"
            / "dynamic_support_trust_region"
            / "dst_03_frozen_corpus_reconstruction.md",
            "task_instruction",
        ),
        (
            PROJECT_ROOT
            / "results"
            / "air_defense_v1"
            / "dynamic_support_trust_region"
            / "dst_01_contract"
            / "research_contract.md",
            "frozen_research_contract",
        ),
        (
            PROJECT_ROOT
            / "results"
            / "air_defense_v1"
            / "dynamic_support_trust_region"
            / "dst_01_contract"
            / "field_dictionary.csv",
            "frozen_field_dictionary",
        ),
        (
            PROJECT_ROOT
            / "results"
            / "air_defense_v1"
            / "dynamic_support_trust_region"
            / "dst_01_contract"
            / "gate_registry.json",
            "frozen_gate_registry",
        ),
        (
            PROJECT_ROOT
            / "rein_learning"
            / "common"
            / "dynamic_support_distance.py",
            "validated_ds_implementation",
        ),
        (Path(__file__), "reconstruction_script"),
        (PROBE_DIR / "probe_manifest.json", "priority_1_manifest"),
        (PROBE_DIR / "probe_states.npz", "priority_1_state_payload"),
        (PROBE_DIR / "probe_summary.csv", "priority_1_summary"),
        (REPLAY_DIR / "experiment_config.json", "priority_2_config"),
        (REPLAY_DIR / "episodes.csv", "audited_aggregate_source"),
        (REPLAY_DIR / "probe_diagnostics.csv", "audited_aggregate_source"),
        (REPLAY_DIR / "runs.csv", "audited_aggregate_source"),
        (TASK10_FROZEN_DIR / "experiment_config.json", "priority_3_config"),
        (TASK10_FROZEN_DIR / "decisions.csv", "audited_state_incomplete_source"),
        (TASK10_FORMAL_DIR / "experiment_config.json", "priority_4_config"),
        (TASK10_FORMAL_DIR / "decisions.csv", "audited_state_incomplete_source"),
        (TASK11_FORMAL_DIR / "experiment_config.json", "priority_4_config"),
        (TASK11_FORMAL_DIR / "decisions.csv", "audited_state_incomplete_source"),
    ]
    inputs.extend(
        (
            MODEL_DIR / MODEL_TEMPLATE.format(seed=seed),
            "priority_4_frozen_policy_model",
        )
        for seed in POLICY_SEEDS
    )
    return {
        "schema_version": 1,
        "task": "DST-03",
        "corpus_semantics": (
            "Frozen Task12 probe states with deterministic Task11 order-012 "
            "policy replay; no environment resampling."
        ),
        "source_kind": "replay",
        "training_performed": False,
        "parquet": {
            "engine": "pyarrow",
            "compression": "zstd",
            "index_written": False,
        },
        "inputs": [manifest_entry(path, role) for path, role in inputs],
        "outputs": [
            manifest_entry(path, "dst_03_output") for path in output_paths
        ],
    }


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
    corpus: PolicyProbeCorpus,
    counters: dict[str, int],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(rows)
    frame.sort_values(
        [
            "scenario",
            "policy_seed",
            "state_hash",
            "unit_position",
            "action_a",
            "action_b",
        ],
        inplace=True,
        ignore_index=True,
    )
    summary = context_summary(frame)
    exclusion_frame = pd.DataFrame.from_records(exclusions)
    report = audit_integrity(frame, corpus, counters)

    parquet_path = output_dir / "ds0_action_pairs.parquet"
    summary_path = output_dir / "context_summary.csv"
    exclusion_path = output_dir / "exclusion_ledger.csv"
    report_path = output_dir / "integrity_report.json"
    manifest_path = output_dir / "source_manifest.json"

    frame.to_parquet(
        parquet_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    exclusion_frame.to_csv(exclusion_path, index=False, encoding="utf-8")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = source_manifest(
        output_paths=(
            parquet_path,
            summary_path,
            exclusion_path,
            report_path,
        )
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    corpus = PolicyProbeCorpus.load(PROBE_DIR, verify_hash=True)
    policy_batches = load_policy_batches(corpus)
    rows, exclusions, counters = build_action_pair_rows(
        corpus,
        policy_batches,
        TASK11_FORMAL_DIR / "experiment_config.json",
    )
    add_file_level_exclusions(exclusions)
    report = write_outputs(
        args.output_dir,
        rows,
        exclusions,
        corpus,
        counters,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
