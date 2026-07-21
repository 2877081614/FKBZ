from __future__ import annotations

from math import ceil
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..envs import AirDefenseResourceAssignmentEnvV1


def engagement_criticality_features(
    env: AirDefenseResourceAssignmentEnvV1,
    conditional_masks: np.ndarray,
) -> dict[str, float | int]:
    masks = np.asarray(conditional_masks, dtype=bool)
    expected_shape = (env.num_defense_units, env.num_targets + 1)
    if masks.shape != expected_shape:
        raise ValueError(f"conditional_masks must have shape {expected_shape}")

    scores: list[float] = []
    damage_potentials: list[float] = []
    times_to_impact: list[float] = []
    threats: list[float] = []
    for unit_index in range(env.num_defense_units):
        for target_index in np.flatnonzero(masks[unit_index, : env.num_targets]):
            target = env.targets[int(target_index)]
            damage_potential = env.target_damage_potential(int(target_index))
            hit_probability = env.hit_probability(unit_index, int(target_index))
            urgency = 1.0 + 5.0 / (1.0 + max(0.0, target.time_to_impact))
            score = (
                damage_potential
                * hit_probability
                * (1.0 + target.threat)
                * urgency
            )
            scores.append(float(score))
            damage_potentials.append(float(damage_potential))
            times_to_impact.append(float(target.time_to_impact))
            threats.append(float(target.threat))
    if not scores:
        return {
            "criticality_score": 0.0,
            "max_damage_potential": 0.0,
            "min_time_to_impact": float("inf"),
            "max_threat": 0.0,
            "legal_relation_count": 0,
        }
    return {
        "criticality_score": max(scores),
        "max_damage_potential": max(damage_potentials),
        "min_time_to_impact": min(times_to_impact),
        "max_threat": max(threats),
        "legal_relation_count": len(scores),
    }


def select_diverse_critical_snapshots(
    records: Sequence[Mapping[str, Any]],
    count: int,
    *,
    seed: int,
    high_fraction: float = 0.8,
    min_episode_step_gap: int = 3,
) -> list[Mapping[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    if len(records) < count:
        raise ValueError("Not enough candidate records")
    if not 0.0 < high_fraction <= 1.0:
        raise ValueError("high_fraction must be in (0, 1]")
    if min_episode_step_gap < 0:
        raise ValueError("min_episode_step_gap must be non-negative")
    required = {"state_id", "criticality_score", "episode_index", "step_index"}
    if any(not required.issubset(record) for record in records):
        raise ValueError("Candidate records are missing criticality metadata")
    if len({str(record["state_id"]) for record in records}) != len(records):
        raise ValueError("Candidate state_id values must be unique")

    ranked = sorted(
        records,
        key=lambda record: (
            -float(record["criticality_score"]),
            int(record["episode_index"]),
            int(record["step_index"]),
        ),
    )
    high_count = min(count, int(ceil(count * high_fraction)))
    selected: list[Mapping[str, Any]] = []
    selected_ids: set[str] = set()
    episode_steps: dict[int, list[int]] = {}
    for record in ranked:
        episode = int(record["episode_index"])
        step = int(record["step_index"])
        if any(
            abs(step - previous) < min_episode_step_gap
            for previous in episode_steps.get(episode, [])
        ):
            continue
        selected.append(record)
        selected_ids.add(str(record["state_id"]))
        episode_steps.setdefault(episode, []).append(step)
        if len(selected) >= high_count:
            break

    if len(selected) < high_count:
        for record in ranked:
            state_id = str(record["state_id"])
            if state_id in selected_ids:
                continue
            selected.append(record)
            selected_ids.add(state_id)
            if len(selected) >= high_count:
                break

    remaining = [
        record for record in records if str(record["state_id"]) not in selected_ids
    ]
    diversity_count = count - len(selected)
    if diversity_count > 0:
        rng = np.random.default_rng(seed)
        order = np.lexsort(
            (
                rng.random(len(remaining)),
                np.asarray([float(record["criticality_score"]) for record in remaining]),
                np.asarray([int(record["step_index"]) for record in remaining]),
            )
        )
        positions = np.linspace(0, len(order) - 1, diversity_count, dtype=int)
        for position in positions:
            selected.append(remaining[int(order[position])])
    return selected[:count]
