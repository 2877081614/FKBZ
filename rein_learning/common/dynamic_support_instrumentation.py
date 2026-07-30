from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from .dynamic_support_distance import (
    dynamic_support_cost_matrix,
    dynamic_support_policy_distance,
    enumerate_feasible_suffixes,
    old_policy_structural_risk,
)
from .policy_probe import PolicyProbeCorpus


DEFAULT_CORE_SCENARIOS = ("time_pressure", "heterogeneity_pressure")
DEFAULT_HIGH_THREAT_THRESHOLD = 0.8


@dataclass(frozen=True)
class FrozenDynamicSupportProbe:
    """Policy-independent state-prefix grid derived from a frozen probe corpus."""

    corpus: PolicyProbeCorpus
    unit_order: tuple[int, ...]
    context_ids: np.ndarray
    state_ids: np.ndarray
    state_indices: np.ndarray
    scenarios: np.ndarray
    unit_positions: np.ndarray
    prefixes: np.ndarray
    anchor_actions: np.ndarray
    legal_action_counts: np.ndarray
    high_threat_reachable: np.ndarray
    eligible_ds: np.ndarray

    def __post_init__(self) -> None:
        size = len(self.context_ids)
        arrays = (
            self.state_ids,
            self.state_indices,
            self.scenarios,
            self.unit_positions,
            self.prefixes,
            self.anchor_actions,
            self.legal_action_counts,
            self.high_threat_reachable,
            self.eligible_ds,
        )
        if size <= 0 or any(len(array) != size for array in arrays):
            raise ValueError("Frozen probe context arrays must have equal nonzero size")
        if len(set(self.context_ids.astype(str))) != size:
            raise ValueError("Frozen probe context ids must be unique")
        num_units = len(self.unit_order)
        if self.prefixes.shape != (size, num_units):
            raise ValueError("Prefixes must be padded to the unit count")
        if self.anchor_actions.shape != (size, num_units):
            raise ValueError("Anchor actions must contain one action per unit")
        if set(self.unit_order) != set(range(num_units)):
            raise ValueError("unit_order must be a permutation of all unit indices")

    @property
    def size(self) -> int:
        return int(len(self.context_ids))

    @property
    def num_units(self) -> int:
        return len(self.unit_order)

    @property
    def num_actions(self) -> int:
        return int(self.corpus.action_masks.shape[1] // self.num_units)

    @property
    def noop_action(self) -> int:
        return self.num_actions - 1

    @property
    def unique_state_indices(self) -> np.ndarray:
        return np.unique(self.state_indices)

    def content_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.corpus.content_sha256().encode("ascii"))
        digest.update(json.dumps(self.unit_order).encode("ascii"))
        for name, array in (
            ("context_ids", self.context_ids),
            ("state_ids", self.state_ids),
            ("state_indices", self.state_indices),
            ("scenarios", self.scenarios),
            ("unit_positions", self.unit_positions),
            ("prefixes", self.prefixes),
            ("anchor_actions", self.anchor_actions),
            ("legal_action_counts", self.legal_action_counts),
            ("high_threat_reachable", self.high_threat_reachable),
            ("eligible_ds", self.eligible_ds),
        ):
            values = np.ascontiguousarray(array)
            digest.update(name.encode("ascii"))
            digest.update(str(values.dtype).encode("ascii"))
            digest.update(json.dumps(values.shape).encode("ascii"))
            if values.dtype.kind in {"U", "O"}:
                digest.update(
                    json.dumps(
                        values.astype(str).tolist(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
            else:
                digest.update(values.tobytes(order="C"))
        return digest.hexdigest()


@dataclass(frozen=True)
class PolicyProbeSnapshot:
    """Deterministic policy values on one frozen state-prefix grid."""

    grid_sha256: str
    context_ids: np.ndarray
    context_probabilities: np.ndarray
    context_actions: np.ndarray
    context_engage_probabilities: np.ndarray
    context_entropies: np.ndarray
    state_ids: np.ndarray
    joint_actions: np.ndarray

    def __post_init__(self) -> None:
        contexts = len(self.context_ids)
        states = len(self.state_ids)
        if self.context_probabilities.ndim != 2:
            raise ValueError("Context probabilities must be two-dimensional")
        if self.context_probabilities.shape[0] != contexts:
            raise ValueError("Context probabilities do not match context ids")
        for array in (
            self.context_actions,
            self.context_engage_probabilities,
            self.context_entropies,
        ):
            if len(array) != contexts:
                raise ValueError("Context snapshot arrays do not align")
        if self.joint_actions.ndim != 2 or len(self.joint_actions) != states:
            raise ValueError("Joint actions do not match state ids")
        if len(set(self.context_ids.astype(str))) != contexts:
            raise ValueError("Snapshot context ids must be unique")


def build_frozen_dynamic_support_probe(
    corpus: PolicyProbeCorpus,
    *,
    unit_order: Sequence[int] = (0, 1, 2),
    scenarios: Iterable[str] = DEFAULT_CORE_SCENARIOS,
    high_threat_threshold: float = DEFAULT_HIGH_THREAT_THRESHOLD,
) -> FrozenDynamicSupportProbe:
    """Enumerate every feasible prefix without selecting on policy outcomes."""

    order = tuple(int(value) for value in unit_order)
    num_units = len(order)
    if set(order) != set(range(num_units)):
        raise ValueError("unit_order must be a permutation")
    if corpus.action_masks.shape[1] % num_units:
        raise ValueError("Probe masks cannot be divided into unit action blocks")
    num_actions = corpus.action_masks.shape[1] // num_units
    if num_actions < 2:
        raise ValueError("Every unit must expose at least one target and no-op")
    selected_scenarios = frozenset(str(value) for value in scenarios)
    if not selected_scenarios:
        raise ValueError("At least one scenario is required")
    if not 0.0 <= high_threat_threshold <= 1.0:
        raise ValueError("high_threat_threshold must lie in [0, 1]")

    records: list[dict[str, Any]] = []
    for state_index in range(corpus.size):
        scenario = str(corpus.scenarios[state_index])
        if scenario not in selected_scenarios:
            continue
        mask = np.asarray(corpus.action_masks[state_index], dtype=bool).reshape(
            num_units,
            num_actions,
        )
        ordered_joint_actions = enumerate_feasible_suffixes(mask, (), order)
        if not ordered_joint_actions:
            raise ValueError("Every frozen state must have a feasible joint action")
        state_id = _probe_state_id(corpus, state_index)
        threats = _target_threats(
            corpus.observations[state_index],
            num_units=num_units,
            num_targets=num_actions - 1,
        )
        for position in range(num_units):
            prefixes = sorted(
                {joint[:position] for joint in ordered_joint_actions}
            )
            for prefix in prefixes:
                completions = [
                    joint
                    for joint in ordered_joint_actions
                    if joint[:position] == prefix
                ]
                anchor_ordered = min(completions)
                anchor_by_unit = np.empty(num_units, dtype=np.int16)
                for order_position, unit_index in enumerate(order):
                    anchor_by_unit[unit_index] = anchor_ordered[order_position]
                current_unit = order[position]
                used_targets = {
                    int(action)
                    for action in prefix
                    if int(action) != num_actions - 1
                }
                conditional_mask = mask[current_unit].copy()
                for target in used_targets:
                    conditional_mask[target] = False
                prefix_padded = np.full(num_units, -1, dtype=np.int16)
                if prefix:
                    prefix_padded[:position] = np.asarray(prefix, dtype=np.int16)
                high_threat = np.flatnonzero(
                    threats >= high_threat_threshold
                )
                reachable = bool(
                    any(conditional_mask[int(target)] for target in high_threat)
                )
                context_payload = (
                    f"{state_id}|{','.join(map(str, order))}|{position}|"
                    f"{','.join(map(str, prefix))}"
                )
                context_id = hashlib.sha256(
                    context_payload.encode("utf-8")
                ).hexdigest()
                records.append(
                    {
                        "context_id": context_id,
                        "state_id": state_id,
                        "state_index": state_index,
                        "scenario": scenario,
                        "unit_position": position,
                        "prefix": prefix_padded,
                        "anchor_actions": anchor_by_unit,
                        "legal_action_count": int(conditional_mask.sum()),
                        "high_threat_reachable": reachable,
                        "eligible_ds": position < num_units - 1,
                    }
                )
    if not records:
        raise ValueError("No frozen probe states match the requested scenarios")
    return FrozenDynamicSupportProbe(
        corpus=corpus,
        unit_order=order,
        context_ids=np.asarray(
            [record["context_id"] for record in records], dtype=np.str_
        ),
        state_ids=np.asarray(
            [record["state_id"] for record in records], dtype=np.str_
        ),
        state_indices=np.asarray(
            [record["state_index"] for record in records], dtype=np.int64
        ),
        scenarios=np.asarray(
            [record["scenario"] for record in records], dtype=np.str_
        ),
        unit_positions=np.asarray(
            [record["unit_position"] for record in records], dtype=np.int8
        ),
        prefixes=np.stack([record["prefix"] for record in records]),
        anchor_actions=np.stack(
            [record["anchor_actions"] for record in records]
        ),
        legal_action_counts=np.asarray(
            [record["legal_action_count"] for record in records], dtype=np.int16
        ),
        high_threat_reachable=np.asarray(
            [record["high_threat_reachable"] for record in records],
            dtype=np.bool_,
        ),
        eligible_ds=np.asarray(
            [record["eligible_ds"] for record in records], dtype=np.bool_
        ),
    )


def evaluate_policy_on_frozen_probe(
    policy_or_model: Any,
    grid: FrozenDynamicSupportProbe,
    *,
    batch_size: int = 512,
) -> PolicyProbeSnapshot:
    """Evaluate a policy without environment calls or persistent RNG changes."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    policy = getattr(policy_or_model, "policy", policy_or_model)
    was_training = bool(getattr(policy, "training", False))
    context_probabilities: list[np.ndarray] = []
    context_actions: list[np.ndarray] = []
    context_engage: list[np.ndarray] = []
    context_entropy: list[np.ndarray] = []
    state_indices = grid.unique_state_indices
    state_ids = np.asarray(
        [_probe_state_id(grid.corpus, int(index)) for index in state_indices],
        dtype=np.str_,
    )
    joint_actions: list[np.ndarray] = []

    rng = _capture_rng_state()
    try:
        policy.eval()
        with torch.no_grad():
            for start in range(0, grid.size, batch_size):
                stop = min(start + batch_size, grid.size)
                indices = grid.state_indices[start:stop]
                observations, _ = policy.obs_to_tensor(
                    grid.corpus.observations[indices]
                )
                masks = grid.corpus.action_masks[indices]
                distribution = policy.get_distribution(observations, masks)
                anchors = torch.as_tensor(
                    grid.anchor_actions[start:stop],
                    device=observations.device,
                    dtype=torch.long,
                )
                probabilities, _ = distribution.conditional_probabilities(
                    anchors
                )
                positions = grid.unit_positions[start:stop]
                units = np.asarray(
                    [grid.unit_order[int(position)] for position in positions],
                    dtype=np.int64,
                )
                row_indices = torch.arange(
                    stop - start, device=observations.device
                )
                unit_indices = torch.as_tensor(
                    units, device=observations.device, dtype=torch.long
                )
                current = probabilities[row_indices, unit_indices, :]
                engage = current[:, : grid.noop_action].sum(dim=1)
                if hasattr(distribution, "engage_logits"):
                    target = current[:, : grid.noop_action].argmax(dim=1)
                    action = torch.where(
                        engage >= 0.5,
                        target,
                        torch.full_like(target, grid.noop_action),
                    )
                else:
                    action = current.argmax(dim=1)
                entropy = -torch.where(
                    current > 0,
                    current * torch.log(current.clamp_min(1e-20)),
                    torch.zeros_like(current),
                ).sum(dim=1)
                context_probabilities.append(
                    current.detach().cpu().numpy().astype(np.float64)
                )
                context_actions.append(
                    action.detach().cpu().numpy().astype(np.int16)
                )
                context_engage.append(
                    engage.detach().cpu().numpy().astype(np.float64)
                )
                context_entropy.append(
                    entropy.detach().cpu().numpy().astype(np.float64)
                )
            for start in range(0, len(state_indices), batch_size):
                indices = state_indices[start : start + batch_size]
                observations, _ = policy.obs_to_tensor(
                    grid.corpus.observations[indices]
                )
                masks = grid.corpus.action_masks[indices]
                distribution = policy.get_distribution(observations, masks)
                actions = distribution.get_actions(deterministic=True)
                joint_actions.append(
                    actions.detach().cpu().numpy().astype(np.int16)
                )
    finally:
        if was_training:
            policy.train()
        _restore_rng_state(rng)
    return PolicyProbeSnapshot(
        grid_sha256=grid.content_sha256(),
        context_ids=grid.context_ids.copy(),
        context_probabilities=np.concatenate(
            context_probabilities, axis=0
        ),
        context_actions=np.concatenate(context_actions),
        context_engage_probabilities=np.concatenate(context_engage),
        context_entropies=np.concatenate(context_entropy),
        state_ids=state_ids,
        joint_actions=np.concatenate(joint_actions, axis=0),
    )


def compute_dynamic_support_update_metrics(
    grid: FrozenDynamicSupportProbe,
    old: PolicyProbeSnapshot,
    new: PolicyProbeSnapshot,
    *,
    update_id: int,
    approx_kl: float,
    clip_fraction: float,
    entropy: float,
    high_threat_threshold: float = DEFAULT_HIGH_THREAT_THRESHOLD,
) -> dict[str, float | int]:
    """Compute the frozen DST-05 update row from two policy snapshots."""

    _validate_snapshot_pair(grid, old, new)
    if update_id < 0:
        raise ValueError("update_id must be nonnegative")
    noop = grid.noop_action
    old_actions = old.context_actions
    new_actions = new.context_actions
    flip = old_actions != new_actions
    old_engaged = old_actions != noop
    new_engaged = new_actions != noop
    actionable = grid.legal_action_counts > 1
    crossings = actionable & (
        (old.context_engage_probabilities >= 0.5)
        != (new.context_engage_probabilities >= 0.5)
    )
    eligible = grid.eligible_ds
    eligible_count = int(eligible.sum())
    if eligible_count == 0:
        raise ValueError("Frozen probe has no DS-eligible prefix contexts")

    weighted_flip: list[float] = []
    policy_distance: list[float] = []
    suffix_changes: list[float] = []
    eligible_indices = np.flatnonzero(eligible)
    for index in eligible_indices:
        state_index = int(grid.state_indices[index])
        mask = np.asarray(
            grid.corpus.action_masks[state_index], dtype=bool
        ).reshape(grid.num_units, grid.num_actions)
        prefix = tuple(
            int(value)
            for value in grid.prefixes[index]
            if int(value) >= 0
        )
        costs = dynamic_support_cost_matrix(
            mask,
            prefix,
            grid.unit_order,
        )
        actions = np.asarray(costs.actions, dtype=np.int64)
        old_probabilities = old.context_probabilities[index, actions]
        new_probabilities = new.context_probabilities[index, actions]
        old_probabilities = old_probabilities / old_probabilities.sum()
        new_probabilities = new_probabilities / new_probabilities.sum()
        risk = old_policy_structural_risk(costs.costs, old_probabilities)
        old_action_index = int(
            np.flatnonzero(actions == int(old_actions[index]))[0]
        )
        new_action_index = int(
            np.flatnonzero(actions == int(new_actions[index]))[0]
        )
        weighted_flip.append(
            float(flip[index]) * float(risk[new_action_index])
        )
        policy_distance.append(
            dynamic_support_policy_distance(
                new_probabilities,
                old_probabilities,
                risk,
            )
        )
        suffix_changes.append(
            float(
                abs(
                    costs.suffix_counts[new_action_index]
                    - costs.suffix_counts[old_action_index]
                )
            )
        )

    old_joint = old.joint_actions
    new_joint = new.joint_actions
    joint_flips = np.any(old_joint != new_joint, axis=1)
    engaged_per_state = np.sum(new_joint != noop, axis=1)
    high_threat_unassigned, high_threat_opportunities = (
        _high_threat_unassigned_by_state(
            grid,
            new,
            high_threat_threshold=high_threat_threshold,
        )
    )
    opportunity_count = int(high_threat_opportunities.sum())
    return {
        "update_id": int(update_id),
        "ppo_update": int(update_id),
        "approx_kl": float(approx_kl),
        "clip_fraction": float(clip_fraction),
        "entropy": float(entropy),
        "engagement_margin_crossings": int(crossings.sum()),
        "engage_to_noop_flips": int((old_engaged & ~new_engaged).sum()),
        "noop_to_engage_flips": int((~old_engaged & new_engaged).sum()),
        "joint_argmax_flips": int(joint_flips.sum()),
        "unweighted_prefix_flip_rate": float(np.mean(flip[eligible])),
        "ds_weighted_flip_mass": float(np.mean(weighted_flip)),
        "suffix_count_change": float(np.mean(suffix_changes)),
        "probe_all_noop_rate": float(np.mean(engaged_per_state == 0)),
        "probe_high_engagement_rate": float(
            np.mean(engaged_per_state >= 2)
        ),
        "probe_high_threat_unassigned_rate": (
            float(
                np.mean(
                    high_threat_unassigned[high_threat_opportunities]
                )
            )
            if opportunity_count
            else float("nan")
        ),
        "ds_policy_distance": float(np.mean(policy_distance)),
        "joint_argmax_flip_rate": float(np.mean(joint_flips)),
        "probe_states": int(len(new.state_ids)),
        "eligible_prefix_contexts": eligible_count,
        "high_threat_opportunity_states": opportunity_count,
    }


def frozen_probe_coverage(
    grid: FrozenDynamicSupportProbe,
    reference: PolicyProbeSnapshot,
) -> dict[str, Any]:
    """Return preregistered structural and reference-margin coverage."""

    if reference.grid_sha256 != grid.content_sha256():
        raise ValueError("Reference snapshot does not match the frozen grid")
    margins = reference.context_engage_probabilities - 0.5
    return {
        "states": int(len(reference.state_ids)),
        "contexts": grid.size,
        "eligible_ds_contexts": int(grid.eligible_ds.sum()),
        "scenario_counts": _counts(grid.scenarios),
        "unit_position_counts": _counts(grid.unit_positions),
        "legal_action_count_counts": _counts(grid.legal_action_counts),
        "engagement_margin_negative": int((margins < 0.0).sum()),
        "engagement_margin_nonnegative": int((margins >= 0.0).sum()),
        "high_threat_reachable_contexts": int(
            grid.high_threat_reachable.sum()
        ),
        "high_threat_unreachable_contexts": int(
            (~grid.high_threat_reachable).sum()
        ),
        "context_ids_unique": (
            len(set(grid.context_ids.astype(str))) == grid.size
        ),
    }


class DynamicSupportInstrumentationCallback(BaseCallback):
    """Opt-in, read-only DST-05 callback evaluated after completed PPO updates."""

    def __init__(
        self,
        grid: FrozenDynamicSupportProbe,
        *,
        batch_size: int = 512,
    ) -> None:
        super().__init__(verbose=0)
        self.grid = grid
        self.batch_size = batch_size
        self.rows: list[dict[str, float | int]] = []
        self._previous: PolicyProbeSnapshot | None = None
        self._last_update = -1

    def _on_training_start(self) -> None:
        self._previous = evaluate_policy_on_frozen_probe(
            self.model,
            self.grid,
            batch_size=self.batch_size,
        )
        self._last_update = int(getattr(self.model, "_n_updates", 0))

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        self._record_completed_update()

    def _on_training_end(self) -> None:
        self._record_completed_update()

    def _record_completed_update(self) -> None:
        update_id = int(getattr(self.model, "_n_updates", 0))
        if update_id <= self._last_update:
            return
        if self._previous is None:
            raise RuntimeError("Instrumentation was not initialized")
        current = evaluate_policy_on_frozen_probe(
            self.model,
            self.grid,
            batch_size=self.batch_size,
        )
        logger_values = getattr(self.model.logger, "name_to_value", {})
        entropy_loss = _as_float(
            logger_values.get("train/entropy_loss")
        )
        self.rows.append(
            compute_dynamic_support_update_metrics(
                self.grid,
                self._previous,
                current,
                update_id=update_id,
                approx_kl=_as_float(
                    logger_values.get("train/approx_kl")
                ),
                clip_fraction=_as_float(
                    logger_values.get("train/clip_fraction")
                ),
                entropy=(
                    -entropy_loss
                    if np.isfinite(entropy_loss)
                    else float("nan")
                ),
            )
        )
        self._previous = current
        self._last_update = update_id


def _validate_snapshot_pair(
    grid: FrozenDynamicSupportProbe,
    old: PolicyProbeSnapshot,
    new: PolicyProbeSnapshot,
) -> None:
    expected = grid.content_sha256()
    if old.grid_sha256 != expected or new.grid_sha256 != expected:
        raise ValueError("Policy snapshots do not match the frozen grid")
    if not np.array_equal(old.context_ids, grid.context_ids):
        raise ValueError("Old snapshot context ids changed")
    if not np.array_equal(new.context_ids, grid.context_ids):
        raise ValueError("New snapshot context ids changed")
    if not np.array_equal(old.state_ids, new.state_ids):
        raise ValueError("Policy snapshots use different frozen states")
    for probabilities in (
        old.context_probabilities,
        new.context_probabilities,
    ):
        if not np.allclose(
            probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0.0
        ):
            raise ValueError("Every conditional probability row must sum to 1")


def _high_threat_unassigned_by_state(
    grid: FrozenDynamicSupportProbe,
    snapshot: PolicyProbeSnapshot,
    *,
    high_threat_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    state_indices = grid.unique_state_indices
    unassigned = np.zeros(len(state_indices), dtype=bool)
    opportunities = np.zeros(len(state_indices), dtype=bool)
    for row, state_index in enumerate(state_indices):
        mask = np.asarray(
            grid.corpus.action_masks[int(state_index)], dtype=bool
        ).reshape(grid.num_units, grid.num_actions)
        threats = _target_threats(
            grid.corpus.observations[int(state_index)],
            num_units=grid.num_units,
            num_targets=grid.noop_action,
        )
        high_targets = {
            int(target)
            for target in np.flatnonzero(
                threats >= high_threat_threshold
            )
            if bool(mask[:, int(target)].any())
        }
        opportunities[row] = bool(high_targets)
        assigned = {
            int(action)
            for action in snapshot.joint_actions[row]
            if int(action) != grid.noop_action
        }
        unassigned[row] = bool(high_targets - assigned)
    return unassigned, opportunities


def _target_threats(
    observation: np.ndarray,
    *,
    num_units: int,
    num_targets: int,
) -> np.ndarray:
    values = np.asarray(observation, dtype=np.float32).reshape(-1)
    fixed = num_targets * 15 + num_units * 15 + 8
    zone_values = len(values) - fixed
    if zone_values <= 0 or zone_values % 7:
        raise ValueError("Observation is incompatible with AirDefense-v1 layout")
    target_start = zone_values
    targets = values[
        target_start : target_start + num_targets * 15
    ].reshape(num_targets, 15)
    return targets[:, 6].astype(np.float64)


def _probe_state_id(corpus: PolicyProbeCorpus, index: int) -> str:
    digest = hashlib.sha256()
    digest.update(
        np.ascontiguousarray(
            corpus.observations[index], dtype=np.float32
        ).tobytes()
    )
    digest.update(
        np.ascontiguousarray(
            corpus.action_masks[index], dtype=np.bool_
        ).tobytes()
    )
    for value in (
        corpus.scenarios[index],
        corpus.sources[index],
        corpus.phases[index],
        corpus.environment_seeds[index],
        corpus.episode_indices[index],
        corpus.step_indices[index],
    ):
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {
        str(name): int(count)
        for name, count in zip(unique.tolist(), counts.tolist())
    }


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state().clone(),
        "torch_cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    array = np.asarray(value)
    return float(array.reshape(-1)[0])
