from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .dynamic_support_instrumentation import (
    FrozenDynamicSupportProbe,
    PolicyProbeSnapshot,
    compute_dynamic_support_update_metrics,
    evaluate_policy_on_frozen_probe,
)


EVALUATION_EPISODE_SEEDS = tuple(range(73_000, 73_050))


@dataclass(frozen=True)
class DS1EventProtocol:
    scenario: str = "heterogeneity_pressure"
    evaluation_episodes: int = 50
    all_noop_threshold: float = 0.98
    actionable_engagement_threshold: float = 0.01
    n_steps: int = 256
    n_epochs: int = 10
    requested_timesteps: int = 10_000
    completed_rollout_updates: int = 40
    actual_timesteps: int = 10_240
    prediction_horizon_min: int = 1
    prediction_horizon_max: int = 3
    small_kl_threshold: float = 0.01

    def __post_init__(self) -> None:
        if self.evaluation_episodes != 50:
            raise ValueError("Formal DS-1 evaluation requires exactly 50 episodes")
        if self.n_steps != 256 or self.n_epochs != 10:
            raise ValueError("Frozen DS-1 PPO cadence is n_steps=256, n_epochs=10")
        if self.completed_rollout_updates * self.n_steps != self.actual_timesteps:
            raise ValueError("Rollout count and actual timesteps are inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluation_seed_bank_sha256(
    seeds: Sequence[int] = EVALUATION_EPISODE_SEEDS,
) -> str:
    normalized = [int(seed) for seed in seeds]
    if len(normalized) != 50 or len(set(normalized)) != 50:
        raise ValueError("Formal evaluation seed bank requires 50 unique seeds")
    payload = json.dumps(
        normalized,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def formal_event_metrics(
    *,
    rollout_update_index: int,
    sb3_n_updates: int,
    num_timesteps: int,
    policy_seed: int,
    evaluation_seed_bank_hash: str,
    evaluation_episodes: int,
    all_noop_episodes: int,
    actionable_decisions: int,
    actionable_engagements: int,
) -> dict[str, Any]:
    """Validate and construct one post-policy formal event evaluation row."""

    if evaluation_episodes != 50:
        raise ValueError("Formal event rows require exactly 50 episodes")
    if not 0 <= all_noop_episodes <= evaluation_episodes:
        raise ValueError("all_noop_episodes is outside the episode count")
    if actionable_decisions < 0 or not 0 <= actionable_engagements <= actionable_decisions:
        raise ValueError("Invalid actionable decision counts")
    if rollout_update_index < 0 or sb3_n_updates < 0 or num_timesteps < 0:
        raise ValueError("Timeline identifiers must be nonnegative")
    if not evaluation_seed_bank_hash:
        raise ValueError("Evaluation seed-bank hash is required")
    all_noop_rate = all_noop_episodes / evaluation_episodes
    actionable_rate = (
        actionable_engagements / actionable_decisions
        if actionable_decisions
        else 0.0
    )
    collapsed = bool(
        all_noop_rate >= 0.98 or actionable_rate < 0.01
    )
    return {
        "rollout_update_index": int(rollout_update_index),
        "sb3_n_updates": int(sb3_n_updates),
        "num_timesteps": int(num_timesteps),
        "policy_seed": int(policy_seed),
        "evaluation_seed_bank_sha256": evaluation_seed_bank_hash,
        "evaluation_episodes": int(evaluation_episodes),
        "all_noop_episodes": int(all_noop_episodes),
        "all_noop_episode_rate": float(all_noop_rate),
        "actionable_decisions": int(actionable_decisions),
        "actionable_engagements": int(actionable_engagements),
        "actionable_engagement_rate": float(actionable_rate),
        "collapse_event_state": collapsed,
    }


def finalize_event_timeline(
    rows: Iterable[dict[str, Any]],
    *,
    expected_seed_bank_sha256: str,
    n_steps: int = 256,
    n_epochs: int = 10,
) -> dict[str, Any]:
    """Freeze first-onset semantics and forward labels on rollout row indices."""

    timeline = [dict(row) for row in rows]
    if not timeline:
        raise ValueError("Event timeline must not be empty")
    indices = [int(row["rollout_update_index"]) for row in timeline]
    if indices != list(range(len(timeline))):
        raise ValueError("rollout_update_index must be consecutive from zero")
    hashes = {str(row["evaluation_seed_bank_sha256"]) for row in timeline}
    if hashes != {expected_seed_bank_sha256}:
        raise ValueError("Evaluation seed-bank hash changed within the timeline")
    policy_seeds = {int(row["policy_seed"]) for row in timeline}
    if len(policy_seeds) != 1:
        raise ValueError("One timeline must contain exactly one policy seed")
    for index, row in enumerate(timeline):
        if int(row["evaluation_episodes"]) != 50:
            raise ValueError("Every formal timeline row requires 50 episodes")
        if int(row["num_timesteps"]) != index * n_steps:
            raise ValueError("num_timesteps is not aligned to rollout updates")
        if int(row["sb3_n_updates"]) != index * n_epochs:
            raise ValueError("sb3_n_updates is not aligned to PPO epochs")

    collapsed = np.asarray(
        [bool(row["collapse_event_state"]) for row in timeline],
        dtype=bool,
    )
    initially_collapsed = bool(collapsed[0])
    onset_candidates = [
        index
        for index in range(1, len(timeline))
        if collapsed[index] and not collapsed[index - 1]
    ]
    first_onset = None if initially_collapsed or not onset_candidates else onset_candidates[0]
    for index, row in enumerate(timeline):
        row["collapse_event_onset"] = bool(
            first_onset is not None and index == first_onset
        )
        row["initially_collapsed"] = initially_collapsed
        row["first_collapse_onset_index"] = first_onset
        row["formal_p2_evidence"] = bool(
            row.get("formal_p2_evidence", True)
        )
        row["event_within_3_updates"] = None
        row["predictor_row_eligible"] = False
        row["exclusion_reason"] = None
        if index == 0:
            row["exclusion_reason"] = "training_baseline"
            continue
        if initially_collapsed:
            row["exclusion_reason"] = "initially_collapsed"
            continue
        if first_onset is not None and index >= first_onset:
            row["exclusion_reason"] = (
                "concurrent_event" if index == first_onset else "post_event"
            )
            continue
        if index + 3 >= len(timeline):
            row["exclusion_reason"] = "right_censored_tail"
            continue
        row["predictor_row_eligible"] = True
        row["event_within_3_updates"] = bool(
            first_onset is not None
            and 1 <= first_onset - index <= 3
        )

    window = event_window_indices(first_onset)
    return {
        "policy_seed": next(iter(policy_seeds)),
        "initially_collapsed": initially_collapsed,
        "first_collapse_onset_index": first_onset,
        "event_bearing_seed": first_onset is not None,
        "baseline_window_indices": window["baseline"],
        "pre_event_window_indices": window["pre_event"],
        "event_median_window_eligible": window["eligible"],
        "rows": timeline,
    }


def event_window_indices(first_onset: int | None) -> dict[str, Any]:
    if first_onset is None or first_onset < 7:
        return {"baseline": [], "pre_event": [], "eligible": False}
    return {
        "baseline": list(range(first_onset - 6, first_onset - 3)),
        "pre_event": list(range(first_onset - 3, first_onset)),
        "eligible": True,
    }


class DS1IntegratedInstrumentationCallback(BaseCallback):
    """Real SB3 callback with rollout timebase and isolated formal evaluation."""

    def __init__(
        self,
        *,
        grid: FrozenDynamicSupportProbe,
        policy_seed: int,
        formal_event_evaluator: Callable[[Any], dict[str, int]],
        protocol: DS1EventProtocol | None = None,
        evaluation_seeds: Sequence[int] = EVALUATION_EPISODE_SEEDS,
        formal_p2_evidence: bool = False,
        batch_size: int = 512,
    ) -> None:
        super().__init__(verbose=0)
        self.grid = grid
        self.policy_seed = int(policy_seed)
        self.formal_event_evaluator = formal_event_evaluator
        self.protocol = protocol or DS1EventProtocol()
        self.evaluation_seeds = tuple(int(seed) for seed in evaluation_seeds)
        self.seed_bank_sha256 = evaluation_seed_bank_sha256(
            self.evaluation_seeds
        )
        self.formal_p2_evidence = bool(formal_p2_evidence)
        self.batch_size = int(batch_size)
        self.update_rows: list[dict[str, Any]] = []
        self.event_rows: list[dict[str, Any]] = []
        self.evaluation_audits: list[dict[str, Any]] = []
        self._previous: PolicyProbeSnapshot | None = None
        self._last_sb3_n_updates = -1
        self._rollout_update_index = 0

    def _on_training_start(self) -> None:
        self._previous = evaluate_policy_on_frozen_probe(
            self.model,
            self.grid,
            batch_size=self.batch_size,
        )
        self._last_sb3_n_updates = int(
            getattr(self.model, "_n_updates", 0)
        )
        if self._last_sb3_n_updates != 0 or int(self.model.num_timesteps) != 0:
            raise ValueError("DST-05.5 smoke must start from update and timestep zero")
        self._record_formal_event(
            rollout_update_index=0,
            sb3_n_updates=0,
            num_timesteps=0,
        )

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        self._record_completed_rollout_update()

    def _on_training_end(self) -> None:
        self._record_completed_rollout_update()

    def _record_completed_rollout_update(self) -> None:
        sb3_updates = int(getattr(self.model, "_n_updates", 0))
        if sb3_updates <= self._last_sb3_n_updates:
            return
        if sb3_updates - self._last_sb3_n_updates != self.protocol.n_epochs:
            raise ValueError("SB3 update counter did not advance by n_epochs")
        if self._previous is None:
            raise RuntimeError("DS-1 instrumentation was not initialized")
        self._rollout_update_index += 1
        expected_timesteps = (
            self._rollout_update_index * self.protocol.n_steps
        )
        if int(self.model.num_timesteps) != expected_timesteps:
            raise ValueError("Training timesteps do not match rollout timebase")
        current = evaluate_policy_on_frozen_probe(
            self.model,
            self.grid,
            batch_size=self.batch_size,
        )
        logger_values = getattr(self.model.logger, "name_to_value", {})
        entropy_loss = _as_float(
            logger_values.get("train/entropy_loss")
        )
        row = compute_dynamic_support_update_metrics(
            self.grid,
            self._previous,
            current,
            update_id=self._rollout_update_index,
            approx_kl=_as_float(logger_values.get("train/approx_kl")),
            clip_fraction=_as_float(
                logger_values.get("train/clip_fraction")
            ),
            entropy=(
                -entropy_loss
                if np.isfinite(entropy_loss)
                else float("nan")
            ),
        )
        row.update(
            {
                "rollout_update_index": self._rollout_update_index,
                "sb3_n_updates": sb3_updates,
                "num_timesteps": int(self.model.num_timesteps),
                "policy_seed": self.policy_seed,
                "evaluation_seed_bank_sha256": self.seed_bank_sha256,
                "formal_p2_evidence": self.formal_p2_evidence,
            }
        )
        event = self._record_formal_event(
            rollout_update_index=self._rollout_update_index,
            sb3_n_updates=sb3_updates,
            num_timesteps=int(self.model.num_timesteps),
        )
        row.update(
            {
                key: value
                for key, value in event.items()
                if key
                in {
                    "evaluation_episodes",
                    "all_noop_episodes",
                    "all_noop_episode_rate",
                    "actionable_decisions",
                    "actionable_engagements",
                    "actionable_engagement_rate",
                    "collapse_event_state",
                }
            }
        )
        self.update_rows.append(row)
        self._previous = current
        self._last_sb3_n_updates = sb3_updates

    def _record_formal_event(
        self,
        *,
        rollout_update_index: int,
        sb3_n_updates: int,
        num_timesteps: int,
    ) -> dict[str, Any]:
        payload = dict(self.formal_event_evaluator(self.model))
        audit = payload.pop("_audit", {})
        row = formal_event_metrics(
            rollout_update_index=rollout_update_index,
            sb3_n_updates=sb3_n_updates,
            num_timesteps=num_timesteps,
            policy_seed=self.policy_seed,
            evaluation_seed_bank_hash=self.seed_bank_sha256,
            evaluation_episodes=int(payload["evaluation_episodes"]),
            all_noop_episodes=int(payload["all_noop_episodes"]),
            actionable_decisions=int(payload["actionable_decisions"]),
            actionable_engagements=int(payload["actionable_engagements"]),
        )
        row["formal_p2_evidence"] = self.formal_p2_evidence
        self.event_rows.append(row)
        self.evaluation_audits.append(
            {
                "rollout_update_index": rollout_update_index,
                **audit,
            }
        )
        return row

    def finalized_event_timeline(self) -> dict[str, Any]:
        return finalize_event_timeline(
            self.event_rows,
            expected_seed_bank_sha256=self.seed_bank_sha256,
            n_steps=self.protocol.n_steps,
            n_epochs=self.protocol.n_epochs,
        )


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    array = np.asarray(value)
    return float(array.reshape(-1)[0])
