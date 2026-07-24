from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import torch

from ..envs.air_defense_v1 import (
    AirDefenseResourceAssignmentEnvV1,
    AirDefenseV1EnvConfig,
    AirDefenseV1StateSnapshot,
)


@dataclass(frozen=True)
class BoundaryCounterfactualProbeConfig:
    max_contexts: int = 2
    repeats: int = 8
    margin_radius: float = 0.62
    minimum_sign_agreement: int = 1
    minimum_informative_repeats: int = 2
    maximum_opposite_repeats: int = 1
    minimum_return_effect: float = 1.0
    engagement_threshold: float = 0.5
    base_seed: int = 73_000
    selection_mode: Literal["boundary", "random"] = "boundary"

    def __post_init__(self) -> None:
        if self.max_contexts < 0:
            raise ValueError("max_contexts must be non-negative")
        if self.repeats <= 0:
            raise ValueError("repeats must be positive")
        if self.margin_radius < 0.0:
            raise ValueError("margin_radius must be non-negative")
        if not 1 <= self.minimum_sign_agreement <= self.repeats:
            raise ValueError("minimum_sign_agreement must be in [1, repeats]")
        if not 1 <= self.minimum_informative_repeats <= self.repeats:
            raise ValueError("minimum_informative_repeats must be in [1, repeats]")
        if not 0 <= self.maximum_opposite_repeats < self.repeats:
            raise ValueError("maximum_opposite_repeats must be in [0, repeats)")
        if self.minimum_return_effect < 0.0:
            raise ValueError("minimum_return_effect must be non-negative")
        if not 0.0 <= self.engagement_threshold <= 1.0:
            raise ValueError("engagement_threshold must be in [0, 1]")
        if self.selection_mode not in {"boundary", "random"}:
            raise ValueError("selection_mode must be 'boundary' or 'random'")


@dataclass(frozen=True)
class BoundaryProbeCandidate:
    rollout_step: int
    unit_index: int
    margin: float
    engage_probability: float


@dataclass(frozen=True)
class BoundaryProbeLabels:
    directions: np.ndarray
    weights: np.ndarray
    mean_deltas: np.ndarray
    selected: np.ndarray
    sign_agreements: np.ndarray
    informative_counts: np.ndarray
    opposite_counts: np.ndarray
    candidates: tuple[BoundaryProbeCandidate, ...]
    extra_transitions: int
    accepted_count: int
    positive_count: int
    negative_count: int

    @property
    def selected_count(self) -> int:
        return len(self.candidates)

    @property
    def acceptance_rate(self) -> float:
        if not self.candidates:
            return 0.0
        return self.accepted_count / len(self.candidates)


def select_boundary_candidates(
    engage_probabilities: np.ndarray,
    actionable: np.ndarray,
    *,
    margin_radius: float,
    max_contexts: int,
) -> tuple[BoundaryProbeCandidate, ...]:
    probabilities = np.asarray(engage_probabilities, dtype=np.float64)
    valid = np.asarray(actionable, dtype=bool)
    if probabilities.shape != valid.shape or probabilities.ndim != 2:
        raise ValueError("Probability and actionable arrays must be matching 2D arrays")
    if max_contexts <= 0:
        return ()

    clipped = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    margins = np.log(clipped) - np.log1p(-clipped)
    candidates = [
        BoundaryProbeCandidate(
            rollout_step=step,
            unit_index=unit,
            margin=float(margins[step, unit]),
            engage_probability=float(probabilities[step, unit]),
        )
        for step, unit in zip(*np.nonzero(valid & (np.abs(margins) <= margin_radius)))
    ]
    candidates.sort(
        key=lambda item: (
            abs(item.margin),
            item.rollout_step,
            item.unit_index,
        )
    )
    return tuple(candidates[:max_contexts])


def paired_direction_gate(
    deltas: Sequence[float],
    *,
    minimum_sign_agreement: int,
    minimum_return_effect: float,
    minimum_informative_repeats: int = 1,
    maximum_opposite_repeats: int | None = None,
) -> tuple[int, bool, float, int]:
    values = np.asarray(tuple(deltas), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("deltas must be a non-empty 1D sequence")
    mean_delta = float(values.mean())
    if mean_delta > 0.0:
        direction = 1
    elif mean_delta < 0.0:
        direction = -1
    else:
        direction = 0
    agreement = int(np.sum(np.sign(values) == direction)) if direction else 0
    informative = int(np.sum(np.sign(values) != 0))
    opposite = informative - agreement
    accepted = bool(
        direction != 0
        and agreement >= minimum_sign_agreement
        and informative >= minimum_informative_repeats
        and agreement > opposite
        and (
            maximum_opposite_repeats is None
            or opposite <= maximum_opposite_repeats
        )
        and abs(mean_delta) >= minimum_return_effect
    )
    return direction, accepted, mean_delta, agreement


def select_random_candidates(
    engage_probabilities: np.ndarray,
    actionable: np.ndarray,
    *,
    max_contexts: int,
    seed: int,
) -> tuple[BoundaryProbeCandidate, ...]:
    probabilities = np.asarray(engage_probabilities, dtype=np.float64)
    valid = np.asarray(actionable, dtype=bool)
    if probabilities.shape != valid.shape or probabilities.ndim != 2:
        raise ValueError("Probability and actionable arrays must be matching 2D arrays")
    indices = np.argwhere(valid)
    if max_contexts <= 0 or indices.size == 0:
        return ()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(
        len(indices),
        size=min(max_contexts, len(indices)),
        replace=False,
    )
    clipped = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    margins = np.log(clipped) - np.log1p(-clipped)
    return tuple(
        BoundaryProbeCandidate(
            rollout_step=int(indices[index, 0]),
            unit_index=int(indices[index, 1]),
            margin=float(margins[tuple(indices[index])]),
            engage_probability=float(probabilities[tuple(indices[index])]),
        )
        for index in chosen
    )


class BoundaryCounterfactualProbeRunner:
    """Generate paired engage/no-op labels from on-policy state snapshots."""

    def __init__(
        self,
        env_config: AirDefenseV1EnvConfig,
        config: BoundaryCounterfactualProbeConfig | None = None,
    ) -> None:
        self.env_config = env_config
        self.config = config or BoundaryCounterfactualProbeConfig()

    @torch.no_grad()
    def generate(
        self,
        *,
        policy: Any,
        snapshots: Sequence[AirDefenseV1StateSnapshot],
        observations: np.ndarray,
        actions: np.ndarray,
        action_masks: np.ndarray,
        rollout_index: int,
    ) -> BoundaryProbeLabels:
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        action_masks = np.asarray(action_masks)
        if observations.ndim != 2 or actions.ndim != 2:
            raise ValueError("Observations and actions must be 2D arrays")
        if len(snapshots) != observations.shape[0]:
            raise ValueError("Snapshot count must match rollout length")

        device = next(policy.parameters()).device
        observation_tensor = torch.as_tensor(observations, device=device)
        mask_tensor = torch.as_tensor(action_masks, device=device)
        action_tensor = torch.as_tensor(actions, device=device)
        distribution = policy.get_distribution(
            observation_tensor,
            action_masks=mask_tensor,
        )
        diagnostics = distribution.hierarchical_diagnostics(action_tensor)
        engage_probabilities = diagnostics["engage_probability"].cpu().numpy()
        actionable = diagnostics["actionable"].bool().cpu().numpy()
        if self.config.selection_mode == "boundary":
            candidates = select_boundary_candidates(
                engage_probabilities,
                actionable,
                margin_radius=self.config.margin_radius,
                max_contexts=self.config.max_contexts,
            )
        else:
            candidates = select_random_candidates(
                engage_probabilities,
                actionable,
                max_contexts=self.config.max_contexts,
                seed=self.config.base_seed + rollout_index,
            )

        shape = actions.shape
        directions = np.zeros(shape, dtype=np.float32)
        weights = np.zeros(shape, dtype=np.float32)
        mean_deltas = np.zeros(shape, dtype=np.float32)
        selected = np.zeros(shape, dtype=bool)
        sign_agreements = np.zeros(shape, dtype=np.int16)
        informative_counts = np.zeros(shape, dtype=np.int16)
        opposite_counts = np.zeros(shape, dtype=np.int16)
        extra_transitions = 0
        positive_count = 0
        negative_count = 0
        probe_env = AirDefenseResourceAssignmentEnvV1(config=self.env_config)
        probe_env.reset(seed=self.config.base_seed)

        for candidate in candidates:
            step = candidate.rollout_step
            unit = candidate.unit_index
            selected[step, unit] = True
            step_observation = observation_tensor[step : step + 1]
            step_mask = mask_tensor[step : step + 1]
            original_action = action_tensor[step : step + 1]
            step_distribution = policy.get_distribution(
                step_observation,
                action_masks=step_mask,
            )
            probabilities, _ = step_distribution.conditional_probabilities(
                original_action
            )
            engage_target = int(
                probabilities[0, unit, : step_distribution.num_targets]
                .argmax()
                .item()
            )
            noop_action, engage_action = self._branch_actions(
                step_distribution,
                original_action,
                unit_index=unit,
                engage_target=engage_target,
            )

            deltas: list[float] = []
            for repeat in range(self.config.repeats):
                tape = self._random_tape(
                    rollout_index=rollout_index,
                    rollout_step=step,
                    unit_index=unit,
                    repeat=repeat,
                )
                noop_return, noop_steps = self._rollout_branch(
                    probe_env,
                    snapshots[step],
                    tape,
                    noop_action,
                    policy,
                )
                engage_return, engage_steps = self._rollout_branch(
                    probe_env,
                    snapshots[step],
                    tape,
                    engage_action,
                    policy,
                )
                extra_transitions += noop_steps + engage_steps
                deltas.append(engage_return - noop_return)

            direction, accepted, mean_delta, agreement = paired_direction_gate(
                deltas,
                minimum_sign_agreement=self.config.minimum_sign_agreement,
                minimum_return_effect=self.config.minimum_return_effect,
                minimum_informative_repeats=(
                    self.config.minimum_informative_repeats
                ),
                maximum_opposite_repeats=(
                    self.config.maximum_opposite_repeats
                ),
            )
            directions[step, unit] = float(direction)
            mean_deltas[step, unit] = mean_delta
            sign_agreements[step, unit] = agreement
            signs = np.sign(np.asarray(deltas))
            informative_counts[step, unit] = int(np.sum(signs != 0))
            opposite_counts[step, unit] = int(
                np.sum((signs != 0) & (signs != direction))
            )
            if accepted:
                weights[step, unit] = 1.0
                if direction > 0:
                    positive_count += 1
                else:
                    negative_count += 1

        probe_env.close()
        accepted_count = positive_count + negative_count
        return BoundaryProbeLabels(
            directions=directions,
            weights=weights,
            mean_deltas=mean_deltas,
            selected=selected,
            sign_agreements=sign_agreements,
            informative_counts=informative_counts,
            opposite_counts=opposite_counts,
            candidates=candidates,
            extra_transitions=extra_transitions,
            accepted_count=accepted_count,
            positive_count=positive_count,
            negative_count=negative_count,
        )

    def _branch_actions(
        self,
        distribution: Any,
        original_action: torch.Tensor,
        *,
        unit_index: int,
        engage_target: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        order = tuple(distribution.unit_order)
        order_position = order.index(unit_index)
        fixed_noop = torch.full_like(original_action, -1)
        fixed_engage = torch.full_like(original_action, -1)
        for prefix_unit in order[:order_position]:
            fixed_noop[:, prefix_unit] = original_action[:, prefix_unit]
            fixed_engage[:, prefix_unit] = original_action[:, prefix_unit]
        fixed_noop[:, unit_index] = distribution.noop_action
        fixed_engage[:, unit_index] = engage_target
        noop_action = distribution.complete_fixed_actions_with_engagement_threshold(
            fixed_noop,
            threshold=self.config.engagement_threshold,
        ).actions
        engage_action = (
            distribution.complete_fixed_actions_with_engagement_threshold(
                fixed_engage,
                threshold=self.config.engagement_threshold,
            ).actions
        )
        return (
            noop_action[0].detach().cpu().numpy(),
            engage_action[0].detach().cpu().numpy(),
        )

    def _random_tape(
        self,
        *,
        rollout_index: int,
        rollout_step: int,
        unit_index: int,
        repeat: int,
    ) -> np.ndarray:
        seed = np.random.SeedSequence(
            [
                self.config.base_seed,
                rollout_index,
                rollout_step,
                unit_index,
                repeat,
            ]
        )
        rng = np.random.default_rng(seed)
        return rng.random(
            (
                self.env_config.max_steps,
                len(self.env_config.targets)
                or self.env_config.num_random_targets,
            ),
            dtype=np.float64,
        )

    @torch.no_grad()
    def _rollout_branch(
        self,
        env: AirDefenseResourceAssignmentEnvV1,
        snapshot: AirDefenseV1StateSnapshot,
        random_tape: np.ndarray,
        first_action: np.ndarray,
        policy: Any,
    ) -> tuple[float, int]:
        env.restore_state(snapshot)
        env.set_hit_random_tape(random_tape)
        observation, reward, terminated, truncated, _ = env.step(first_action)
        total_return = float(reward)
        steps = 1
        device = next(policy.parameters()).device
        while not (terminated or truncated):
            observation_tensor = torch.as_tensor(
                observation[None, :],
                device=device,
                dtype=torch.float32,
            )
            action_mask = torch.as_tensor(
                env.action_masks()[None, :],
                device=device,
            )
            distribution = policy.get_distribution(
                observation_tensor,
                action_masks=action_mask,
            )
            action = distribution.sample_with_engagement_threshold(
                self.config.engagement_threshold
            ).actions[0]
            observation, reward, terminated, truncated, _ = env.step(
                action.detach().cpu().numpy()
            )
            total_return += float(reward)
            steps += 1
        return total_return, steps
