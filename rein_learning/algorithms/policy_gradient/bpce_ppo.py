from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from stable_baselines3.common.utils import explained_variance

from ...common.boundary_counterfactual_probe import (
    BoundaryCounterfactualProbeConfig,
    BoundaryCounterfactualProbeRunner,
)
from ...envs.air_defense_v1 import AirDefenseResourceAssignmentEnvV1
from .factorized_engagement_ppo import (
    FactorizedEngagementActorCriticPolicy,
    FactorizedEngagementMaskablePPO,
)


class BPCEMaskableRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    action_masks: torch.Tensor
    bpce_directions: torch.Tensor
    bpce_weights: torch.Tensor


class BPCEMaskableRolloutBuffer(MaskableRolloutBuffer):
    """Maskable rollout buffer with per-unit counterfactual direction labels."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        action_space = args[2] if len(args) >= 3 else kwargs["action_space"]
        if not isinstance(action_space, spaces.MultiDiscrete):
            raise ValueError("BPCE requires a MultiDiscrete action space")
        self.num_units = int(len(action_space.nvec))
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        self.bpce_directions = np.zeros(
            (self.buffer_size, self.n_envs, self.num_units),
            dtype=np.float32,
        )
        self.bpce_weights = np.zeros_like(self.bpce_directions)
        super().reset()

    def set_probe_labels(
        self,
        directions: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        expected = (self.buffer_size, self.n_envs, self.num_units)
        direction_values = np.asarray(directions, dtype=np.float32)
        weight_values = np.asarray(weights, dtype=np.float32)
        if direction_values.shape != expected or weight_values.shape != expected:
            raise ValueError(f"BPCE labels must have shape {expected}")
        self.bpce_directions[...] = direction_values
        self.bpce_weights[...] = weight_values

    def get(
        self, batch_size: int | None = None
    ) -> Generator[BPCEMaskableRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling"
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            for tensor_name in (
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "action_masks",
                "bpce_directions",
                "bpce_weights",
            ):
                self.__dict__[tensor_name] = self.swap_and_flatten(
                    self.__dict__[tensor_name]
                )
            self.generator_ready = True
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
        for start in range(0, self.buffer_size * self.n_envs, batch_size):
            batch_indices = indices[start : start + batch_size]
            data = (
                self.observations[batch_indices],
                self.actions[batch_indices],
                self.values[batch_indices].flatten(),
                self.log_probs[batch_indices].flatten(),
                self.advantages[batch_indices].flatten(),
                self.returns[batch_indices].flatten(),
                self.action_masks[batch_indices].reshape(-1, self.mask_dims),
                self.bpce_directions[batch_indices].reshape(-1, self.num_units),
                self.bpce_weights[batch_indices].reshape(-1, self.num_units),
            )
            yield BPCEMaskableRolloutBufferSamples(
                *map(self.to_torch, data)
            )


class BoundaryProbedCounterfactualEngagementPPO(
    FactorizedEngagementMaskablePPO
):
    """Joint PPO with sparse boundary-probed engagement ranking supervision."""

    def __init__(
        self,
        policy: Any,
        env: Any,
        *args: Any,
        counterfactual_loss_coef: float = 0.05,
        probe_interval: int = 2,
        probe_max_contexts: int = 2,
        probe_repeats: int = 8,
        probe_margin_radius: float = 0.62,
        probe_minimum_sign_agreement: int = 1,
        probe_minimum_informative_repeats: int = 2,
        probe_maximum_opposite_repeats: int = 1,
        probe_minimum_return_effect: float = 1.0,
        probe_base_seed: int = 73_000,
        probe_selection_mode: str = "boundary",
        **kwargs: Any,
    ) -> None:
        if counterfactual_loss_coef < 0.0:
            raise ValueError("counterfactual_loss_coef must be non-negative")
        if probe_interval <= 0:
            raise ValueError("probe_interval must be positive")
        self.counterfactual_loss_coef = float(counterfactual_loss_coef)
        self.probe_interval = int(probe_interval)
        self.probe_config = BoundaryCounterfactualProbeConfig(
            max_contexts=probe_max_contexts,
            repeats=probe_repeats,
            margin_radius=probe_margin_radius,
            minimum_sign_agreement=probe_minimum_sign_agreement,
            minimum_informative_repeats=probe_minimum_informative_repeats,
            maximum_opposite_repeats=probe_maximum_opposite_repeats,
            minimum_return_effect=probe_minimum_return_effect,
            base_seed=probe_base_seed,
            selection_mode=probe_selection_mode,
        )
        self._bpce_rollout_index = 0
        self._bpce_probe_runner: BoundaryCounterfactualProbeRunner | None = None
        self.bpce_extra_transitions = 0
        self.bpce_probe_rollouts = 0
        self.bpce_selected_contexts = 0
        self.bpce_accepted_contexts = 0
        self.bpce_positive_contexts = 0
        self.bpce_negative_contexts = 0
        self.bpce_selected_abs_delta_sum = 0.0
        self.bpce_selected_sign_agreement_sum = 0.0
        self.bpce_effect_pass_count = 0
        self.bpce_agreement_pass_count = 0
        self.bpce_informative_repeat_sum = 0
        self.bpce_opposite_repeat_sum = 0
        self.last_bpce_probe_diagnostics: dict[str, float] = {}
        self.last_bpce_training_diagnostics: dict[str, float] = {}
        self.bpce_auxiliary_train_calls = 0
        self.bpce_auxiliary_loss_sum = 0.0
        kwargs.setdefault("rollout_buffer_class", BPCEMaskableRolloutBuffer)
        super().__init__(policy, env, *args, **kwargs)
        self.action_generator_signature = {
            **self.action_generator_signature,
            "optimizer": {
                "type": "boundary_probed_counterfactual_engagement_ppo",
                "counterfactual_loss_coef": self.counterfactual_loss_coef,
                "probe_interval": self.probe_interval,
                "probe_config": {
                    "max_contexts": self.probe_config.max_contexts,
                    "repeats": self.probe_config.repeats,
                    "margin_radius": self.probe_config.margin_radius,
                    "minimum_sign_agreement": (
                        self.probe_config.minimum_sign_agreement
                    ),
                    "minimum_informative_repeats": (
                        self.probe_config.minimum_informative_repeats
                    ),
                    "maximum_opposite_repeats": (
                        self.probe_config.maximum_opposite_repeats
                    ),
                    "minimum_return_effect": (
                        self.probe_config.minimum_return_effect
                    ),
                    "selection_mode": self.probe_config.selection_mode,
                },
            },
        }

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_bpce_probe_runner"]

    @classmethod
    def load(
        cls,
        path: str | Path,
        env: Any | None = None,
        **kwargs: Any,
    ) -> "BoundaryProbedCounterfactualEngagementPPO":
        model = super().load(path, env=env, **kwargs)
        if not isinstance(model.policy, FactorizedEngagementActorCriticPolicy):
            raise ValueError("Saved model does not contain the factorized policy")
        return model

    def collect_rollouts(
        self,
        env: Any,
        callback: Any,
        rollout_buffer: Any,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        if self.n_envs != 1:
            raise ValueError("BPCE v0 currently requires exactly one environment")
        base_env = self._base_air_defense_env(env)
        base_env.start_state_snapshot_recording()
        try:
            completed = super().collect_rollouts(
                env,
                callback,
                rollout_buffer,
                n_rollout_steps,
                use_masking,
            )
        finally:
            snapshots = base_env.stop_state_snapshot_recording()
        if not completed:
            return False
        if not isinstance(rollout_buffer, BPCEMaskableRolloutBuffer):
            raise TypeError("BPCE requires BPCEMaskableRolloutBuffer")
        if len(snapshots) != n_rollout_steps:
            raise RuntimeError("Recorded snapshot count does not match rollout length")

        should_probe = (
            self.counterfactual_loss_coef > 0.0
            and self.probe_config.max_contexts > 0
            and self._bpce_rollout_index % self.probe_interval == 0
        )
        if should_probe:
            if self._bpce_probe_runner is None:
                self._bpce_probe_runner = BoundaryCounterfactualProbeRunner(
                    base_env.config,
                    self.probe_config,
                )
            old_training_mode = self.policy.training
            self.policy.set_training_mode(False)
            labels = self._bpce_probe_runner.generate(
                policy=self.policy,
                snapshots=snapshots,
                observations=rollout_buffer.observations[:, 0],
                actions=rollout_buffer.actions[:, 0],
                action_masks=rollout_buffer.action_masks[:, 0],
                rollout_index=self._bpce_rollout_index,
            )
            self.policy.set_training_mode(old_training_mode)
            rollout_buffer.set_probe_labels(
                labels.directions[:, None, :],
                labels.weights[:, None, :],
            )
            self.bpce_extra_transitions += labels.extra_transitions
            self.bpce_probe_rollouts += 1
            self.bpce_selected_contexts += labels.selected_count
            self.bpce_accepted_contexts += labels.accepted_count
            self.bpce_positive_contexts += labels.positive_count
            self.bpce_negative_contexts += labels.negative_count
            selected_abs_deltas = np.abs(labels.mean_deltas[labels.selected])
            selected_agreements = labels.sign_agreements[labels.selected]
            selected_informative = labels.informative_counts[labels.selected]
            selected_opposite = labels.opposite_counts[labels.selected]
            self.bpce_selected_abs_delta_sum += float(
                selected_abs_deltas.sum()
            )
            self.bpce_selected_sign_agreement_sum += float(
                selected_agreements.sum()
            )
            self.bpce_informative_repeat_sum += int(
                selected_informative.sum()
            )
            self.bpce_opposite_repeat_sum += int(selected_opposite.sum())
            self.bpce_effect_pass_count += int(
                np.sum(
                    selected_abs_deltas
                    >= self.probe_config.minimum_return_effect
                )
            )
            self.bpce_agreement_pass_count += int(
                np.sum(
                    selected_agreements
                    >= self.probe_config.minimum_sign_agreement
                )
            )
            self.last_bpce_probe_diagnostics = {
                "selected_count": float(labels.selected_count),
                "accepted_count": float(labels.accepted_count),
                "acceptance_rate": float(labels.acceptance_rate),
                "positive_count": float(labels.positive_count),
                "negative_count": float(labels.negative_count),
                "extra_transitions": float(labels.extra_transitions),
                "selected_mean_abs_delta": float(
                    np.abs(labels.mean_deltas[labels.selected]).mean()
                    if labels.selected_count
                    else 0.0
                ),
                "selected_mean_sign_agreement": float(
                    labels.sign_agreements[labels.selected].mean()
                    if labels.selected_count
                    else 0.0
                ),
                "effect_pass_rate": float(
                    np.mean(
                        np.abs(labels.mean_deltas[labels.selected])
                        >= self.probe_config.minimum_return_effect
                    )
                    if labels.selected_count
                    else 0.0
                ),
                "agreement_pass_rate": float(
                    np.mean(
                        labels.sign_agreements[labels.selected]
                        >= self.probe_config.minimum_sign_agreement
                    )
                    if labels.selected_count
                    else 0.0
                ),
                "selected_mean_informative_repeats": float(
                    selected_informative.mean()
                    if labels.selected_count
                    else 0.0
                ),
                "selected_mean_opposite_repeats": float(
                    selected_opposite.mean()
                    if labels.selected_count
                    else 0.0
                ),
                "cumulative_extra_transitions": float(
                    self.bpce_extra_transitions
                ),
                "cumulative_probe_rollouts": float(self.bpce_probe_rollouts),
                "cumulative_selected_count": float(
                    self.bpce_selected_contexts
                ),
                "cumulative_accepted_count": float(
                    self.bpce_accepted_contexts
                ),
                "cumulative_acceptance_rate": float(
                    self.bpce_accepted_contexts
                    / max(1, self.bpce_selected_contexts)
                ),
                "cumulative_positive_count": float(
                    self.bpce_positive_contexts
                ),
                "cumulative_negative_count": float(
                    self.bpce_negative_contexts
                ),
                "cumulative_mean_abs_delta": float(
                    self.bpce_selected_abs_delta_sum
                    / max(1, self.bpce_selected_contexts)
                ),
                "cumulative_mean_sign_agreement": float(
                    self.bpce_selected_sign_agreement_sum
                    / max(1, self.bpce_selected_contexts)
                ),
                "cumulative_effect_pass_rate": float(
                    self.bpce_effect_pass_count
                    / max(1, self.bpce_selected_contexts)
                ),
                "cumulative_agreement_pass_rate": float(
                    self.bpce_agreement_pass_count
                    / max(1, self.bpce_selected_contexts)
                ),
                "cumulative_mean_informative_repeats": float(
                    self.bpce_informative_repeat_sum
                    / max(1, self.bpce_selected_contexts)
                ),
                "cumulative_mean_opposite_repeats": float(
                    self.bpce_opposite_repeat_sum
                    / max(1, self.bpce_selected_contexts)
                ),
            }
        else:
            self.last_bpce_probe_diagnostics = {
                "selected_count": 0.0,
                "accepted_count": 0.0,
                "acceptance_rate": 0.0,
                "positive_count": 0.0,
                "negative_count": 0.0,
                "extra_transitions": 0.0,
                "selected_mean_abs_delta": 0.0,
                "selected_mean_sign_agreement": 0.0,
                "effect_pass_rate": 0.0,
                "agreement_pass_rate": 0.0,
                "selected_mean_informative_repeats": 0.0,
                "selected_mean_opposite_repeats": 0.0,
                "cumulative_extra_transitions": float(
                    self.bpce_extra_transitions
                ),
                "cumulative_probe_rollouts": float(self.bpce_probe_rollouts),
                "cumulative_selected_count": float(
                    self.bpce_selected_contexts
                ),
                "cumulative_accepted_count": float(
                    self.bpce_accepted_contexts
                ),
                "cumulative_acceptance_rate": float(
                    self.bpce_accepted_contexts
                    / max(1, self.bpce_selected_contexts)
                ),
                "cumulative_positive_count": float(
                    self.bpce_positive_contexts
                ),
                "cumulative_negative_count": float(
                    self.bpce_negative_contexts
                ),
                "cumulative_mean_abs_delta": float(
                    self.bpce_selected_abs_delta_sum
                    / max(1, self.bpce_selected_contexts)
                ),
                "cumulative_mean_sign_agreement": float(
                    self.bpce_selected_sign_agreement_sum
                    / max(1, self.bpce_selected_contexts)
                ),
                "cumulative_effect_pass_rate": float(
                    self.bpce_effect_pass_count
                    / max(1, self.bpce_selected_contexts)
                ),
                "cumulative_agreement_pass_rate": float(
                    self.bpce_agreement_pass_count
                    / max(1, self.bpce_selected_contexts)
                ),
                "cumulative_mean_informative_repeats": float(
                    self.bpce_informative_repeat_sum
                    / max(1, self.bpce_selected_contexts)
                ),
                "cumulative_mean_opposite_repeats": float(
                    self.bpce_opposite_repeat_sum
                    / max(1, self.bpce_selected_contexts)
                ),
            }
        self._bpce_rollout_index += 1
        return True

    def train(self) -> None:
        active_labels = bool(
            self.counterfactual_loss_coef > 0.0
            and np.any(self.rollout_buffer.bpce_weights > 0.0)
        )
        if not active_labels:
            super().train()
            self.last_bpce_training_diagnostics = {
                "auxiliary_loss": 0.0,
                "active_label_rate": 0.0,
                "joint_gradient_norm": 0.0,
                "auxiliary_gradient_norm": 0.0,
                "gradient_cosine": 0.0,
                "cumulative_auxiliary_train_calls": float(
                    self.bpce_auxiliary_train_calls
                ),
                "cumulative_mean_auxiliary_loss": float(
                    self.bpce_auxiliary_loss_sum
                    / max(1, self.bpce_auxiliary_train_calls)
                ),
            }
            self._record_bpce_logs()
            return

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(
                self._current_progress_remaining
            )

        entropy_losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        clip_fractions: list[float] = []
        auxiliary_losses: list[float] = []
        active_label_rates: list[float] = []
        approximate_kls: list[float] = []
        joint_gradient_norm = 0.0
        auxiliary_gradient_norm = 0.0
        gradient_cosine = 0.0
        gradient_diagnostics_recorded = False
        continue_training = True
        loss = torch.zeros((), device=self.device)

        for epoch in range(self.n_epochs):
            approximate_kls = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss = -torch.min(
                    advantages * ratio,
                    advantages
                    * torch.clamp(ratio, 1 - clip_range, 1 + clip_range),
                ).mean()
                policy_losses.append(policy_loss.item())
                clip_fractions.append(
                    torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                )

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                entropy_loss = (
                    -torch.mean(-log_prob)
                    if entropy is None
                    else -torch.mean(entropy)
                )
                entropy_losses.append(entropy_loss.item())

                distribution = self.policy.get_distribution(
                    rollout_data.observations,
                    action_masks=rollout_data.action_masks,
                )
                diagnostics = distribution.hierarchical_diagnostics(
                    actions.long()
                )
                probabilities = diagnostics["engage_probability"].clamp(
                    1e-8, 1.0 - 1e-8
                )
                margins = torch.log(probabilities) - torch.log1p(-probabilities)
                weights = rollout_data.bpce_weights
                directions = rollout_data.bpce_directions
                active_weight = weights.sum()
                auxiliary_loss = (
                    weights * F.softplus(-directions * margins)
                ).sum() / active_weight.clamp_min(1.0)
                auxiliary_losses.append(auxiliary_loss.item())
                active_label_rates.append(
                    float((weights > 0.0).float().mean().item())
                )
                if not gradient_diagnostics_recorded:
                    (
                        joint_gradient_norm,
                        auxiliary_gradient_norm,
                        gradient_cosine,
                    ) = self._gradient_diagnostics(
                        policy_loss,
                        auxiliary_loss,
                    )
                    gradient_diagnostics_recorded = True

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.counterfactual_loss_coef * auxiliary_loss
                )
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approximate_kl = torch.mean(
                        (torch.exp(log_ratio) - 1) - log_ratio
                    ).cpu().numpy()
                    approximate_kls.append(approximate_kl)
                if (
                    self.target_kl is not None
                    and approximate_kl > 1.5 * self.target_kl
                ):
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm,
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record(
            "train/policy_gradient_loss", np.mean(policy_losses)
        )
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approximate_kls))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record(
            "train/n_updates", self._n_updates, exclude="tensorboard"
        )
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        mean_auxiliary_loss = float(np.mean(auxiliary_losses))
        self.bpce_auxiliary_train_calls += 1
        self.bpce_auxiliary_loss_sum += mean_auxiliary_loss
        self.last_bpce_training_diagnostics = {
            "auxiliary_loss": mean_auxiliary_loss,
            "active_label_rate": float(np.mean(active_label_rates)),
            "joint_gradient_norm": joint_gradient_norm,
            "auxiliary_gradient_norm": auxiliary_gradient_norm,
            "gradient_cosine": gradient_cosine,
            "cumulative_auxiliary_train_calls": float(
                self.bpce_auxiliary_train_calls
            ),
            "cumulative_mean_auxiliary_loss": float(
                self.bpce_auxiliary_loss_sum
                / self.bpce_auxiliary_train_calls
            ),
        }
        self._record_bpce_logs()

    def _record_bpce_logs(self) -> None:
        for key, value in self.last_bpce_probe_diagnostics.items():
            self.logger.record(f"train/bpce_probe_{key}", value)
        for key, value in self.last_bpce_training_diagnostics.items():
            self.logger.record(f"train/bpce_{key}", value)

    def _gradient_diagnostics(
        self,
        joint_loss: torch.Tensor,
        auxiliary_loss: torch.Tensor,
    ) -> tuple[float, float, float]:
        parameters = tuple(
            parameter
            for parameter in self.policy.parameters()
            if parameter.requires_grad
        )
        joint_grads = torch.autograd.grad(
            joint_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        auxiliary_grads = torch.autograd.grad(
            auxiliary_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        joint_vector = torch.cat(
            [
                torch.zeros_like(parameter).flatten()
                if gradient is None
                else gradient.flatten()
                for parameter, gradient in zip(parameters, joint_grads)
            ]
        )
        auxiliary_vector = torch.cat(
            [
                torch.zeros_like(parameter).flatten()
                if gradient is None
                else gradient.flatten()
                for parameter, gradient in zip(parameters, auxiliary_grads)
            ]
        )
        joint_norm = torch.linalg.vector_norm(joint_vector)
        auxiliary_norm = torch.linalg.vector_norm(auxiliary_vector)
        denominator = (joint_norm * auxiliary_norm).clamp_min(1e-12)
        cosine = torch.dot(joint_vector, auxiliary_vector) / denominator
        return (
            float(joint_norm.item()),
            float(auxiliary_norm.item()),
            float(cosine.item()),
        )

    @staticmethod
    def _base_air_defense_env(vec_env: Any) -> AirDefenseResourceAssignmentEnvV1:
        if not hasattr(vec_env, "envs") or len(vec_env.envs) != 1:
            raise ValueError("BPCE requires a single local vector environment")
        base_env = vec_env.envs[0].unwrapped
        if not isinstance(base_env, AirDefenseResourceAssignmentEnvV1):
            raise TypeError("BPCE requires AirDefenseResourceAssignmentEnvV1")
        return base_env
