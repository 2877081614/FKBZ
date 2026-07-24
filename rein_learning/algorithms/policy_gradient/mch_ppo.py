from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.utils import explained_variance

from ...common.masked_context_support import MaskedContextSupportIndex
from ...models import HierarchicalMaskedQCritic
from .factorized_engagement_ppo import (
    FactorizedEngagementActorCriticPolicy,
    FactorizedEngagementMaskablePPO,
)


@dataclass(frozen=True)
class CounterfactualAdvantageBatch:
    engagement: torch.Tensor
    target: torch.Tensor
    engagement_reliability: torch.Tensor
    target_reliability: torch.Tensor
    engagement_support: torch.Tensor
    target_support: torch.Tensor
    actionable: torch.Tensor
    engaged: torch.Tensor


class MaskedCounterfactualHierarchicalPPO(FactorizedEngagementMaskablePPO):
    """PPO with masked counterfactual credit and hierarchical clipping.

    Frozen hierarchical Q critics provide per-unit engagement and conditional
    target advantages. The actor uses separate PPO ratios for both factors,
    while the ordinary rollout return continues to train the state-value head.
    """

    def __init__(
        self,
        policy: Any,
        env: Any,
        *args: Any,
        q_critic_paths: Iterable[str | Path] | None = None,
        engagement_loss_coef: float = 1.0,
        target_loss_coef: float = 1.0,
        **kwargs: Any,
    ) -> None:
        paths = tuple(str(Path(path).resolve()) for path in (q_critic_paths or ()))
        deferred_load = kwargs.get("_init_setup_model") is False
        if not paths and not deferred_load:
            raise ValueError("MCH-PPO requires at least one frozen Q-Critic checkpoint")
        if engagement_loss_coef < 0.0 or target_loss_coef < 0.0:
            raise ValueError("MCH-PPO loss coefficients must be non-negative")
        self.q_critic_paths = paths
        self.engagement_loss_coef = float(engagement_loss_coef)
        self.target_loss_coef = float(target_loss_coef)
        self._q_critics: list[HierarchicalMaskedQCritic] = []
        self._q_normalizations: list[dict[str, float]] = []
        super().__init__(policy, env, *args, **kwargs)
        if self.q_critic_paths:
            self._load_q_critics()
        self.action_generator_signature = {
            **self.action_generator_signature,
            "optimizer": {
                "type": "masked_counterfactual_hierarchical_ppo",
                "engagement_loss_coef": self.engagement_loss_coef,
                "target_loss_coef": self.target_loss_coef,
                "q_critic_ensemble_size": len(self._q_critics),
            },
        }

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + [
            "_q_critics",
            "_q_normalizations",
        ]

    def _load_q_critics(self) -> None:
        if not isinstance(self.policy, FactorizedEngagementActorCriticPolicy):
            raise TypeError("MCH-PPO requires the factorized engagement policy")
        self._q_critics = []
        self._q_normalizations = []
        expected_layout = self.policy.observation_layout.signature()
        for raw_path in self.q_critic_paths:
            path = Path(raw_path)
            if not path.is_file():
                raise FileNotFoundError(f"Q-Critic checkpoint not found: {path}")
            payload = torch.load(path, map_location=self.device, weights_only=False)
            signature = payload.get("signature", {})
            if signature.get("observation_layout") != expected_layout:
                raise ValueError(f"Q-Critic layout is incompatible: {path}")
            critic = HierarchicalMaskedQCritic(self.policy.observation_layout)
            critic.load_state_dict(payload["state_dict"])
            critic.to(self.device)
            critic.eval()
            for parameter in critic.parameters():
                parameter.requires_grad_(False)
            normalization = payload.get("normalization", {})
            required = {
                "engagement_mean",
                "engagement_std",
                "target_mean",
                "target_std",
            }
            if not required.issubset(normalization):
                raise ValueError(f"Q-Critic normalization is incomplete: {path}")
            self._q_critics.append(critic)
            self._q_normalizations.append(
                {key: float(normalization[key]) for key in required}
            )

    @staticmethod
    def _normalize_valid(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(values)
        selected = values[valid]
        if selected.numel() == 0:
            return result
        centered = selected - selected.mean()
        scale = selected.std(unbiased=False)
        if bool(scale > 1e-8):
            centered = centered / (scale + 1e-8)
        result[valid] = centered
        return result

    @staticmethod
    def _ensemble_reliability(values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=0)
        mean_magnitude = values.abs().mean(dim=0)
        return (mean.abs() / (mean_magnitude + 1e-8)).clamp(0.0, 1.0)

    @torch.no_grad()
    def _counterfactual_advantages(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        old_distribution: Any,
    ) -> CounterfactualAdvantageBatch:
        actions = actions.long().reshape(-1, old_distribution.num_units)
        diagnostics = old_distribution.hierarchical_diagnostics(actions)
        probabilities, dynamic_masks = old_distribution.conditional_probabilities(
            actions
        )
        batch_size, num_units = actions.shape
        num_targets = old_distribution.num_targets
        noop_action = old_distribution.noop_action
        dtype = observations.dtype

        engagement_advantages = torch.zeros(
            (batch_size, num_units), device=self.device, dtype=dtype
        )
        target_advantages = torch.zeros_like(engagement_advantages)
        engagement_reliability = torch.zeros_like(engagement_advantages)
        target_reliability = torch.zeros_like(engagement_advantages)
        engagement_support = torch.ones_like(engagement_advantages)
        target_support = torch.zeros_like(engagement_advantages)
        prefix_occupancy = torch.zeros(
            (batch_size, num_targets), device=self.device, dtype=dtype
        )
        batch_indices = torch.arange(batch_size, device=self.device)

        for unit_index in old_distribution.unit_order:
            legal_mask = dynamic_masks[:, unit_index, :].bool()
            unit_indices = torch.full(
                (batch_size,), unit_index, device=self.device, dtype=torch.long
            )
            engagement_predictions = []
            target_predictions = []
            for critic, normalization in zip(
                self._q_critics, self._q_normalizations
            ):
                engagement = critic.forward_engagement(
                    observations,
                    unit_indices,
                    prefix_occupancy,
                    legal_mask,
                )
                engagement = (
                    engagement * normalization["engagement_std"]
                    + normalization["engagement_mean"]
                )
                engagement_predictions.append(engagement)

                candidate_values = torch.zeros(
                    (batch_size, num_targets), device=self.device, dtype=dtype
                )
                for target_index in range(num_targets):
                    legal_rows = legal_mask[:, target_index]
                    if not bool(torch.any(legal_rows)):
                        continue
                    candidates = torch.full(
                        (int(legal_rows.sum().item()),),
                        target_index,
                        device=self.device,
                        dtype=torch.long,
                    )
                    predicted = critic.forward_target(
                        observations[legal_rows],
                        unit_indices[legal_rows],
                        candidates,
                        prefix_occupancy[legal_rows],
                        legal_mask[legal_rows],
                    )
                    predicted = (
                        predicted * normalization["target_std"]
                        + normalization["target_mean"]
                    )
                    candidate_values[legal_rows, target_index] = predicted
                target_predictions.append(candidate_values)

            q_engagement = torch.stack(engagement_predictions)
            q_targets = torch.stack(target_predictions)
            engage_probability = diagnostics["engage_probability"][:, unit_index]
            selected_engage = diagnostics["selected_engage"][:, unit_index].long()
            engagement_baseline = (
                (1.0 - engage_probability)[None, :] * q_engagement[:, :, 0]
                + engage_probability[None, :] * q_engagement[:, :, 1]
            )
            selected_engagement_values = torch.gather(
                q_engagement,
                dim=2,
                index=selected_engage[None, :, None].expand(
                    q_engagement.shape[0], -1, 1
                ),
            ).squeeze(2)
            critic_engagement_advantages = (
                selected_engagement_values - engagement_baseline
            )
            engagement_advantages[:, unit_index] = (
                critic_engagement_advantages.mean(dim=0)
            )
            engagement_reliability[:, unit_index] = self._ensemble_reliability(
                critic_engagement_advantages
            )

            conditional_target_probabilities = torch.where(
                engage_probability[:, None] > 0.0,
                probabilities[:, unit_index, :num_targets]
                / engage_probability[:, None].clamp_min(1e-12),
                torch.zeros_like(q_targets[0]),
            )
            target_baseline = torch.sum(
                conditional_target_probabilities[None, :, :] * q_targets,
                dim=2,
            )
            selected_actions = actions[:, unit_index]
            engaged = selected_actions != noop_action
            safe_actions = selected_actions.clamp_max(num_targets - 1)
            chosen_target_values = torch.gather(
                q_targets,
                dim=2,
                index=safe_actions[None, :, None].expand(
                    q_targets.shape[0], -1, 1
                ),
            ).squeeze(2)
            critic_target_advantages = chosen_target_values - target_baseline
            target_advantages[:, unit_index] = torch.where(
                engaged,
                critic_target_advantages.mean(dim=0),
                torch.zeros_like(target_baseline[0]),
            )
            target_reliability[:, unit_index] = torch.where(
                engaged,
                self._ensemble_reliability(critic_target_advantages),
                torch.zeros_like(target_baseline[0]),
            )
            unit_engagement_support, unit_target_support = (
                self._context_support_scores(
                    observations,
                    unit_indices,
                    safe_actions,
                    prefix_occupancy,
                    legal_mask,
                    engaged,
                )
            )
            engagement_support[:, unit_index] = unit_engagement_support
            target_support[:, unit_index] = unit_target_support

            selected_target = engaged
            prefix_occupancy[batch_indices[selected_target], safe_actions[selected_target]] = 1.0

        actionable = diagnostics["actionable"].bool()
        engaged = diagnostics["selected_engage"].bool()
        engagement_advantages = self._normalize_valid(
            engagement_advantages, actionable
        )
        target_advantages = self._normalize_valid(target_advantages, engaged)
        return CounterfactualAdvantageBatch(
            engagement=engagement_advantages,
            target=target_advantages,
            engagement_reliability=engagement_reliability,
            target_reliability=target_reliability,
            engagement_support=engagement_support,
            target_support=target_support,
            actionable=actionable,
            engaged=engaged,
        )

    def _context_support_scores(
        self,
        observations: torch.Tensor,
        unit_indices: torch.Tensor,
        selected_actions: torch.Tensor,
        prefix_occupancy: torch.Tensor,
        legal_action_masks: torch.Tensor,
        engaged: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del unit_indices, selected_actions, prefix_occupancy, legal_action_masks
        engagement_support = torch.ones(
            observations.shape[0], device=observations.device, dtype=observations.dtype
        )
        return engagement_support, engaged.to(observations.dtype)

    def _actor_advantages(
        self,
        rollout_advantages: torch.Tensor,
        counterfactual: CounterfactualAdvantageBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        del rollout_advantages
        diagnostics = {
            "engagement_reliability": self._valid_mean(
                counterfactual.engagement_reliability,
                counterfactual.actionable,
            ),
            "target_reliability": self._valid_mean(
                counterfactual.target_reliability,
                counterfactual.engaged,
            ),
            "engagement_residual_abs": self._valid_mean(
                counterfactual.engagement.abs(), counterfactual.actionable
            ),
            "target_residual_abs": self._valid_mean(
                counterfactual.target.abs(), counterfactual.engaged
            ),
            "engagement_gate_active_rate": 1.0,
            "target_gate_active_rate": 1.0,
            "engagement_support": self._valid_mean(
                counterfactual.engagement_support, counterfactual.actionable
            ),
            "target_support": self._valid_mean(
                counterfactual.target_support, counterfactual.engaged
            ),
        }
        return counterfactual.engagement, counterfactual.target, diagnostics

    @staticmethod
    def _valid_mean(values: torch.Tensor, valid: torch.Tensor) -> float:
        selected = values[valid]
        return float(selected.mean().item()) if selected.numel() else 0.0

    def train(self) -> None:
        if not self._q_critics:
            self._load_q_critics()
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        old_policy = deepcopy(self.policy).to(self.device)
        old_policy.set_training_mode(False)
        for parameter in old_policy.parameters():
            parameter.requires_grad_(False)

        entropy_losses: list[float] = []
        policy_losses: list[float] = []
        engagement_losses: list[float] = []
        target_losses: list[float] = []
        value_losses: list[float] = []
        engagement_clip_fractions: list[float] = []
        target_clip_fractions: list[float] = []
        engagement_valid_rates: list[float] = []
        target_valid_rates: list[float] = []
        engagement_reliabilities: list[float] = []
        target_reliabilities: list[float] = []
        engagement_residuals: list[float] = []
        target_residuals: list[float] = []
        engagement_gate_rates: list[float] = []
        target_gate_rates: list[float] = []
        engagement_supports: list[float] = []
        target_supports: list[float] = []
        anchor_kls: list[float] = []
        anchor_penalties: list[float] = []
        anchor_excess_rates: list[float] = []
        all_approx_kl_divs: list[float] = []
        continue_training = True
        loss = torch.zeros((), device=self.device)

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions.long().reshape(
                    -1, self.policy.observation_layout.num_units
                )
                with torch.no_grad():
                    old_distribution = old_policy.get_distribution(
                        rollout_data.observations,
                        action_masks=rollout_data.action_masks,
                    )
                    old_diagnostics = old_distribution.hierarchical_diagnostics(
                        actions
                    )
                    anchor_diagnostics = self._anchor_diagnostics(
                        rollout_data.observations,
                        actions,
                        rollout_data.action_masks,
                        old_diagnostics,
                    )
                    counterfactual = self._counterfactual_advantages(
                        rollout_data.observations,
                        actions,
                        old_distribution,
                    )
                    (
                        engagement_advantages,
                        target_advantages,
                        advantage_diagnostics,
                    ) = self._actor_advantages(
                        rollout_data.advantages,
                        counterfactual,
                    )
                    actionable = counterfactual.actionable
                    engaged = counterfactual.engaged

                new_distribution = self.policy.get_distribution(
                    rollout_data.observations,
                    action_masks=rollout_data.action_masks,
                )
                new_evaluation = new_distribution.evaluate(actions)
                new_diagnostics = new_distribution.hierarchical_diagnostics(actions)
                values = self.policy.predict_values(
                    rollout_data.observations
                ).flatten()

                engagement_ratio = torch.exp(
                    new_diagnostics["engagement_log_prob"]
                    - old_diagnostics["engagement_log_prob"]
                )
                target_ratio = torch.exp(
                    new_diagnostics["target_log_prob"]
                    - old_diagnostics["target_log_prob"]
                )
                engagement_surrogate = torch.minimum(
                    engagement_advantages * engagement_ratio,
                    engagement_advantages
                    * torch.clamp(engagement_ratio, 1 - clip_range, 1 + clip_range),
                )
                target_surrogate = torch.minimum(
                    target_advantages * target_ratio,
                    target_advantages
                    * torch.clamp(target_ratio, 1 - clip_range, 1 + clip_range),
                )
                engagement_loss = -engagement_surrogate[actionable].mean()
                if bool(torch.any(engaged)):
                    target_loss = -target_surrogate[engaged].mean()
                else:
                    target_loss = new_diagnostics["target_log_prob"].sum() * 0.0
                policy_loss = (
                    self.engagement_loss_coef * engagement_loss
                    + self.target_loss_coef * target_loss
                )
                regularization_loss, regularization_diagnostics = (
                    self._policy_regularization(
                        new_diagnostics,
                        old_diagnostics,
                        anchor_diagnostics,
                        actionable,
                    )
                )
                policy_loss = policy_loss + regularization_loss

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                entropy_loss = -new_evaluation.entropy.mean()
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with torch.no_grad():
                    log_ratio = new_evaluation.log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean(
                        (torch.exp(log_ratio) - 1) - log_ratio
                    ).item()

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

                policy_losses.append(policy_loss.item())
                engagement_losses.append(engagement_loss.item())
                target_losses.append(target_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                engagement_clip_fractions.append(
                    ((torch.abs(engagement_ratio - 1) > clip_range) & actionable)
                    .float()
                    .sum()
                    .div(actionable.float().sum().clamp_min(1.0))
                    .item()
                )
                target_clip_fractions.append(
                    ((torch.abs(target_ratio - 1) > clip_range) & engaged)
                    .float()
                    .sum()
                    .div(engaged.float().sum().clamp_min(1.0))
                    .item()
                )
                engagement_valid_rates.append(actionable.float().mean().item())
                target_valid_rates.append(engaged.float().mean().item())
                engagement_reliabilities.append(
                    advantage_diagnostics["engagement_reliability"]
                )
                target_reliabilities.append(
                    advantage_diagnostics["target_reliability"]
                )
                engagement_residuals.append(
                    advantage_diagnostics["engagement_residual_abs"]
                )
                target_residuals.append(
                    advantage_diagnostics["target_residual_abs"]
                )
                engagement_gate_rates.append(
                    advantage_diagnostics["engagement_gate_active_rate"]
                )
                target_gate_rates.append(
                    advantage_diagnostics["target_gate_active_rate"]
                )
                engagement_supports.append(
                    advantage_diagnostics["engagement_support"]
                )
                target_supports.append(advantage_diagnostics["target_support"])
                anchor_kls.append(regularization_diagnostics["anchor_kl"])
                anchor_penalties.append(
                    regularization_diagnostics["anchor_penalty"]
                )
                anchor_excess_rates.append(
                    regularization_diagnostics["anchor_excess_rate"]
                )
                all_approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        self.last_mch_training_diagnostics = {
            "mch_engagement_reliability": float(
                np.mean(engagement_reliabilities)
            ),
            "mch_target_reliability": float(np.mean(target_reliabilities)),
            "mch_engagement_residual_abs": float(
                np.mean(engagement_residuals)
            ),
            "mch_target_residual_abs": float(np.mean(target_residuals)),
            "mch_engagement_gate_active_rate": float(
                np.mean(engagement_gate_rates)
            ),
            "mch_target_gate_active_rate": float(np.mean(target_gate_rates)),
            "mch_engagement_support": float(np.mean(engagement_supports)),
            "mch_target_support": float(np.mean(target_supports)),
            "mch_anchor_kl": float(np.mean(anchor_kls)),
            "mch_anchor_penalty": float(np.mean(anchor_penalties)),
            "mch_anchor_excess_rate": float(np.mean(anchor_excess_rates)),
        }
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(policy_losses))
        self.logger.record("train/mch_engagement_loss", np.mean(engagement_losses))
        self.logger.record("train/mch_target_loss", np.mean(target_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(all_approx_kl_divs))
        self.logger.record(
            "train/mch_engagement_clip_fraction",
            np.mean(engagement_clip_fractions),
        )
        self.logger.record(
            "train/mch_target_clip_fraction", np.mean(target_clip_fractions)
        )
        self.logger.record(
            "train/mch_actionable_factor_rate", np.mean(engagement_valid_rates)
        )
        self.logger.record(
            "train/mch_engaged_factor_rate", np.mean(target_valid_rates)
        )
        self.logger.record(
            "train/mch_engagement_reliability",
            self.last_mch_training_diagnostics["mch_engagement_reliability"],
        )
        self.logger.record(
            "train/mch_target_reliability",
            self.last_mch_training_diagnostics["mch_target_reliability"],
        )
        self.logger.record(
            "train/mch_engagement_residual_abs",
            self.last_mch_training_diagnostics["mch_engagement_residual_abs"],
        )
        self.logger.record(
            "train/mch_target_residual_abs",
            self.last_mch_training_diagnostics["mch_target_residual_abs"],
        )
        self.logger.record(
            "train/mch_engagement_gate_active_rate",
            self.last_mch_training_diagnostics[
                "mch_engagement_gate_active_rate"
            ],
        )
        self.logger.record(
            "train/mch_target_gate_active_rate",
            self.last_mch_training_diagnostics["mch_target_gate_active_rate"],
        )
        for name in (
            "mch_engagement_support",
            "mch_target_support",
            "mch_anchor_kl",
            "mch_anchor_penalty",
            "mch_anchor_excess_rate",
        ):
            self.logger.record(
                f"train/{name}", self.last_mch_training_diagnostics[name]
            )
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def _anchor_diagnostics(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
        old_diagnostics: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        del observations, actions, action_masks
        return old_diagnostics

    def _policy_regularization(
        self,
        new_diagnostics: dict[str, torch.Tensor],
        old_diagnostics: dict[str, torch.Tensor],
        anchor_diagnostics: dict[str, torch.Tensor],
        actionable: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del old_diagnostics, anchor_diagnostics, actionable
        zero = new_diagnostics["engage_probability"].sum() * 0.0
        return zero, {
            "anchor_kl": 0.0,
            "anchor_penalty": 0.0,
            "anchor_excess_rate": 0.0,
        }


class ReliabilityGatedMCHPPO(MaskedCounterfactualHierarchicalPPO):
    """MCH-PPO retaining GAE and adding bounded reliable CF residuals."""

    def __init__(
        self,
        policy: Any,
        env: Any,
        *args: Any,
        engagement_residual_coef: float = 0.5,
        target_residual_coef: float = 0.5,
        residual_clip: float = 0.5,
        reliability_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        if engagement_residual_coef < 0.0 or target_residual_coef < 0.0:
            raise ValueError("RG-MCH residual coefficients must be non-negative")
        if residual_clip <= 0.0:
            raise ValueError("RG-MCH residual_clip must be positive")
        if not 0.0 <= reliability_threshold <= 1.0:
            raise ValueError("RG-MCH reliability_threshold must be in [0, 1]")
        self.engagement_residual_coef = float(engagement_residual_coef)
        self.target_residual_coef = float(target_residual_coef)
        self.residual_clip = float(residual_clip)
        self.reliability_threshold = float(reliability_threshold)
        super().__init__(policy, env, *args, **kwargs)
        self.action_generator_signature = {
            **self.action_generator_signature,
            "optimizer": {
                **self.action_generator_signature["optimizer"],
                "type": "reliability_gated_mch_ppo",
                "engagement_residual_coef": self.engagement_residual_coef,
                "target_residual_coef": self.target_residual_coef,
                "residual_clip": self.residual_clip,
                "reliability_threshold": self.reliability_threshold,
                "base_advantage": "normalized_on_policy_gae",
            },
        }

    def _actor_advantages(
        self,
        rollout_advantages: torch.Tensor,
        counterfactual: CounterfactualAdvantageBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        normalized_gae = self._normalize_vector(rollout_advantages)
        base = normalized_gae[:, None].expand_as(counterfactual.engagement)
        engagement_residual = torch.clamp(
            self.engagement_residual_coef
            * counterfactual.engagement_reliability
            * counterfactual.engagement,
            -self.residual_clip,
            self.residual_clip,
        )
        target_residual = torch.clamp(
            self.target_residual_coef
            * counterfactual.target_reliability
            * counterfactual.target,
            -self.residual_clip,
            self.residual_clip,
        )
        engagement = self._normalize_valid(
            base + engagement_residual, counterfactual.actionable
        )
        target = self._normalize_valid(
            base + target_residual, counterfactual.engaged
        )
        engagement_gate = (
            counterfactual.engagement_reliability >= self.reliability_threshold
        )
        target_gate = (
            counterfactual.target_reliability >= self.reliability_threshold
        )
        diagnostics = {
            "engagement_reliability": self._valid_mean(
                counterfactual.engagement_reliability,
                counterfactual.actionable,
            ),
            "target_reliability": self._valid_mean(
                counterfactual.target_reliability,
                counterfactual.engaged,
            ),
            "engagement_residual_abs": self._valid_mean(
                engagement_residual.abs(), counterfactual.actionable
            ),
            "target_residual_abs": self._valid_mean(
                target_residual.abs(), counterfactual.engaged
            ),
            "engagement_gate_active_rate": self._valid_mean(
                engagement_gate.float(), counterfactual.actionable
            ),
            "target_gate_active_rate": self._valid_mean(
                target_gate.float(), counterfactual.engaged
            ),
            "engagement_support": self._valid_mean(
                counterfactual.engagement_support, counterfactual.actionable
            ),
            "target_support": self._valid_mean(
                counterfactual.target_support, counterfactual.engaged
            ),
        }
        return engagement, target, diagnostics

    @staticmethod
    def _normalize_vector(values: torch.Tensor) -> torch.Tensor:
        centered = values - values.mean()
        scale = values.std(unbiased=False)
        return centered / (scale + 1e-8) if bool(scale > 1e-8) else centered


class SupportAnchoredRGMCHPPO(ReliabilityGatedMCHPPO):
    """RG-MCH with train-support reliability and cumulative actor anchoring."""

    def __init__(
        self,
        policy: Any,
        env: Any,
        *args: Any,
        support_dataset_path: str | Path | None = None,
        anchor_kl_budget: float = 0.10,
        anchor_kl_coef: float = 1.0,
        **kwargs: Any,
    ) -> None:
        deferred_load = kwargs.get("_init_setup_model") is False
        if support_dataset_path is None and not deferred_load:
            raise ValueError("SA-RG-MCH requires a support_dataset_path")
        if anchor_kl_budget < 0.0 or anchor_kl_coef < 0.0:
            raise ValueError("SA-RG-MCH anchor parameters must be non-negative")
        self.support_dataset_path = (
            str(Path(support_dataset_path).resolve())
            if support_dataset_path is not None
            else ""
        )
        self.anchor_kl_budget = float(anchor_kl_budget)
        self.anchor_kl_coef = float(anchor_kl_coef)
        self._support_index: MaskedContextSupportIndex | None = None
        self._engagement_anchor_policy: FactorizedEngagementActorCriticPolicy | None = None
        super().__init__(policy, env, *args, **kwargs)
        if self.support_dataset_path:
            self._load_support_index()
        self._ensure_engagement_anchor()
        self.action_generator_signature = {
            **self.action_generator_signature,
            "optimizer": {
                **self.action_generator_signature["optimizer"],
                "type": "support_anchored_reliability_gated_mch_ppo",
                "support": (
                    self._support_index.signature()
                    if self._support_index is not None
                    else {"dataset_path": self.support_dataset_path}
                ),
                "combined_reliability": "ensemble_agreement_times_context_support",
                "anchor_kl_budget": self.anchor_kl_budget,
                "anchor_kl_coef": self.anchor_kl_coef,
            },
        }

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + [
            "_support_index",
            "_engagement_anchor_policy",
        ]

    def _load_support_index(self) -> None:
        self._support_index = MaskedContextSupportIndex.from_npz(
            self.support_dataset_path,
            num_units=self.policy.observation_layout.num_units,
            device=self.device,
        )

    def _ensure_engagement_anchor(self) -> None:
        if self._engagement_anchor_policy is not None:
            return
        policy = getattr(self, "policy", None)
        if not isinstance(policy, FactorizedEngagementActorCriticPolicy):
            return
        self._engagement_anchor_policy = deepcopy(policy).to(self.device)
        self._engagement_anchor_policy.set_training_mode(False)
        for parameter in self._engagement_anchor_policy.parameters():
            parameter.requires_grad_(False)

    def _context_support_scores(
        self,
        observations: torch.Tensor,
        unit_indices: torch.Tensor,
        selected_actions: torch.Tensor,
        prefix_occupancy: torch.Tensor,
        legal_action_masks: torch.Tensor,
        engaged: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._support_index is None:
            self._load_support_index()
        assert self._support_index is not None
        engagement_support = self._support_index.engagement_scores(
            observations,
            unit_indices,
            prefix_occupancy,
            legal_action_masks,
        )
        target_support = self._support_index.target_scores(
            observations,
            unit_indices,
            selected_actions,
            prefix_occupancy,
            legal_action_masks,
        )
        return engagement_support, torch.where(
            engaged, target_support, torch.zeros_like(target_support)
        )

    def _actor_advantages(
        self,
        rollout_advantages: torch.Tensor,
        counterfactual: CounterfactualAdvantageBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        ensemble_engagement = counterfactual.engagement_reliability
        ensemble_target = counterfactual.target_reliability
        supported = replace(
            counterfactual,
            engagement_reliability=(
                ensemble_engagement * counterfactual.engagement_support
            ),
            target_reliability=ensemble_target * counterfactual.target_support,
        )
        engagement, target, diagnostics = super()._actor_advantages(
            rollout_advantages, supported
        )
        diagnostics.update(
            {
                "ensemble_engagement_reliability": self._valid_mean(
                    ensemble_engagement, counterfactual.actionable
                ),
                "ensemble_target_reliability": self._valid_mean(
                    ensemble_target, counterfactual.engaged
                ),
            }
        )
        return engagement, target, diagnostics

    def _anchor_diagnostics(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
        old_diagnostics: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        del old_diagnostics
        self._ensure_engagement_anchor()
        assert self._engagement_anchor_policy is not None
        with torch.no_grad():
            distribution = self._engagement_anchor_policy.get_distribution(
                observations, action_masks=action_masks
            )
            return distribution.hierarchical_diagnostics(actions)

    def _policy_regularization(
        self,
        new_diagnostics: dict[str, torch.Tensor],
        old_diagnostics: dict[str, torch.Tensor],
        anchor_diagnostics: dict[str, torch.Tensor],
        actionable: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del old_diagnostics
        epsilon = 1e-6
        anchor_probability = anchor_diagnostics["engage_probability"].clamp(
            epsilon, 1.0 - epsilon
        )
        current_probability = new_diagnostics["engage_probability"].clamp(
            epsilon, 1.0 - epsilon
        )
        kl = (
            anchor_probability
            * torch.log(anchor_probability / current_probability)
            + (1.0 - anchor_probability)
            * torch.log(
                (1.0 - anchor_probability) / (1.0 - current_probability)
            )
        )
        selected = kl[actionable]
        if selected.numel() == 0:
            zero = kl.sum() * 0.0
            return zero, {
                "anchor_kl": 0.0,
                "anchor_penalty": 0.0,
                "anchor_excess_rate": 0.0,
            }
        excess = torch.relu(selected - self.anchor_kl_budget)
        penalty = self.anchor_kl_coef * torch.square(excess).mean()
        return penalty, {
            "anchor_kl": float(selected.mean().detach().item()),
            "anchor_penalty": float(penalty.detach().item()),
            "anchor_excess_rate": float(
                (selected > self.anchor_kl_budget).float().mean().detach().item()
            ),
        }
