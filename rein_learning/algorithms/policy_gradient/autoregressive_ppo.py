from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs

from ...models.autoregressive_action_head import (
    AutoregressiveMaskedMultiCategorical,
)


def autoregressive_action_generator_signature(
    num_units: int,
    unit_order: tuple[int, ...] | list[int] | None = None,
) -> dict[str, object]:
    if num_units <= 0:
        raise ValueError("num_units must be positive")
    order = (
        tuple(range(num_units))
        if unit_order is None
        else tuple(int(value) for value in unit_order)
    )
    if len(order) != num_units or set(order) != set(range(num_units)):
        raise ValueError("unit_order must be a permutation of all unit indices")
    return {
        "type": "autoregressive_conflict_free",
        "unit_order": list(order),
        "conditional_target_mask": True,
        "joint_log_prob": "sum_of_conditional_log_probs",
        "environment_steps_per_joint_action": 1,
    }


AUTOREGRESSIVE_ACTION_GENERATOR_SIGNATURE = (
    autoregressive_action_generator_signature(3)
)


class AutoregressiveMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    """Maskable actor-critic policy with prefix-conditioned unit actions."""

    action_generator_signature = AUTOREGRESSIVE_ACTION_GENERATOR_SIGNATURE

    def __init__(
        self,
        *args: Any,
        unit_order: tuple[int, ...] | list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.action_space, spaces.MultiDiscrete):
            raise ValueError("Autoregressive policy requires a MultiDiscrete space")
        self._action_dims = tuple(int(value) for value in self.action_space.nvec)
        if len(set(self._action_dims)) != 1:
            raise ValueError("All per-unit action dimensions must be equal")
        self.unit_order = tuple(
            autoregressive_action_generator_signature(
                len(self._action_dims),
                unit_order,
            )["unit_order"]
        )
        self.action_generator_signature = autoregressive_action_generator_signature(
            len(self._action_dims),
            self.unit_order,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data["unit_order"] = self.unit_order
        return data

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_pi, latent_vf = self._latent_features(obs)
        values = self.value_net(latent_vf)
        distribution = self._autoregressive_distribution(latent_pi, action_masks)
        evaluation = distribution.sample(deterministic=deterministic)
        return evaluation.actions, values, evaluation.log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_pi, latent_vf = self._latent_features(obs)
        distribution = self._autoregressive_distribution(latent_pi, action_masks)
        evaluation = distribution.evaluate(actions)
        return self.value_net(latent_vf), evaluation.log_prob, evaluation.entropy

    def get_distribution(
        self,
        obs: PyTorchObs,
        action_masks: np.ndarray | torch.Tensor | None = None,
    ) -> AutoregressiveMaskedMultiCategorical:
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._autoregressive_distribution(latent_pi, action_masks)

    def _predict(
        self,
        observation: PyTorchObs,
        deterministic: bool = False,
        action_masks: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.get_distribution(
            observation,
            action_masks,
        ).get_actions(deterministic=deterministic)

    def _latent_features(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            return self.mlp_extractor(features)
        pi_features, vf_features = features
        return (
            self.mlp_extractor.forward_actor(pi_features),
            self.mlp_extractor.forward_critic(vf_features),
        )

    def _autoregressive_distribution(
        self,
        latent_pi: torch.Tensor,
        action_masks: np.ndarray | torch.Tensor | None,
    ) -> AutoregressiveMaskedMultiCategorical:
        if action_masks is None:
            raise ValueError("Autoregressive policy requires base action masks")
        return AutoregressiveMaskedMultiCategorical(
            self.action_net(latent_pi),
            self._action_dims,
            action_masks,
            unit_order=self.unit_order,
        )


class AutoregressiveMaskablePPO(MaskablePPO):
    """Maskable PPO whose policy factors one joint action by unit order."""

    expected_action_generator_signature = AUTOREGRESSIVE_ACTION_GENERATOR_SIGNATURE

    def __init__(self, policy: Any, env: Any, *args: Any, **kwargs: Any) -> None:
        if policy == "MlpPolicy":
            policy = AutoregressiveMaskableActorCriticPolicy
        if policy is not AutoregressiveMaskableActorCriticPolicy:
            raise ValueError(
                "AutoregressiveMaskablePPO requires "
                "AutoregressiveMaskableActorCriticPolicy"
            )
        self.action_generator_signature = deepcopy(AUTOREGRESSIVE_ACTION_GENERATOR_SIGNATURE)
        super().__init__(policy, env, *args, **kwargs)
        if hasattr(self, "policy") and isinstance(
            self.policy,
            AutoregressiveMaskableActorCriticPolicy,
        ):
            self.action_generator_signature = deepcopy(
                self.policy.action_generator_signature
            )
        elif hasattr(self, "action_space") and isinstance(
            self.action_space,
            spaces.MultiDiscrete,
        ):
            self.action_generator_signature = (
                autoregressive_action_generator_signature(
                    len(self.action_space.nvec)
                )
            )

    @classmethod
    def load(
        cls,
        path: str | Path,
        env: Any | None = None,
        **kwargs: Any,
    ) -> "AutoregressiveMaskablePPO":
        model = super().load(path, env=env, **kwargs)
        if not isinstance(model.policy, AutoregressiveMaskableActorCriticPolicy):
            raise ValueError("Saved model does not contain the autoregressive policy")
        expected_signature = model.policy.action_generator_signature
        if model.action_generator_signature != expected_signature:
            raise ValueError("Saved model has an incompatible action generator signature")
        return model
