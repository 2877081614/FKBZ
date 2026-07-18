from __future__ import annotations

from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.type_aliases import PyTorchObs
from torch import nn

from ...models import (
    AirDefenseV1ObservationLayout,
    AutoregressiveMaskedMultiCategorical,
    RoleConditionedActionHeadConfig,
    RoleConditionedAirDefenseActionHead,
)
from .autoregressive_ppo import (
    AutoregressiveMaskableActorCriticPolicy,
    autoregressive_action_generator_signature,
)


def role_conditioned_action_generator_signature(
    layout: AirDefenseV1ObservationLayout,
    unit_order: tuple[int, ...] | list[int] | None = None,
    head_config: RoleConditionedActionHeadConfig | None = None,
) -> dict[str, object]:
    signature = autoregressive_action_generator_signature(
        layout.num_units,
        unit_order,
    )
    signature["type"] = "role_conditioned_autoregressive_conflict_free"
    signature["actor_head"] = (
        head_config or RoleConditionedActionHeadConfig()
    ).signature()
    signature["observation_layout"] = layout.signature()
    return signature


def policy_parameter_counts(policy: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in policy.parameters())
    mlp_extractor = getattr(policy, "mlp_extractor", None)
    action_net = getattr(policy, "action_net", None)
    value_net = getattr(policy, "value_net", None)
    actor_modules = [
        getattr(mlp_extractor, "policy_net", None),
        action_net,
    ]
    critic_modules = [
        getattr(mlp_extractor, "value_net", None),
        value_net,
    ]
    actor = sum(
        parameter.numel()
        for module in actor_modules
        if module is not None
        for parameter in module.parameters()
    )
    critic = sum(
        parameter.numel()
        for module in critic_modules
        if module is not None
        for parameter in module.parameters()
    )
    return {
        "actor_parameters": int(actor),
        "critic_parameters": int(critic),
        "shared_parameters": int(total - actor - critic),
        "total_parameters": int(total),
    }


class RoleConditionedAutoregressiveActorCriticPolicy(
    AutoregressiveMaskableActorCriticPolicy
):
    """Autoregressive policy with a shared unit-target relation actor."""

    def __init__(
        self,
        *args: Any,
        unit_order: tuple[int, ...] | list[int] | None = None,
        entity_embedding_dim: int = 32,
        context_dim: int = 96,
        relation_hidden_dim: int = 64,
        **kwargs: Any,
    ) -> None:
        self.role_head_config = RoleConditionedActionHeadConfig(
            entity_embedding_dim=entity_embedding_dim,
            context_dim=context_dim,
            relation_hidden_dim=relation_hidden_dim,
        )
        requested_net_arch = kwargs.get("net_arch")
        if isinstance(requested_net_arch, dict):
            critic_arch = list(requested_net_arch.get("vf", (128, 128)))
        elif requested_net_arch is None:
            critic_arch = [128, 128]
        else:
            critic_arch = list(requested_net_arch)
        kwargs["net_arch"] = {"pi": [], "vf": critic_arch}
        super().__init__(*args, unit_order=unit_order, **kwargs)
        self.action_generator_signature = role_conditioned_action_generator_signature(
            self.observation_layout,
            self.unit_order,
            self.role_head_config,
        )

    def _build(self, lr_schedule: Any) -> None:
        self._build_mlp_extractor()
        self.observation_layout = AirDefenseV1ObservationLayout.infer(
            self.observation_space,
            self.action_space,
        )
        self.action_net = RoleConditionedAirDefenseActionHead(
            self.observation_layout,
            self.role_head_config,
        )
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: np.sqrt(2),
                self.value_net: 1.0,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))
            self.action_net.pair_output.apply(
                partial(self.init_weights, gain=0.01)
            )
            self.action_net.noop_output.apply(
                partial(self.init_weights, gain=0.01)
            )

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            {
                "entity_embedding_dim": self.role_head_config.entity_embedding_dim,
                "context_dim": self.role_head_config.context_dim,
                "relation_hidden_dim": self.role_head_config.relation_hidden_dim,
            }
        )
        return data

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, latent_vf = self._latent_features(obs)
        values = self.value_net(latent_vf)
        distribution = self._role_distribution(obs, action_masks)
        evaluation = distribution.sample(deterministic=deterministic)
        return evaluation.actions, values, evaluation.log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, latent_vf = self._latent_features(obs)
        distribution = self._role_distribution(obs, action_masks)
        evaluation = distribution.evaluate(actions)
        return self.value_net(latent_vf), evaluation.log_prob, evaluation.entropy

    def get_distribution(
        self,
        obs: PyTorchObs,
        action_masks: np.ndarray | torch.Tensor | None = None,
    ) -> AutoregressiveMaskedMultiCategorical:
        return self._role_distribution(obs, action_masks)

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

    def _role_distribution(
        self,
        observation: torch.Tensor,
        action_masks: np.ndarray | torch.Tensor | None,
    ) -> AutoregressiveMaskedMultiCategorical:
        if action_masks is None:
            raise ValueError("Role-conditioned policy requires base action masks")
        logits = self.action_net(observation, action_masks)
        return AutoregressiveMaskedMultiCategorical(
            logits,
            self._action_dims,
            action_masks,
            unit_order=self.unit_order,
        )

    def parameter_counts(self) -> dict[str, int]:
        return policy_parameter_counts(self)


class RoleConditionedAutoregressiveMaskablePPO(MaskablePPO):
    """Maskable PPO using the role-conditioned autoregressive policy."""

    def __init__(self, policy: Any, env: Any, *args: Any, **kwargs: Any) -> None:
        if policy == "MlpPolicy":
            policy = RoleConditionedAutoregressiveActorCriticPolicy
        if policy is not RoleConditionedAutoregressiveActorCriticPolicy:
            raise ValueError(
                "RoleConditionedAutoregressiveMaskablePPO requires its "
                "role-conditioned policy"
            )
        self.action_generator_signature: dict[str, object] = {}
        super().__init__(policy, env, *args, **kwargs)
        if hasattr(self, "policy") and isinstance(
            self.policy,
            RoleConditionedAutoregressiveActorCriticPolicy,
        ):
            self.action_generator_signature = deepcopy(
                self.policy.action_generator_signature
            )

    @classmethod
    def load(
        cls,
        path: str | Path,
        env: Any | None = None,
        **kwargs: Any,
    ) -> "RoleConditionedAutoregressiveMaskablePPO":
        model = super().load(path, env=env, **kwargs)
        if not isinstance(
            model.policy,
            RoleConditionedAutoregressiveActorCriticPolicy,
        ):
            raise ValueError("Saved model does not contain the role-conditioned policy")
        if model.action_generator_signature != model.policy.action_generator_signature:
            raise ValueError("Saved model has an incompatible action generator signature")
        return model
