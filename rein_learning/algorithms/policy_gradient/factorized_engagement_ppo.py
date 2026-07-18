from __future__ import annotations

from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.type_aliases import PyTorchObs
from torch import nn

from ...models import (
    AirDefenseV1ObservationLayout,
    FactorizedEngagementActionHeadConfig,
    FactorizedEngagementAirDefenseActionHead,
    FactorizedEngagementAutoregressiveDistribution,
)
from .autoregressive_ppo import (
    AutoregressiveMaskableActorCriticPolicy,
    autoregressive_action_generator_signature,
)
from .role_conditioned_autoregressive_ppo import policy_parameter_counts


def factorized_engagement_action_generator_signature(
    layout: AirDefenseV1ObservationLayout,
    unit_order: tuple[int, ...] | list[int] | None = None,
    head_config: FactorizedEngagementActionHeadConfig | None = None,
) -> dict[str, object]:
    signature = autoregressive_action_generator_signature(
        layout.num_units, unit_order
    )
    signature["type"] = "factorized_engagement_autoregressive_conflict_free"
    signature["probability_schema"] = {
        "noop": "1-sigmoid(engage_logit)",
        "target": "sigmoid(engage_logit)*softmax(legal_target_logits)",
        "entropy": "exact_final_discrete_distribution",
        "deterministic_rule": "bernoulli_argmax_then_target_argmax",
    }
    signature["actor_head"] = (
        head_config or FactorizedEngagementActionHeadConfig()
    ).signature()
    signature["observation_layout"] = layout.signature()
    return signature


class FactorizedEngagementActorCriticPolicy(
    AutoregressiveMaskableActorCriticPolicy
):
    """Role-conditioned policy separating engagement from target selection."""

    def __init__(
        self,
        *args: Any,
        unit_order: tuple[int, ...] | list[int] | None = None,
        entity_embedding_dim: int = 32,
        context_dim: int = 96,
        relation_hidden_dim: int = 64,
        initial_engage_bias: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.factorized_head_config = FactorizedEngagementActionHeadConfig(
            entity_embedding_dim=entity_embedding_dim,
            context_dim=context_dim,
            relation_hidden_dim=relation_hidden_dim,
            initial_engage_bias=initial_engage_bias,
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
        self.action_generator_signature = (
            factorized_engagement_action_generator_signature(
                self.observation_layout,
                self.unit_order,
                self.factorized_head_config,
            )
        )

    def _build(self, lr_schedule: Any) -> None:
        self._build_mlp_extractor()
        self.observation_layout = AirDefenseV1ObservationLayout.infer(
            self.observation_space, self.action_space
        )
        self.action_net = FactorizedEngagementAirDefenseActionHead(
            self.observation_layout, self.factorized_head_config
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
            self.action_net.engage_output.apply(
                partial(self.init_weights, gain=0.01)
            )
        nn.init.constant_(
            self.action_net.engage_output.bias,
            self.factorized_head_config.initial_engage_bias,
        )

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(asdict_factorized_config(self.factorized_head_config))
        return data

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, latent_vf = self._latent_features(obs)
        values = self.value_net(latent_vf)
        evaluation = self._factorized_distribution(obs, action_masks).sample(
            deterministic=deterministic
        )
        return evaluation.actions, values, evaluation.log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, latent_vf = self._latent_features(obs)
        evaluation = self._factorized_distribution(obs, action_masks).evaluate(
            actions
        )
        return self.value_net(latent_vf), evaluation.log_prob, evaluation.entropy

    def get_distribution(
        self,
        obs: PyTorchObs,
        action_masks: np.ndarray | torch.Tensor | None = None,
    ) -> FactorizedEngagementAutoregressiveDistribution:
        return self._factorized_distribution(obs, action_masks)

    def _predict(
        self,
        observation: PyTorchObs,
        deterministic: bool = False,
        action_masks: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.get_distribution(observation, action_masks).get_actions(
            deterministic=deterministic
        )

    def _factorized_distribution(
        self,
        observation: torch.Tensor,
        action_masks: np.ndarray | torch.Tensor | None,
    ) -> FactorizedEngagementAutoregressiveDistribution:
        if action_masks is None:
            raise ValueError("Factorized policy requires base action masks")
        target_logits, engage_logits = self.action_net(observation, action_masks)
        return FactorizedEngagementAutoregressiveDistribution(
            target_logits,
            engage_logits,
            self._action_dims,
            action_masks,
            unit_order=self.unit_order,
        )

    def parameter_counts(self) -> dict[str, int]:
        return policy_parameter_counts(self)


def asdict_factorized_config(
    config: FactorizedEngagementActionHeadConfig,
) -> dict[str, int | float]:
    return {
        "entity_embedding_dim": config.entity_embedding_dim,
        "context_dim": config.context_dim,
        "relation_hidden_dim": config.relation_hidden_dim,
        "initial_engage_bias": config.initial_engage_bias,
    }


class FactorizedEngagementMaskablePPO(MaskablePPO):
    """Maskable PPO using the factorized engagement policy."""

    def __init__(self, policy: Any, env: Any, *args: Any, **kwargs: Any) -> None:
        if policy == "MlpPolicy":
            policy = FactorizedEngagementActorCriticPolicy
        if policy is not FactorizedEngagementActorCriticPolicy:
            raise ValueError(
                "FactorizedEngagementMaskablePPO requires its factorized policy"
            )
        self.action_generator_signature: dict[str, object] = {}
        super().__init__(policy, env, *args, **kwargs)
        if hasattr(self, "policy") and isinstance(
            self.policy, FactorizedEngagementActorCriticPolicy
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
    ) -> "FactorizedEngagementMaskablePPO":
        model = super().load(path, env=env, **kwargs)
        if not isinstance(model.policy, FactorizedEngagementActorCriticPolicy):
            raise ValueError("Saved model does not contain the factorized policy")
        if model.action_generator_signature != model.policy.action_generator_signature:
            raise ValueError("Saved model has an incompatible action generator signature")
        return model
