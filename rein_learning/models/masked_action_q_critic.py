from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .air_defense_observation_layout import AirDefenseV1ObservationLayout


@dataclass(frozen=True)
class MaskedActionQCriticConfig:
    hidden_dims: tuple[int, ...] = (256, 128)
    include_entity_features: bool = True
    include_prefix_occupancy: bool = True
    include_legal_mask: bool = True

    def __post_init__(self) -> None:
        if not self.hidden_dims or min(self.hidden_dims) <= 0:
            raise ValueError("hidden_dims must contain positive values")

    def signature(self) -> dict[str, object]:
        return {
            "type": "masked_action_q_critic_mlp",
            **asdict(self),
        }


class MaskedActionQCritic(nn.Module):
    """Estimate Q(s, prefix, unit, candidate) without graph operations."""

    UNIT_POSITION = slice(0, 2)
    TARGET_POSITION = slice(0, 2)

    def __init__(
        self,
        layout: AirDefenseV1ObservationLayout,
        config: MaskedActionQCriticConfig | None = None,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.config = config or MaskedActionQCriticConfig()
        self.num_actions = layout.num_targets + 1

        input_dim = layout.observation_dim + layout.num_units + self.num_actions
        if self.config.include_entity_features:
            input_dim += layout.unit_feature_dim + layout.target_feature_dim + 4
        if self.config.include_prefix_occupancy:
            input_dim += layout.num_targets
        if self.config.include_legal_mask:
            input_dim += self.num_actions

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in self.config.hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), nn.Tanh()))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        observations: torch.Tensor,
        unit_indices: torch.Tensor,
        candidate_actions: torch.Tensor,
        prefix_occupancy: torch.Tensor,
        legal_action_masks: torch.Tensor,
    ) -> torch.Tensor:
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        batch_size = observations.shape[0]
        unit_indices = unit_indices.long().reshape(-1)
        candidate_actions = candidate_actions.long().reshape(-1)
        if unit_indices.shape[0] != batch_size or candidate_actions.shape[0] != batch_size:
            raise ValueError("Unit/action batches must match observations")
        if bool(torch.any((unit_indices < 0) | (unit_indices >= self.layout.num_units))):
            raise ValueError("unit_indices contain an invalid unit")
        if bool(torch.any((candidate_actions < 0) | (candidate_actions >= self.num_actions))):
            raise ValueError("candidate_actions contain an invalid action")
        if prefix_occupancy.shape != (batch_size, self.layout.num_targets):
            raise ValueError("prefix_occupancy has the wrong shape")
        if legal_action_masks.shape != (batch_size, self.num_actions):
            raise ValueError("legal_action_masks has the wrong shape")
        batch_indices = torch.arange(batch_size, device=observations.device)
        if not bool(torch.all(legal_action_masks.bool()[batch_indices, candidate_actions])):
            raise ValueError("Every candidate action must be legal")

        structured = self.layout.split(observations)
        unit_one_hot = torch.nn.functional.one_hot(
            unit_indices, num_classes=self.layout.num_units
        ).to(observations.dtype)
        action_one_hot = torch.nn.functional.one_hot(
            candidate_actions, num_classes=self.num_actions
        ).to(observations.dtype)
        features = [observations, unit_one_hot, action_one_hot]

        if self.config.include_entity_features:
            unit_features = structured.units[batch_indices, unit_indices]
            target_indices = candidate_actions.clamp_max(self.layout.num_targets - 1)
            target_features = structured.targets[batch_indices, target_indices]
            is_noop = candidate_actions == self.layout.num_targets
            target_features = torch.where(
                is_noop[:, None], torch.zeros_like(target_features), target_features
            )
            unit_position = unit_features[:, self.UNIT_POSITION]
            target_position = target_features[:, self.TARGET_POSITION]
            delta = target_position - unit_position
            distance = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
            relative = torch.cat(
                (delta, distance, is_noop.to(observations.dtype)[:, None]), dim=1
            )
            features.extend((unit_features, target_features, relative))
        if self.config.include_prefix_occupancy:
            features.append(prefix_occupancy.to(observations.dtype))
        if self.config.include_legal_mask:
            features.append(legal_action_masks.to(observations.dtype))
        return self.network(torch.cat(features, dim=1)).squeeze(1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def signature(self) -> dict[str, object]:
        return {
            **self.config.signature(),
            "observation_layout": self.layout.signature(),
            "num_actions": self.num_actions,
            "parameter_count": self.parameter_count(),
        }
