from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .air_defense_observation_layout import AirDefenseV1ObservationLayout


@dataclass(frozen=True)
class RiskAwareEngagementCriticConfig:
    hidden_dims: tuple[int, ...] = (256, 128)

    def __post_init__(self) -> None:
        if not self.hidden_dims or min(self.hidden_dims) <= 0:
            raise ValueError("hidden_dims must contain positive values")

    def signature(self) -> dict[str, object]:
        return {"type": "risk_aware_engagement_critic_mlp", **asdict(self)}


class RiskAwareEngagementCritic(nn.Module):
    """Estimate no-op and engage utility for one autoregressive unit context."""

    def __init__(
        self,
        layout: AirDefenseV1ObservationLayout,
        config: RiskAwareEngagementCriticConfig | None = None,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.config = config or RiskAwareEngagementCriticConfig()
        self.num_actions = layout.num_targets + 1
        input_dim = (
            layout.observation_dim
            + layout.num_units
            + layout.unit_feature_dim
            + layout.num_targets
            + self.num_actions
        )
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in self.config.hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), nn.Tanh()))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        observations: torch.Tensor,
        unit_indices: torch.Tensor,
        prefix_occupancy: torch.Tensor,
        legal_action_masks: torch.Tensor,
    ) -> torch.Tensor:
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        batch_size = observations.shape[0]
        unit_indices = unit_indices.long().reshape(-1)
        if unit_indices.shape[0] != batch_size:
            raise ValueError("Unit batch must match observations")
        if bool(torch.any((unit_indices < 0) | (unit_indices >= self.layout.num_units))):
            raise ValueError("unit_indices contain an invalid unit")
        if prefix_occupancy.shape != (batch_size, self.layout.num_targets):
            raise ValueError("prefix_occupancy has the wrong shape")
        if legal_action_masks.shape != (batch_size, self.num_actions):
            raise ValueError("legal_action_masks has the wrong shape")
        if not bool(torch.all(legal_action_masks[:, self.layout.num_targets].bool())):
            raise ValueError("No-op must be legal for every engagement context")

        structured = self.layout.split(observations)
        batch_indices = torch.arange(batch_size, device=observations.device)
        unit_one_hot = torch.nn.functional.one_hot(
            unit_indices, num_classes=self.layout.num_units
        ).to(observations.dtype)
        unit_features = structured.units[batch_indices, unit_indices]
        features = torch.cat(
            (
                observations,
                unit_one_hot,
                unit_features,
                prefix_occupancy.to(observations.dtype),
                legal_action_masks.to(observations.dtype),
            ),
            dim=1,
        )
        return self.network(features)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def signature(self) -> dict[str, object]:
        return {
            **self.config.signature(),
            "observation_layout": self.layout.signature(),
            "num_actions": self.num_actions,
            "parameter_count": self.parameter_count(),
        }
