from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
from torch import nn

from .air_defense_observation_layout import AirDefenseV1ObservationLayout


class StateConditionedEngagementOutput(NamedTuple):
    safety_gain: torch.Tensor
    cost_delta: torch.Tensor
    budget_multiplier: torch.Tensor
    score: torch.Tensor


@dataclass(frozen=True)
class StateConditionedEngagementValueConfig:
    hidden_dims: tuple[int, ...] = (256, 128)
    budget_mode: str = "state_budget"
    max_budget_multiplier: float = 5.0

    def __post_init__(self) -> None:
        if not self.hidden_dims or min(self.hidden_dims) <= 0:
            raise ValueError("hidden_dims must contain positive values")
        if self.budget_mode not in {"safety_only", "global_budget", "state_budget"}:
            raise ValueError("Unsupported budget_mode")
        if self.max_budget_multiplier <= 0.0:
            raise ValueError("max_budget_multiplier must be positive")

    def signature(self) -> dict[str, object]:
        return {"type": "state_conditioned_engagement_value", **asdict(self)}


class StateConditionedEngagementValue(nn.Module):
    """Estimate paired safety gain, resource cost, and a constrained score."""

    def __init__(
        self,
        layout: AirDefenseV1ObservationLayout,
        config: StateConditionedEngagementValueConfig | None = None,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.config = config or StateConditionedEngagementValueConfig()
        self.num_actions = layout.num_targets + 1
        input_dim = (
            layout.observation_dim
            + layout.num_units
            + layout.unit_feature_dim
            + layout.num_targets
            + self.num_actions
            + 1
        )
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in self.config.hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), nn.Tanh()))
            previous_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.safety_head = nn.Linear(previous_dim, 1)
        self.cost_head = nn.Linear(previous_dim, 1)
        if self.config.budget_mode == "state_budget":
            self.budget_head: nn.Module | None = nn.Linear(previous_dim, 1)
            self.global_budget = None
        elif self.config.budget_mode == "global_budget":
            self.budget_head = None
            self.global_budget = nn.Parameter(torch.tensor(-2.0))
        else:
            self.budget_head = None
            self.global_budget = None

    def forward(
        self,
        observations: torch.Tensor,
        unit_indices: torch.Tensor,
        prefix_occupancy: torch.Tensor,
        legal_action_masks: torch.Tensor,
        margin_logits: torch.Tensor,
    ) -> StateConditionedEngagementOutput:
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        batch_size = observations.shape[0]
        unit_indices = unit_indices.long().reshape(-1)
        margin_logits = margin_logits.reshape(-1).to(observations.dtype)
        if unit_indices.shape[0] != batch_size or margin_logits.shape[0] != batch_size:
            raise ValueError("Context inputs must match observations")
        invalid_units = (unit_indices < 0) | (unit_indices >= self.layout.num_units)
        if bool(torch.any(invalid_units)):
            raise ValueError("unit_indices contain an invalid unit")
        if prefix_occupancy.shape != (batch_size, self.layout.num_targets):
            raise ValueError("prefix_occupancy has the wrong shape")
        if legal_action_masks.shape != (batch_size, self.num_actions):
            raise ValueError("legal_action_masks has the wrong shape")

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
                margin_logits.unsqueeze(1),
            ),
            dim=1,
        )
        hidden = self.encoder(features)
        safety_gain = self.safety_head(hidden).squeeze(1)
        cost_delta = self.cost_head(hidden).squeeze(1)
        if self.config.budget_mode == "state_budget":
            assert self.budget_head is not None
            raw_budget = self.budget_head(hidden).squeeze(1)
            budget = torch.nn.functional.softplus(raw_budget)
        elif self.config.budget_mode == "global_budget":
            assert self.global_budget is not None
            budget = torch.nn.functional.softplus(self.global_budget).expand(batch_size)
        else:
            budget = torch.zeros_like(safety_gain)
        budget = budget.clamp(max=self.config.max_budget_multiplier)
        score = safety_gain - budget * torch.relu(cost_delta)
        return StateConditionedEngagementOutput(
            safety_gain=safety_gain,
            cost_delta=cost_delta,
            budget_multiplier=budget,
            score=score,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def signature(self) -> dict[str, object]:
        return {
            **self.config.signature(),
            "observation_layout": self.layout.signature(),
            "parameter_count": self.parameter_count(),
        }
