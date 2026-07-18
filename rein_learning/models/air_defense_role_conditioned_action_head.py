from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .air_defense_observation_layout import AirDefenseV1ObservationLayout


@dataclass(frozen=True)
class RoleConditionedActionHeadConfig:
    entity_embedding_dim: int = 32
    context_dim: int = 96
    relation_hidden_dim: int = 64

    def __post_init__(self) -> None:
        if min(asdict(self).values()) <= 0:
            raise ValueError("Role-conditioned head dimensions must be positive")

    def signature(self) -> dict[str, object]:
        return {
            "type": "shared_unit_target_relation_mlp",
            **{key: int(value) for key, value in asdict(self).items()},
            "unit_index_embedding": False,
            "shared_noop_head": True,
        }


class RoleConditionedAirDefenseActionHead(nn.Module):
    """Permutation-equivariant shared scorer for AirDefense unit actions."""

    TARGET_ALIVE_INDEX = 13
    UNIT_POSITION = slice(0, 2)
    UNIT_RANGE_INDEX = 6
    TARGET_POSITION = slice(0, 2)

    def __init__(
        self,
        layout: AirDefenseV1ObservationLayout,
        config: RoleConditionedActionHeadConfig | None = None,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.config = config or RoleConditionedActionHeadConfig()
        embedding_dim = self.config.entity_embedding_dim

        self.zone_encoder = nn.Sequential(
            nn.Linear(layout.zone_feature_dim, embedding_dim),
            nn.Tanh(),
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(layout.target_feature_dim, embedding_dim),
            nn.Tanh(),
        )
        self.unit_encoder = nn.Sequential(
            nn.Linear(layout.unit_feature_dim, embedding_dim),
            nn.Tanh(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(layout.global_feature_dim, embedding_dim),
            nn.Tanh(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(4 * embedding_dim, self.config.context_dim),
            nn.Tanh(),
        )

        pair_input_dim = (
            2 * embedding_dim + self.config.context_dim + 4
        )
        self.pair_hidden = nn.Sequential(
            nn.Linear(pair_input_dim, self.config.relation_hidden_dim),
            nn.Tanh(),
        )
        self.pair_output = nn.Linear(self.config.relation_hidden_dim, 1)

        noop_input_dim = 2 * embedding_dim + self.config.context_dim
        self.noop_hidden = nn.Sequential(
            nn.Linear(noop_input_dim, self.config.relation_hidden_dim),
            nn.Tanh(),
        )
        self.noop_output = nn.Linear(self.config.relation_hidden_dim, 1)

    @property
    def num_actions(self) -> int:
        return self.layout.num_targets + 1

    def forward(
        self,
        observation: torch.Tensor,
        action_masks: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        structured = self.layout.split(observation)
        zone_embeddings = self.zone_encoder(structured.zones)
        target_embeddings = self.target_encoder(structured.targets)
        unit_embeddings = self.unit_encoder(structured.units)
        global_embedding = self.global_encoder(structured.global_features)

        alive = structured.targets[:, :, self.TARGET_ALIVE_INDEX] > 0.5
        target_context = self._masked_mean(target_embeddings, alive)
        context = self.context_encoder(
            torch.cat(
                (
                    zone_embeddings.mean(dim=1),
                    target_context,
                    unit_embeddings.mean(dim=1),
                    global_embedding,
                ),
                dim=1,
            )
        )

        batch_size = observation.shape[0]
        units = unit_embeddings.unsqueeze(2).expand(
            -1,
            -1,
            self.layout.num_targets,
            -1,
        )
        targets = target_embeddings.unsqueeze(1).expand(
            -1,
            self.layout.num_units,
            -1,
            -1,
        )
        pair_context = context[:, None, None, :].expand(
            -1,
            self.layout.num_units,
            self.layout.num_targets,
            -1,
        )
        pair_features = torch.cat(
            (
                units,
                targets,
                pair_context,
                self._relative_features(structured.units, structured.targets),
            ),
            dim=3,
        )
        pair_logits = self.pair_output(self.pair_hidden(pair_features)).squeeze(-1)

        masks = torch.as_tensor(
            action_masks,
            device=observation.device,
            dtype=torch.bool,
        ).reshape(-1, self.layout.num_units, self.num_actions)
        if masks.shape[0] == 1 and batch_size > 1:
            masks = masks.expand(batch_size, -1, -1)
        if masks.shape[0] != batch_size:
            raise ValueError("Action-mask batch does not match observation batch")
        legal_target_masks = masks[:, :, : self.layout.num_targets]
        legal_target_context = self._per_unit_masked_target_mean(
            target_embeddings,
            legal_target_masks,
        )
        noop_context = context[:, None, :].expand(
            -1,
            self.layout.num_units,
            -1,
        )
        noop_features = torch.cat(
            (unit_embeddings, legal_target_context, noop_context),
            dim=2,
        )
        noop_logits = self.noop_output(self.noop_hidden(noop_features))
        return torch.cat((pair_logits, noop_logits), dim=2).reshape(batch_size, -1)

    def signature(self) -> dict[str, object]:
        return {
            **self.config.signature(),
            "observation_layout": self.layout.signature(),
        }

    @staticmethod
    def _masked_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = mask.to(values.dtype).unsqueeze(-1)
        totals = (values * weights).sum(dim=1)
        counts = weights.sum(dim=1).clamp_min(1.0)
        return totals / counts

    @staticmethod
    def _per_unit_masked_target_mean(
        target_embeddings: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        expanded_targets = target_embeddings.unsqueeze(1)
        weights = masks.to(target_embeddings.dtype).unsqueeze(-1)
        totals = (expanded_targets * weights).sum(dim=2)
        counts = weights.sum(dim=2).clamp_min(1.0)
        return totals / counts

    def _relative_features(
        self,
        unit_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        unit_positions = unit_features[:, :, self.UNIT_POSITION].unsqueeze(2)
        target_positions = target_features[:, :, self.TARGET_POSITION].unsqueeze(1)
        delta = target_positions - unit_positions
        distance = torch.linalg.vector_norm(delta, dim=3, keepdim=True)
        unit_range = unit_features[:, :, self.UNIT_RANGE_INDEX].unsqueeze(2).unsqueeze(3)
        range_margin = unit_range - distance
        return torch.cat((delta, distance, range_margin), dim=3)
