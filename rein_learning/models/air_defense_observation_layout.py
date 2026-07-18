from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from gymnasium import spaces


@dataclass(frozen=True)
class StructuredAirDefenseV1Observation:
    zones: torch.Tensor
    targets: torch.Tensor
    units: torch.Tensor
    global_features: torch.Tensor

    def flatten(self) -> torch.Tensor:
        batch_size = self.global_features.shape[0]
        return torch.cat(
            (
                self.zones.reshape(batch_size, -1),
                self.targets.reshape(batch_size, -1),
                self.units.reshape(batch_size, -1),
                self.global_features,
            ),
            dim=1,
        )


@dataclass(frozen=True)
class AirDefenseV1ObservationLayout:
    """Frozen AirDefense v1 observation slices used by structured policies."""

    num_zones: int
    num_targets: int
    num_units: int
    zone_feature_dim: int = 7
    target_feature_dim: int = 15
    unit_feature_dim: int = 15
    global_feature_dim: int = 8

    def __post_init__(self) -> None:
        values = (
            self.num_zones,
            self.num_targets,
            self.num_units,
            self.zone_feature_dim,
            self.target_feature_dim,
            self.unit_feature_dim,
            self.global_feature_dim,
        )
        if any(value <= 0 for value in values):
            raise ValueError("All observation layout dimensions must be positive")

    @property
    def observation_dim(self) -> int:
        return (
            self.num_zones * self.zone_feature_dim
            + self.num_targets * self.target_feature_dim
            + self.num_units * self.unit_feature_dim
            + self.global_feature_dim
        )

    @classmethod
    def infer(
        cls,
        observation_space: spaces.Space,
        action_space: spaces.Space,
    ) -> "AirDefenseV1ObservationLayout":
        if not isinstance(observation_space, spaces.Box):
            raise ValueError("AirDefense v1 requires a Box observation space")
        if len(observation_space.shape) != 1:
            raise ValueError("AirDefense v1 observations must be one-dimensional")
        if not isinstance(action_space, spaces.MultiDiscrete):
            raise ValueError("AirDefense v1 requires a MultiDiscrete action space")
        action_dims = tuple(int(value) for value in action_space.nvec)
        if not action_dims or len(set(action_dims)) != 1 or action_dims[0] < 2:
            raise ValueError("AirDefense v1 unit action dimensions must match")

        num_units = len(action_dims)
        num_targets = action_dims[0] - 1
        fixed_dim = num_targets * 15 + num_units * 15 + 8
        zone_values = int(observation_space.shape[0]) - fixed_dim
        if zone_values <= 0 or zone_values % 7 != 0:
            raise ValueError(
                "Observation shape is incompatible with the frozen "
                "AirDefense v1 layout"
            )
        layout = cls(
            num_zones=zone_values // 7,
            num_targets=num_targets,
            num_units=num_units,
        )
        if layout.observation_dim != int(observation_space.shape[0]):
            raise ValueError("Inferred observation layout has the wrong dimension")
        return layout

    def split(self, observation: torch.Tensor) -> StructuredAirDefenseV1Observation:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if observation.ndim != 2 or observation.shape[1] != self.observation_dim:
            raise ValueError(
                f"Expected observations shaped [batch, {self.observation_dim}], "
                f"got {tuple(observation.shape)}"
            )

        batch_size = observation.shape[0]
        zone_end = self.num_zones * self.zone_feature_dim
        target_end = zone_end + self.num_targets * self.target_feature_dim
        unit_end = target_end + self.num_units * self.unit_feature_dim
        return StructuredAirDefenseV1Observation(
            zones=observation[:, :zone_end].reshape(
                batch_size,
                self.num_zones,
                self.zone_feature_dim,
            ),
            targets=observation[:, zone_end:target_end].reshape(
                batch_size,
                self.num_targets,
                self.target_feature_dim,
            ),
            units=observation[:, target_end:unit_end].reshape(
                batch_size,
                self.num_units,
                self.unit_feature_dim,
            ),
            global_features=observation[:, unit_end:],
        )

    def signature(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}
