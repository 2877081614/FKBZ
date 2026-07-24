from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class _SupportSpace:
    references: torch.Tensor
    center: torch.Tensor
    scale: torch.Tensor
    distance_scale: torch.Tensor


class MaskedContextSupportIndex:
    """Nearest-neighbor support for masked unit-prefix decision contexts."""

    def __init__(
        self,
        *,
        engagement: _SupportSpace,
        target: _SupportSpace,
        num_units: int,
        num_actions: int,
        train_row_count: int,
        dataset_path: str,
    ) -> None:
        self.engagement = engagement
        self.target = target
        self.num_units = int(num_units)
        self.num_actions = int(num_actions)
        self.train_row_count = int(train_row_count)
        self.dataset_path = dataset_path

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        num_units: int,
        device: torch.device | str,
        split: str = "train",
        std_floor: float = 0.05,
        distance_quantile: float = 0.95,
    ) -> "MaskedContextSupportIndex":
        if std_floor <= 0.0:
            raise ValueError("std_floor must be positive")
        if not 0.0 < distance_quantile <= 1.0:
            raise ValueError("distance_quantile must be in (0, 1]")
        dataset_path = Path(path).resolve()
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Support dataset not found: {dataset_path}")
        data = np.load(dataset_path, allow_pickle=False)
        required = {
            "observations",
            "unit_indices",
            "candidate_actions",
            "prefix_occupancy",
            "legal_action_masks",
            "splits",
        }
        if not required.issubset(data.files):
            missing = sorted(required - set(data.files))
            raise ValueError(f"Support dataset is missing fields: {missing}")
        selected = np.asarray(data["splits"]) == split
        if not np.any(selected):
            raise ValueError(f"Support dataset contains no split={split!r} rows")
        observations = np.asarray(data["observations"][selected], dtype=np.float32)
        unit_indices = np.asarray(data["unit_indices"][selected], dtype=np.int64)
        candidate_actions = np.asarray(
            data["candidate_actions"][selected], dtype=np.int64
        )
        prefix_occupancy = np.asarray(
            data["prefix_occupancy"][selected], dtype=np.float32
        )
        legal_action_masks = np.asarray(
            data["legal_action_masks"][selected], dtype=np.float32
        )
        num_actions = int(legal_action_masks.shape[1])
        if np.any((unit_indices < 0) | (unit_indices >= num_units)):
            raise ValueError("Support dataset contains an invalid unit index")
        if np.any((candidate_actions < 0) | (candidate_actions >= num_actions)):
            raise ValueError("Support dataset contains an invalid candidate action")
        unit_one_hot = np.eye(num_units, dtype=np.float32)[unit_indices]
        engagement_features = np.concatenate(
            (
                observations,
                unit_one_hot,
                prefix_occupancy,
                legal_action_masks,
            ),
            axis=1,
        )
        action_one_hot = np.eye(num_actions, dtype=np.float32)[candidate_actions]
        target_features = np.concatenate(
            (engagement_features, action_one_hot), axis=1
        )
        return cls(
            engagement=cls._build_space(
                engagement_features,
                device=device,
                std_floor=std_floor,
                distance_quantile=distance_quantile,
            ),
            target=cls._build_space(
                target_features,
                device=device,
                std_floor=std_floor,
                distance_quantile=distance_quantile,
            ),
            num_units=num_units,
            num_actions=num_actions,
            train_row_count=int(selected.sum()),
            dataset_path=str(dataset_path),
        )

    @staticmethod
    def _build_space(
        features: np.ndarray,
        *,
        device: torch.device | str,
        std_floor: float,
        distance_quantile: float,
    ) -> _SupportSpace:
        unique = np.unique(features, axis=0)
        if unique.shape[0] < 2:
            raise ValueError("Support space requires at least two unique contexts")
        center = unique.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = unique.std(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(scale, std_floor)
        standardized = (unique - center) / scale
        reference_tensor = torch.as_tensor(
            standardized, device=device, dtype=torch.float32
        )
        distances = torch.cdist(reference_tensor, reference_tensor) / np.sqrt(
            reference_tensor.shape[1]
        )
        distances.fill_diagonal_(torch.inf)
        nearest = distances.min(dim=1).values
        distance_scale = torch.quantile(nearest, distance_quantile).clamp_min(
            1e-6
        )
        return _SupportSpace(
            references=reference_tensor,
            center=torch.as_tensor(center, device=device),
            scale=torch.as_tensor(scale, device=device),
            distance_scale=distance_scale,
        )

    def engagement_scores(
        self,
        observations: torch.Tensor,
        unit_indices: torch.Tensor,
        prefix_occupancy: torch.Tensor,
        legal_action_masks: torch.Tensor,
    ) -> torch.Tensor:
        features = self._context_features(
            observations,
            unit_indices,
            prefix_occupancy,
            legal_action_masks,
        )
        return self._scores(features, self.engagement)

    def target_scores(
        self,
        observations: torch.Tensor,
        unit_indices: torch.Tensor,
        candidate_actions: torch.Tensor,
        prefix_occupancy: torch.Tensor,
        legal_action_masks: torch.Tensor,
    ) -> torch.Tensor:
        features = self._context_features(
            observations,
            unit_indices,
            prefix_occupancy,
            legal_action_masks,
        )
        candidates = candidate_actions.long().reshape(-1)
        if bool(torch.any((candidates < 0) | (candidates >= self.num_actions))):
            raise ValueError("candidate_actions contain an invalid action")
        action_one_hot = F.one_hot(
            candidates, num_classes=self.num_actions
        ).to(features.dtype)
        return self._scores(torch.cat((features, action_one_hot), dim=1), self.target)

    def _context_features(
        self,
        observations: torch.Tensor,
        unit_indices: torch.Tensor,
        prefix_occupancy: torch.Tensor,
        legal_action_masks: torch.Tensor,
    ) -> torch.Tensor:
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        batch_size = observations.shape[0]
        units = unit_indices.long().reshape(-1)
        if units.shape[0] != batch_size:
            raise ValueError("unit_indices must match observations")
        if bool(torch.any((units < 0) | (units >= self.num_units))):
            raise ValueError("unit_indices contain an invalid unit")
        if prefix_occupancy.shape[0] != batch_size:
            raise ValueError("prefix_occupancy must match observations")
        if legal_action_masks.shape != (batch_size, self.num_actions):
            raise ValueError("legal_action_masks have the wrong shape")
        unit_one_hot = F.one_hot(units, num_classes=self.num_units).to(
            observations.dtype
        )
        return torch.cat(
            (
                observations,
                unit_one_hot,
                prefix_occupancy.to(observations.dtype),
                legal_action_masks.to(observations.dtype),
            ),
            dim=1,
        )

    @staticmethod
    def _scores(features: torch.Tensor, space: _SupportSpace) -> torch.Tensor:
        standardized = (features - space.center) / space.scale
        distances = torch.cdist(standardized, space.references) / np.sqrt(
            standardized.shape[1]
        )
        nearest = distances.min(dim=1).values
        return torch.exp(
            -np.log(2.0) * torch.square(nearest / space.distance_scale)
        ).clamp(0.0, 1.0)

    def signature(self) -> dict[str, Any]:
        return {
            "type": "masked_context_nearest_neighbor_support",
            "dataset_path": self.dataset_path,
            "split": "train",
            "train_row_count": self.train_row_count,
            "engagement_reference_count": int(
                self.engagement.references.shape[0]
            ),
            "target_reference_count": int(self.target.references.shape[0]),
            "std_floor": 0.05,
            "distance_quantile": 0.95,
            "score_at_distance_scale": 0.5,
        }
