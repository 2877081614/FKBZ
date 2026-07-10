from __future__ import annotations

import torch
from torch import nn


class DiscreteQNetwork(nn.Module):
    """MLP Q-network for discrete observations encoded as one-hot vectors."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = num_states

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size

        layers.append(nn.Linear(input_dim, num_actions))
        self.network = nn.Sequential(*layers)
        self.num_states = num_states

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 1:
            raise ValueError("states must be a 1D tensor of discrete state ids")
        one_hot_states = torch.nn.functional.one_hot(
            states.long(),
            num_classes=self.num_states,
        ).float()
        return self.network(one_hot_states)


class VectorQNetwork(nn.Module):
    """MLP Q-network for vector observations and discrete actions."""

    def __init__(
        self,
        observation_dim: int,
        num_actions: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")

        layers: list[nn.Module] = []
        input_dim = observation_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size

        layers.append(nn.Linear(input_dim, num_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2:
            raise ValueError("observations must be a 2D tensor")
        return self.network(observations.float())
