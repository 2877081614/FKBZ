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
