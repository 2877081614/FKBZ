from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..buffers import (
    ReplayBuffer,
    TransitionBatch,
    VectorReplayBuffer,
    VectorTransitionBatch,
)
from ..models import DiscreteQNetwork, VectorQNetwork


@dataclass
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 32
    buffer_capacity: int = 10_000
    min_replay_size: int = 100
    target_update_interval: int = 100
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    hidden_sizes: tuple[int, ...] = (64, 64)
    device: str = "auto"


class DQNAgent:
    """Standard DQN agent for discrete-state, discrete-action tasks."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        config: DQNConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.config = config or DQNConfig()
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed or 0)

        self.device = self._resolve_device(self.config.device)
        self.q_network = DiscreteQNetwork(
            num_states=num_states,
            num_actions=num_actions,
            hidden_sizes=self.config.hidden_sizes,
        ).to(self.device)
        self.target_network = DiscreteQNetwork(
            num_states=num_states,
            num_actions=num_actions,
            hidden_sizes=self.config.hidden_sizes,
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.replay_buffer = ReplayBuffer(self.config.buffer_capacity, seed=seed)
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate,
        )
        self.loss_fn: nn.Module = nn.MSELoss()
        self.update_steps = 0

    def select_action(self, state: int, *, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.config.epsilon:
            return int(self.rng.integers(self.num_actions))

        with torch.no_grad():
            states = torch.tensor([state], dtype=torch.long, device=self.device)
            q_values = self.q_network(states)
            return int(torch.argmax(q_values, dim=1).item())

    def store_transition(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self) -> float | None:
        if len(self.replay_buffer) < self.config.min_replay_size:
            return None
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        batch = self.replay_buffer.sample(self.config.batch_size)
        loss = self._update_from_batch(batch)
        self.update_steps += 1

        if self.update_steps % self.config.target_update_interval == 0:
            self.sync_target_network()

        return loss

    def sync_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self) -> float:
        self.config.epsilon = max(
            self.config.epsilon_min,
            self.config.epsilon * self.config.epsilon_decay,
        )
        return self.config.epsilon

    def _update_from_batch(self, batch: TransitionBatch) -> float:
        states = torch.tensor(batch.states, dtype=torch.long, device=self.device)
        actions = torch.tensor(batch.actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(batch.next_states, dtype=torch.long, device=self.device)
        dones = torch.tensor(batch.dones, dtype=torch.float32, device=self.device)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1).values
            target_q = rewards + self.config.gamma * (1.0 - dones) * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)


class VectorDQNAgent:
    """DQN agent for vector observations, discrete actions, and optional action masks."""

    def __init__(
        self,
        observation_dim: int,
        num_actions: int,
        config: DQNConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.observation_dim = observation_dim
        self.num_actions = num_actions
        self.config = config or DQNConfig(hidden_sizes=(128, 128))
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed or 0)

        self.device = self._resolve_device(self.config.device)
        self.q_network = VectorQNetwork(
            observation_dim=observation_dim,
            num_actions=num_actions,
            hidden_sizes=self.config.hidden_sizes,
        ).to(self.device)
        self.target_network = VectorQNetwork(
            observation_dim=observation_dim,
            num_actions=num_actions,
            hidden_sizes=self.config.hidden_sizes,
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.replay_buffer = VectorReplayBuffer(self.config.buffer_capacity, seed=seed)
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate,
        )
        self.loss_fn: nn.Module = nn.MSELoss()
        self.update_steps = 0

    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray | None = None,
        *,
        greedy: bool = False,
    ) -> int:
        mask = self._normalize_action_mask(action_mask)
        legal_actions = np.flatnonzero(mask)
        if legal_actions.size == 0:
            raise ValueError("action_mask must contain at least one legal action")

        if not greedy and self.rng.random() < self.config.epsilon:
            return int(self.rng.choice(legal_actions))

        with torch.no_grad():
            state_tensor = torch.tensor(
                np.asarray(state, dtype=np.float32)[None, :],
                dtype=torch.float32,
                device=self.device,
            )
            q_values = self.q_network(state_tensor).squeeze(0).cpu().numpy()
            q_values[mask == 0] = -np.inf
            return int(np.argmax(q_values))

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray,
    ) -> None:
        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
            self._normalize_action_mask(next_action_mask),
        )

    def update(self) -> float | None:
        if len(self.replay_buffer) < self.config.min_replay_size:
            return None
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        batch = self.replay_buffer.sample(self.config.batch_size)
        loss = self._update_from_batch(batch)
        self.update_steps += 1

        if self.update_steps % self.config.target_update_interval == 0:
            self.sync_target_network()

        return loss

    def sync_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self) -> float:
        self.config.epsilon = max(
            self.config.epsilon_min,
            self.config.epsilon * self.config.epsilon_decay,
        )
        return self.config.epsilon

    def _update_from_batch(self, batch: VectorTransitionBatch) -> float:
        states = torch.tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(batch.next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.dones, dtype=torch.float32, device=self.device)
        next_action_masks = torch.tensor(
            batch.next_action_masks,
            dtype=torch.bool,
            device=self.device,
        )

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.masked_fill(~next_action_masks, -torch.inf)
            next_q = next_q_values.max(dim=1).values
            next_q = torch.where(torch.isfinite(next_q), next_q, torch.zeros_like(next_q))
            target_q = rewards + self.config.gamma * (1.0 - dones) * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def _normalize_action_mask(self, action_mask: np.ndarray | None) -> np.ndarray:
        if action_mask is None:
            return np.ones(self.num_actions, dtype=np.int8)
        mask = np.asarray(action_mask, dtype=np.int8)
        if mask.shape != (self.num_actions,):
            raise ValueError(
                f"action_mask must have shape ({self.num_actions},), got {mask.shape}"
            )
        return mask

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
