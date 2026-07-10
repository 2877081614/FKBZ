from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np


@dataclass(frozen=True)
class TransitionBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


@dataclass(frozen=True)
class VectorTransitionBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    next_action_masks: np.ndarray


class ReplayBuffer:
    """Fixed-size replay buffer for off-policy algorithms."""

    def __init__(self, capacity: int, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.storage: Deque[tuple[int, int, float, int, bool]] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        self.storage.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> TransitionBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self.storage):
            raise ValueError("batch_size cannot exceed buffer size")

        indices = self.rng.choice(len(self.storage), size=batch_size, replace=False)
        transitions = [self.storage[int(index)] for index in indices]
        states, actions, rewards, next_states, dones = zip(*transitions)

        return TransitionBatch(
            states=np.asarray(states, dtype=np.int64),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            next_states=np.asarray(next_states, dtype=np.int64),
            dones=np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.storage)


class VectorReplayBuffer:
    """Fixed-size replay buffer for vector-observation off-policy algorithms."""

    def __init__(self, capacity: int, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.storage: Deque[
            tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]
        ] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray,
    ) -> None:
        self.storage.append(
            (
                np.asarray(state, dtype=np.float32).copy(),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32).copy(),
                bool(done),
                np.asarray(next_action_mask, dtype=np.int8).copy(),
            )
        )

    def sample(self, batch_size: int) -> VectorTransitionBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self.storage):
            raise ValueError("batch_size cannot exceed buffer size")

        indices = self.rng.choice(len(self.storage), size=batch_size, replace=False)
        transitions = [self.storage[int(index)] for index in indices]
        states, actions, rewards, next_states, dones, next_action_masks = zip(*transitions)

        return VectorTransitionBatch(
            states=np.stack(states).astype(np.float32),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            next_states=np.stack(next_states).astype(np.float32),
            dones=np.asarray(dones, dtype=np.float32),
            next_action_masks=np.stack(next_action_masks).astype(np.int8),
        )

    def __len__(self) -> int:
        return len(self.storage)
