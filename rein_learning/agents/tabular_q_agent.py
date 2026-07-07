from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..algorithms.tabular import q_learning_update


@dataclass
class QLearningConfig:
    learning_rate: float = 0.1
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995


class TabularQLearningAgent:
    """Classic epsilon-greedy Q-learning agent for discrete spaces."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        config: QLearningConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.config = config or QLearningConfig()
        self.q_table = np.zeros((num_states, num_actions), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: int, *, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.config.epsilon:
            return int(self.rng.integers(self.num_actions))
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
    ) -> float:
        return q_learning_update(
            q_table=self.q_table,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminated=terminated,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
        )

    def decay_epsilon(self) -> float:
        self.config.epsilon = max(
            self.config.epsilon_min,
            self.config.epsilon * self.config.epsilon_decay,
        )
        return self.config.epsilon
