"""
经验回放池 (Section 3.1, Table 1: capacity = 1e6)
"""
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """Fixed-size experience replay buffer"""

    def __init__(self, capacity=int(1e6)):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, actions, rewards, next_obs, dones):
        """
        Args:
            obs:      dict {agent_i: np.ndarray}
            actions:  dict {agent_i: int}
            rewards:  dict {agent_i: float}
            next_obs: dict {agent_i: np.ndarray}
            dones:    bool
        """
        self.buffer.append({
            'obs': {k: v.copy() for k, v in obs.items()},
            'actions': dict(actions),
            'rewards': dict(rewards),
            'next_obs': {k: v.copy() for k, v in next_obs.items()},
            'dones': dones,
        })

    def sample(self, batch_size):
        """Random sample a batch"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return batch

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, batch_size):
        return len(self) >= batch_size
