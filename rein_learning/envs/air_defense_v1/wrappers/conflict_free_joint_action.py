from __future__ import annotations

from itertools import product
from typing import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..centralized_env import AirDefenseResourceAssignmentEnvV1


class ConflictFreeJointActionCodec:
    """Map conflict-free unit assignments to one deterministic discrete index."""

    def __init__(self, *, num_units: int, num_targets: int) -> None:
        if num_units <= 0:
            raise ValueError("num_units must be positive")
        if num_targets <= 0:
            raise ValueError("num_targets must be positive")

        self.num_units = int(num_units)
        self.num_targets = int(num_targets)
        self.noop_action = self.num_targets
        self._joint_actions = tuple(
            action
            for action in product(
                range(self.num_targets + 1),
                repeat=self.num_units,
            )
            if self._is_conflict_free(action)
        )
        self._action_to_index = {
            action: index for index, action in enumerate(self._joint_actions)
        }
        self._action_array = np.asarray(self._joint_actions, dtype=np.int64)
        self._action_array.setflags(write=False)

    def __len__(self) -> int:
        return len(self._joint_actions)

    @property
    def joint_actions(self) -> tuple[tuple[int, ...], ...]:
        return self._joint_actions

    @property
    def action_array(self) -> np.ndarray:
        return self._action_array

    @property
    def all_noop_index(self) -> int:
        return self._action_to_index[(self.noop_action,) * self.num_units]

    def encode(self, joint_action: Sequence[int] | np.ndarray) -> int:
        action_array = np.asarray(joint_action, dtype=np.int64).reshape(-1)
        if action_array.size != self.num_units:
            raise ValueError(
                f"Expected {self.num_units} unit actions, got {action_array.size}"
            )
        action = tuple(int(value) for value in action_array)
        try:
            return self._action_to_index[action]
        except KeyError as exc:
            raise ValueError(
                f"Joint action {action} is out of range or assigns one target "
                "to multiple units"
            ) from exc

    def decode(self, encoded_action: int | np.integer) -> np.ndarray:
        index = int(encoded_action)
        if not 0 <= index < len(self):
            raise ValueError(
                f"Encoded action {index} is outside Discrete({len(self)})"
            )
        return self._action_array[index].copy()

    def _is_conflict_free(self, action: Sequence[int]) -> bool:
        active_targets = [value for value in action if value != self.noop_action]
        return len(active_targets) == len(set(active_targets))


class ConflictFreeJointActionWrapper(gym.Wrapper):
    """Expose only one-to-one joint assignments as a Discrete action space."""

    def __init__(self, env: AirDefenseResourceAssignmentEnvV1) -> None:
        super().__init__(env)
        self.codec = ConflictFreeJointActionCodec(
            num_units=env.num_defense_units,
            num_targets=env.num_targets,
        )
        self.action_space = spaces.Discrete(len(self.codec))
        self.observation_space = env.observation_space

    @property
    def base_env(self) -> AirDefenseResourceAssignmentEnvV1:
        return self.env

    @property
    def num_defense_units(self) -> int:
        return self.base_env.num_defense_units

    @property
    def num_targets(self) -> int:
        return self.base_env.num_targets

    @property
    def noop_action(self) -> int:
        return self.base_env.noop_action

    def action_mask(self) -> np.ndarray:
        base_mask = self.base_env.action_mask().astype(bool, copy=False)
        unit_indices = np.arange(self.num_defense_units)[None, :]
        return np.all(
            base_mask[unit_indices, self.codec.action_array],
            axis=1,
        )

    def action_masks(self) -> np.ndarray:
        return self.action_mask()

    def step(
        self,
        action: int | np.integer | np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        action_array = np.asarray(action)
        if action_array.size != 1:
            raise ValueError(f"Expected one encoded action, got shape {action_array.shape}")
        encoded_action = int(action_array.reshape(-1)[0])
        if not self.action_space.contains(encoded_action):
            raise ValueError(
                f"Encoded action {encoded_action} is outside {self.action_space}"
            )

        decoded_action = self.codec.decode(encoded_action)
        observation, reward, terminated, truncated, info = self.base_env.step(
            decoded_action
        )
        wrapped_info = dict(info)
        wrapped_info.update(
            {
                "encoded_joint_action": encoded_action,
                "decoded_joint_action": decoded_action.copy(),
            }
        )
        return observation, reward, terminated, truncated, wrapped_info
