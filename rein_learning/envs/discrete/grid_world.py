from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
from gymnasium import spaces


@dataclass(frozen=True)
class GridWorldConfig:
    size: int = 5
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (4, 4)
    traps: tuple[tuple[int, int], ...] = ((1, 3), (2, 1), (3, 3))
    max_steps: int = 50
    step_reward: float = -1.0
    goal_reward: float = 10.0
    trap_reward: float = -10.0
    wall_reward: float = -2.0


class SmallGridWorldEnv(gym.Env):
    """Small discrete GridWorld environment for testing RL algorithms."""

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    ACTIONS: dict[int, tuple[int, int]] = {
        0: (-1, 0),  # up
        1: (0, 1),  # right
        2: (1, 0),  # down
        3: (0, -1),  # left
    }
    ACTION_NAMES = {0: "up", 1: "right", 2: "down", 3: "left"}

    def __init__(
        self,
        config: GridWorldConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        self.config = config or GridWorldConfig()
        self.render_mode = render_mode

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self._validate_config()
        self.observation_space = spaces.Discrete(self.config.size * self.config.size)
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        self.agent_pos = self.config.start
        self.steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)
        self.agent_pos = self.config.start
        self.steps = 0
        observation = self._position_to_state(self.agent_pos)
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected one of {list(self.ACTIONS)}")

        self.steps += 1
        row_delta, col_delta = self.ACTIONS[int(action)]
        row, col = self.agent_pos
        candidate_pos = (row + row_delta, col + col_delta)

        hit_wall = not self._is_in_bounds(candidate_pos)
        if hit_wall:
            next_pos = self.agent_pos
            reward = self.config.wall_reward
        else:
            next_pos = candidate_pos
            reward = self.config.step_reward

        self.agent_pos = next_pos
        terminated = False

        if self.agent_pos == self.config.goal:
            reward = self.config.goal_reward
            terminated = True
        elif self.agent_pos in self.config.traps:
            reward = self.config.trap_reward
            terminated = True

        truncated = self.steps >= self.config.max_steps and not terminated
        observation = self._position_to_state(self.agent_pos)
        info = self._get_info()
        info["action_name"] = self.ACTION_NAMES[int(action)]
        info["hit_wall"] = hit_wall

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self) -> str | None:
        grid = [["." for _ in range(self.config.size)] for _ in range(self.config.size)]

        for trap in self.config.traps:
            trap_row, trap_col = trap
            grid[trap_row][trap_col] = "X"

        start_row, start_col = self.config.start
        goal_row, goal_col = self.config.goal
        agent_row, agent_col = self.agent_pos

        grid[start_row][start_col] = "S"
        grid[goal_row][goal_col] = "G"
        grid[agent_row][agent_col] = "A"

        output = "\n".join(" ".join(row) for row in grid)

        if self.render_mode == "human":
            print(output)
            print()
            return None

        return output

    def close(self) -> None:
        return None

    def _position_to_state(self, position: tuple[int, int]) -> int:
        row, col = position
        return row * self.config.size + col

    def _state_to_position(self, state: int) -> tuple[int, int]:
        return divmod(state, self.config.size)

    def _is_in_bounds(self, position: tuple[int, int]) -> bool:
        row, col = position
        return 0 <= row < self.config.size and 0 <= col < self.config.size

    def _get_info(self) -> dict[str, Any]:
        state = self._position_to_state(self.agent_pos)
        return {
            "agent_pos": self.agent_pos,
            "state": state,
            "steps": self.steps,
            "distance_to_goal": self._manhattan_distance(self.agent_pos, self.config.goal),
        }

    def _manhattan_distance(
        self,
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    def _validate_config(self) -> None:
        positions = [self.config.start, self.config.goal, *self.config.traps]
        for position in positions:
            if not self._is_in_bounds(position):
                raise ValueError(f"Position {position} is outside the grid")

        if self.config.start == self.config.goal:
            raise ValueError("start and goal must be different positions")

        if self.config.start in self.config.traps:
            raise ValueError("start cannot be a trap")

        if self.config.goal in self.config.traps:
            raise ValueError("goal cannot be a trap")
