import numpy as np


def q_learning_update(
    q_table: np.ndarray,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    terminated: bool,
    learning_rate: float,
    gamma: float,
) -> float:
    """Apply one tabular Q-learning update and return the TD error."""
    best_next_q = 0.0 if terminated else float(np.max(q_table[next_state]))
    target = reward + gamma * best_next_q
    td_error = target - q_table[state, action]
    q_table[state, action] += learning_rate * td_error
    return float(td_error)
