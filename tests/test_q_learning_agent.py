import numpy as np

from rein_learning.agents import QLearningConfig, TabularQLearningAgent
from rein_learning.algorithms.tabular import q_learning_update


def test_q_learning_update_moves_value_toward_td_target() -> None:
    q_table = np.zeros((3, 2), dtype=np.float32)
    q_table[1, 0] = 4.0

    td_error = q_learning_update(
        q_table=q_table,
        state=0,
        action=1,
        reward=1.0,
        next_state=1,
        terminated=False,
        learning_rate=0.5,
        gamma=0.9,
    )

    assert np.isclose(td_error, 4.6)
    assert np.isclose(q_table[0, 1], 2.3)


def test_tabular_agent_greedy_action_uses_q_table() -> None:
    agent = TabularQLearningAgent(
        num_states=4,
        num_actions=3,
        config=QLearningConfig(epsilon=0.0),
        seed=0,
    )
    agent.q_table[2] = np.asarray([0.0, 5.0, 1.0])

    action = agent.select_action(2, greedy=True)

    assert action == 1


def test_tabular_agent_decays_epsilon_to_minimum() -> None:
    agent = TabularQLearningAgent(
        num_states=2,
        num_actions=2,
        config=QLearningConfig(epsilon=0.2, epsilon_min=0.1, epsilon_decay=0.1),
        seed=0,
    )

    epsilon = agent.decay_epsilon()

    assert epsilon == 0.1
