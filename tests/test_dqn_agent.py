import torch
import numpy as np

from rein_learning.agents import DQNAgent, DQNConfig, VectorDQNAgent
from rein_learning.models import DiscreteQNetwork, VectorQNetwork


def test_discrete_q_network_outputs_action_values() -> None:
    network = DiscreteQNetwork(num_states=5, num_actions=3, hidden_sizes=(8,))

    q_values = network(torch.tensor([0, 1, 2], dtype=torch.long))

    assert q_values.shape == (3, 3)


def test_dqn_agent_selects_greedy_action() -> None:
    agent = DQNAgent(
        num_states=4,
        num_actions=2,
        config=DQNConfig(
            epsilon=0.0,
            min_replay_size=1,
            batch_size=1,
            hidden_sizes=(8,),
            device="cpu",
        ),
        seed=0,
    )

    action = agent.select_action(0, greedy=True)

    assert action in {0, 1}


def test_dqn_agent_updates_from_replay_buffer() -> None:
    agent = DQNAgent(
        num_states=5,
        num_actions=2,
        config=DQNConfig(
            epsilon=0.0,
            min_replay_size=4,
            batch_size=4,
            target_update_interval=2,
            hidden_sizes=(8,),
            device="cpu",
        ),
        seed=0,
    )

    for index in range(8):
        state = index % 5
        next_state = (index + 1) % 5
        agent.store_transition(state, index % 2, 1.0, next_state, False)

    loss = agent.update()

    assert loss is not None
    assert loss >= 0.0


def test_vector_q_network_outputs_action_values() -> None:
    network = VectorQNetwork(observation_dim=4, num_actions=3, hidden_sizes=(8,))

    q_values = network(torch.zeros((2, 4), dtype=torch.float32))

    assert q_values.shape == (2, 3)


def test_vector_dqn_agent_respects_action_mask_when_exploring() -> None:
    agent = VectorDQNAgent(
        observation_dim=4,
        num_actions=3,
        config=DQNConfig(
            epsilon=1.0,
            hidden_sizes=(8,),
            device="cpu",
        ),
        seed=0,
    )

    actions = {
        agent.select_action(
            np.zeros(4, dtype=np.float32),
            np.asarray([0, 1, 0], dtype=np.int8),
        )
        for _ in range(10)
    }

    assert actions == {1}


def test_vector_dqn_agent_updates_from_replay_buffer() -> None:
    agent = VectorDQNAgent(
        observation_dim=4,
        num_actions=2,
        config=DQNConfig(
            epsilon=0.0,
            min_replay_size=4,
            batch_size=4,
            target_update_interval=2,
            hidden_sizes=(8,),
            device="cpu",
        ),
        seed=0,
    )

    for index in range(8):
        state = np.full(4, index, dtype=np.float32)
        next_state = np.full(4, index + 1, dtype=np.float32)
        agent.store_transition(
            state,
            index % 2,
            1.0,
            next_state,
            False,
            np.asarray([1, 1], dtype=np.int8),
        )

    loss = agent.update()

    assert loss is not None
    assert loss >= 0.0
