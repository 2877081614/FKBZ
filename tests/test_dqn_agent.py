import torch

from rein_learning.agents import DQNAgent, DQNConfig
from rein_learning.models import DiscreteQNetwork


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
