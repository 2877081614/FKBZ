from rein_learning.buffers import ReplayBuffer


def test_replay_buffer_adds_and_samples_batch() -> None:
    buffer = ReplayBuffer(capacity=10, seed=0)
    for index in range(5):
        buffer.add(index, index % 2, float(index), index + 1, False)

    batch = buffer.sample(batch_size=3)

    assert len(buffer) == 5
    assert batch.states.shape == (3,)
    assert batch.actions.shape == (3,)
    assert batch.rewards.shape == (3,)
    assert batch.next_states.shape == (3,)
    assert batch.dones.shape == (3,)


def test_replay_buffer_respects_capacity() -> None:
    buffer = ReplayBuffer(capacity=2, seed=0)
    buffer.add(0, 0, 0.0, 1, False)
    buffer.add(1, 0, 0.0, 2, False)
    buffer.add(2, 0, 0.0, 3, True)

    assert len(buffer) == 2
    batch = buffer.sample(batch_size=2)
    assert 0 not in set(batch.states.tolist())
