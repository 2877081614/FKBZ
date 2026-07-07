from rein_learning.envs import GridWorldConfig, SmallGridWorldEnv


def test_reset_returns_start_state() -> None:
    env = SmallGridWorldEnv()

    obs, info = env.reset(seed=0)

    assert env.observation_space.n == 25
    assert env.action_space.n == 4
    assert obs == 0
    assert info["agent_pos"] == (0, 0)


def test_wall_keeps_agent_in_place() -> None:
    env = SmallGridWorldEnv()
    env.reset()

    obs, reward, terminated, truncated, info = env.step(0)

    assert obs == 0
    assert reward == -2.0
    assert not terminated
    assert not truncated
    assert info["hit_wall"] is True


def test_reaching_goal_terminates_episode() -> None:
    env = SmallGridWorldEnv()
    env.reset()

    result = None
    for action in [1, 1, 2, 2, 1, 1, 2, 2]:
        result = env.step(action)

    assert result is not None
    obs, reward, terminated, truncated, info = result
    assert obs == 24
    assert reward == 10.0
    assert terminated
    assert not truncated
    assert info["agent_pos"] == (4, 4)


def test_stepping_on_trap_terminates_episode() -> None:
    env = SmallGridWorldEnv()
    env.reset()

    env.step(1)
    env.step(1)
    env.step(1)
    obs, reward, terminated, truncated, info = env.step(2)

    assert obs == 8
    assert reward == -10.0
    assert terminated
    assert not truncated
    assert info["agent_pos"] == (1, 3)


def test_max_steps_truncates_episode() -> None:
    config = GridWorldConfig(max_steps=2)
    env = SmallGridWorldEnv(config=config)
    env.reset()

    env.step(1)
    obs, reward, terminated, truncated, info = env.step(3)

    assert obs == 0
    assert reward == -1.0
    assert not terminated
    assert truncated
    assert info["steps"] == 2
