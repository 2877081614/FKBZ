from rein_learning.trainers import (
    train_grid_world_dqn,
    train_grid_world_q_learning,
    train_grid_world_reinforce,
)


def test_trainer_functions_are_importable() -> None:
    assert callable(train_grid_world_q_learning)
    assert callable(train_grid_world_dqn)
    assert callable(train_grid_world_reinforce)
