from typing import Any


def run_grid_world_dqn() -> None:
    from .grid_world_dqn import main

    main()


def run_air_defense_dqn() -> None:
    from .air_defense_dqn import main

    main()


def run_air_defense_v1_ppo() -> None:
    from .air_defense_v1_ppo import main

    main()


def run_grid_world_q_learning() -> None:
    from .grid_world_q_learning import main

    main()


def run_grid_world_reinforce() -> None:
    from .grid_world_reinforce import main

    main()


def train_grid_world_dqn(*args: Any, **kwargs: Any) -> Any:
    from .grid_world_dqn import train

    return train(*args, **kwargs)


def train_air_defense_dqn(*args: Any, **kwargs: Any) -> Any:
    from .air_defense_dqn import train

    return train(*args, **kwargs)


def train_air_defense_v1_ppo(*args: Any, **kwargs: Any) -> Any:
    from .air_defense_v1_ppo import train

    return train(*args, **kwargs)


def train_grid_world_q_learning(*args: Any, **kwargs: Any) -> Any:
    from .grid_world_q_learning import train

    return train(*args, **kwargs)


def train_grid_world_reinforce(*args: Any, **kwargs: Any) -> Any:
    from .grid_world_reinforce import train

    return train(*args, **kwargs)


__all__ = [
    "run_air_defense_dqn",
    "run_air_defense_v1_ppo",
    "run_grid_world_dqn",
    "run_grid_world_q_learning",
    "run_grid_world_reinforce",
    "train_air_defense_dqn",
    "train_air_defense_v1_ppo",
    "train_grid_world_dqn",
    "train_grid_world_q_learning",
    "train_grid_world_reinforce",
]
