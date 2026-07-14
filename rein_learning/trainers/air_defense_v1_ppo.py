from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from ..envs import AirDefenseResourceAssignmentEnvV1, AirDefenseV1EnvConfig


AlgorithmName = Literal["ppo", "maskable_ppo"]


@dataclass(frozen=True)
class AirDefenseV1PPOConfig:
    """Training hyperparameters for SB3 PPO-style baselines."""

    total_timesteps: int = 20_000
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: tuple[int, ...] = (128, 128)
    seed: int = 42
    device: str = "cpu"
    verbose: int = 1
    tensorboard_log: str | None = None
    progress_bar: bool = False


def default_air_defense_v1_ppo_config() -> AirDefenseV1PPOConfig:
    return AirDefenseV1PPOConfig()


def train_ppo(
    *,
    env_config: AirDefenseV1EnvConfig | None = None,
    train_config: AirDefenseV1PPOConfig | None = None,
    save_path: str | Path | None = None,
    callback: Any | None = None,
    tb_log_name: str | None = None,
) -> Any:
    """Train a plain PPO baseline on the centralized v1.0 environment."""

    return _train_sb3_model(
        algorithm="ppo",
        env_config=env_config,
        train_config=train_config,
        save_path=save_path,
        callback=callback,
        tb_log_name=tb_log_name,
    )


def train_maskable_ppo(
    *,
    env_config: AirDefenseV1EnvConfig | None = None,
    train_config: AirDefenseV1PPOConfig | None = None,
    save_path: str | Path | None = None,
    callback: Any | None = None,
    tb_log_name: str | None = None,
) -> Any:
    """Train Maskable PPO using the environment's action_masks() interface."""

    return _train_sb3_model(
        algorithm="maskable_ppo",
        env_config=env_config,
        train_config=train_config,
        save_path=save_path,
        callback=callback,
        tb_log_name=tb_log_name,
    )


def train(
    algorithm: AlgorithmName = "maskable_ppo",
    *,
    env_config: AirDefenseV1EnvConfig | None = None,
    train_config: AirDefenseV1PPOConfig | None = None,
    save_path: str | Path | None = None,
    callback: Any | None = None,
    tb_log_name: str | None = None,
) -> Any:
    if algorithm == "ppo":
        return train_ppo(
            env_config=env_config,
            train_config=train_config,
            save_path=save_path,
            callback=callback,
            tb_log_name=tb_log_name,
        )
    if algorithm == "maskable_ppo":
        return train_maskable_ppo(
            env_config=env_config,
            train_config=train_config,
            save_path=save_path,
            callback=callback,
            tb_log_name=tb_log_name,
        )
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def evaluate_air_defense_v1_model(
    model: Any,
    *,
    env_factory: Callable[[], AirDefenseResourceAssignmentEnvV1] | None = None,
    env_config: AirDefenseV1EnvConfig | None = None,
    episodes: int = 30,
    seed: int = 1_000,
    deterministic: bool = True,
    use_action_masks: bool | None = None,
) -> dict[str, float]:
    """Evaluate a trained SB3-style model with the same metrics as rule baselines."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if env_factory is None:
        env_factory = lambda: AirDefenseResourceAssignmentEnvV1(config=env_config)
    if use_action_masks is None:
        use_action_masks = _looks_like_maskable_model(model)

    episode_metrics = []
    for episode_index in range(episodes):
        episode_seed = seed + episode_index
        env = env_factory()
        obs, initial_info = env.reset(seed=episode_seed)
        initial_ammo = int(initial_info["ammo_remaining"])
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        shots = 0
        hits = 0
        invalid_actions = 0
        info = initial_info

        while not (terminated or truncated):
            if use_action_masks:
                action, _ = model.predict(
                    obs,
                    deterministic=deterministic,
                    action_masks=env.action_masks(),
                )
            else:
                action, _ = model.predict(obs, deterministic=deterministic)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            shots += int(info["shots"])
            hits += int(info["hits"])
            invalid_actions += int(info["invalid_actions"])

        episode_metrics.append(
            {
                "total_reward": total_reward,
                "steps": steps,
                "num_targets": env.num_targets,
                "num_intercepted": int(info["num_intercepted"]),
                "num_leaked": int(info["num_leaked"]),
                "total_damage": float(info["total_damage"]),
                "ammo_used": initial_ammo - int(info["ammo_remaining"]),
                "shots": shots,
                "hits": hits,
                "invalid_actions": invalid_actions,
                "success": float(info["num_alive"] == 0 and info["total_damage"] == 0.0),
            }
        )
        env.close()

    return _aggregate_episode_metrics(episode_metrics)


def _train_sb3_model(
    *,
    algorithm: AlgorithmName,
    env_config: AirDefenseV1EnvConfig | None,
    train_config: AirDefenseV1PPOConfig | None,
    save_path: str | Path | None,
    callback: Any | None,
    tb_log_name: str | None,
) -> Any:
    config = train_config or default_air_defense_v1_ppo_config()
    env = AirDefenseResourceAssignmentEnvV1(config=env_config)
    model_class = _load_algorithm_class(algorithm)
    model = model_class(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs={"net_arch": list(config.net_arch)},
        seed=config.seed,
        device=config.device,
        verbose=config.verbose,
        tensorboard_log=config.tensorboard_log,
    )
    learn_kwargs: dict[str, Any] = {
        "total_timesteps": config.total_timesteps,
        "progress_bar": config.progress_bar,
        "callback": callback,
    }
    if tb_log_name is not None:
        learn_kwargs["tb_log_name"] = tb_log_name
    model.learn(**learn_kwargs)
    if save_path is not None:
        save_model(model, save_path)
    env.close()
    return model


def save_model(model: Any, save_path: str | Path) -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return path


def _load_algorithm_class(algorithm: AlgorithmName) -> Any:
    if algorithm == "ppo":
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:  # pragma: no cover - exercised only without deps
            raise ImportError(
                "stable-baselines3 is required for PPO. "
                "Install the rein-learning environment dependencies first."
            ) from exc
        return PPO
    if algorithm == "maskable_ppo":
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as exc:  # pragma: no cover - exercised only without deps
            raise ImportError(
                "sb3-contrib is required for Maskable PPO. "
                "Install the rein-learning environment dependencies first."
            ) from exc
        return MaskablePPO
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _looks_like_maskable_model(model: Any) -> bool:
    return model.__class__.__name__ == "MaskablePPO"


def _aggregate_episode_metrics(
    episode_metrics: list[dict[str, float]],
) -> dict[str, float]:
    rewards = np.asarray([metrics["total_reward"] for metrics in episode_metrics])
    steps = np.asarray([metrics["steps"] for metrics in episode_metrics])
    intercepted = np.asarray([metrics["num_intercepted"] for metrics in episode_metrics])
    leaked = np.asarray([metrics["num_leaked"] for metrics in episode_metrics])
    targets = np.asarray([metrics["num_targets"] for metrics in episode_metrics])
    damage = np.asarray([metrics["total_damage"] for metrics in episode_metrics])
    ammo_used = np.asarray([metrics["ammo_used"] for metrics in episode_metrics])
    shots = np.asarray([metrics["shots"] for metrics in episode_metrics])
    hits = np.asarray([metrics["hits"] for metrics in episode_metrics])
    invalid_actions = np.asarray([metrics["invalid_actions"] for metrics in episode_metrics])
    success = np.asarray([metrics["success"] for metrics in episode_metrics])

    return {
        "episodes": float(len(episode_metrics)),
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_steps": float(np.mean(steps)),
        "success_rate": float(np.mean(success)),
        "intercept_rate": float(np.sum(intercepted) / np.sum(targets)),
        "leak_rate": float(np.sum(leaked) / np.sum(targets)),
        "avg_total_damage": float(np.mean(damage)),
        "avg_ammo_used": float(np.mean(ammo_used)),
        "avg_shots": float(np.mean(shots)),
        "hit_rate_per_shot": float(np.sum(hits) / max(1, np.sum(shots))),
        "avg_invalid_actions": float(np.mean(invalid_actions)),
    }


def main() -> None:
    config = default_air_defense_v1_ppo_config()
    model = train_maskable_ppo(train_config=config)
    metrics = evaluate_air_defense_v1_model(
        model,
        episodes=20,
        seed=1_000,
        use_action_masks=True,
    )
    print("AirDefense v1 Maskable PPO evaluation:")
    print(
        f"avg_reward={metrics['avg_reward']:.2f}, "
        f"success={metrics['success_rate']:.2f}, "
        f"intercept={metrics['intercept_rate']:.2f}, "
        f"leak={metrics['leak_rate']:.2f}, "
        f"damage={metrics['avg_total_damage']:.2f}, "
        f"shots={metrics['avg_shots']:.2f}, "
        f"hit_per_shot={metrics['hit_rate_per_shot']:.2f}, "
        f"invalid={metrics['avg_invalid_actions']:.2f}"
    )


if __name__ == "__main__":
    main()
