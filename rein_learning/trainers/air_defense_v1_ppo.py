from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal

import numpy as np

from ..common import (
    DEFAULT_HIGH_THREAT_THRESHOLD,
    AirDefenseV1DecisionTracker,
    AirDefenseV1DiagnosticsTracker,
    aggregate_air_defense_v1_episode_metrics,
)
from ..envs import (
    AirDefenseResourceAssignmentEnvV1,
    AirDefenseV1EnvConfig,
    ConflictFreeJointActionWrapper,
)


AlgorithmName = Literal[
    "ppo",
    "maskable_ppo",
    "conflict_free_maskable_ppo",
    "autoregressive_maskable_ppo",
    "role_conditioned_autoregressive_ppo",
    "factorized_engagement_autoregressive_ppo",
]


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


def train_conflict_free_maskable_ppo(
    *,
    env_config: AirDefenseV1EnvConfig | None = None,
    train_config: AirDefenseV1PPOConfig | None = None,
    save_path: str | Path | None = None,
    callback: Any | None = None,
    tb_log_name: str | None = None,
) -> Any:
    """Train Maskable PPO on the conflict-free Discrete joint action space."""

    return _train_sb3_model(
        algorithm="conflict_free_maskable_ppo",
        env_config=env_config,
        train_config=train_config,
        save_path=save_path,
        callback=callback,
        tb_log_name=tb_log_name,
    )


def train_autoregressive_maskable_ppo(
    *,
    env_config: AirDefenseV1EnvConfig | None = None,
    train_config: AirDefenseV1PPOConfig | None = None,
    save_path: str | Path | None = None,
    callback: Any | None = None,
    tb_log_name: str | None = None,
    unit_order: tuple[int, ...] | None = None,
) -> Any:
    """Train Maskable PPO with prefix-conditioned conflict-free actions."""

    return _train_sb3_model(
        algorithm="autoregressive_maskable_ppo",
        env_config=env_config,
        train_config=train_config,
        save_path=save_path,
        callback=callback,
        tb_log_name=tb_log_name,
        unit_order=unit_order,
    )


def train_role_conditioned_autoregressive_ppo(
    *,
    env_config: AirDefenseV1EnvConfig | None = None,
    train_config: AirDefenseV1PPOConfig | None = None,
    save_path: str | Path | None = None,
    callback: Any | None = None,
    tb_log_name: str | None = None,
    unit_order: tuple[int, ...] | None = None,
) -> Any:
    """Train the shared role-conditioned autoregressive policy."""

    return _train_sb3_model(
        algorithm="role_conditioned_autoregressive_ppo",
        env_config=env_config,
        train_config=train_config,
        save_path=save_path,
        callback=callback,
        tb_log_name=tb_log_name,
        unit_order=unit_order,
    )


def train_factorized_engagement_autoregressive_ppo(
    *,
    env_config: AirDefenseV1EnvConfig | None = None,
    train_config: AirDefenseV1PPOConfig | None = None,
    save_path: str | Path | None = None,
    callback: Any | None = None,
    tb_log_name: str | None = None,
    unit_order: tuple[int, ...] | None = None,
) -> Any:
    """Train the engagement-target factorized autoregressive policy."""

    return _train_sb3_model(
        algorithm="factorized_engagement_autoregressive_ppo",
        env_config=env_config,
        train_config=train_config,
        save_path=save_path,
        callback=callback,
        tb_log_name=tb_log_name,
        unit_order=unit_order,
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
    if algorithm == "conflict_free_maskable_ppo":
        return train_conflict_free_maskable_ppo(
            env_config=env_config,
            train_config=train_config,
            save_path=save_path,
            callback=callback,
            tb_log_name=tb_log_name,
        )
    if algorithm == "autoregressive_maskable_ppo":
        return train_autoregressive_maskable_ppo(
            env_config=env_config,
            train_config=train_config,
            save_path=save_path,
            callback=callback,
            tb_log_name=tb_log_name,
        )
    if algorithm == "role_conditioned_autoregressive_ppo":
        return train_role_conditioned_autoregressive_ppo(
            env_config=env_config,
            train_config=train_config,
            save_path=save_path,
            callback=callback,
            tb_log_name=tb_log_name,
        )
    if algorithm == "factorized_engagement_autoregressive_ppo":
        return train_factorized_engagement_autoregressive_ppo(
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
    high_threat_threshold: float = DEFAULT_HIGH_THREAT_THRESHOLD,
    episode_metrics_callback: Callable[
        [dict[str, float | int | bool]], None
    ]
    | None = None,
    decision_trace_callback: Callable[[int, dict[str, object]], None]
    | None = None,
    leak_attribution_callback: Callable[[int, dict[str, object]], None]
    | None = None,
) -> dict[str, float]:
    """Evaluate a trained SB3-style model with the same metrics as rule baselines."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if env_factory is None:
        env_factory = lambda: AirDefenseResourceAssignmentEnvV1(config=env_config)
    if use_action_masks is None:
        use_action_masks = _looks_like_maskable_model(model)

    episode_metrics: list[dict[str, float | int | bool]] = []
    for episode_index in range(episodes):
        episode_seed = seed + episode_index
        env = env_factory()
        obs, initial_info = env.reset(seed=episode_seed)
        base_env = env.unwrapped
        if not isinstance(base_env, AirDefenseResourceAssignmentEnvV1):
            raise TypeError(
                "AirDefense v1 evaluation requires an "
                "AirDefenseResourceAssignmentEnvV1 base environment"
            )
        unit_order = _model_unit_order(model, base_env.num_defense_units)
        decision_tracker = AirDefenseV1DecisionTracker(
            unit_order=unit_order,
            num_units=base_env.num_defense_units,
            num_targets=base_env.num_targets,
            high_threat_threshold=high_threat_threshold,
        )
        initial_ammo = int(initial_info["ammo_remaining"])
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        shots = 0
        hits = 0
        invalid_actions = 0
        unit_decisions = 0
        actionable_decisions = 0
        engagements = 0
        actionable_engagements = 0
        decision_time_seconds = 0.0
        diagnostics = AirDefenseV1DiagnosticsTracker(
            high_threat_threshold=high_threat_threshold
        )
        info = initial_info

        while not (terminated or truncated):
            base_action_mask = np.asarray(base_env.action_masks(), dtype=bool).reshape(
                base_env.num_defense_units, -1
            )
            actionable_units = np.any(
                base_action_mask[:, : base_env.num_targets], axis=1
            )
            decision_started = perf_counter()
            if use_action_masks:
                action, _ = model.predict(
                    obs,
                    deterministic=deterministic,
                    action_masks=env.action_masks(),
                )
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            decision_time_seconds += perf_counter() - decision_started

            joint_action = _decode_evaluation_action(env, action)
            joint_action_array = np.asarray(joint_action, dtype=np.int64).reshape(-1)
            engaged_units = joint_action_array != base_env.num_targets
            unit_decisions += base_env.num_defense_units
            actionable_decisions += int(np.sum(actionable_units))
            engagements += int(np.sum(engaged_units))
            actionable_engagements += int(
                np.sum(engaged_units & actionable_units)
            )
            decision_rows = decision_tracker.before_step(base_env, joint_action)
            obs, reward, terminated, truncated, info = env.step(action)
            decision_tracker.after_step(base_env, info, decision_rows)
            if decision_trace_callback is not None:
                for row in decision_rows:
                    decision_trace_callback(episode_index, row.copy())
            total_reward += float(reward)
            steps += 1
            shots += int(info["shots"])
            hits += int(info["hits"])
            invalid_actions += int(info["invalid_actions"])
            diagnostics.record_step(info)

        ammo_used = initial_ammo - int(info["ammo_remaining"])
        if leak_attribution_callback is not None:
            for row in decision_tracker.finalize_leak_attributions(base_env):
                leak_attribution_callback(episode_index, row.copy())
        raw_metrics: dict[str, float | int | bool] = {
            "total_reward": total_reward,
            "steps": steps,
            "num_targets": base_env.num_targets,
            "num_intercepted": int(info["num_intercepted"]),
            "num_leaked": int(info["num_leaked"]),
            "total_damage": float(info["total_damage"]),
            "ammo_used": ammo_used,
            "shots": shots,
            "hits": hits,
            "invalid_actions": invalid_actions,
            "unit_decisions": unit_decisions,
            "actionable_decisions": actionable_decisions,
            "engagements": engagements,
            "actionable_engagements": actionable_engagements,
            "all_noop_episode": bool(
                actionable_decisions > 0 and engagements == 0
            ),
            "success": bool(
                info["num_alive"] == 0 and info["total_damage"] == 0.0
            ),
            "decision_time_seconds": decision_time_seconds,
            "decision_time_ms": 1_000.0 * decision_time_seconds / steps,
            **diagnostics.finalize(base_env, ammo_used=ammo_used),
        }
        episode_metrics.append(raw_metrics)
        if episode_metrics_callback is not None:
            episode_metrics_callback(raw_metrics.copy())
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
    unit_order: tuple[int, ...] | None = None,
) -> Any:
    config = train_config or default_air_defense_v1_ppo_config()
    env = _create_training_environment(algorithm, env_config)
    model_class = _load_algorithm_class(algorithm)
    policy: Any = "MlpPolicy"
    if algorithm in {
        "autoregressive_maskable_ppo",
        "role_conditioned_autoregressive_ppo",
        "factorized_engagement_autoregressive_ppo",
    }:
        from ..algorithms.policy_gradient.autoregressive_ppo import (
            AutoregressiveMaskableActorCriticPolicy,
        )

        policy = AutoregressiveMaskableActorCriticPolicy
    if algorithm == "role_conditioned_autoregressive_ppo":
        from ..algorithms.policy_gradient.role_conditioned_autoregressive_ppo import (
            RoleConditionedAutoregressiveActorCriticPolicy,
        )

        policy = RoleConditionedAutoregressiveActorCriticPolicy
    if algorithm == "factorized_engagement_autoregressive_ppo":
        from ..algorithms.policy_gradient.factorized_engagement_ppo import (
            FactorizedEngagementActorCriticPolicy,
        )

        policy = FactorizedEngagementActorCriticPolicy
    policy_kwargs: dict[str, Any] = {"net_arch": list(config.net_arch)}
    if algorithm in {
        "autoregressive_maskable_ppo",
        "role_conditioned_autoregressive_ppo",
        "factorized_engagement_autoregressive_ppo",
    }:
        policy_kwargs["unit_order"] = unit_order
    model = model_class(
        policy,
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
        policy_kwargs=policy_kwargs,
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
    if algorithm in {"maskable_ppo", "conflict_free_maskable_ppo"}:
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as exc:  # pragma: no cover - exercised only without deps
            raise ImportError(
                "sb3-contrib is required for Maskable PPO. "
                "Install the rein-learning environment dependencies first."
            ) from exc
        return MaskablePPO
    if algorithm == "autoregressive_maskable_ppo":
        from ..algorithms.policy_gradient.autoregressive_ppo import (
            AutoregressiveMaskablePPO,
        )

        return AutoregressiveMaskablePPO
    if algorithm == "role_conditioned_autoregressive_ppo":
        from ..algorithms.policy_gradient.role_conditioned_autoregressive_ppo import (
            RoleConditionedAutoregressiveMaskablePPO,
        )

        return RoleConditionedAutoregressiveMaskablePPO
    if algorithm == "factorized_engagement_autoregressive_ppo":
        from ..algorithms.policy_gradient.factorized_engagement_ppo import (
            FactorizedEngagementMaskablePPO,
        )

        return FactorizedEngagementMaskablePPO
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _create_training_environment(
    algorithm: AlgorithmName,
    env_config: AirDefenseV1EnvConfig | None,
) -> Any:
    env = AirDefenseResourceAssignmentEnvV1(config=env_config)
    if algorithm == "conflict_free_maskable_ppo":
        return ConflictFreeJointActionWrapper(env)
    return env


def _looks_like_maskable_model(model: Any) -> bool:
    return model.__class__.__name__ in {
        "MaskablePPO",
        "AutoregressiveMaskablePPO",
        "RoleConditionedAutoregressiveMaskablePPO",
        "FactorizedEngagementMaskablePPO",
    }


def _model_unit_order(model: Any, num_units: int) -> tuple[int, ...]:
    signature = getattr(model, "action_generator_signature", None)
    if isinstance(signature, dict) and "unit_order" in signature:
        return tuple(int(value) for value in signature["unit_order"])
    return tuple(range(num_units))


def _decode_evaluation_action(env: Any, action: Any) -> Any:
    if isinstance(env, ConflictFreeJointActionWrapper):
        encoded_action = int(np.asarray(action).reshape(-1)[0])
        return env.codec.decode(encoded_action)
    return action


def _aggregate_episode_metrics(
    episode_metrics: list[dict[str, float | int | bool]],
) -> dict[str, float]:
    return aggregate_air_defense_v1_episode_metrics(episode_metrics)


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
