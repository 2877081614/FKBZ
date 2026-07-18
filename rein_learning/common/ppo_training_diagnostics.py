from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .policy_probe import PolicyProbeCorpus, evaluate_policy_probe


class PPOTrainingDiagnosticsCallback(BaseCallback):
    """Record PPO optimization and frozen-state policy statistics."""

    LOGGER_KEYS = {
        "policy_loss": "train/policy_gradient_loss",
        "value_loss": "train/value_loss",
        "entropy_loss": "train/entropy_loss",
        "approx_kl": "train/approx_kl",
        "clip_fraction": "train/clip_fraction",
        "explained_variance": "train/explained_variance",
    }

    def __init__(
        self,
        *,
        method: str,
        train_scenario: str,
        train_seed: int,
        record_freq: int,
        probe_corpus: PolicyProbeCorpus | None = None,
    ) -> None:
        super().__init__(verbose=0)
        if record_freq <= 0:
            raise ValueError("record_freq must be positive")
        self.method = method
        self.train_scenario = train_scenario
        self.train_seed = train_seed
        self.record_freq = record_freq
        self.probe_corpus = probe_corpus
        self.training_rows: list[dict[str, Any]] = []
        self.probe_rows: list[dict[str, Any]] = []
        self._next_record = record_freq
        self._last_record = -1
        self._rollout_statistics = self._empty_rollout_statistics()

    def _on_training_start(self) -> None:
        self._record(0)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        rollout_buffer = self.model.rollout_buffer
        advantages = np.asarray(rollout_buffer.advantages, dtype=np.float64).reshape(-1)
        self._rollout_statistics = {
            "advantage_mean": float(np.mean(advantages)),
            "advantage_std": float(np.std(advantages)),
            "positive_advantage_rate": float(np.mean(advantages > 0.0)),
        }

    def _on_rollout_start(self) -> None:
        if self.num_timesteps >= self._next_record:
            self._record(self.num_timesteps)
            while self._next_record <= self.num_timesteps:
                self._next_record += self.record_freq

    def _on_training_end(self) -> None:
        if self._last_record != self.num_timesteps:
            self._record(self.num_timesteps)

    def _record(self, timesteps: int) -> None:
        common = {
            "method": self.method,
            "train_scenario": self.train_scenario,
            "train_seed": self.train_seed,
            "timesteps": int(timesteps),
        }
        logger_values = getattr(self.model.logger, "name_to_value", {})
        row: dict[str, Any] = {**common}
        for output_name, logger_name in self.LOGGER_KEYS.items():
            row[output_name] = _as_float(logger_values.get(logger_name))
        row.update(self._rollout_statistics)
        row.update(self._gradient_norms())
        self.training_rows.append(row)

        if self.probe_corpus is not None:
            for probe_row in evaluate_policy_probe(
                self.model, self.probe_corpus, deterministic=True
            ):
                self.probe_rows.append({**common, **probe_row})
        self._last_record = int(timesteps)

    def _gradient_norms(self) -> dict[str, float]:
        policy = self.model.policy
        mlp_extractor = getattr(policy, "mlp_extractor", None)
        actor_modules = (
            getattr(mlp_extractor, "policy_net", None),
            getattr(policy, "action_net", None),
        )
        critic_modules = (
            getattr(mlp_extractor, "value_net", None),
            getattr(policy, "value_net", None),
        )
        return {
            "actor_gradient_norm": _module_gradient_norm(actor_modules),
            "critic_gradient_norm": _module_gradient_norm(critic_modules),
        }

    @staticmethod
    def _empty_rollout_statistics() -> dict[str, float]:
        return {
            "advantage_mean": float("nan"),
            "advantage_std": float("nan"),
            "positive_advantage_rate": float("nan"),
        }


def _module_gradient_norm(modules: tuple[Any, ...]) -> float:
    squared_norm = 0.0
    found = False
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            found = True
            squared_norm += float(parameter.grad.detach().norm(2).item()) ** 2
    return sqrt(squared_norm) if found else float("nan")


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    array = np.asarray(value)
    return float(array.reshape(-1)[0])
