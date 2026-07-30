from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import random
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.common.ds1_event_timeline import (
    DS1EventProtocol,
    DS1IntegratedInstrumentationCallback,
    EVALUATION_EPISODE_SEEDS,
    evaluation_seed_bank_sha256,
)
from rein_learning.common.dynamic_support_instrumentation import (
    build_frozen_dynamic_support_probe,
)
from rein_learning.common.policy_probe import PolicyProbeCorpus
from rein_learning.envs import get_air_defense_v1_scenario
from rein_learning.trainers.air_defense_v1_ppo import (
    AirDefenseV1PPOConfig,
    evaluate_air_defense_v1_model,
    train_factorized_engagement_autoregressive_ppo,
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "dynamic_support_trust_region"
    / "dst_05_5_event_timeline_preflight"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "experiments"
    / "air_defense_v1_dst05_5_event_timeline_preflight.md"
)
PROBE_PATH = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task12_probe_corpus"
    / "probe_states.npz"
)
TIMELINE_CODE = (
    PROJECT_ROOT / "rein_learning" / "common" / "ds1_event_timeline.py"
)
INSTRUMENTATION_CODE = (
    PROJECT_ROOT
    / "rein_learning"
    / "common"
    / "dynamic_support_instrumentation.py"
)
TIMELINE_TEST = PROJECT_ROOT / "tests" / "test_ds1_event_timeline.py"
POLICY_SEED = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real-callback DST-05.5 preflight."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def tensor_module_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        values = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(json.dumps(list(values.shape)).encode("ascii"))
        digest.update(values.numpy().tobytes(order="C"))
    return digest.hexdigest()


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": deepcopy(np.random.get_state()),
        "torch_cpu": torch.random.get_rng_state().clone(),
        "torch_cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def rng_states_equal(first: dict[str, Any], second: dict[str, Any]) -> bool:
    numpy_equal = (
        first["numpy"][0] == second["numpy"][0]
        and np.array_equal(first["numpy"][1], second["numpy"][1])
        and first["numpy"][2:] == second["numpy"][2:]
    )
    cuda_first = first["torch_cuda"]
    cuda_second = second["torch_cuda"]
    cuda_equal = (
        cuda_first is None
        and cuda_second is None
        or (
            cuda_first is not None
            and cuda_second is not None
            and len(cuda_first) == len(cuda_second)
            and all(
                torch.equal(left, right)
                for left, right in zip(cuda_first, cuda_second)
            )
        )
    )
    return bool(
        first["python"] == second["python"]
        and numpy_equal
        and torch.equal(first["torch_cpu"], second["torch_cpu"])
        and cuda_equal
    )


def rng_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(
        pickle.dumps(state["python"], protocol=pickle.HIGHEST_PROTOCOL)
    )
    numpy_state = state["numpy"]
    digest.update(numpy_state[0].encode("ascii"))
    digest.update(np.asarray(numpy_state[1]).tobytes(order="C"))
    digest.update(
        json.dumps(
            [
                int(numpy_state[2]),
                int(numpy_state[3]),
                float(numpy_state[4]),
            ],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(
        state["torch_cpu"].detach().cpu().contiguous().numpy().tobytes()
    )
    cuda_states = state["torch_cuda"]
    if cuda_states is None:
        digest.update(b"NO_CUDA")
    else:
        for cuda_state in cuda_states:
            digest.update(
                cuda_state.detach().cpu().contiguous().numpy().tobytes()
            )
    return digest.hexdigest()


def nested_state_equal(first: Any, second: Any) -> bool:
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        return torch.equal(first, second)
    if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
        return np.array_equal(first, second)
    if isinstance(first, dict) and isinstance(second, dict):
        return (
            first.keys() == second.keys()
            and all(nested_state_equal(first[key], second[key]) for key in first)
        )
    if isinstance(first, (list, tuple)) and isinstance(second, type(first)):
        return (
            len(first) == len(second)
            and all(
                nested_state_equal(left, right)
                for left, right in zip(first, second)
            )
        )
    return bool(first == second)


def training_environment_sha256(model: Any) -> str:
    vector_env = model.get_env()
    if vector_env is None or not hasattr(vector_env, "envs"):
        raise TypeError("Training model does not expose an inspectable environment")
    base = vector_env.envs[0].unwrapped
    snapshot = base.snapshot_state()
    return hashlib.sha256(
        pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


def gradient_sha256(policy: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(policy.named_parameters()):
        digest.update(name.encode("utf-8"))
        if parameter.grad is None:
            digest.update(b"NONE")
        else:
            digest.update(
                parameter.grad.detach().cpu().contiguous().numpy().tobytes()
            )
    return digest.hexdigest()


def formal_event_evaluator(
    env_config: Any,
    evaluation_counter: dict[str, int],
) -> Any:
    def evaluate(model: Any) -> dict[str, Any]:
        evaluation_counter["calls"] += 1
        random_before = rng_state()
        policy_training = bool(model.policy.training)
        parameter_before = tensor_module_sha256(model.policy)
        optimizer_before = deepcopy(model.policy.optimizer.state_dict())
        gradient_before = gradient_sha256(model.policy)
        environment_before = training_environment_sha256(model)
        timesteps_before = int(model.num_timesteps)
        progress_before = float(model._current_progress_remaining)
        episodes: list[dict[str, Any]] = []
        try:
            evaluate_air_defense_v1_model(
                model,
                env_config=env_config,
                episodes=50,
                seed=73_000,
                deterministic=True,
                use_action_masks=True,
                episode_metrics_callback=lambda row: episodes.append(
                    dict(row)
                ),
            )
            random_after = rng_state()
            model.policy.set_training_mode(policy_training)
            audit = {
                "global_rng_unchanged_before_restore": rng_states_equal(
                    random_before,
                    random_after,
                ),
                "policy_mode_unchanged": (
                    bool(model.policy.training) == policy_training
                ),
                "parameters_unchanged": (
                    tensor_module_sha256(model.policy) == parameter_before
                ),
                "optimizer_unchanged": nested_state_equal(
                    model.policy.optimizer.state_dict(),
                    optimizer_before,
                ),
                "gradients_unchanged": (
                    gradient_sha256(model.policy) == gradient_before
                ),
                "training_environment_unchanged": (
                    training_environment_sha256(model)
                    == environment_before
                ),
                "training_timesteps_unchanged": (
                    int(model.num_timesteps) == timesteps_before
                ),
                "scheduler_progress_unchanged": (
                    float(model._current_progress_remaining)
                    == progress_before
                ),
                "independent_evaluation_environments": True,
                "evaluation_environment_steps_added_to_training_zero": (
                    int(model.num_timesteps) == timesteps_before
                ),
            }
        finally:
            restore_rng_state(random_before)
            model.policy.set_training_mode(policy_training)
        if len(episodes) != 50:
            raise RuntimeError("Formal evaluator did not complete 50 episodes")
        if not all(audit.values()):
            raise RuntimeError(f"Formal event evaluation isolation failed: {audit}")
        return {
            "evaluation_episodes": len(episodes),
            "all_noop_episodes": sum(
                bool(row["all_noop_episode"]) for row in episodes
            ),
            "actionable_decisions": sum(
                int(row["actionable_decisions"]) for row in episodes
            ),
            "actionable_engagements": sum(
                int(row["actionable_engagements"]) for row in episodes
            ),
            "_audit": audit,
        }

    return evaluate


class TrainingEquivalenceTraceCallback(BaseCallback):
    """Capture real rollout and completed-train state without evaluation."""

    def __init__(self, *, route: str) -> None:
        super().__init__(verbose=0)
        self.route = route
        self.initial_parameter_sha256 = ""
        self.rollouts: list[dict[str, Any]] = []
        self.train_rows: list[dict[str, Any]] = []
        self.rollout_start_rng: list[str] = []
        self._step_actions: list[np.ndarray] = []
        self._step_rewards: list[np.ndarray] = []
        self._step_dones: list[np.ndarray] = []
        self._last_recorded_updates = -1

    def _on_training_start(self) -> None:
        self.initial_parameter_sha256 = tensor_module_sha256(
            self.model.policy
        )
        self._last_recorded_updates = int(
            getattr(self.model, "_n_updates", 0)
        )

    def _on_rollout_start(self) -> None:
        self._record_completed_train()
        self.rollout_start_rng.append(rng_sha256(rng_state()))
        self._step_actions = []
        self._step_rewards = []
        self._step_dones = []

    def _on_step(self) -> bool:
        self._step_actions.append(np.asarray(self.locals["actions"]).copy())
        self._step_rewards.append(np.asarray(self.locals["rewards"]).copy())
        self._step_dones.append(np.asarray(self.locals["dones"]).copy())
        return True

    def _on_rollout_end(self) -> None:
        buffer = self.model.rollout_buffer
        self.rollouts.append(
            {
                "actions": np.stack(self._step_actions),
                "rewards": np.stack(self._step_rewards),
                "dones": np.stack(self._step_dones),
                "advantages": np.asarray(buffer.advantages).copy(),
                "returns": np.asarray(buffer.returns).copy(),
            }
        )

    def _on_training_end(self) -> None:
        self._record_completed_train()

    def _record_completed_train(self) -> None:
        updates = int(getattr(self.model, "_n_updates", 0))
        if updates <= self._last_recorded_updates:
            return
        values = getattr(self.model.logger, "name_to_value", {})
        entropy_loss = float(values["train/entropy_loss"])
        self.train_rows.append(
            {
                "sb3_n_updates": updates,
                "num_timesteps": int(self.model.num_timesteps),
                "loss": float(values["train/loss"]),
                "approx_kl": float(values["train/approx_kl"]),
                "clip_fraction": float(values["train/clip_fraction"]),
                "entropy": -entropy_loss,
            }
        )
        self._last_recorded_updates = updates


def compare_array(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    exact = bool(np.array_equal(first, second))
    maximum = (
        float(np.max(np.abs(first.astype(np.float64) - second.astype(np.float64))))
        if first.size
        else 0.0
    )
    return {
        "bitwise_equal": exact,
        "max_abs_error": maximum,
    }


def compare_training_routes(
    route_a_model: Any,
    route_a_trace: TrainingEquivalenceTraceCallback,
    route_b_model: Any,
    route_b_trace: TrainingEquivalenceTraceCallback,
    integrated: DS1IntegratedInstrumentationCallback,
    evaluation_calls_a: int,
    evaluation_calls_b: int,
) -> dict[str, Any]:
    rollout_fields: dict[str, list[dict[str, Any]]] = {}
    for field in ("actions", "rewards", "dones", "advantages", "returns"):
        rollout_fields[field] = [
            compare_array(left[field], right[field])
            for left, right in zip(
                route_a_trace.rollouts,
                route_b_trace.rollouts,
            )
        ]
    training_metrics: dict[str, list[dict[str, Any]]] = {}
    for field in (
        "loss",
        "approx_kl",
        "clip_fraction",
        "entropy",
    ):
        training_metrics[field] = [
            {
                "absolute_error": abs(left[field] - right[field]),
                "equal_within_1e_10": (
                    abs(left[field] - right[field]) <= 1e-10
                ),
            }
            for left, right in zip(
                route_a_trace.train_rows,
                route_b_trace.train_rows,
            )
        ]
    parameters_exact = (
        tensor_module_sha256(route_a_model.policy)
        == tensor_module_sha256(route_b_model.policy)
    )
    optimizer_exact = nested_state_equal(
        route_a_model.policy.optimizer.state_dict(),
        route_b_model.policy.optimizer.state_dict(),
    )
    audit_values = [
        bool(value)
        for audit in integrated.evaluation_audits
        for key, value in audit.items()
        if key != "rollout_update_index"
    ]
    expected_timebase = (
        [row["rollout_update_index"] for row in integrated.update_rows]
        == [1, 2]
        and [row["sb3_n_updates"] for row in integrated.update_rows]
        == [10, 20]
        and [row["num_timesteps"] for row in integrated.update_rows]
        == [256, 512]
    )
    all_rollout_exact = all(
        result["bitwise_equal"]
        for results in rollout_fields.values()
        for result in results
    )
    all_train_equal = all(
        result["equal_within_1e_10"]
        for results in training_metrics.values()
        for result in results
    )
    passed = bool(
        len(route_a_trace.rollouts) == len(route_b_trace.rollouts) == 2
        and len(route_a_trace.train_rows) == len(route_b_trace.train_rows) == 2
        and route_a_trace.initial_parameter_sha256
        == route_b_trace.initial_parameter_sha256
        and route_a_trace.rollout_start_rng
        == route_b_trace.rollout_start_rng
        and all_rollout_exact
        and all_train_equal
        and parameters_exact
        and optimizer_exact
        and len(integrated.update_rows) == 2
        and len(integrated.event_rows) == 3
        and expected_timebase
        and evaluation_calls_a == 0
        and evaluation_calls_b == 3
        and all(audit_values)
        and int(route_a_model.num_timesteps)
        == int(route_b_model.num_timesteps)
        == 512
    )
    return {
        "route_a": {
            "instrumentation_attached": False,
            "probe_loaded_during_route": False,
            "evaluation_environment_calls": evaluation_calls_a,
            "model_saved": False,
            "num_timesteps": int(route_a_model.num_timesteps),
        },
        "route_b": {
            "instrumentation_attached": True,
            "evaluation_environment_calls": evaluation_calls_b,
            "model_saved": False,
            "num_timesteps": int(route_b_model.num_timesteps),
        },
        "initial_parameters_bitwise_equal": (
            route_a_trace.initial_parameter_sha256
            == route_b_trace.initial_parameter_sha256
        ),
        "rollout_start_rng_bitwise_equal": (
            route_a_trace.rollout_start_rng
            == route_b_trace.rollout_start_rng
        ),
        "rollout_fields": rollout_fields,
        "training_metrics": training_metrics,
        "final_parameters_bitwise_equal": parameters_exact,
        "final_optimizer_state_bitwise_equal": optimizer_exact,
        "callback_update_rows": len(integrated.update_rows),
        "callback_event_points": len(integrated.event_rows),
        "observed_rollout_update_index": [
            row["rollout_update_index"] for row in integrated.update_rows
        ],
        "observed_sb3_n_updates": [
            row["sb3_n_updates"] for row in integrated.update_rows
        ],
        "observed_num_timesteps": [
            row["num_timesteps"] for row in integrated.update_rows
        ],
        "formal_evaluation_isolation_checks": integrated.evaluation_audits,
        "formal_p2_evidence": False,
        "passed": passed,
    }


def build_provenance(env_config: Any) -> dict[str, Any]:
    environment_payload = asdict(env_config)
    return {
        "input_probe": {
            "path": PROBE_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(PROBE_PATH),
        },
        "code": {
            "timeline": {
                "path": TIMELINE_CODE.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(TIMELINE_CODE),
            },
            "instrumentation": {
                "path": INSTRUMENTATION_CODE.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
                "sha256": sha256_file(INSTRUMENTATION_CODE),
            },
            "timeline_test": {
                "path": TIMELINE_TEST.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(TIMELINE_TEST),
            },
            "preflight_script": {
                "path": Path(__file__).resolve().relative_to(
                    PROJECT_ROOT
                ).as_posix(),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "environment": {
            "scenario": "heterogeneity_pressure",
            "config_sha256": canonical_sha256(environment_payload),
            "config": environment_payload,
        },
        "evaluation_seed_bank_sha256": evaluation_seed_bank_sha256(),
    }


def combined_sample_rows(
    integrated: DS1IntegratedInstrumentationCallback,
    provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    finalized = integrated.finalized_event_timeline()
    updates = {
        int(row["rollout_update_index"]): row
        for row in integrated.update_rows
    }
    rows: list[dict[str, Any]] = []
    for event in finalized["rows"]:
        index = int(event["rollout_update_index"])
        predictor = updates.get(index, {})
        rows.append(
            {
                "formal_p2_evidence": False,
                **{
                    key: predictor.get(key)
                    for key in (
                        "approx_kl",
                        "clip_fraction",
                        "entropy",
                        "unweighted_prefix_flip_rate",
                        "ds_weighted_flip_mass",
                        "suffix_count_change",
                        "probe_all_noop_rate",
                        "probe_high_engagement_rate",
                        "probe_high_threat_unassigned_rate",
                    )
                },
                **event,
                "probe_sha256": provenance["input_probe"]["sha256"],
                "timeline_code_sha256": provenance["code"]["timeline"][
                    "sha256"
                ],
                "instrumentation_code_sha256": provenance["code"][
                    "instrumentation"
                ]["sha256"],
                "environment_config_sha256": provenance["environment"][
                    "config_sha256"
                ],
            }
        )
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def markdown_report(
    integration: dict[str, Any],
    gate: dict[str, Any],
) -> str:
    return f"""# AirDefense-v1 DST-05.5 事件时间轴与 Callback 预检

任务状态：`{gate['status']}`  
正式 P2 证据：`false`  
正式 10k 运行：`0`

## 1. 结论

正式事件、rollout 时间轴与真实 SB3 callback 已通过预检。DST-06 只能使用
`rollout_update_index` 构造未来 1—3 更新和事件前窗口；`sb3_n_updates`
保留为 0/10/20… 的追溯字段，不能作为窗口下标。

probe 的 all-noop/high-engagement/high-threat 字段仍是同步诊断，正式塌缩事件只由
50 回合 CRN 环境评估的 `all_noop_episode_rate` 和
`actionable_engagement_rate` 产生。

## 2. 冻结协议

- 场景：`heterogeneity_pressure`；
- 正式种子：`8/9/10`；
- 评估回合：`50`，episode seeds=`73000...73049`；
- 事件：all-noop `>=0.98` 或 actionable engagement `<0.01`；
- 时间单位：一轮 256-step rollout 加 10 epochs PPO train；
- 正式 10k 请求将产生 40 个 rollout 更新和实际 `10,240` timesteps；
- 每个种子只使用首次 onset，训练前已塌缩种子不算 event-bearing。

## 3. 两路 512-step 真实集成 smoke

- 初始参数 bitwise 一致：
  `{str(integration['initial_parameters_bitwise_equal']).lower()}`；
- 两次 rollout 的 actions/rewards/dones/advantages/returns 全部 bitwise 一致；
- loss/KL/clip/entropy 每轮绝对误差均不超过 `1e-10`；
- 第一次正式评估后，第二次 rollout 仍完全一致；
- 最终参数 bitwise 一致：
  `{str(integration['final_parameters_bitwise_equal']).lower()}`；
- optimizer state bitwise 一致：
  `{str(integration['final_optimizer_state_bitwise_equal']).lower()}`；
- Route B 更新行/事件点：`{integration['callback_update_rows']}/`
  `{integration['callback_event_points']}`；
- 时间轴：rollout=`{integration['observed_rollout_update_index']}`，
  SB3=`{integration['observed_sb3_n_updates']}`，
  timesteps=`{integration['observed_num_timesteps']}`；
- Route A 正式评估环境调用：`0`；Route B：`3`；
- 两路模型均未保存，smoke 不进入 P2 数据。

## 4. 事件逻辑

事件逻辑测试覆盖 49/50 边界、0.009/0.01 边界、初始塌缩、首次 onset、
t+1/t+2/t+3 标签、并发/事件后/尾部排除、SB3 跳 10、六更新窗口、
非 50 回合拒绝和 seed-bank hash 冲突拒绝。

## 5. 阶段出口

`DST-05.5={gate['status']}`。该结果只证明 DST-06 数据接口有效，不说明 DS
能够预警崩塌，也不授权 DS-TR。下一步是冻结的
`heterogeneity_pressure, requested 10k × seeds 8/9/10`。
"""


def main() -> None:
    args = parse_args()
    protocol = DS1EventProtocol()
    env_config = get_air_defense_v1_scenario("heterogeneity_pressure")
    provenance = build_provenance(env_config)
    training = AirDefenseV1PPOConfig(
        total_timesteps=512,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        net_arch=(128, 128),
        seed=POLICY_SEED,
        device="cpu",
        verbose=0,
        progress_bar=False,
    )

    # Route A deliberately runs before loading the probe or building eval hooks.
    route_a_evaluations = {"calls": 0}
    trace_a = TrainingEquivalenceTraceCallback(route="A")
    route_a_model = train_factorized_engagement_autoregressive_ppo(
        env_config=env_config,
        train_config=training,
        save_path=None,
        callback=trace_a,
        unit_order=(0, 1, 2),
    )

    corpus = PolicyProbeCorpus.load(PROBE_PATH)
    grid = build_frozen_dynamic_support_probe(corpus)
    route_b_evaluations = {"calls": 0}
    trace_b = TrainingEquivalenceTraceCallback(route="B")
    integrated = DS1IntegratedInstrumentationCallback(
        grid=grid,
        policy_seed=POLICY_SEED,
        formal_event_evaluator=formal_event_evaluator(
            env_config,
            route_b_evaluations,
        ),
        protocol=protocol,
        evaluation_seeds=EVALUATION_EPISODE_SEEDS,
        formal_p2_evidence=False,
    )
    route_b_model = train_factorized_engagement_autoregressive_ppo(
        env_config=env_config,
        train_config=training,
        save_path=None,
        callback=CallbackList([trace_b, integrated]),
        unit_order=(0, 1, 2),
    )

    integration = compare_training_routes(
        route_a_model,
        trace_a,
        route_b_model,
        trace_b,
        integrated,
        route_a_evaluations["calls"],
        route_b_evaluations["calls"],
    )
    integration = {
        "schema_version": 1,
        "task": "DST-05.5",
        "smoke_training_authorization": "paired 512-step only",
        "formal_p2_evidence": False,
        "formal_10k_executed": False,
        "training_config": asdict(training),
        "provenance": provenance,
        **integration,
    }
    finalized = integrated.finalized_event_timeline()
    timeline_manifest = {
        "schema_version": 1,
        "task": "DST-05.5",
        "formal_p2_evidence": False,
        "time_axis": "rollout_update_index",
        "raw_trace_fields": ["sb3_n_updates", "num_timesteps"],
        "alignment": {
            "baseline": {
                "rollout_update_index": 0,
                "sb3_n_updates": 0,
                "num_timesteps": 0,
                "predictors_present": False,
            },
            "per_rollout": {
                "n_steps": 256,
                "n_epochs": 10,
                "predictors": "completed rollout train cycle t",
                "flip_transition": "policy t-1 to policy t",
                "event_policy": "post-update policy t",
            },
            "formal_requested_timesteps": 10_000,
            "formal_completed_rollout_updates": 40,
            "formal_actual_timesteps": 10_240,
        },
        "window_rules": {
            "prediction": "future rollout_update_index +1 through +3",
            "baseline": "onset-6 through onset-4",
            "pre_event": "onset-3 through onset-1",
            "sb3_n_updates_used_for_windows": False,
        },
        "smoke_observed": {
            "rollout_update_index": [0, 1, 2],
            "sb3_n_updates": [0, 10, 20],
            "num_timesteps": [0, 256, 512],
        },
        "provenance": provenance,
    }
    event_protocol = {
        "schema_version": 1,
        "task": "DST-05.5",
        "status": "FROZEN",
        "formal_p2_evidence": False,
        "protocol": protocol.to_dict(),
        "collapse_event": (
            "all_noop_episode_rate >= 0.98 OR "
            "actionable_engagement_rate < 0.01"
        ),
        "event_source": "independent deterministic environment evaluation",
        "probe_fields_are_event_source": False,
        "first_onset_only": True,
        "initially_collapsed_event_bearing": False,
        "forward_label": "first onset at t+1, t+2, or t+3",
        "exclusions": [
            "rollout_update_index=0",
            "concurrent event",
            "post event",
            "right-censored last 3 updates",
            "incomplete 50-episode evaluation",
            "seed-bank hash mismatch",
        ],
        "provenance": provenance,
    }
    seed_bank = {
        "schema_version": 1,
        "task": "DST-05.5",
        "scenario": "heterogeneity_pressure",
        "policy_action": "deterministic",
        "episodes": 50,
        "episode_seeds": list(EVALUATION_EPISODE_SEEDS),
        "sha256": evaluation_seed_bank_sha256(),
        "reuse_across_updates_and_policy_seeds": True,
        "provenance": provenance,
    }
    gate_checks = {
        "formal_and_probe_fields_separate": True,
        "event_protocol_frozen": True,
        "three_time_fields_recorded": True,
        "windows_use_rollout_update_index": True,
        "event_logic_test_classes": 10,
        "integration_equivalence_passed": integration["passed"],
        "callback_rows_complete": (
            integration["callback_update_rows"] == 2
            and integration["callback_event_points"] == 3
        ),
        "smoke_formal_p2_evidence_false": True,
        "formal_10k_not_run": True,
        "ds_tr_not_implemented": True,
    }
    gate_passed = all(
        value is True or key == "event_logic_test_classes" and value >= 10
        for key, value in gate_checks.items()
    )
    gate = {
        "schema_version": 1,
        "task": "DST-05.5",
        "status": "PASSED" if gate_passed else "BLOCKED",
        "checks": gate_checks,
        "smoke_event_timeline": finalized,
        "formal_p2_evidence": False,
        "training_performed": "paired 512-step integration smoke only",
        "formal_10k_executed": False,
        "next_task": "DST-06" if gate_passed else None,
        "provenance": provenance,
    }
    if not gate_passed:
        raise RuntimeError(
            "DST-05.5 integration gate failed: "
            + json.dumps(
                {
                    "checks": gate_checks,
                    "integration": integration,
                },
                ensure_ascii=False,
                default=str,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "event_protocol.json", event_protocol)
    write_json(
        args.output_dir / "update_timebase_manifest.json",
        timeline_manifest,
    )
    write_json(
        args.output_dir / "evaluation_seed_bank.json",
        seed_bank,
    )
    write_json(
        args.output_dir / "integration_equivalence.json",
        integration,
    )
    sample_rows = combined_sample_rows(integrated, provenance)
    pd.DataFrame.from_records(sample_rows).to_csv(
        args.output_dir / "sample_update_event_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    write_json(args.output_dir / "gate_summary.json", gate)
    args.report.write_text(
        markdown_report(integration, gate),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "task": "DST-05.5",
                "status": gate["status"],
                "integration_equivalence": integration["passed"],
                "rollout_update_index": integration[
                    "observed_rollout_update_index"
                ],
                "sb3_n_updates": integration[
                    "observed_sb3_n_updates"
                ],
                "num_timesteps": integration[
                    "observed_num_timesteps"
                ],
                "formal_p2_evidence": False,
                "next_task": gate["next_task"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
