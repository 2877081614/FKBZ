from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rein_learning.algorithms.policy_gradient.factorized_engagement_ppo import (
    FactorizedEngagementMaskablePPO,
)
from rein_learning.common.dynamic_support_instrumentation import (
    build_frozen_dynamic_support_probe,
    compute_dynamic_support_update_metrics,
    evaluate_policy_on_frozen_probe,
    frozen_probe_coverage,
)
from rein_learning.common.policy_probe import PolicyProbeCorpus


PROBE_DIR = (
    PROJECT_ROOT / "results" / "air_defense_v1" / "task12_probe_corpus"
)
TASK12_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "task12_factorized_screening_30k_3seeds"
)
REFERENCE_MODEL = (
    TASK12_DIR
    / "models"
    / "medium"
    / "factorized_engagement_ar_ppo_order_012_seed8.zip"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "air_defense_v1"
    / "dynamic_support_trust_region"
    / "dst_05_instrumentation"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "experiments"
    / "air_defense_v1_ds_update_instrumentation_audit.md"
)
SEED = 20260730


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the zero-training DST-05 instrumentation audit."
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


def parameter_sha256(policy: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        values = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(json.dumps(list(values.shape)).encode("ascii"))
        digest.update(values.numpy().tobytes(order="C"))
    return digest.hexdigest()


def snapshots_equal(first: Any, second: Any) -> dict[str, Any]:
    discrete_equal = bool(
        np.array_equal(first.context_ids, second.context_ids)
        and np.array_equal(first.context_actions, second.context_actions)
        and np.array_equal(first.state_ids, second.state_ids)
        and np.array_equal(first.joint_actions, second.joint_actions)
    )
    maximum_error = max(
        float(
            np.max(
                np.abs(
                    first.context_probabilities
                    - second.context_probabilities
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    first.context_engage_probabilities
                    - second.context_engage_probabilities
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    first.context_entropies
                    - second.context_entropies
                )
            )
        ),
    )
    return {
        "discrete_events_bitwise_identical": discrete_equal,
        "continuous_max_abs_error": maximum_error,
        "continuous_tolerance": 1e-8,
        "passed": discrete_equal and maximum_error <= 1e-8,
    }


def policy_parameters_equal(
    first: torch.nn.Module,
    second: torch.nn.Module,
) -> tuple[bool, float]:
    maximum = 0.0
    exact = True
    first_state = first.state_dict()
    second_state = second.state_dict()
    if first_state.keys() != second_state.keys():
        return False, float("inf")
    for name in first_state:
        left = first_state[name].detach().cpu()
        right = second_state[name].detach().cpu()
        exact = exact and torch.equal(left, right)
        if left.is_floating_point():
            maximum = max(
                maximum,
                float(torch.max(torch.abs(left - right)).item()),
            )
    return exact, maximum


def rng_signature() -> dict[str, str]:
    numpy_state = np.random.get_state()
    return {
        "python": hashlib.sha256(
            repr(random.getstate()).encode("utf-8")
        ).hexdigest(),
        "numpy": hashlib.sha256(
            numpy_state[1].tobytes()
            + repr(numpy_state[0:1] + numpy_state[2:]).encode("utf-8")
        ).hexdigest(),
        "torch": hashlib.sha256(
            torch.random.get_rng_state().cpu().numpy().tobytes()
        ).hexdigest(),
    }


def synthetic_update(
    model: FactorizedEngagementMaskablePPO,
    corpus: PolicyProbeCorpus,
    *,
    instrument: Any | None,
) -> dict[str, Any]:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model.policy.train()
    rng_before = rng_signature()
    snapshot_before = None
    if instrument is not None:
        snapshot_before = evaluate_policy_on_frozen_probe(
            model,
            instrument,
        )
    rng_after_instrument = rng_signature()
    observations, _ = model.policy.obs_to_tensor(
        corpus.observations[:32]
    )
    masks = corpus.action_masks[:32]
    actions, _, _ = model.policy.forward(
        observations,
        deterministic=False,
        action_masks=masks,
    )
    values, log_prob, entropy = model.policy.evaluate_actions(
        observations,
        actions,
        action_masks=torch.as_tensor(
            masks,
            device=observations.device,
            dtype=torch.bool,
        ),
    )
    loss = (
        -log_prob.mean()
        + 0.5 * values.square().mean()
        - 0.01 * entropy.mean()
    )
    optimizer = torch.optim.SGD(model.policy.parameters(), lr=1e-6)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    snapshot_after = (
        evaluate_policy_on_frozen_probe(model, instrument)
        if instrument is not None
        else None
    )
    return {
        "actions": actions.detach().cpu().numpy(),
        "loss": float(loss.detach().cpu().item()),
        "rng_before": rng_before,
        "rng_after_instrument": rng_after_instrument,
        "snapshot_before": snapshot_before,
        "snapshot_after": snapshot_after,
    }


def margin_coverage_evidence() -> dict[str, Any]:
    dynamics = pd.read_csv(TASK12_DIR / "probe_dynamics.csv")
    selected = dynamics[
        dynamics["method"].eq("factorized_engagement_ar_ppo_order_012")
        & dynamics["timesteps"].eq(0)
        & dynamics["probe_scenario"].isin(
            ("time_pressure", "heterogeneity_pressure")
        )
    ].copy()
    mixed = selected[
        selected["deterministic_engagement_rate"].between(
            0.0, 1.0, inclusive="neither"
        )
    ]
    return {
        "source": (
            TASK12_DIR / "probe_dynamics.csv"
        ).relative_to(PROJECT_ROOT).as_posix(),
        "source_sha256": sha256_file(TASK12_DIR / "probe_dynamics.csv"),
        "filter": (
            "factorized_engagement_ar_ppo_order_012, timesteps=0, "
            "core scenarios"
        ),
        "rows": int(len(selected)),
        "mixed_margin_rows": int(len(mixed)),
        "mixed_margin_policy_seeds": sorted(
            int(value) for value in mixed["train_seed"].unique()
        ),
        "minimum_deterministic_engagement_rate": float(
            selected["deterministic_engagement_rate"].min()
        ),
        "maximum_deterministic_engagement_rate": float(
            selected["deterministic_engagement_rate"].max()
        ),
        "passed": bool(len(mixed) > 0),
    }


def replay_feasibility() -> dict[str, Any]:
    dynamics = pd.read_csv(TASK12_DIR / "training_dynamics.csv")
    dynamics = dynamics[
        dynamics["method"].eq("factorized_engagement_ar_ppo_order_012")
    ]
    runs = pd.read_csv(TASK12_DIR / "runs.csv")
    runs = runs[
        runs["method"].eq("factorized_engagement_ar_ppo_order_012")
    ]
    models: dict[int, dict[str, Any]] = {}
    for _, row in runs.iterrows():
        seed = int(row["train_seed"])
        model_path = Path(str(row["model_path"]))
        models[seed] = {
            "path": model_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(model_path),
            "completed_timesteps": int(row["training_timesteps"]),
            "checkpoint_role": "final_only",
        }
    per_seed = {
        str(seed): {
            "logged_update_diagnostic_rows": int(
                len(dynamics[dynamics["train_seed"].eq(seed)])
            ),
            "saved_weight_snapshots": int(seed in models),
            "saved_weight_updates": (
                [models[seed]["completed_timesteps"]]
                if seed in models
                else []
            ),
        }
        for seed in (8, 9, 10)
    }
    return {
        "schema_version": 1,
        "task": "DST-05",
        "status": "PASS_WITH_SHORT_RUN_REQUIRED",
        "replay_insufficient": True,
        "checkpoint_to_update_alignment_exact": False,
        "target_sequence": {
            "method": "factorized_engagement_ar_ppo_order_012",
            "scenario": "heterogeneity_pressure",
            "policy_seeds": [8, 9, 10],
            "required": "consecutive completed PPO updates",
        },
        "task12_existing_evidence": {
            "training_scenario": "medium",
            "models": models,
            "per_seed": per_seed,
            "diagnostic_cadence_timesteps": 2048,
            "diagnostic_logs_contain_weights": False,
            "diagnostic_logs_contain_ds_weighted_flip_mass": False,
        },
        "exclusions": [
            {
                "source": "Task12 final models",
                "reason": (
                    "one final weight snapshot per seed cannot reconstruct "
                    "before-after completed updates"
                ),
            },
            {
                "source": "Task12 aggregate probe_dynamics.csv",
                "reason": (
                    "aggregate margins omit context-level old/new "
                    "probabilities and DS-weighted flips"
                ),
            },
            {
                "source": "other experiment final models",
                "reason": (
                    "different runs are not a temporal checkpoint sequence"
                ),
            },
        ],
        "dst06_new_training_required": True,
        "dst06_frozen_budget": "heterogeneity_pressure, 10k x seeds 8/9/10",
        "additional_seeds_allowed": False,
        "training_performed_by_dst05": False,
        "next_task": "DST-06",
    }


def write_sample_csv(path: Path, row: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def markdown_report(
    probe_manifest: dict[str, Any],
    equivalence: dict[str, Any],
    replay: dict[str, Any],
) -> str:
    coverage = probe_manifest["coverage"]
    margin = probe_manifest["historical_margin_coverage"]
    return f"""# AirDefense-v1 DST-05 更新级诊断仪表与可重放性

任务状态：`PASSED`  
训练：`0`（仅执行一次不保存的冻结批次合成梯度等价性测试）  
策略或环境语义修改：`0`

## 1. 结论

更新级只读仪表已经实现并通过不干扰性验证。现有 Task12 日志记录了优化统计和
冻结 probe 聚合量，但每个种子只有最终权重，不能恢复相邻 PPO 更新。因此
`replay_insufficient=true`，DST-06 需要执行预注册的
`heterogeneity_pressure, 10k × seeds 8/9/10` 短跑。

这不是 P2 阳性或阴性结果；DST-05 只解决测量与可重放性。

## 2. 冻结 probe

- 原始状态：`{probe_manifest['source']['num_states']}`，核心场景状态：
  `{coverage['states']}`；
- 策略无关状态—前缀上下文：`{coverage['contexts']}`，其中 DS 合格上下文：
  `{coverage['eligible_ds_contexts']}`；
- 位置覆盖：`{json.dumps(coverage['unit_position_counts'], ensure_ascii=False)}`；
- 合法动作数覆盖：`{json.dumps(coverage['legal_action_count_counts'], ensure_ascii=False)}`；
- 高威胁可达/不可达上下文：`{coverage['high_threat_reachable_contexts']}/`
  `{coverage['high_threat_unreachable_contexts']}`；
- 历史 timestep-0 聚合中同时包含 margin 两侧的核心场景行：
  `{margin['mixed_margin_rows']}`，涉及种子
  `{margin['mixed_margin_policy_seeds']}`。

probe 于 2026-07-18 已生成；本任务只按场景和所有可行前缀做确定性展开，没有
按历史塌缩位置、奖励、Q 或结果标签筛选。

## 3. 指标定义

`unweighted_prefix_flip_rate` 是 DS 合格唯一上下文上的 argmax 变化率。
`ds_weighted_flip_mass` 定义为
`mean[1(a_old != a_new) * r_old(a_new)]`，其中 `r_old` 使用 DST-01 冻结的
Jaccard 结构风险；同时额外记录完整概率质量形式的 `ds_policy_distance`。
`suffix_count_change` 是新旧 argmax 所选动作精确可行后缀数量之差的绝对值均值。
`update_id` 与 DST-01 字段字典中的 `ppo_update` 是同值别名，均指已经完成的
PPO update。`probe_all_noop_rate` 是三单元全 no-op 的状态比例，
`probe_high_engagement_rate` 是至少两个单元 engagement 的状态比例；
`probe_high_threat_unassigned_rate` 以存在合法高威胁目标的状态为分母。

最后位置只参与普通 margin/argmax 边界统计，不进入 DS 加权指标。所有聚合以唯一
`context_id` 为单位，不读取动作对表，因此动作对复制不能改变结果。

## 4. 不干扰性

- 初始参数一致：`{str(equivalence['initial_parameters_bitwise_equal']).lower()}`；
- 仪表前后训练 RNG 一致：`{str(equivalence['instrumentation_rng_unchanged']).lower()}`；
- 合成 rollout actions 一致：`{str(equivalence['rollout_actions_bitwise_equal']).lower()}`；
- loss 绝对差：`{equivalence['loss_absolute_difference']:.12g}`；
- 更新后参数 bitwise 一致：
  `{str(equivalence['updated_parameters_bitwise_equal']).lower()}`；
- 两次 probe 重放离散事件一致：
  `{str(equivalence['replay_repeatability']['discrete_events_bitwise_identical']).lower()}`；
- 连续指标最大误差：
  `{equivalence['replay_repeatability']['continuous_max_abs_error']:.12g}`；
- 环境 step 调用：`0`。

仪表是显式 opt-in callback；未附加时不建立 probe 网格、不加载 probe，也不写文件。

## 5. 历史重放判定

Task12 的 seeds 8/9/10 各有 16 行训练诊断，但每个种子只有一个 30,208 步最终
模型；日志不含权重、上下文级概率或 DS 指标。不同实验的最终模型没有被拼接为
伪时间序列。故现有证据不足以执行 P2，下一步是 DST-06 预注册短跑，而不是从
聚合曲线推断先行性。

## 6. 交付物

```text
results/air_defense_v1/dynamic_support_trust_region/dst_05_instrumentation/
  probe_manifest.json
  instrumentation_equivalence.json
  replay_feasibility.json
  sample_update_metrics.csv
```
"""


def main() -> None:
    args = parse_args()
    corpus = PolicyProbeCorpus.load(PROBE_DIR)
    grid = build_frozen_dynamic_support_probe(corpus)
    reference = FactorizedEngagementMaskablePPO.load(
        REFERENCE_MODEL,
        device="cpu",
    )
    parameter_before = parameter_sha256(reference.policy)
    first = evaluate_policy_on_frozen_probe(reference, grid)
    second = evaluate_policy_on_frozen_probe(reference, grid)
    parameter_after = parameter_sha256(reference.policy)
    repeatability = snapshots_equal(first, second)
    coverage = frozen_probe_coverage(grid, first)
    source_manifest = json.loads(
        (PROBE_DIR / "probe_manifest.json").read_text(encoding="utf-8")
    )
    probe_manifest = {
        "schema_version": 1,
        "task": "DST-05",
        "frozen_before_dst05": True,
        "selection_uses_failure_outcomes": False,
        "selection_rule": (
            "all task12 frozen states in time_pressure and "
            "heterogeneity_pressure; enumerate every feasible prefix"
        ),
        "unit_order": [0, 1, 2],
        "high_threat_threshold": 0.8,
        "source": {
            "path": (
                PROBE_DIR / "probe_states.npz"
            ).relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(PROBE_DIR / "probe_states.npz"),
            "content_sha256": corpus.content_sha256(),
            "num_states": corpus.size,
            "original_manifest": source_manifest,
        },
        "context_grid_sha256": grid.content_sha256(),
        "coverage": coverage,
        "historical_margin_coverage": margin_coverage_evidence(),
        "probe_training_isolation": {
            "source_created_before_dst01": True,
            "used_as_training_batch": False,
            "environment_steps_called_by_instrumentation": 0,
            "policy_independent_context_enumeration": True,
        },
    }

    disabled = FactorizedEngagementMaskablePPO.load(
        REFERENCE_MODEL,
        device="cpu",
    )
    enabled = FactorizedEngagementMaskablePPO.load(
        REFERENCE_MODEL,
        device="cpu",
    )
    initial_equal, initial_error = policy_parameters_equal(
        disabled.policy,
        enabled.policy,
    )
    disabled_result = synthetic_update(
        disabled,
        corpus,
        instrument=None,
    )
    enabled_result = synthetic_update(
        enabled,
        corpus,
        instrument=grid,
    )
    updated_equal, updated_error = policy_parameters_equal(
        disabled.policy,
        enabled.policy,
    )
    sample_metrics = compute_dynamic_support_update_metrics(
        grid,
        enabled_result["snapshot_before"],
        enabled_result["snapshot_after"],
        update_id=1,
        approx_kl=float("nan"),
        clip_fraction=float("nan"),
        entropy=float(
            np.mean(enabled_result["snapshot_after"].context_entropies)
        ),
    )
    sample_row = {
        "sample_kind": "synthetic_frozen_batch_equivalence_fixture",
        "formal_p2_evidence": False,
        **sample_metrics,
    }
    equivalence = {
        "schema_version": 1,
        "task": "DST-05",
        "status": "PASS",
        "reference_model": {
            "path": REFERENCE_MODEL.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(REFERENCE_MODEL),
        },
        "training_performed": False,
        "test_update": (
            "one in-memory frozen-batch synthetic gradient step; "
            "not saved and not a PPO or environment rollout"
        ),
        "initial_parameters_bitwise_equal": initial_equal,
        "initial_parameter_max_abs_error": initial_error,
        "reference_parameters_unchanged_by_probe": (
            parameter_before == parameter_after
        ),
        "instrumentation_rng_unchanged": (
            enabled_result["rng_before"]
            == enabled_result["rng_after_instrument"]
        ),
        "rollout_actions_bitwise_equal": bool(
            np.array_equal(
                disabled_result["actions"],
                enabled_result["actions"],
            )
        ),
        "loss_disabled": disabled_result["loss"],
        "loss_enabled": enabled_result["loss"],
        "loss_absolute_difference": abs(
            disabled_result["loss"] - enabled_result["loss"]
        ),
        "updated_parameters_bitwise_equal": updated_equal,
        "updated_parameter_max_abs_error": updated_error,
        "replay_repeatability": repeatability,
        "context_ids_unchanged": bool(
            np.array_equal(first.context_ids, second.context_ids)
        ),
        "k0_k1_k2_fields_reconstructible": all(
            field in sample_row
            for field in (
                "approx_kl",
                "clip_fraction",
                "entropy",
                "unweighted_prefix_flip_rate",
                "ds_weighted_flip_mass",
            )
        ),
        "event_timestamp_semantics": (
            "callback reads model._n_updates after completed PPO updates"
        ),
        "context_aggregation": (
            "one row per unique frozen context_id; no action-pair input"
        ),
        "environment_steps_called": 0,
        "callback_is_opt_in": True,
        "passed": bool(
            initial_equal
            and parameter_before == parameter_after
            and enabled_result["rng_before"]
            == enabled_result["rng_after_instrument"]
            and np.array_equal(
                disabled_result["actions"],
                enabled_result["actions"],
            )
            and abs(
                disabled_result["loss"] - enabled_result["loss"]
            )
            <= 1e-12
            and updated_equal
            and repeatability["passed"]
        ),
    }
    if not equivalence["passed"]:
        raise RuntimeError("Instrumentation equivalence audit failed")
    replay = replay_feasibility()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "probe_manifest.json").write_text(
        json.dumps(probe_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "instrumentation_equivalence.json").write_text(
        json.dumps(equivalence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "replay_feasibility.json").write_text(
        json.dumps(replay, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_sample_csv(
        args.output_dir / "sample_update_metrics.csv",
        sample_row,
    )
    args.report.write_text(
        markdown_report(probe_manifest, equivalence, replay),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "task": "DST-05",
                "status": "PASS",
                "instrumentation_equivalence": equivalence["passed"],
                "replay_insufficient": replay["replay_insufficient"],
                "dst06_new_training_required": replay[
                    "dst06_new_training_required"
                ],
                "context_grid_sha256": grid.content_sha256(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
