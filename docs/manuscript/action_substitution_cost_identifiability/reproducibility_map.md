# W1-05 方法复现映射

更新时间：2026-07-28  
规则：只映射仓库中已经存在的代码、配置、输入和输出，不构成公开发布承诺

## 1. 运行入口

正式实验入口为：

```powershell
conda run -n rein-learning python scripts/run_air_defense_v1_action_substitution_confirmation.py --device cuda
```

正式结果目录已标记为 completed，脚本默认阻止无意覆盖。账本修正重跑参数只接受
首轮被判为 `invalid_cost_ledger_fix_only` 的冻结状态，不应用于当前已完成结果。

## 2. 方法到材料映射

| 方法步骤 | 代码 | 配置 | 输入 | 输出 | 正文位置 |
| --- | --- | --- | --- | --- | --- |
| AirDefense v1 环境与转移 | `rein_learning/envs/air_defense_v1/centralized_env.py`、`config.py`、`entities.py` | `rein_learning/envs/air_defense_v1/scenarios.py` | 环境 seed、场景名 | 观测、奖励分量、资源成本、终止状态 | PF01；S1 |
| 无冲突联合动作 | `rein_learning/envs/air_defense_v1/wrappers/conflict_free_joint_action.py` | 单元顺序 0/1/2 | 基础合法掩码、前缀目标占用 | 条件合法动作掩码 | PF01-PF02；S1-S2 |
| factorized joint PPO | `rein_learning/algorithms/policy_gradient/factorized_engagement_ppo.py`、`rein_learning/models/factorized_engagement_action_head.py` | `rein_learning/trainers/air_defense_v1_ppo.py::AirDefenseV1PPOConfig` | 环境观测与动态掩码 | 冻结来源策略 | PF02；S2 |
| 正式模型准备与留存 | `scripts/run_air_defense_v1_action_substitution_confirmation.py::_prepare_models` | `results/air_defense_v1/action_substitution_confirmation/experiment_config.json` | scenarios × seeds 17/18/19 | `source_model_manifest.json`、`source_model_training_log.csv`、`source_models/` | P02；S2 |
| 种子用途审计 | `scripts/run_air_defense_v1_action_substitution_confirmation.py::_seed_usage_audit` | 同上 | 项目内已有模型名、结果名和文本 | `seed_usage_audit.json` | P02；S2 |
| 候选上下文收集与配额选择 | `rein_learning/common/action_substitution_confirmation.py::collect_confirmation_contexts` | `confirmation_config` | 冻结策略、每块 24 个候选回合 | `context_selection.csv` | P02；S3 |
| 上下文身份和旧 hash 检查 | `rein_learning/common/action_substitution_confirmation.py::validate_confirmation_contexts` | `probability_tolerance=1e-12` | 前置审计 hash、选中上下文 | `context_identity_check.csv` | P02、P04；S3 |
| N/E 成对回放与 CRN | `rein_learning/common/action_substitution_confirmation.py::audit_confirmation_context` | repeats 32、branch seed 1293000 | 环境快照、环境随机带、策略 uniform tape | 目标账本与 repeat 汇总 | M01-M03、P03；S4 |
| 完整成本账本 | `rein_learning/common/action_substitution_confirmation.py::_cost_ledger` | `decomposition_tolerance=1e-6` | N/E 当前及未来逐单元成本 | `repeat_cost_ledger.csv`、`repeat_marginal_metrics.csv` | M04-M08；S5 |
| context 与 block 区间 | `rein_learning/common/action_substitution_confirmation.py::component_interval`、`grouped_summary_rows` | `confidence_z=1.96` | repeat/context 均值 | `context_substitution_estimates.csv`、`block_summary.csv` | P03-P05；S7 |
| P-C1/P-C2/P-C3 判定 | `rein_learning/common/action_substitution_confirmation.py::summarize_confirmation` | 冻结门槛 | context、repeat、ledger 数据 | `gate_summary.json` | P04-P05；S7 |
| 资源类型与场景边界汇总 | `rein_learning/common/action_substitution_confirmation.py::grouped_summary_rows` | 分组字段 | context 估计 | `resource_type_summary.csv`、`scenario_boundary_summary.csv` | P03；Results 6.5 |
| 软件回归 | `tests/test_action_substitution_confirmation.py` | 测试配置 | 实现代码 | pytest 结果及 `software_tests_passed` | P04；S7 |
| 首轮账本修正审计 | `scripts/run_air_defense_v1_action_substitution_confirmation.py::_archive_pre_correction` 与账本实现 | `ledger_correction_rerun=true` | 首轮 future-only 账本 | `pre_ledger_correction/`、修正后正式目录 | Methods 5.5；科研完整性披露 |

## 3. 前置审计材料

| 用途 | 代码/报告 | 输出 |
| --- | --- | --- |
| A/B/C 标签语义 | `rein_learning/common/bpce_label_semantics.py`；`docs/experiments/air_defense_v1_bpce_label_semantics_audit.md` | `results/air_defense_v1/bpce_label_semantics_audit/` |
| 短视窗安全—资源标签 | 对应任务报告 `docs/task_guides/next_research_phase_bpce_short_horizon_component_label_audit.md` | `results/air_defense_v1/bpce_short_horizon_label_audit/` |
| R1 动作替代发现 | `rein_learning/common/action_substitution_opportunity_cost.py`；`docs/experiments/air_defense_v1_action_substitution_opportunity_cost_audit.md` | `results/air_defense_v1/action_substitution_opportunity_cost_audit/` |

前置材料只承担问题收窄和机制发现；R2 的独立确认结论必须从正式
`action_substitution_confirmation/` 目录复核。

