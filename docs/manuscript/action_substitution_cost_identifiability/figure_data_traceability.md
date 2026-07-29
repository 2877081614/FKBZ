# W1-06 图表数据追溯

更新时间：2026-07-28  
生成脚本：`figures/source/generate_figures.py`  
数据规则：冻结结果只读；无新增 rollout；无视觉选点；无手工中间数

## 1. 统计与方向

- \(N\) 为当前探针 no-op，\(E\) 为指定合法 engage；
- \(\Delta C_{\mathrm{episode}}=C_{\mathrm{episode}}(E)-C_{\mathrm{episode}}(N)\)；
- 替代量方向为 \(N-E\)；
- context 区间以 32 个 paired repeat 为样本；
- block、场景和资源类型区间以 context 为样本；
- 所有区间为 \(\bar{x}\pm1.96s/\sqrt n\)；
- `ledger row` 只用于精确目标边缘化和代数校验，不作为独立 context。

## 2. 逐面板追溯

| Panel | 数据文件 | 字段 | 过滤 | 聚合 | 面板单位 | 导出 |
| --- | --- | --- | --- | --- | --- | --- |
| Fig. 1a | 环境与无冲突动作实现 | 单元顺序、动态掩码、目标占用 | order 0/1/2 | 示意，无经验聚合 | joint action | `figure_1_measurement_problem.*` |
| Fig. 1b | `formula_and_direction_freeze.md`、确认实现 | N/E 身份、随机带、动态后缀 | 冻结上下文 | 示意 | paired intervention | 同上 |
| Fig. 1c | `formula_and_direction_freeze.md` | \(C_{\mathrm{direct}}\)、\(Sub_{\mathrm{cost,total}}\)、\(\Delta C\) | 无 | 冻结恒等式 | algebraic quantity | 同上 |
| Fig. 2a | `experiment_config.json`、确认实现 | snapshot、CRN、stochastic continuation | 正式 R2 协议 | 示意 | repeat protocol | `figure_2_protocol_and_identity.*` |
| Fig. 2b | 确认实现 | `target_probability` | 全部合法目标 | \(\sum_kp_kE[\cdot\mid k]\) | target-conditional expectation | 同上 |
| Fig. 2c | `formula_and_direction_freeze.md` | 三个替代分量 | 无 | 精确相加 | ledger identity | 同上 |
| Fig. 2d | R2 `gate_summary.json` | `maximum_future_only_decomposition_error`、`maximum_extended_decomposition_error`、`target_ledger_rows` | 正式结果 | 最大绝对残差；受影响行为预修正账本中 \(|residual|>10^{-6}\) 的计数 | ledger row maximum/count | 同上 |
| Fig. 3a | R1 `context_opportunity_estimates.csv` | `sub_shot_mean/lower/upper`、`policy_seed` | `time_pressure/resource` | 每 context 32 repeats 的均值与区间 | context，\(n=18\) | `figure_3_discovery_and_confirmation.*` |
| Fig. 3b | R2 `context_substitution_estimates.csv` | `sub_shot_mean/lower/upper`、`policy_seed` | `time_pressure/resource` | 每 context 32 repeats 的均值与区间 | context，\(n=18\) | 同上 |
| Fig. 3c | R2 `gate_summary.json` | `P-C2.seed_block_intervals` | `time_pressure/resource` | 每 seed 内 6 个 context 的均值与区间 | seed block，3 blocks | 同上 |
| Fig. 3d | R1/R2 `gate_summary.json` | R1 `nonpositive_total_cost_contexts`/`explained_nonpositive_cost_contexts`；R2 `nonpositive_contexts`/`nonpositive_with_positive_sub_cost` | 两阶段均为 `time_pressure/resource` | context 计数 | R1 11/11；R2 7/7 | 同上 |
| Fig. 4a | R2 `context_substitution_estimates.csv` | `direct_cost_mean`、三个替代分量、`episode_cost_delta_mean` | `time_pressure/resource` | 18 个 context 等权均值 | context aggregate | `figure_4_cost_composition.*` |
| Fig. 4b | 同 Fig. 4a | same、future probe、future other | 同上 | same/total 与 future/total | context-equal composition | 同上 |
| Fig. 4c | 同 Fig. 4a | `rho_sub_mean`、`episode_cost_delta_mean`、`cost_sign_masked_rate`、`resource_type` | 同上 | 不聚合，不删点 | context，\(n=18\) | 同上 |
| Fig. 5a | R2 `scenario_boundary_summary.csv` | `rho_sub_mean_aggregate/lower/upper` | `slot=resource` | 每场景 18 个 context 的均值与区间 | scenario-context aggregate | `figure_5_scenario_resource_boundaries.*` |
| Fig. 5b | R2 `resource_type_summary.csv` | `sub_shot_mean_aggregate/lower/upper` | `time_pressure/resource` | 每类型 9 个 context 的均值与区间 | resource-type context aggregate | 同上 |
| Fig. 5c | R2 `gate_summary.json` | `P-C3.*.masked_contexts` | `time_pressure/resource` | context 计数，与冻结门槛 3 比较 | context，9/type | 同上 |
| Fig. 5d | R2 `gate_summary.json` | `mechanism_gates`、`decision` | 正式结果 | 原样映射 PASS/FAIL | gate | 同上 |

`*` 表示同名 SVG、PDF、TIFF 和 `_preview.png`。每张图的 JSON 元数据位于
`figures/metadata/`，包含结论、图型、面板源、字段、过滤、统计单位和导出文件。

## 3. 面板源数据

| 文件 | 用途 | 生成方式 |
| --- | --- | --- |
| `figures/source/figure_2_residual_data.csv` | Fig. 2d | 从 R2 gate JSON 提取 |
| `figure_3_r1_context_data.csv` | Fig. 3a | R1 `time_pressure/resource` 固定过滤 |
| `figure_3_r2_context_data.csv` | Fig. 3b、3d | R2 同一固定过滤 |
| `figure_3_seed_block_data.csv` | Fig. 3c | R2 P-C2 seed block |
| `figure_3_nonpositive_explanation_data.csv` | Fig. 3d | R1/R2 非正累计成本 context 计数 |
| `figure_4_component_data.csv` | Fig. 4a-b | 18 context 等权聚合 |
| `figure_4_context_data.csv` | Fig. 4c | 18 context 全量输出 |
| `figure_5_scenario_data.csv` | Fig. 5a | 三场景 resource 槽全量输出 |
| `figure_5_resource_data.csv` | Fig. 5b-c | time/resource 两类型全量输出 |

这些 CSV 是绘图脚本的确定性派生产物，不替代 `results/` 中的数值权威。

## 4. 表格追溯

| Table | 数据文件/字段 | 过滤与聚合 | 导出 |
| --- | --- | --- | --- |
| Table 1 | R2 `experiment_config.json`、`source_model_count` | 正式配置原样汇总 | `table_1_task_policy_protocol.csv/.md` |
| Table 2 | R1/R2 gate、R2 manifest | discovery/confirmation 分行；不合并 seed | `table_2_independence_integrity.csv/.md` |
| Table 3 | R2 `mechanism_gates`、P-C2、P-C3 | 按冻结代码判据映射 | `table_3_confirmation_gates.csv/.md` |
| Table 4 | scenario/resource summary CSV | `slot=resource`；资源类型再限 `time_pressure` | `table_4_scenario_resource_boundaries.csv/.md` |
| Table S1 | label/short-horizon gate JSON | A/B/C 和短视窗标签原样计数 | `table_s1_label_semantics.csv/.md` |
| Table S2 | R1 gate JSON | P-R1/P-R2/P-R3 与冻结路线决策 | `table_s2_resource_restoration_negative.csv/.md` |
| Table S3 | R2 gate、预修正归档、experiment config | 首轮与唯一重跑审计 | `table_s3_ledger_correction_integrity.csv/.md` |

## 5. 缺失值与排除

- 没有按效果量、误差线或视觉布局删除 context；
- `first_substitution_step` 的空值不进入当前五张主图；
- \(\rho_{\mathrm{sub}}\) 仅在正直接成本上定义；正式确认账本全部满足该条件；
- Fig. 3 只使用预注册的 `time_pressure/resource` 主确认切片；
- Fig. 5 同时保留 missile 和 laser，未选取单一有利资源类型；
- 所有失败门控保持可见。

## 6. 确定性重建

```powershell
conda run -n rein-learning python docs/manuscript/action_substitution_cost_identifiability/figures/source/generate_figures.py
```

脚本执行后检查五张图的四种导出均非空、SVG 包含可编辑 `<text>`、R2 主切片
恰有 18 个 context 且 missile/laser 各 9 个。
