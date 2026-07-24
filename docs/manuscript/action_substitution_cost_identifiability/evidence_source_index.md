# 主张—证据来源索引

更新时间：2026-07-24  
状态：W1-01 冻结  
主张层唯一权威：`docs/project/first_innovation_claim_evidence_matrix.md`

## 1. 权威层级

1. **数值权威**：冻结结果目录中的 JSON/CSV。每个关键数字只绑定一个 Evidence ID 和一个唯一文件。
2. **叙事核查**：`docs/experiments/` 下的正式实验报告，用于说明协议、过程和解释，不另立数值版本。
3. **主张权威**：第一创新 Claim–Evidence 矩阵，唯一决定 C1–C8 的支持状态与论文表述边界。
4. **任务指导**：`docs/task_guides/` 只规定流程和门控，不作为实验数字来源。

下游文稿可重复展示数值，但必须回指本索引中的 Evidence ID，不得把图表、
汇报稿或任务文档升级为新的权威来源。

## 2. 数值证据索引

| Evidence ID | Claim ID | 冻结事实 | 唯一数值权威与字段 | 统计单位 | 允许用途 | 外推边界 |
| --- | --- | --- | --- | --- | --- | --- |
| EV-R1-01 | C1 | time/resource 中 `18/18` 上下文的 \(Sub_{\mathrm{shot}}\) 均值及 95% 下界为正 | `results/air_defense_v1/action_substitution_opportunity_cost_audit/gate_summary.json`：`P-R1.time_resource_contexts`、`positive_mean_sub_shot`、`positive_lower_sub_shot` | context | Results、Figure、Supplement | 旧策略种子 `8/9/10`；只用于 R1 机制复核 |
| EV-R1-02 | C1 | `11/11` 个累计成本差非正上下文均由正替代成本解释 | 同上：`P-R1.nonpositive_total_cost_contexts`、`explained_nonpositive_cost_contexts`、`explained_fraction` | context | Results、Discussion | 不证明任意上下文都会出现符号掩盖 |
| EV-R1-03 | C5 | 可靠资源机会标签仅为 time `5/18`、heterogeneity `2/18`，可靠资源类型仅 missile | 同上：`reliable_resource_contexts`、`reliable_resource_unit_types` | context | Results、Boundary、Supplement | 否决通用弹药机会价值，不训练 opportunity oracle |
| EV-R2-01 | C2、C3 | 新来源模型 `9` 个 | `results/air_defense_v1/action_substitution_confirmation/experiment_config.json`：`source_model_count` | model | Methods、Supplement | 仅为场景×种子 `3×3` 的冻结模型 |
| EV-R2-02 | C3 | 新上下文 `108` 个，旧 hash 重叠为零 | `results/air_defense_v1/action_substitution_confirmation/gate_summary.json`：`context_count`、`integrity_gates.old_hash_overlap_zero` | context | Methods、Results、Supplement | hash 独立性不等同于环境分布外泛化 |
| EV-R2-03 | C2、C3 | `3,456` 条重复记录、`7,776` 条目标账本、`157,485` 个额外 transition | 同上：`repeat_rows`、`target_ledger_rows`、`actual_extra_transitions` | repeat / ledger row / transition | Methods、Supplement | 目标账本行不是独立 context |
| EV-R2-04 | C2 | 首轮 `287/7,776` 条账本受 future-only 公式遗漏影响 | `results/air_defense_v1/action_substitution_confirmation/pre_ledger_correction/repeat_cost_ledger.csv`：筛选 `abs(protocol_residual)>1e-6` | ledger row | Methods、Audit、Supplement | 只描述被纠正的首轮账本，不得作为最终效果结果 |
| EV-R2-05 | C2 | 原 future-only 最大残差 `2.0` | `results/air_defense_v1/action_substitution_confirmation/gate_summary.json`：`maximum_future_only_decomposition_error` | ledger row maximum | Methods、Audit | 该值用于证明同一步项不可省略 |
| EV-R2-06 | C2 | 完整分解最大误差 `8.881784197001252e-16` | 同上：`maximum_protocol_decomposition_error` | ledger row maximum | Results、Methods、Figure、Supplement | 数值恒等式不自动证明统计估计无偏 |
| EV-R2-07 | C2、C3 | 确认期间 Actor 最大参数差 `0.0` | 同上：`maximum_actor_parameter_difference` | parameter maximum | Methods、Integrity | 只证明该确认过程未更新 Actor |
| EV-R2-08 | C3 | 新种子 time/resource 中 `13/18` 上下文的 \(Sub_{\mathrm{shot}}\) 95% 下界为正 | 同上：`P-C2.positive_lower_sub_shot` | context | Results、Figure | 不外推到任意策略种子、算法或环境 |
| EV-R2-09 | C3 | seeds `17/18/19` 三个 block 的 95% 下界均为正；下界依次为 `0.757/0.029/0.166` | 同上：`P-C2.positive_block_lower_seeds`、`seed_block_intervals` | seed block | Results、Figure | block 仅限 time/resource |
| EV-R2-10 | C1、C3 | `7/7` 个累计成本均值非正上下文具有正 \(Sub_{\mathrm{cost,total}}\) | 同上：`P-C2.nonpositive_contexts`、`nonpositive_with_positive_sub_cost` | context | Results、Discussion | 是机制一致性，不是全场景发生率 |
| EV-R2-11 | C4 | time/resource 的 missile 为 `2/9` 掩盖上下文，laser 为 `5/9` | 同上：`P-C3.missile.masked_contexts`、`P-C3.laser.masked_contexts`、各自 `contexts` | resource-type context | Results、Boundary、Figure | 否决“跨资源类型普遍同强度” |
| EV-R2-12 | C4 | time/resource 的 \(\rho_{\mathrm{sub}}\)：missile `0.571`、laser `1.175` | `results/air_defense_v1/action_substitution_confirmation/resource_type_summary.csv`：筛选 `scenario=time_pressure, slot=resource`，字段 `rho_sub_mean_aggregate` | resource-type context aggregate | Results、Figure | 资源类型比较只在 time/resource 内成立 |
| EV-R2-13 | C3、C4 | resource 槽位场景 \(\rho_{\mathrm{sub}}\)：medium `0.747`、time `0.873`、heterogeneity `0.972` | `results/air_defense_v1/action_substitution_confirmation/scenario_boundary_summary.csv`：筛选 `slot=resource`，字段 `rho_sub_mean_aggregate` | scenario context aggregate | Results、Figure | 场景聚合不能替代 seed/block 不确定性 |
| EV-BPCE-01 | C5 | time/resource 短视窗可行动标签为 `0 ENGAGE / 0 STOP / 18 AMBIGUOUS` | `results/air_defense_v1/bpce_short_horizon_label_audit/gate_summary.json`：`slot_counts.time_pressure/resource` | context | Results、Boundary | 仅针对已冻结的短视窗标签协议 |
| EV-BPCE-02 | C6、C7 | BPCE 候选运行中有 `2` 个塌缩运行，机制总门控失败 | `results/air_defense_v1/bpce_ppo_mechanism_stress_test/bpce_stress_summary.json`：`collapsed_candidate_run_count`、`mechanism_gate_passed` | trained run / gate | Results、Limitations | 作为失败机制证据，不得表述为稳定在线改进 |

## 3. 非数值边界证据

| Boundary ID | Claim ID | 冻结结论 | 唯一主张权威 | 允许用途 |
| --- | --- | --- | --- | --- |
| BD-01 | C7 | 当前贡献是成本测量与可辨识性贡献，不是已完成的 PPO 算法创新 | `docs/project/first_innovation_claim_evidence_matrix.md`：C7 | Abstract、Introduction、Discussion、Limitations |
| BD-02 | C8 | 尚无证据表明 GNN 可直接修复当前信用问题，GNN 继续冻结 | 同上：C8 | Discussion、Future Work |
| BD-03 | C4 | 资源类型只支持条件性结论，跨类型普遍成立的表述被否决 | 同上：C4 | Abstract、Results、Discussion |

## 4. 正式报告与追溯关系

| 报告 | 正式章节 | 作用 | 对应证据 |
| --- | --- | --- | --- |
| `docs/experiments/air_defense_v1_action_substitution_confirmation.md` | §2 独立性协议、§3 成本账本修正、§4 数据完整性 | 新模型、上下文、账本行数、旧 hash、Actor 冻结及公式修正 | EV-R2-01 至 EV-R2-07 |
| 同上 | §6 P-C2：独立确认 | 新种子上下文、block 与非正成本解释 | EV-R2-08 至 EV-R2-10 |
| 同上 | §7 P-C3、§8 场景适用边界 | 资源类型和场景条件边界 | EV-R2-11 至 EV-R2-13 |
| `docs/experiments/air_defense_v1_action_substitution_opportunity_cost_audit.md` | §5 P-R1、§6 P-R2、§8 决策 | R1 替代机制与机会价值适用边界 | EV-R1-01 至 EV-R1-03 |
| `docs/experiments/air_defense_v1_bpce_label_semantics_audit.md` | §4–§8 | BPCE 标签语义来源和早期可辨识性审计 | C5、C6 的背景证据 |
| `docs/experiments/air_defense_v1_bpce_short_horizon_label_audit.md` | §6 槽位诊断、§7 门控结果 | 短视窗标签可行动性门控 | EV-BPCE-01 |
| `docs/experiments/air_defense_v1_bpce_ppo_stress_test.md` | §5 主要结果、§7 机制门控、§8 研究结论 | 在线候选的塌缩与总门控失败 | EV-BPCE-02 |
| `docs/project/first_innovation_claim_evidence_matrix.md` | §2 Claim–Evidence、§4–§5 表述边界 | C1–C8 支持状态、允许与禁止表述 | BD-01 至 BD-03 |

## 5. 可复核提取

EV-R2-04 的 `287` 为对首轮冻结 CSV 的确定性计数，不是新实验：

```powershell
$rows = Import-Csv `
  results/air_defense_v1/action_substitution_confirmation/pre_ledger_correction/repeat_cost_ledger.csv
($rows | Where-Object {
  [math]::Abs([double]$_.protocol_residual) -gt 1e-6
}).Count
```

输出固定为 `287`。其余证据均直接读取表中列或 JSON 键，不进行二次模型拟合。
