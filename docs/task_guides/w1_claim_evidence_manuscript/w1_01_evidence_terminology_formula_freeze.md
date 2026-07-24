# W1-01：证据、术语与公式冻结

更新时间：2026-07-24  
任务状态：PASSED  
前置任务：无  
后续任务：W1-02、W1-03  
允许并行：无  
任务性质：只读核查与文档整理，不新增实验

## 1. 目标

建立全文唯一可复用的证据、术语和公式基础，使后续任务不再各自解释 N/E
方向、替代成本范围或关键数字。

本任务不重做 [第一创新 Claim–Evidence 矩阵](../../project/first_innovation_claim_evidence_matrix.md)；
该文件继续作为主张层唯一权威源。

## 2. 输入

- [R2 动作替代独立确认](../../experiments/air_defense_v1_action_substitution_confirmation.md)
- [R1 动作替代与机会成本审计](../../experiments/air_defense_v1_action_substitution_opportunity_cost_audit.md)
- [BPCE 标签语义审计](../../experiments/air_defense_v1_bpce_label_semantics_audit.md)
- [BPCE 短视窗标签审计](../../experiments/air_defense_v1_bpce_short_horizon_label_audit.md)
- [第一创新 Claim–Evidence 矩阵](../../project/first_innovation_claim_evidence_matrix.md)
- 冻结结果目录 `results/air_defense_v1/action_substitution_confirmation/`

## 3. 工作内容

### 3.1 建立证据来源索引

为每个可进入论文的关键数字记录：

| 字段 | 内容 |
| --- | --- |
| Evidence ID | 稳定编号 |
| Claim ID | C1–C8 |
| 正式报告 | 文件路径和章节 |
| 结果文件 | CSV/JSON 路径 |
| 数据字段 | 列名或键 |
| 统计单位 | context、block、seed 或 ledger row |
| 允许用途 | Results、Methods、Figure、Supplement |
| 边界 | 不允许外推的范围 |

禁止把任务指导报告作为数字来源。

### 3.2 冻结术语账本

至少冻结：

- dynamic legal-action masking；
- conflict-free autoregressive joint action；
- factorized joint PPO；
- paired counterfactual trajectories；
- common random numbers；
- probe direct cost；
- same-step other-unit substitution；
- future probe substitution；
- future other-unit substitution；
- total substitution cost；
- future substituted shots；
- episode-level cumulative cost difference；
- substitution ratio；
- cost-sign masking；
- identifiability boundary。

`action substitution` 暂作工作术语；W1-02 必须核验其与既有文献术语是否冲突。

### 3.3 冻结 N/E 方向

```text
Delta_C_episode
:= total_cost(E) - total_cost(N)
```

任何后续文件不得只写“counterfactual difference”而省略方向。

### 3.4 冻结完整成本分解

```text
Sub_cost_total
:= Sub_cost_same
 + Sub_cost_future_probe
 + Sub_cost_future_other
```

```text
Sub_cost_same
:= current_other_cost(N) - current_other_cost(E)
```

```text
Delta_C_episode
= C_direct - Sub_cost_total
```

```text
rho_sub
:= Sub_cost_total / C_direct
```

```text
cost_sign_masked
:= (C_direct > 0)
   and (Delta_C_episode <= 0)
```

必须单独说明：

- `Sub_shot` 只统计当前步之后的未来射击；
- `Sub_cost_total` 包含同一步和未来替代；
- 两者不可在图表和正文中互换。

### 3.5 核查关键事实

至少逐项核对：

```text
来源模型：9/9
新上下文：108/108
旧 hash 重叠：0
目标成本账本：7,776
首轮受影响账本：287/7,776
原 future-only 最大残差：2.0
修正后最大误差：8.88e-16
新种子块正下界：3/3
time/missile 掩盖上下文：2
time/laser 掩盖上下文：5
```

若正式报告与结果文件不一致，不自行裁决，立即标记为完整性问题。

## 4. 交付物

在 `docs/manuscript/action_substitution_cost_identifiability/` 创建：

```text
terminology_ledger.md
evidence_source_index.md
formula_and_direction_freeze.md
evidence_conflict_log.md
```

`evidence_conflict_log.md` 即使没有冲突也必须存在，并写明核查范围与“未发现冲突”。

## 5. 验收门控 T01

全部满足才通过：

- 每个关键数字有唯一权威来源；
- N/E 方向只有一种写法；
- 完整三类替代进入主公式；
- `Sub_shot` 与 `Sub_cost_total` 范围明确；
- 术语账本包含中文、英文、符号和使用边界；
- 未创建第二份权威 Claim–Evidence 矩阵；
- 所有冲突均已解决或明确升级。

## 6. 停止条件

出现任一情况则状态改为 `BLOCKED`：

- 修正恒等式不能从冻结账本复核；
- 新旧上下文 hash 实际不为零重叠；
- Actor 参数冻结记录与正式报告冲突；
- N/E 身份在代码、数据和报告之间不一致。

不得通过新增实验绕过数据定义冲突。

## 7. 移交

向 W1-02 和 W1-03 提供：

- 四个交付文件；
- T01 判定；
- 尚未解决的术语候选；
- 不允许下游使用的数字或表达。

## 8. 执行结果

执行时间：2026-07-24  
门控结论：`T01 PASS`

已生成：

- `docs/manuscript/action_substitution_cost_identifiability/terminology_ledger.md`
- `docs/manuscript/action_substitution_cost_identifiability/evidence_source_index.md`
- `docs/manuscript/action_substitution_cost_identifiability/formula_and_direction_freeze.md`
- `docs/manuscript/action_substitution_cost_identifiability/evidence_conflict_log.md`

复核结论：

- 来源模型 `9/9`、新上下文 `108/108`、旧 hash 重叠 `0`；
- 最终账本 `7,776` 条，首轮受影响 `287` 条；
- future-only 最大残差 `2.0`，完整公式最大误差 `8.88e-16`；
- Actor 最大参数差 `0.0`；
- 新种子 `17/18/19` 的 time/resource block 下界均为正；
- time/resource 中 missile 与 laser 的掩盖上下文分别为 `2/9` 和 `5/9`。

唯一升级项为“action substitution”与既有文献术语、优先权的关系，已登记为
`CF-07 / ESCALATED_W1-02`。该项不改变公式或数值，不阻塞 W1-01，但 W1-02
完成前禁止使用“首次提出”类表述。

## 9. 下游移交

W1-02 和 W1-03 必须直接复用四份冻结文件。若下游需要修改术语、N/E 方向、
成本公式或关键数字，应先回到本任务登记冲突，不得在文稿中静默改写。
