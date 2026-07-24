# W1-03：稿件架构与可追溯设计

更新时间：2026-07-24  
任务状态：PASSED（T03）  
前置任务：W1-01 通过 T01；W1-02 判定 L1、L2 或 L3  
后续任务：W1-04、W1-05、W1-06  
允许并行：本任务完成后，下游三个任务并行  
任务性质：论证架构、章节任务和追溯关系设计

## 1. 目标

在写正文之前固定：

- 论文到底回答什么问题；
- 每节只承担什么功能；
- 哪些证据进主文、哪些进补充材料；
- 每项 Claim 映射到哪个段落、图表和数据源。

本任务不写完整正文，不制作正式图表。

## 2. 输入

- W1-01 的术语、公式和证据索引；
- W1-02 的 `paper_positioning_decision.md`；
- W1-02 的 `literature_evidence_matrix.md`；
- [第一创新 Claim–Evidence 矩阵](../../project/first_innovation_claim_evidence_matrix.md)。

## 3. 故事压缩

形成并冻结：

1. 一句话版本：中心洞见和边界；
2. 三句话版本：Problem、Method、Insight；
3. 一段话版本：加入决定性证据，不堆叠历史流水账。

三种版本必须在以下方面一致：

- AirDefense v1 范围；
- 冻结 factorized PPO；
- 同一步与未来替代；
- 跨新种子复现；
- 场景/资源类型条件边界；
- 非算法改进定位。

## 4. 推荐章节架构

```text
1. Introduction
2. Related Work
3. Problem Formulation and Evaluation Framework
4. Paired Counterfactual Cost Decomposition
5. Experimental Protocol
6. Results
   6.1 Local resource-credit ambiguity
   6.2 Paired counterfactual action substitution
   6.3 Exact same-step and future decomposition
   6.4 Independent replication
   6.5 Scenario and resource-type boundaries
   6.6 Resource-restoration negative boundary
7. Discussion
8. Limitations
9. Conclusion
```

若 L2 或 L3 改变稿件形态，可调整章节层级，但不得改变证据先于解释的顺序。

## 5. 段落工作表

每个计划段落记录：

| Paragraph ID | Section | 单一功能 | Claim ID | Evidence ID | Figure/Table | 边界句 |
| --- | --- | --- | --- | --- | --- | --- |
| 待填写 | — | context/gap/method/result/comparison/mechanism/implication/limitation | — | — | — | — |

若一个段落承担两个功能，拆成两个段落。

## 6. 主文与补充材料边界

主文优先保留：

- 局部信用测量问题；
- N/E 和 CRN 设计；
- 完整成本恒等式；
- R1 机制发现；
- R2 独立确认；
- 场景和资源类型边界；
- 资源恢复负结果。

补充材料优先承接：

- 全部历史变体流水账；
- smoke 结果；
- 重复超参数；
- 首轮无效账本逐行结果；
- 不改变主张的次级指标；
- 完整前置标签审计表。

不利结果不能仅因为不利而移入补充材料。

## 7. 稿件追溯矩阵

新建但不复制权威主张定义：

| Claim ID | Evidence ID | 章节 | Paragraph ID | Figure/Table | 允许动词 | 禁止外推 |
| --- | --- | --- | --- | --- | --- | --- |
| C1–C8 | 待映射 | 待映射 | 待映射 | 待映射 | show/suggest 等 | 必填 |

W1-04 至 W1-08 只填写自己负责的位置，不修改 Claim 的科学内容。

## 8. 交付物

```text
story_compression.md
manuscript_outline.md
paragraph_job_map.md
main_vs_supplement_plan.md
manuscript_traceability_matrix.md
section_handoff_contracts.md
```

`section_handoff_contracts.md` 为 W1-04、W1-05、W1-06 分别列出：

- 输入文件；
- 负责章节；
- 不得修改的共享定义；
- 预期输出；
- 更新追溯矩阵的字段。

## 9. 验收门控 T03

- L 判定已被落实到稿件形态；
- 一句话、三句话和一段话范围一致；
- 每个计划段落只有一个功能；
- C1–C8 均有主文、补充材料或明确不使用的去向；
- 主文没有按项目时间顺序组织；
- Results、Methods、Figures 的边界互不冲突；
- 摘要和标题仍被标记为最后写；
- 不存在未经查新支持的优先权表述。

## 10. 移交

向 W1-04、W1-05、W1-06 同时移交全部六个文件，并声明三项任务可以并行。
若后续需要改变章节架构，必须先更新 `manuscript_outline.md` 和追溯矩阵，再
通知三个下游任务。

## 11. 执行结果（2026-07-24）

### 11.1 架构决策

已落实 W1-02 的 **L2** 判定：当前稿件按较大方法论文中的测量、诊断与资源
信用分解模块设计，不按独立通用 PPO 算法论文组织。

正文采用：

```text
测量问题
→ 三分量配对反事实成本账本
→ 代数完整性
→ 新策略种子独立确认
→ 场景/资源类型边界
→ 机会成本和在线算法负边界
```

未采用项目任务执行时间线作为论文结构。

### 11.2 交付件

已在公共稿件目录生成：

- `story_compression.md`
- `manuscript_outline.md`
- `paragraph_job_map.md`
- `main_vs_supplement_plan.md`
- `manuscript_traceability_matrix.md`
- `section_handoff_contracts.md`

### 11.3 T03 验收

| 验收条件 | 结果 |
| --- | --- |
| L2 落实到稿件形态 | PASS |
| 一句话、三句话和一段话范围一致 | PASS |
| 每个 Paragraph ID 只有一个功能 | PASS |
| C1-C8 均有主文、补充材料或明确不使用去向 | PASS |
| 主文不按项目时间顺序组织 | PASS |
| Results、Methods、Figures 文件所有权和职责无冲突 | PASS |
| 标题和摘要标记为最后写 | PASS |
| 无未经查新支持的优先权表述 | PASS |
| 未新建实验或修改冻结科学结论 | PASS |

### 11.4 移交

W1-04、W1-05、W1-06 的输入、负责章节、禁止修改项、预期输出和追溯矩阵
更新字段均已冻结。三个任务可以并行启动。

禁止下游假设：

- 不得把 L2 改写为独立通用算法论文；
- 不得把 C4-C6 的否决状态删去或改成部分成功；
- 不得把 GNN 写成已验证修复；
- 不得改变 N/E 方向、三分量公式或统计单位；
- 不得提前定稿标题和摘要。
