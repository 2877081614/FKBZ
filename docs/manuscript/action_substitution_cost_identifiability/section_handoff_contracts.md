# W1-03 下游章节移交契约

更新时间：2026-07-24  
状态：W1-04、W1-05、W1-06 可并行启动  
共享架构：`manuscript_outline.md`

## 1. 公共冻结项

三个下游任务均不得修改：

- C1-C8 的支持、否决或未验证状态；
- L2 稿件定位；
- N/E 分支身份与
  \(\Delta C_{\mathrm{episode}}=C_{\mathrm{episode}}(E)-C_{\mathrm{episode}}(N)\)；
- 三分量成本恒等式及同一步项；
- `context`、`block`、`repeat`、`ledger row`、`seed` 的统计单位；
- W1-01 术语账本；
- F1-F3 支持、F4-F6 否决；
- 标题和摘要延后至 W1-08。

发现冲突时，优先回到 `evidence_source_index.md`，不得由下游文档自行平均或改写。

## 2. W1-04：Results 中文证据稿

### 输入文件

- `story_compression.md`
- `manuscript_outline.md`
- `paragraph_job_map.md`
- `manuscript_traceability_matrix.md`
- `main_vs_supplement_plan.md`
- `evidence_source_index.md`
- R1/R2/BPCE 正式实验报告和冻结结果

### 负责章节

- §6.1 局部资源信用混合；
- §6.2 同一步与未来精确分解；
- §6.3 新策略种子独立确认；
- §6.4 场景和资源类型边界；
- §6.5 资源恢复负边界；
- §6.6 在线算法负证据。

### 不得修改

- 方法定义和公式；
- 图表聚合口径；
- “精确”只修饰代数恒等式；
- C4-C6 的失败状态；
- 不能把观察和解释混在同一段。

### 预期输出

```text
results_draft_zh.md
results_evidence_notes.md
supplementary_results_outline.md
```

### 追溯矩阵更新

- `章节`
- `Paragraph ID`
- `Figure/Table`

只更新 R01-R13 的最终落点，不修改 Evidence ID 和禁止外推列。

### 验收提示

- 每段以待回答的问题或结果主张开头；
- 每个数字带 Evidence ID；
- R1 与 R2 的发现/确认职责分开；
- 资源类型失败进入主文；
- 不做 Discussion 式机制推断。

## 3. W1-05：Methods 与研究完整性稿

### 输入文件

- `story_compression.md`
- `manuscript_outline.md`
- `paragraph_job_map.md`
- `manuscript_traceability_matrix.md`
- `formula_and_direction_freeze.md`
- `terminology_ledger.md`
- `evidence_source_index.md`
- 实现代码、实验配置和正式协议报告

### 负责章节

- §3 Problem Formulation and Evaluation Scope；
- §4 Paired Counterfactual Resource-Cost Decomposition；
- §5 Experimental Protocol；
- 补充 Methods S1-S4；
- reproducibility 与 research integrity 说明。

### 不得修改

- N/E 方向；
- `Sub_cost_total` 的三个组成；
- 同一步项不可省略；
- CRN 仅作方差缩减；
- 目标条件概率精确边缘化定义；
- Actor 冻结和统计单位；
- 不得把代码实现字段短名升级为论文规范名。

### 预期输出

```text
methods_draft_zh.md
supplementary_methods.md
research_integrity_disclosure.md
reproducibility_map.md
```

### 追溯矩阵更新

- PF01-PF04、M01-M09、P01-P05 的最终章节和 Paragraph ID；
- Table 1 及补充方法表号；
- 不修改 Claim 状态和数值权威。

### 验收提示

- 每个方法模块说明动机、机制和在证据链中的角色；
- 输入、输出、干预、随机性和估计量可复现；
- 统计单位不混用；
- 明确首轮账本修正和唯一重跑；
- 不用“标准方法”“常规分析”等不可复核表述。

## 4. W1-06：Figures、Tables 与数据追溯

### 输入文件

- `main_vs_supplement_plan.md`
- `manuscript_traceability_matrix.md`
- `paragraph_job_map.md`
- `formula_and_direction_freeze.md`
- `evidence_source_index.md`
- 对应 JSON/CSV 数值权威

### 负责内容

- Fig. 1 至 Fig. 4；
- Table 1、Table 2；
- Supplementary figures/tables；
- 图注、数据提取脚本和 figure-data 映射。

### 不得修改

- 主张和公式；
- 场景、槽位、资源类型筛选口径；
- 置信区间统计单位；
- missile/laser 必须同时显示；
- 失败门控不能从图表中移除；
- 不得从实验报告手工录入替代权威数据。

### 预期输出

```text
figure_table_plan.md
figures/
tables/
figure_data_traceability.md
figure_caption_draft_zh.md
table_caption_draft_zh.md
figure_qa_report.md
```

### 追溯矩阵更新

- 最终 Figure/Table 编号；
- panel 到 Evidence ID 的映射；
- 源文件和字段；
- 生成脚本与输出文件。

### 验收提示

- 每个 panel 只支撑一个主要结论；
- 图中样本单位和误差线定义可见；
- Fig. 2 明确 N/E 方向和恒等式；
- Fig. 4 显示条件边界而非单一有利切片；
- 图表可由权威数据确定性重建。

## 5. 并行边界

W1-04、W1-05、W1-06 可以并行，但文件所有权互斥：

| 文件/目录 | 主负责人 | 其他任务 |
| --- | --- | --- |
| `results_draft_zh.md` | W1-04 | 只读 |
| `methods_draft_zh.md`、补充方法、完整性声明 | W1-05 | 只读 |
| `figure_table_plan.md`、`figures/`、`tables/` | W1-06 | 只读 |
| `manuscript_traceability_matrix.md` | W1-03 主控 | 仅更新各自字段 |
| `terminology_ledger.md` | W1-01 主控 | 只能提交变更请求 |

## 6. 移交格式

三个任务完成时均报告：

```text
任务编号：
状态：
已生成文件：
已填写 Paragraph ID：
已填写 Figure/Table：
引用的 Evidence ID：
未解决问题：
未改变的冻结边界：
```
