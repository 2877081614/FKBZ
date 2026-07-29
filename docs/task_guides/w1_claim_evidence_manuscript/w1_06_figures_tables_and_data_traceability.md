# W1-06：Figures、Tables 与数据追溯

更新时间：2026-07-24  
任务状态：PASSED（T06）
前置任务：W1-03 通过 T03  
后续任务：W1-07、W1-09  
允许并行：W1-04、W1-05  
任务性质：图表故事、确定性数据重聚合与可复现导出

## 1. 目标

建立一套每张主图只回答一个科学问题、每个数值都能回溯到冻结数据的图表体系。

允许使用现有冻结数据进行只读分析和确定性重绘；不允许新增 rollout、筛选
有利上下文或手工修改数据。

## 2. 输入

- W1-01 的证据索引和术语公式；
- W1-03 的章节架构、追溯矩阵和主文/补充材料边界；
- W1-04 的结果段落占位可在并行过程中同步；
- 冻结 CSV/JSON 结果。

## 3. 主图计划

### Figure 1：测量问题与反事实分支

- 动态合法动作和自回归联合动作；
- N/E 当前步差异；
- 同一步后缀替代；
- 未来 probe/other 替代；
- 正直接成本为何可对应零值或负累计差。

唯一结论：累计回合成本可能偏置局部动作成本读出。

### Figure 2：审计协议与完整恒等式

- 状态快照与 CRN；
- 合法目标精确边缘化；
- stochastic continuation；
- 直接成本和三类替代；
- 账本残差验证。

### Figure 3：R1 发现与 R2 独立确认

- 旧/新种子的 `Sub_shot`；
- 块级 95% 下界；
- 非正 `Delta_C_episode` 的替代解释；
- 清楚区分发现和确认数据。

### Figure 4：同一步与未来替代组成

- `C_direct`；
- `Sub_cost_same`；
- 未来 probe/other 替代；
- 约 17% 同一步与 83% 未来组成；
- `rho_sub` 与符号掩盖。

### Figure 5：场景和资源类型边界

- 三场景 `rho_sub`；
- missile/laser 的 `Sub_shot` 和下界；
- 掩盖上下文；
- 可见地标注 P-C3 未通过。

## 4. 表格计划

| 表 | 内容 |
| --- | --- |
| Table 1 | 任务、来源策略和反事实协议 |
| Table 2 | R1/R2 独立性与完整性 |
| Table 3 | P-C1/P-C2/P-C3 门控 |
| Table 4 | 场景和资源类型边界 |
| Supplementary Table 1 | 标签语义与短视窗前置审计 |
| Supplementary Table 2 | 资源恢复负结果 |
| Supplementary Table 3 | 首轮账本修正与完整性检查 |

## 5. 数据追溯要求

每个面板记录：

| Panel ID | 数据文件 | 字段 | 过滤条件 | 聚合公式 | 绘图脚本 | 导出文件 |
| --- | --- | --- | --- | --- | --- | --- |
| 待填写 | — | — | — | — | — | — |

规则：

- 脚本读取冻结结果，不复制手工中间数字；
- 聚合规则必须与正式报告一致；
- 缺失值和排除规则写入图表元数据；
- 不按视觉效果删点；
- 误差线、下界和样本单位在图注说明；
- 同一符号和颜色在全篇保持一致；
- 主图与补充图不得给出互相冲突的统计范围。

## 6. 图注要求

每个图注至少包含：

- 检验问题；
- 场景和资源类型；
- 种子/模型数量；
- context、repeat 或 ledger 的统计单位；
- 汇总统计和区间定义；
- N/E 方向；
- 该图不支持的外推。

## 7. 交付物

```text
figure_table_plan.md
figure_data_traceability.md
figure_caption_draft_zh.md
table_caption_draft_zh.md
figures/
  source/
  exported/
  metadata/
```

若本任务只完成故事板而未生成正式图，必须在计划中明确 `PLANNED`，不得用空白
占位冒充完成。

## 8. 验收门控 T06

- 每张主图只有一个主要结论；
- 每个面板有数据、字段、过滤、聚合和脚本来源；
- P-C3 失败和资源类型边界可见；
- `Sub_shot` 与 `Sub_cost_total` 未混用；
- R1/R2 发现与确认视觉上可区分；
- 图注包含统计单位和适用边界；
- 无新增实验、选点或结果美化；
- 所有导出可由脚本确定性重现。

## 9. 移交

通过 T06 后向 W1-07 提供解释所需图表结论，向 W1-09 提供全部图表、图注、
追溯文件和未完成项清单。

## 10. 执行结果

完成日期：2026-07-28

### 10.1 已生成交付物

```text
docs/manuscript/action_substitution_cost_identifiability/
  figure_table_plan.md
  figure_data_traceability.md
  figure_caption_draft_zh.md
  table_caption_draft_zh.md
  figure_qa_report.md
  figures/
    source/generate_figures.py
    source/figure_*_data.csv
    exported/figure_*.svg
    exported/figure_*.pdf
    exported/figure_*.tiff
    exported/figure_*_preview.png
    metadata/figure_*_metadata.json
  tables/
    exported/table_*.csv
    exported/table_*.md
    metadata/table_manifest.json
```

主图共 5 张，每张均导出 SVG、PDF、600 dpi TIFF 和 PNG preview；主表 4 张、
补充表 3 张，均同时导出 CSV 和 Markdown。脚本只读冻结结果。

### 10.2 编号和追溯同步

- 将 W1-03 过载的原 Fig. 2 拆为最终 Fig. 2 协议/恒等式与 Fig. 4 成本组成；
- 原边界图顺延为 Fig. 5；
- 更新 `main_vs_supplement_plan.md`、`paragraph_job_map.md`、
  `manuscript_traceability_matrix.md`、`results_draft_zh.md` 和
  `supplementary_results_outline.md` 中的图表落点；
- 未改变 Claim 状态、Evidence 数字、N/E 方向、统计单位或聚合口径。

### 10.3 T06 验收记录

| 检查项 | 结果 |
| --- | --- |
| 五张主图分别具有单一核心结论 | PASS |
| 18 个面板均有数据/公式、字段、过滤、聚合和脚本来源 | PASS |
| P-C3 FAIL、missile 2/9 和 laser 5/9 可见 | PASS |
| \(Sub_{\mathrm{shot}}\) 与 \(Sub_{\mathrm{cost,total}}\) 分轴且未混用 | PASS |
| R1 空心点与 R2 实心点清楚区分发现/确认 | PASS |
| 图注包含方向、场景、模型/seed、统计单位、区间和外推边界 | PASS |
| 无新增 rollout、视觉删点或手工中间数 | PASS |
| 五图 SVG/PDF/TIFF/PNG 均非空 | PASS |
| SVG 保留可编辑文字 | PASS |
| 五张 TIFF 均为 600 dpi | PASS |
| 七张表可由同一脚本确定性生成 | PASS |
| 自动化结构验收 | PASS（48/48） |
| 五张 PNG 原尺寸视觉 QA | PASS |

### 10.4 移交

```text
任务编号：W1-06
状态：PASSED（T06）
已生成文件：5 主图、7 表、源数据、生成脚本、元数据、图注/表注和 QA 记录
已通过门控：T06
未解决问题：补充结果提纲中的 Table S4-S7 仍为 PLANNED，不阻塞主文图表
禁止下游假设：不得隐藏 P-C3，不能把 R1 当独立确认，不能把 Sub_shot 当总替代成本
下一接收任务：W1-07、W1-09
```
