# W1-02：系统查新与稿件定位门控

更新时间：2026-07-24  
任务状态：PASSED（L2）  
前置任务：W1-01 通过 T01  
后续任务：W1-03、W1-07、W1-08  
允许并行：完成检索协议后可分主题检索  
任务性质：系统文献检索、优先权核验与稿件定位

## 1. 目标

回答两个决定后续投入方向的问题：

1. 相邻工作是否已经同时覆盖当前 Problem、Method、Evidence 和 Insight；
2. 当前成果应定位为独立论文、较大方法论文组成部分，还是学位论文章节。

本任务不是为了堆积参考文献，而是尽早发现独立投稿定位是否会进入死路。

## 2. 输入

- W1-01 的 `terminology_ledger.md`
- W1-01 的 `evidence_source_index.md`
- [第一创新 Claim–Evidence 矩阵](../../project/first_innovation_claim_evidence_matrix.md)
- [W1 总纲](../next_research_phase_claim_evidence_freeze_and_manuscript_drafting.md)

## 3. 检索主题

至少覆盖：

1. multi-agent credit assignment；
2. counterfactual baselines 与 difference rewards；
3. temporal credit assignment 和 delayed action effects；
4. common-random-number paired simulation；
5. action substitution、action displacement、policy-induced substitution；
6. sequential/autoregressive joint action allocation；
7. action masking 下的反事实评估；
8. resource shadow price、opportunity cost、constrained MARL；
9. episode-return label validity、measurement bias、identifiability。

检索中发现规范术语与 W1-01 不一致时，不直接覆盖术语账本，而提交变更建议。

## 4. 检索协议

每组检索记录：

| 项目 | 要求 |
| --- | --- |
| 数据库 | 明确名称 |
| 日期 | 精确到日 |
| 检索式 | 完整可复制 |
| 时间范围 | 明确起止年份 |
| 类型限制 | 论文、预印本、学位论文等 |
| 初筛 | 数量与标题判断规则 |
| 复筛 | 摘要/全文纳入理由 |
| 原始来源 | DOI、arXiv ID 或出版社链接 |
| 排除理由 | 与问题、方法或证据无关的具体原因 |

只引用实际阅读过的来源。综述可用于导航，但最接近工作的判定必须阅读原始论文。

## 5. 四层比较

对最接近的工作逐项填写：

| 文献 | Problem | Method | Evidence | Insight | 重叠 | 差异 | 风险 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 待检索 | — | — | — | — | — | — | — |

差异判断不得只依靠：

- 环境名称不同；
- 模型名称不同；
- 使用了不同缩写；
- 多做了几个种子；
- “我们的图更完整”。

可形成实质差异的层次包括：

- 既有工作没有识别相同的测量对象混叠；
- 既有方法未分离同一步后缀替代和未来替代；
- 既有证据没有验证成本符号边界；
- 既有洞见未指出 rollout 数量与结构性混叠的区别。

## 6. 创新压力测试

### 6.1 重复性

- 是否已有工作直接研究累计回合成本作为局部信用的偏置；
- 是否已有等价的动作替代成本恒等式；
- 是否已有相同的动态掩码序列分配边界。

### 6.2 动机

- 为什么 CRN 和账本分解针对的是测量问题；
- 为什么普通方差降低不足以解决该问题；
- 为什么资源类型边界是结论而不是实验噪声。

### 6.3 可证伪性

核验 F1–F6 是否在论文中保持原支持/否决状态，不因文献定位而静默改写。

### 6.4 故事压缩

检索后试写：

- 一句话；
- Problem–Method–Insight 三句话；
- 不超过 120 词的一段话。

三种尺度都不能依赖“首次”才能成立。

## 7. 定位门控 L

| 判定 | 条件 | 后续 |
| --- | --- | --- |
| L1 | 相邻工作未同时覆盖当前 Problem、完整分解 Method 和条件性 Insight | 按独立论文执行 |
| L2 | Problem 已知，但完整分解或独立边界仍有实质差异 | 定位为较大方法论文组成部分 |
| L3 | Problem–Method–Insight 已基本覆盖，仅 AirDefense v1 证据不同 | 定位为学位论文章节/技术报告 |
| L4 | 关键来源未取得或差异仍依赖猜测 | 暂停 W1-03 以后任务 |

L2/L3 不触发新实验。若必须形成算法创新，应另立研究任务。

## 8. 交付物

在公共稿件目录创建：

```text
literature_search_protocol.md
literature_search_log.md
literature_evidence_matrix.md
closest_work_comparison.md
paper_positioning_decision.md
novelty_evolution_log.md
```

`paper_positioning_decision.md` 必须写明：

- L1/L2/L3/L4；
- 判定依据；
- 三项贡献的保留、收窄或删除；
- 允许和禁止的优先权措辞；
- 后续稿件形态。

## 9. 验收条件

- 检索可复核；
- 最接近工作阅读了原始来源；
- 四层比较完成；
- `action substitution` 术语已核验；
- 三项贡献不依赖形容词或优先权口号；
- 给出唯一 L 判定；
- 对 L2/L3 的收窄有版本记录；
- 没有虚构引用或把综述当作原始证据。

## 10. 移交

向 W1-03 提交定位决策和最终三层论证；向 W1-07/W1-08 提交按技术主题组织的
文献矩阵。若判定 L4，只移交缺口清单，不启动后续写作。

## 11. 执行结果（2026-07-24）

### 11.1 门禁判定

**L2：定位为较大方法论文中的测量、诊断与资源信用分解模块。**

理由：

- 反事实信用、后续动作介导效应和顺序团队信用已有直接原始工作；
- 当前动态合法掩码、同一步后缀、三分量资源成本恒等式和条件化符号边界
  仍构成 Method/Insight 层的实质差异；
- 差异不只来自 AirDefense 场景，但不足以支撑独立通用算法优先权。

### 11.2 交付件

- `literature_search_protocol.md`
- `literature_search_log.md`
- `literature_evidence_matrix.md`
- `closest_work_comparison.md`
- `paper_positioning_decision.md`
- `novelty_evolution_log.md`

交付目录：
`docs/manuscript/action_substitution_cost_identifiability/`

### 11.3 验收记录

| 条件 | 结果 |
| --- | --- |
| 检索式、日期、范围和筛选规则可复核 | PASS |
| 最近工作回到原始出版页/预印本 | PASS |
| Problem–Method–Evidence–Insight 四层比较 | PASS |
| `action substitution` 术语核验 | PASS；提交术语变更建议，未覆盖 W1-01 账本 |
| 三项贡献去除“首次”等口号 | PASS |
| 唯一 L 判定 | PASS：L2 |
| L2 收窄形成版本记录 | PASS |
| F1-F6 原支持/否决状态保持 | PASS |
| 新实验 | 未启动，符合任务约束 |

### 11.4 移交

- W1-03 使用 L2 和修订后三项贡献组织稿件主论证；
- W1-07/W1-08 使用 24 篇原始工作证据矩阵组织相关工作；
- 术语评审时考虑将正文总称改为
  `downstream action-mediated cost substitution`；
- 在线算法创新和 GNN 不从本任务自动恢复，须另立研究门控。
