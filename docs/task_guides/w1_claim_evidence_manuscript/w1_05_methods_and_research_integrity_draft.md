# W1-05：Methods 与研究完整性稿

更新时间：2026-07-24  
任务状态：NOT_STARTED  
前置任务：W1-03 通过 T03  
后续任务：W1-08、W1-09  
允许并行：W1-04、W1-06  
任务性质：方法复现与研究完整性披露

## 1. 目标

完成足以复现任务、N/E 反事实协议、成本账本和独立确认的中文 Methods 稿，
同时完整记录首轮账本修正。

Methods 回答“如何完成”，不解释结果为何重要。

## 2. 输入

- W1-01 的术语、公式、证据索引；
- W1-03 的章节架构和追溯矩阵；
- R2 正式报告、任务协议、代码和冻结结果；
- R1 与标签审计的正式方法记录。

## 3. 方法章节

### 3.1 AirDefense v1 与任务形式化

写清：

- 状态、动作、资源、目标和保护区；
- missile/laser 成本和交战约束；
- 动态合法目标与目标占用；
- episode 终止；
- 奖励和成本分量；
- 适用场景和规模。

### 3.2 来源策略与联合动作

- factorized joint PPO；
- engagement/target 因子化；
- dynamic legal-action masking；
- conflict-free autoregressive suffix；
- strict joint PPO fallback；
- 来源模型训练、保留和不筛选原则。

factorized PPO 是来源策略，不是本论文提出的新算法。

### 3.3 成对反事实轨迹

- 状态快照；
- N/E 当前步身份；
- 环境 CRN 随机带；
- 策略 uniform tape；
- 合法目标精确边缘化；
- stochastic continuation；
- Actor 冻结和参数一致性。

第一次出现 N/E 时写出明确差值方向。

### 3.4 前置标签语义审计

只保留收窄最终问题所必需的：

- A/B/C；
- target 边缘化；
- deterministic/stochastic continuation；
- 完整回合与短视窗；
- ENGAGE/STOP/AMBIGUOUS；
- 为什么问题由标签语义收窄到累计成本测量。

不得把全部历史审计写成主要方法模块。

### 3.5 完整成本账本

必须给出：

```text
Sub_cost_total
:= Sub_cost_same
 + Sub_cost_future_probe
 + Sub_cost_future_other
```

```text
Delta_C_episode
= C_direct - Sub_cost_total
```

并定义：

- 所有当前/未来字段；
- probe/other 子分解；
- `Sub_shot`；
- `rho_sub`；
- cost-sign masking；
- 数值容差；
- 完整性门槛。

### 3.6 独立确认协议

- seeds 17/18/19；
- 三个场景；
- 9 个来源模型；
- 108 个零重叠上下文；
- 每块 3 missile + 3 laser；
- 每上下文 32 次重复；
- P-C1/P-C2/P-C3；
- 9/9 模型无条件保留。

### 3.7 统计与门控

明确：

- 分析单位；
- 块级和上下文级统计；
- 95% 下界；
- 预注册阈值；
- 多层数据的汇总关系；
- 哪些指标用于发现、哪些用于确认。

## 4. 研究完整性披露

主文至少一段，补充材料完整记录：

1. 原 future-only 恒等式；
2. `287/7776` 条目出现非零残差；
3. 最大残差 `2.0`；
4. 根因为同一步其他单元替代；
5. 扩展恒等式误差 `8.88e-16`；
6. 未修改模型、上下文、随机带和门槛；
7. 首轮无效结果归档；
8. 仅一次完整重跑。

不得写成“常规代码修复”而省略科学定义变化，也不得写成“预先假设了同一步替代”。

## 5. 复现材料映射

建立：

| 方法步骤 | 代码 | 配置 | 输入数据 | 输出数据 | 正文位置 |
| --- | --- | --- | --- | --- | --- |
| 待填写 | — | — | — | — | — |

只记录实际存在的文件，不承诺尚未决定的代码、权重或数据发布。

## 6. 交付物

```text
methods_draft_zh.md
supplementary_methods.md
research_integrity_disclosure.md
reproducibility_map.md
```

并更新追溯矩阵中的 Methods 和 Supplement 位置。

## 7. 验收门控 T05

- 同行可复现 N/E、CRN 和成本账本；
- N/E 方向与 W1-01 一致；
- 三类替代成本完整；
- 前置审计没有喧宾夺主；
- 独立确认的模型、状态、资源配额和重复完整；
- 首轮修正透明且不带事后合理化；
- 没有虚构发布承诺；
- Methods 不混入 Discussion 解释。

## 8. 停止与移交

代码、报告和结果字段不一致时，记录问题并暂停相应小节，不自行修改科学定义。

通过 T05 后向 W1-08、W1-09 移交四个文件和追溯矩阵更新。

