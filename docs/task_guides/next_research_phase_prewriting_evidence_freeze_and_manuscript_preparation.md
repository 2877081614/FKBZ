# 下一项工作建议：写作前证据冻结与论文准备

更新时间：2026-07-23  
建议状态：可立即启动；最终主张等待 R2 独立确认  
任务编号：W0  
任务优先级：与 R2 并行的 P0 写作准备  
论文类型：研究论文  
目标期刊：暂按通用英文期刊组织，具体期刊待定  
任务性质：证据整理、贡献边界冻结与论文结构搭建，不产生新实验结论

## 1. 决策摘要

项目尚未完全进入正式论文定稿阶段，而是处于：

> **R2 独立确认之前的写作准备与证据冻结阶段。**

当前可以立即开始：

- 汇总已完成实验的可用证据；
- 建立术语表和 claim–evidence 矩阵；
- 按“证据先行”顺序搭建 Results、Methods 和 Discussion；
- 形成图表故事板；
- 规划系统文献检索；
- 为 R2 三种可能结果预留明确写作分支。

当前不得定稿：

- 论文标题；
- 摘要；
- 最终贡献列表；
- Discussion 中的外推；
- Conclusion 中的普适性表述；
- “跨场景”“跨资源类型”“通用”或“首次”等结论。

R2 完成后，必须先依据 P-C1/P-C2/P-C3 冻结主张范围，再完成标题、摘要、Introduction 末段和 Conclusion。

本任务不替代：

- [R2：动作替代独立确认与适用边界审计](./next_research_phase_action_substitution_independent_confirmation.md)

两项工作关系为：

```text
W0：整理已经成立的证据、方法和边界
                    ↘
                     R2正式结果
                    ↗
W1：冻结最终主张、完成论文全文
```

## 2. 为什么现在可以开始写作准备

虽然最终主张仍等待 R2，但以下事实已经通过正式实验，可以直接进入论文材料库：

1. 动态合法动作掩码、无冲突自回归动作和 strict joint PPO fallback 已通过软件与机制验证；
2. masked-argmax target 存在 regret，但不是当前可靠 engagement 方向错误的主要来源；
3. deterministic continuation 在低功效上下文中不能稳定代表冻结随机策略的条件期望；
4. 完整回合 engagement 标签缺少稳定资源停止方向；
5. 目标相关短视窗只将可操作标签由27增加到31，仍不能形成跨种子双向覆盖；
6. `time_pressure/resource` 的累计成本差受未来射击替代混叠；
7. 原 R1 的18/18个上下文具有可靠正 `Sub_shot`，11个非正累计成本差全部完成成本重构；
8. 恢复弹药扩大行动集合，但没有形成跨场景、跨种子、跨资源类型的可靠安全机会价值；
9. 通用机会成本 oracle、BPCE/MCH-PPO 在线辅助和GNN路线均已按预注册规则停止。

这些结果已经足以撰写：

- 研究背景和问题演化；
- 环境与动作结构；
- 成对反事实评估方法；
- 标签语义审计；
- 成本恒等分解；
- 已否决机制；
- 当前证据边界。

R2 只决定：

- 动作替代是否可以成为论文主要正贡献；
- 主张是跨资源类型、资源类型条件，还是仅限旧种子；
- 论文应进入正式投稿写作，还是只形成阶段性技术报告。

## 3. 写作阶段定位

### 3.1 当前阶段 W0

目标：

> 在不使用未完成 R2 结果的条件下，完成全部可复用证据、术语、方法和章节骨架。

W0 不写“我们的独立实验进一步证明……”，而使用：

```text
[R2 evidence pending]
[Claim scope pending independent confirmation]
[Resource-type boundary pending]
```

### 3.2 R2 后阶段 W1

R2 完成后，根据唯一门控结果执行：

| R2结果 | W1写作路径 |
| --- | --- |
| P-C1/P-C2/P-C3通过 | 冻结跨新种子、跨资源类型的动作替代测量失真主张 |
| P-C1/P-C2通过，P-C3失败 | 冻结场景或资源类型条件主张 |
| P-C1通过，P-C2失败 | 将P-R1降级为旧种子条件发现，重新评估第一创新问题 |
| P-C1失败 | 不进入主论文定稿，先修复成本账本与分支语义 |

## 4. 当前论文核心论证

### 4.1 一句话工作论证

> 在动态掩码防空资源分配中，本研究通过共同随机数反事实轨迹和逐时刻成本分解发现，回合累计资源成本会混合当前直接消耗与未来动作替代；该失真的独立适用范围仍等待 R2 确认。

该句是 W0 的工作论证，不是最终摘要句。

### 4.2 完整论证链

```text
动态防空资源分配需要同时决定是否交战、目标和执行单元
                    ↓
局部反事实信用被用于缓解联合PPO的种子分叉
                    ↓
多轮受控审计表明标签问题不主要来自target argmax
                    ↓
随机后续与短视窗仍不能形成稳定双向资源停止标签
                    ↓
逐步成本账本发现当前射击替代未来射击
                    ↓
累计成本差因此不能稳定代表当前动作资源消耗
                    ↓
资源恢复扩大行动集合，但不稳定转化为安全收益
                    ↓
贡献收窄为资源反事实信用的测量失真与可辨识性边界
                    ↓
R2决定该机制能否跨新种子和资源类型成立
```

## 5. Problem–Method–Insight

| 层次 | W0工作表述 |
| --- | --- |
| Problem | 动态序列资源分配中的回合累计成本差同时包含当前直接消耗和后续动作替代，使局部资源信用可能出现零值或反向。 |
| Method | 使用动态合法目标精确边缘化、冻结策略随机延续、共同随机数 N/E 分支和逐时刻成本账本，将当前直接成本与未来替代成本分解。 |
| Insight | 资源成本标签的有效性取决于未来替代比率；增加 rollout 只能降低采样误差，不能消除估计量本身的结构性混叠。 |

W0 可以写 Problem 和 Method。Insight 的普适范围必须等待 R2。

## 6. 术语表冻结

写作前建立统一术语账本，全文只使用以下规范名称。

| 中文概念 | 英文规范名 | 缩写/符号 | 使用边界 |
| --- | --- | --- | --- |
| 防空资源智能分配 | intelligent air-defence resource allocation | — | 任务总称 |
| 动态合法动作掩码 | dynamic legal-action masking | — | 不简写为普通 action mask |
| 无冲突自回归联合动作 | conflict-free autoregressive joint action | — | 描述动作结构 |
| 因子化联合PPO | factorized joint PPO | — | 安全主干/来源策略 |
| 成对反事实轨迹 | paired counterfactual trajectories | — | N/E及相关分支总称 |
| 共同随机数 | common random numbers | CRN | 环境和策略随机带控制 |
| 当前直接成本 | current direct cost | `C_direct` | 当前被测动作产生的成本 |
| 未来替代成本 | future substitution cost | `Sub_cost` | N相对E多产生的未来成本 |
| 未来替代射击 | future substituted shots | `Sub_shot` | N相对E多产生的未来射击 |
| 回合累计成本差 | episode-level cumulative cost difference | `Delta_C_episode` | E减N的整回合成本 |
| 替代比率 | substitution ratio | `rho_sub` | `Sub_cost/C_direct` |
| 成本符号掩盖 | cost-sign masking | — | `C_direct>0`但`Delta_C_episode<=0` |
| 动作替代 | action substitution | — | 当前动作替代未来动作的机制 |
| 弹药机会安全价值 | ammunition opportunity safety value | — | 仅用于R1 E/E-R负结果 |
| 可辨识性边界 | identifiability boundary | — | 不等同于算法性能边界 |
| 不明确标签 | ambiguous label | — | 不译为 uncertain class，除非定义改变 |

以下名称仅用于历史机制路线，不得作为最终成功算法名称：

```text
MCH-PPO
RG-MCH-PPO
SA-RG-MCH
BPCE-PPO
```

全文必须区分：

- `resource cost`：环境中实际射击成本；
- `opportunity value`：保留资源产生的未来安全价值；
- `cost-value prediction`：模型预测；
- `cost difference`：N/E实测差值。

不得混用。

## 7. Claim–Evidence 矩阵

W0 必须首先创建正式矩阵，建议初始内容如下。

| Claim ID | 候选主张 | 当前证据 | 状态 | R2作用 |
| --- | --- | --- | --- | --- |
| C1 | 动态掩码和无冲突自回归动作可以严格消除非法分配与目标冲突 | 软件测试、30k/100k基线和相关实验 | 已支持 | 无 |
| C2 | target argmax不是BPCE方向失真的主要来源 | A/B一致率0.901，可靠反转0/24 | 已支持但限当前上下文 | 无 |
| C3 | deterministic continuation在低功效上下文中不稳定 | B/C一致率0.778，低于门槛 | 已支持 | 无 |
| C4 | 短视窗不能恢复通用双向engagement标签 | 31/72可操作，time为5/2 | 已支持 | 无 |
| C5 | time/resource累计成本差受未来射击替代混叠 | P-R1为18/18，11/11成本重构 | 当前数据支持 | 独立确认 |
| C6 | 恢复弹药扩大行动集合但不稳定改善安全结果 | P-R2/P-R3失败 | 已支持的负结果 | 无 |
| C7 | 动作替代测量失真跨新策略种子成立 | 尚无 | 待证据 | P-C2 |
| C8 | 动作替代测量失真跨missile/laser成立 | 尚无 | 待证据 | P-C3 |
| C9 | 通用机会成本oracle可以稳定训练 | 证据否决 | 禁止使用 | 不再测试 |
| C10 | BPCE/MCH-PPO解决all-noop或过度交战 | 正式门控失败 | 禁止使用 | 不再测试 |

每一段 Results、Discussion 和 Conclusion 都必须能够回溯到至少一个 Claim ID。

## 8. 论文结构建议

### 8.1 论文类型

建议定位为：

> **以测量有效性、机制审计和可辨识性边界为主的研究论文。**

暂不定位为：

- 新PPO算法论文；
- GNN防空分配论文；
- 机会成本预测模型论文；
- 纯性能benchmark论文。

### 8.2 章节架构

```text
1. Introduction
2. Related Work
3. Problem Formulation and Evaluation Framework
4. Paired Counterfactual Cost Decomposition
5. Experimental Protocol
6. Results
   6.1 Structured policy and failure symptom
   6.2 Target and continuation semantic audit
   6.3 Full-return versus short-window identifiability
   6.4 Action substitution and exact cost decomposition
   6.5 Resource restoration and opportunity-value boundary
   6.6 Independent confirmation and resource-type boundary [R2 pending]
7. Discussion
8. Limitations
9. Conclusion
```

如果目标期刊采用 Nature-family 风格，可将 Related Work 合并到 Introduction，并将 Methods 后置。具体期刊确定前不做格式性重排。

## 9. 推荐写作顺序

研究论文应从证据开始，不能先写一个过度承诺的 Introduction。

冻结顺序：

```text
1. Terminology ledger
2. Claim–evidence matrix
3. Results（已完成证据）
4. Methods
5. Figure/table plan
6. R2结果占位及三分支文本
7. Discussion
8. Introduction
9. Conclusion
10. Title
11. Abstract
```

标题和摘要最后写，因为它们必须服从 R2 后的最终主张范围。

## 10. Results 写作任务

### 10.1 可立即撰写

#### Results 1：结构化策略与问题症状

段落任务：

- 说明动态掩码和无冲突动作结构已经成立；
- 报告联合 PPO 仍出现 all-noop/高成本分叉；
- 将问题从动作合法性收窄到 engagement 信用与资源成本测量。

禁止：

- 把 factorized PPO 写成最终优胜算法；
- 详细堆叠所有历史任务指标；
- 用失败模型数量替代机制论证。

#### Results 2：标签语义审计

核心证据：

```text
A/B一致率：0.901
可靠target方向反转：0/24
B/C一致率：0.778
随机后续可靠标签：25/72
```

段落工作：

- target argmax regret存在但不是主要方向错误；
- deterministic continuation 在低功效上下文失稳；
- 观察与解释分段书写。

#### Results 3：短视窗审计

核心证据：

```text
短视窗：15 ENGAGE / 16 STOP / 41 AMBIGUOUS
完整回合：14 ENGAGE / 13 STOP / 45 AMBIGUOUS
time_pressure：5/2
heterogeneity_pressure：10/14
```

段落工作：

- 短视窗只增加4个可操作标签；
- 异质资源场景条件有效；
- time/resource 仍为0/0/18；
- 不把“短视窗失败”解释为“局部后果完全无效”。

#### Results 4：动作替代与成本分解

核心证据：

```text
time/resource mean(Sub_shot)>0：18/18
lower95(Sub_shot)>0：18/18
非正累计成本差：11
替代解释：11/11
平均Sub_shot：0.990
平均Sub_cost：1.995
首次替代时刻：2.86步
成本重构误差：4.00e-15
```

该节是当前最强正结果，但结尾必须写：

```text
Whether this mechanism generalizes to previously unused source-policy seeds
and resource types is evaluated in the independent confirmation study.
```

#### Results 5：资源恢复负结果

核心证据：

```text
time资源槽可靠机会：5/18
heterogeneity资源槽可靠机会：2/18
异质可靠机会全部来自seed9
7个可靠资源上下文全部为missile
```

段落工作：

- 区分行动集合扩大和最终安全改善；
- 明确机会价值不能构成通用监督；
- 将负结果写成主张边界，而不是附录中的失败尝试。

### 10.2 等待 R2

Results 6 仅保留：

```text
[R2 data integrity]
[P-C1 cost decomposition]
[P-C2 independent replication]
[P-C3 missile/laser boundary]
[medium/time/heterogeneity boundary]
```

正式结果生成前不得预写“confirmed”“generalized”或“robust”。

## 11. Methods 写作任务

Methods 可以在 R2 前基本完成。

建议结构：

### 11.1 AirDefense v1

- 资源类型、弹药、成本、命中概率和冷却；
- 动态目标、保护区和time-to-impact；
- 动作合法性和目标占用约束；
- episode终止与奖励分量。

### 11.2 Factorized joint policy

- engagement/target 因子化；
- 无冲突自回归前缀；
- 动态合法掩码；
- strict joint PPO fallback。

仅描述可复现结构，不宣称其稳定解决种子分叉。

### 11.3 Paired counterfactual trajectories

- 状态快照；
- N/E分支；
- 环境共同随机带；
- 策略uniform tape；
- 合法目标精确边缘化；
- Actor冻结。

### 11.4 Label semantic audits

- A/B/C定义；
- 完整回合与事件窗；
- ENGAGE/STOP/AMBIGUOUS；
- 置信区间与预注册门控。

### 11.5 Cost decomposition

```text
Delta_C_episode = C_direct - Sub_cost
rho_sub = Sub_cost / C_direct
```

- 当前/未来成本边界；
- probe/other-units分解；
- cost-sign masking定义；
- 首次替代时刻。

### 11.6 Resource restoration audit

- E/E-R当前步身份；
- 只恢复一枚弹药；
- 不退还即时成本；
- 机会安全价值定义；
- 终止步不可观测处理。

### 11.7 Independent confirmation

先写协议，不写结果：

- 种子17/18/19；
- 三场景；
- 108个新上下文；
- observation hash零重叠；
- missile/laser配额；
- P-C1/P-C2/P-C3。

## 12. Figure 与 Table 故事板

### Figure 1：任务与测量问题

建议内容：

- 当前强制 engage；
- no-op 后未来射击；
- 当前直接成本；
- 未来替代成本；
- 为什么累计差可以接近0。

结论：

> Episode-level cost differences can mask a positive current action cost.

### Figure 2：标签语义审计链

面板建议：

- A/B target边缘化一致性；
- B/C continuation分歧；
- 完整回合与短视窗可操作标签；
- 场景×种子双向覆盖。

### Figure 3：动作替代主结果

面板建议：

- context级 `Sub_shot`；
- `C_direct`、`Sub_cost`、`Delta_C_episode` 分解；
- 首次替代时刻；
- 非正累计成本差的重构。

### Figure 4：行动集合与安全价值断裂

面板建议：

- `Reuse_probe`；
- `OptionEdge`；
- `AmmoGain_D/L`；
- 场景×种子可靠机会数量。

### Figure 5：R2独立边界

当前占位：

- 新种子 `rho_sub`；
- missile/laser；
- 三场景；
- cost-sign masking比例。

### 主表

| 表格 | 内容 |
| --- | --- |
| Table 1 | 环境、资源类型和来源策略 |
| Table 2 | 各阶段预注册命题与门控结果 |
| Table 3 | N/E成本分解与替代统计 |
| Table 4 | R2独立确认和边界结果 |

图表只服务于 Claim ID，不绘制没有明确结论的装饰性图。

## 13. Introduction 写作占位

Introduction 在 Results 骨架完成后写。

建议采用“技术瓶颈”结构：

1. 动态资源分配需要在安全收益和不可恢复资源之间进行序列决策；
2. 局部反事实信用常用回合价值差评价当前动作；
3. 在序列策略中，当前动作会改变未来动作数量与分工，因此累计成本差可能混合直接消耗和动作替代；
4. 现有项目中的多轮在线失败使这一测量问题可观测；
5. 本文使用共同随机数轨迹、标签语义审计和成本恒等分解研究其适用边界。

Introduction 末段贡献列表必须等待 R2。

当前只能使用候选表述：

```text
We investigate...
We decompose...
We evaluate whether...
```

不得提前使用：

```text
We establish a general...
We solve...
Our algorithm consistently...
```

## 14. Related Work 任务

正式写作前需完成系统查新，至少覆盖：

1. 多智能体反事实信用分配；
2. 分层/参数化动作 PPO；
3. 动态动作掩码与组合分配；
4. 资源约束、对偶价值和shadow price；
5. sequential treatment/action substitution；
6. common-random-number counterfactual evaluation；
7. negative results、identifiability和offline policy evaluation。

每类相关工作按机制组织，不按作者逐篇罗列。

相关工作矩阵至少包含：

| 文献 | 问题 | 估计量 | 是否分离当前/未来成本 | 动态动作替代 | 与本项目差异 |
| --- | --- | --- | --- | --- | --- |

在系统检索和原文核查完成前，不使用：

```text
first
previously unexplored
no prior work
```

## 15. Discussion 写作任务

Discussion 建议按以下顺序：

1. 中心发现：累计成本读出包含动作替代；
2. 为什么增加 rollout 不能修复结构混叠；
3. 为什么行动集合扩大不保证最终安全改善；
4. 与反事实信用、资源约束和离线价值估计的关系；
5. 场景、种子和资源类型边界；
6. 对在线辅助设计的含义；
7. 不能外推到最优策略影子价格；
8. 对未来研究的有限建议。

必须主动讨论竞争解释：

- 单位成本异质性；
- 来源策略行为差异；
- all-noop导致的状态分布变化；
- episode剩余长度；
- 目标紧迫度；
- 其他单元替代覆盖；
- 固定策略而非最优策略。

Discussion 不得把相关性写成因果机制，只有共同随机数配对干预和恒等分解支持的部分才能使用 `show` 或 `demonstrate`。

## 16. Conclusion 和 Abstract 门控

### 16.1 Conclusion

只有 R2 后才能写最终版本，必须包含：

```text
贡献
决定性证据
适用意义
明确边界
```

不加入新数据，不重新叙述全部失败实验。

### 16.2 Abstract

最后撰写，结构为：

```text
context
measurement gap
paired decomposition method
decisive result
independent confirmation
bounded implication
```

若 R2 条件通过，摘要必须直接包含条件：

```text
under time pressure
for missile resources
under frozen factorized policies
```

### 16.3 Title

当前只保留方向，不冻结标题。

候选模式：

```text
Action substitution masks resource cost in dynamic air-defence allocation
Paired counterfactual cost decomposition reveals action substitution in sequential resource allocation
Identifiability limits of episode-level resource credit under dynamic action substitution
```

标题最终选择取决于 R2：

- 跨资源类型通过：可使用较宽的动态资源分配表述；
- 仅missile通过：标题必须包含条件或避免通用名词；
- 独立确认失败：标题不能使用 `reveals` 或跨种子表述。

## 17. W0 验收标准

W0 完成必须同时满足：

- 术语账本完成并在全部草稿中一致；
- claim–evidence 矩阵覆盖所有主要主张；
- 所有数字均可追溯到正式实验报告或结果文件；
- Methods 不包含无法复现的模糊步骤；
- 已完成 Results 1–5 的证据骨架；
- Results 6 保留 R2 占位，不虚构结果；
- 图表故事板中的每个面板对应明确 Claim ID；
- 相关工作检索计划完成，但未核查文献不进入正文；
- Discussion 明确列出竞争解释和边界；
- 标题、摘要、最终贡献和Conclusion保持未冻结；
- 全文不把失败的 BPCE/MCH-PPO 表述为成功算法；
- 不使用未经查新的“首次”主张。

## 18. R2 后 W1 验收标准

W1 只有在 R2 正式报告完成后启动。

必须完成：

1. 将 P-C1/P-C2/P-C3 写入 claim–evidence 矩阵；
2. 选择通用、条件或降级写作分支；
3. 更新 Results 6；
4. 重写 Introduction 末段贡献；
5. 完成 Discussion 中的外推边界；
6. 冻结 Conclusion；
7. 最后撰写 Title 和 Abstract；
8. 执行逐句 claim–evidence 审计；
9. 执行审稿人视角的拒稿风险检查；
10. 确定目标期刊后调整格式、字数和相关工作位置。

## 19. 建议产物

W0 建议建立：

```text
docs/manuscript/action_substitution_cost_identifiability/
  manuscript_plan.md
  terminology_ledger.md
  claim_evidence_matrix.md
  evidence_index.md
  figure_table_plan.md
  methods_draft.md
  results_draft.md
  discussion_scaffold.md
  related_work_search_plan.md
  r2_result_placeholders.md
```

R2 后 W1 再创建：

```text
docs/manuscript/action_substitution_cost_identifiability/
  manuscript_draft_zh.md
  manuscript_draft_en.md
  supplementary_methods.md
  supplementary_results.md
  submission_checklist.md
```

本任务报告本身不要求立即创建全部产物；只有在 W0 正式启动后按顺序生成。

## 20. 审稿压力点

| 风险 | 可能质疑 | 写作处理 |
| --- | --- | --- |
| 负结果包装 | 失败算法是否被重新包装为创新？ | 把算法失败作为问题来源，主要贡献限定为测量机制与边界。 |
| 恒等式过于简单 | 成本分解是否只是记账？ | 强调实证发现是未来替代足以改变标签符号，并提供独立确认与边界。 |
| 同一环境过拟合 | 是否只在AirDefense v1成立？ | 明确范围；R2使用新种子、新状态、三场景和资源类型配额。 |
| 缺少算法提升 | 为什么值得发表？ | 论证无效资源信用会误导多类在线辅助设计；不虚构性能提升。 |
| 失败实验过多 | 论文是否成为实验流水账？ | 只保留支持因果收窄链条的阶段，其余移至补充材料。 |
| 机会价值失败 | 是否否定所有资源价值建模？ | 仅否定当前冻结策略、当前环境中的通用机会安全监督。 |
| 结论外推 | 是否声称适用于所有MARL？ | 限定为动态掩码、序列资源分配和策略条件反事实评估。 |
| 相关工作不足 | action substitution是否已有成熟定义？ | 完成系统查新后再冻结创新措辞，不使用“首次”。 |

## 21. 创新演化记录

| 阶段 | 原主张 | 新证据 | 当前写作处理 |
| --- | --- | --- | --- |
| MCH/BPCE | 反事实分层信用可稳定在线PPO | 多个候选出现all-noop或高成本分叉 | 不作为成功算法贡献 |
| 标签语义A/A2 | 随机后续或短视窗可恢复双向标签 | 25/72与31/72，跨种子覆盖失败 | 作为可辨识性收窄证据 |
| R1动作替代 | 累计成本差受未来动作替代 | time/resource 18/18、11/11重构 | 当前最强机制证据 |
| R1机会价值 | 行动集合扩大可形成安全机会成本 | P-R2/P-R3失败 | 作为关键负结果和边界 |
| R2待完成 | 动作替代跨新种子与资源类型成立 | 尚无 | 决定最终贡献范围 |

## 22. 建议执行顺序

```text
1. 建立manuscript工作目录
2. 冻结术语账本
3. 建立claim–evidence矩阵
4. 建立正式证据索引
5. 撰写Results 1–5证据骨架
6. 撰写Methods完整初稿
7. 规划Figures/Tables
8. 建立R2三分支结果占位
9. 编写Discussion骨架与竞争解释
10. 制定系统文献检索方案
11. 等待并读取R2正式结果
12. 冻结最终主张范围
13. 完成Introduction、Conclusion、Title和Abstract
14. 执行claim–evidence与审稿风险审计
15. 确定期刊后完成英文稿和投稿格式
```

步骤1至10可以与 R2 并行。步骤11至15必须等待 R2 正式门控完成。

## 23. 当前完成定义

当前“下一项工作”不是立即交付一篇完整论文，而是：

> **在 R2 运行期间完成全部不依赖最终确认结果的写作基础，使 R2 一旦结束即可按唯一证据分支冻结主张并进入正式全文写作。**

该安排可以避免两个风险：

1. 等待实验期间完全停滞；
2. 在独立确认前把条件性发现提前写成普适创新。
