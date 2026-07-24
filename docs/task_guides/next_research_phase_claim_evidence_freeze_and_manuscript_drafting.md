# 下一项工作建议：W1 主张—证据冻结与论文正文写作

更新时间：2026-07-24  
建议状态：可立即启动  
任务编号：W1  
任务优先级：P0  
论文类型：测量有效性、反事实信用与可辨识性边界研究论文  
目标期刊：暂按通用英文期刊组织，待系统查新和稿件定位后确定  
任务性质：文献检索、证据冻结、图表规划与论文写作；默认不新增实验

拆分执行入口：
[W1 主张—证据冻结与论文写作任务包](./w1_claim_evidence_manuscript/README.md)

本报告保留为 W1 总纲。实际执行按任务包中的 W1-01 至 W1-10 分项领取、
验收和移交，避免在单一任务中同时混合查新、数据核查、章节写作和投稿审计。

## 1. 决策摘要

R2 动作替代独立确认已经完成：

- P-C1 成本分解通过；
- P-C2 跨新策略种子确认通过；
- P-C3 跨资源类型的统一符号掩盖结论未通过；
- 第一创新贡献已经收敛为场景和资源类型条件下的测量与可辨识性结论；
- 追加种子、资源恢复、机会成本网络、BPCE/MCH-PPO 和 GNN 路线继续停止。

因此，项目当前不再处于“等待 R2 的写作准备阶段”，而正式进入：

> **W1：系统查新、主张—证据冻结与证据优先的论文正文写作。**

W1 不以尽快写出一版摘要为目标，而以形成一篇主张可追溯、证据不越界、
公式无歧义、能够接受审稿压力测试的完整稿件为目标。

本阶段必须保留以下两条约束：

1. 当前贡献是测量、机制审计和可辨识性边界贡献，不是已经优于 PPO 的新算法；
2. “首次”“通用”“跨资源一致”等表述必须等待系统文献检索或已经被证据否决。

W1 的入口材料为：

- [第一创新 Claim–Evidence 矩阵](../project/first_innovation_claim_evidence_matrix.md)
- [R2 动作替代独立确认正式报告](../experiments/air_defense_v1_action_substitution_confirmation.md)
- [R1 动作替代与机会成本审计](../experiments/air_defense_v1_action_substitution_opportunity_cost_audit.md)
- [BPCE 标签语义审计](../experiments/air_defense_v1_bpce_label_semantics_audit.md)
- [BPCE 短视窗标签审计](../experiments/air_defense_v1_bpce_short_horizon_label_audit.md)
- [研究创新路线图](../project/research_innovation_roadmap.md)
- [学术项目进度](../project/academic_project_progress.md)

原 W0 报告继续保留为 R2 完成前的历史计划，不覆盖、不改写：

- [W0 写作前证据冻结与论文准备](./next_research_phase_prewriting_evidence_freeze_and_manuscript_preparation.md)

## 2. 当前项目阶段

### 2.1 已完成

以下研究判断已经完成正式门控，不在 W1 中重新打开：

1. 动态合法动作掩码和无冲突自回归联合动作已通过软件与机制验证；
2. target argmax 不是当前 BPCE 方向失效的主要来源；
3. deterministic continuation 不能稳定代表冻结随机策略的条件期望；
4. 短视窗标签不能恢复跨场景、跨种子的通用双向 engagement 监督；
5. 当前动作会通过同一步后缀单元和后续策略动作产生替代；
6. 累计资源成本差因而不能稳定等同于当前动作的局部直接成本；
7. 该替代机制已经在新策略种子、新状态和三个场景中独立确认；
8. 替代是否足以改变成本标签符号受场景和资源类型约束；
9. 弹药恢复没有形成通用安全机会价值；
10. 当前 BPCE/MCH-PPO 不构成通过门控的在线算法贡献。

### 2.2 尚未完成

W1 必须解决以下写作问题：

1. 既有文献是否已经在相同问题、方法和洞见层面覆盖当前贡献；
2. 当前成果适合独立研究论文、短论文、方法论文组成部分还是学位论文章节；
3. 哪些历史实验属于主文因果收敛链，哪些应移入补充材料；
4. 每项主张对应哪一份正式证据、哪张图、哪张表和哪一段正文；
5. 如何透明披露首轮成本账本公式不完整及唯一一次修正重跑；
6. 如何用英文表达条件性结论而不把“机制存在”误写为“所有资源均发生符号反转”。

## 3. W1 目标与非目标

### 3.1 总目标

> 在不追加机制实验的前提下，完成系统文献定位、证据与术语冻结、主图和表格
> 故事板、中文证据稿、英文完整初稿及投稿前主张审计。

### 3.2 必须完成

- 建立带检索式、时间和纳入规则的系统文献记录；
- 冻结一条核心论证、三项主要贡献和明确的禁止表述；
- 将现有 Claim–Evidence 矩阵扩展为章节—图表—数据文件可追溯关系；
- 冻结 N/E 分支语义、成本方向、符号和分解公式；
- 先完成 Results 和 Methods，再完成其他章节；
- 将首轮无效账本和修正过程写入可审计的方法记录；
- 形成可供审稿人复核的局限性和竞争解释清单；
- 完成至少一版中文证据稿和一版英文完整初稿；
- 在目标期刊确定后执行格式和篇幅适配。

### 3.3 默认不执行

- 不增加新的策略种子；
- 不重新选择更有利的来源模型或上下文；
- 不扩大资源恢复实验；
- 不训练机会成本网络；
- 不恢复 BPCE、MCH-PPO 或其变体；
- 不实现 GNN；
- 不为获得更漂亮的结果修改既有门槛；
- 不在查新完成前使用“首次”“首个”或等价优先权表述；
- 不把历史失败模型数量当作主要贡献证据。

## 4. 冻结的一句话论证

### 4.1 中文工作版本

> 在 AirDefense v1 冻结 factorized PPO 的动态掩码序列分配中，本研究使用
> 共同随机数成对反事实轨迹和逐时刻成本账本表明，同一步与未来动作替代会
> 系统性偏置回合累计资源成本对当前动作的局部信用读出；该机制可跨全新
> 策略种子复现，但其是否改变成本标签符号受场景和资源类型约束。

### 4.2 英文论证模板

```text
In dynamically masked sequential allocation with frozen factorized PPO policies
in AirDefense v1, paired counterfactual trajectories with common random numbers
and a step-wise cost ledger show that same-step and future action substitution
systematically biases episode-level cumulative resource cost as a readout of
local action credit; the mechanism replicates across previously unused policy
seeds, whereas cost-sign changes remain scenario- and resource-dependent.
```

该英文句是写作模板，不是最终摘要句。完成文献门控和全文后仍需校准术语、
时态和目标期刊风格。

## 5. Problem–Method–Insight 冻结

| 层次 | W1 冻结表述 |
| --- | --- |
| Problem | 动态掩码序列资源分配中，回合累计成本差同时包含当前直接消耗、同一步联合动作替代和未来策略替代，因而可能错误表示当前动作的局部资源信用。 |
| Method | 在冻结策略下使用状态快照、共同随机数 N/E 分支、合法目标精确边缘化、stochastic continuation 和逐时刻成本账本，分离直接成本与三类替代成本，并用全新策略种子和零重叠状态进行独立确认。 |
| Insight | 增加 rollout 只能降低采样误差，不能消除累计成本读出的结构性混叠；标签符号是否被改变取决于直接成本与动作替代的相对强度，因此测量有效性具有场景和资源类型边界。 |

全文中的方法、结果、讨论和贡献列表必须服务于该三层结构。若某一节无法回到
Problem、Method 或 Insight，应删除、移入补充材料或说明其必要性。

## 6. 论文贡献工作版本

在系统查新完成前，以下是“内容已受证据支持、优先权尚待核验”的工作版本：

1. 识别动态掩码序列资源分配中的一种局部信用测量混叠：当前动作会替代
   同一步自回归后缀单元和后续策略动作，使回合累计资源成本低估或掩盖
   当前直接消耗。
2. 建立冻结策略、共同随机数成对反事实轨迹和逐时刻成本账本相结合的审计
   协议，将回合成本差精确分解为当前直接成本与三类动作替代成本。
3. 使用 9 个新来源模型、108 个与旧正式数据零重叠的上下文和 7,776 条目标
   成本账本独立确认动作替代，同时确定成本符号失真的场景和资源类型边界。

以下内容不是独立贡献：

- factorized joint PPO 本身；
- 动态 action mask 本身；
- all-noop 现象本身；
- 失败的 BPCE/MCH-PPO 变体数量；
- 弹药恢复实验本身；
- 尚未实现的 GNN；
- “我们进行了大量实验”。

## 7. 可证伪主张与现有判定

| 编号 | 可证伪主张 | 支持条件 | 失败条件 | 当前判定 |
| --- | --- | --- | --- | --- |
| F1 | 回合累计成本差混合当前直接成本和动作替代 | 替代项可重构非正或被压低的累计成本差 | 替代项为零或不能解释差值 | 支持 |
| F2 | 完整联合动作替代可被精确分解 | 全部账本满足修正恒等式，误差低于门槛 | 存在未解释残差 | 支持，最大误差 `8.88e-16` |
| F3 | 动作替代可跨新策略种子复现 | 至少两个新种子块的下界为正 | 新种子结果主要由单个优势种子驱动 | 支持，3/3 新种子块下界为正 |
| F4 | 成本符号掩盖跨 missile/laser 普遍成立 | 两类均达到预注册掩盖上下文门槛 | 任一类型未达到门槛 | 否决，missile 为 2 个、laser 为 5 个 |
| F5 | 恢复弹药具有通用安全机会价值 | 跨场景、跨种子、跨资源类型稳定成立 | 证据集中在单一种子或资源类型 | 否决 |
| F6 | 当前路线已经形成稳定在线算法改进 | 独立训练门控稳定优于基线 | all-noop、高成本或跨种子分叉持续存在 | 否决 |

F4–F6 的否决结果必须作为边界保留，不能从论文中删除后再扩大正面主张。

## 8. 术语与公式冻结

### 8.1 规范术语

| 中文概念 | 英文规范名 | 符号/缩写 | 使用边界 |
| --- | --- | --- | --- |
| 防空资源智能分配 | intelligent air-defence resource allocation | — | 任务总称 |
| 动态合法动作掩码 | dynamic legal-action masking | — | 不简化成静态 action mask |
| 无冲突自回归联合动作 | conflict-free autoregressive joint action | — | 强调联合动作内部的执行顺序 |
| 因子化联合 PPO | factorized joint PPO | — | 来源策略，不作为新算法贡献 |
| 成对反事实轨迹 | paired counterfactual trajectories | — | N/E 分支及其随机性控制 |
| 共同随机数 | common random numbers | CRN | 环境与策略随机带配对 |
| 被测单元直接成本 | probe direct cost | `C_direct` | E 与 N 当前被测单元成本差 |
| 同一步其他单元替代 | same-step other-unit substitution | `Sub_cost_same` | 联合动作后缀单元在当前步的成本差 |
| 未来被测单元替代 | future probe substitution | `Sub_cost_future_probe` | 当前步之后被测单元成本差 |
| 未来其他单元替代 | future other-unit substitution | `Sub_cost_future_other` | 当前步之后其他单元成本差 |
| 总替代成本 | total substitution cost | `Sub_cost_total` | 三类替代成本之和 |
| 未来替代射击 | future substituted shots | `Sub_shot` | 只统计当前步之后，不含同一步后缀替代 |
| 回合累计成本差 | episode-level cumulative cost difference | `Delta_C_episode` | `total_cost(E)-total_cost(N)` |
| 替代比率 | substitution ratio | `rho_sub` | `Sub_cost_total/C_direct` |
| 成本符号掩盖 | cost-sign masking | — | `C_direct>0` 且 `Delta_C_episode<=0` |
| 可辨识性边界 | identifiability boundary | — | 指测量有效性，不等同算法性能边界 |

### 8.2 N/E 方向

全文第一次出现 N/E 时必须定义：

```text
Delta_C_episode
:= total_cost(E) - total_cost(N)
```

不得只写“counterfactual difference”而不说明差值方向。

### 8.3 修正后的完整成本账本

```text
Sub_cost_total
:= Sub_cost_same
 + Sub_cost_future_probe
 + Sub_cost_future_other
```

其中：

```text
Sub_cost_same
:= current_other_cost(N) - current_other_cost(E)
```

主恒等式：

```text
Delta_C_episode
= C_direct - Sub_cost_total
```

替代比率：

```text
rho_sub
:= Sub_cost_total / C_direct
```

符号掩盖：

```text
cost_sign_masked
:= (C_direct > 0)
   and (Delta_C_episode <= 0)
```

### 8.4 必须解释的统计口径

- `Sub_shot` 仍只统计未来射击，不含同一步后缀单元替代；
- `Sub_cost_total` 同时包含同一步和未来替代；
- R1 的 future-only 解释未被推翻，但只覆盖总替代成本的一部分；
- R2 的 time/resource 平均总替代成本为 `0.864`；
- 其中同一步其他单元替代为 `0.147`，未来替代为 `0.718`；
- 约 83% 的替代成本来自当前步之后，约 17% 来自联合动作内部同一步替代。

任何图、表、正文和补充材料不得混用 `Sub_shot` 与 `Sub_cost_total` 的统计范围。

## 9. 证据来源与优先级

出现数字、定义或解释冲突时，按以下顺序处理：

1. R2 正式实验报告及其冻结结果文件；
2. 第一创新 Claim–Evidence 矩阵；
3. R1 正式实验报告；
4. 标签语义和短视窗正式实验报告；
5. 项目路线图与阶段进度；
6. W0/W1 任务指导报告；
7. 临时笔记、终端输出和未归档草稿。

任务指导报告不是科学结果来源。正文中的关键数字必须回溯到正式报告或冻结
CSV/JSON 文件。

现有 [第一创新 Claim–Evidence 矩阵](../project/first_innovation_claim_evidence_matrix.md)
继续作为主张层的唯一权威矩阵。W1 不复制另一份可能发生漂移的主张矩阵，
而是新建“稿件可追溯矩阵”，增加以下字段：

| Claim ID | 正式证据文件 | 数据字段 | 主文段落 | Figure/Table | 允许动词 | 边界句 |
| --- | --- | --- | --- | --- | --- | --- |
| C1–C8 | 待逐项填入 | 待逐项填入 | 待写作后绑定 | 待绑定 | show/suggest 等 | 必填 |

## 10. W1-A：系统查新与稿件定位门控

### 10.1 检索主题

至少覆盖：

1. multi-agent credit assignment；
2. counterfactual baselines 与 difference rewards；
3. temporal credit assignment 和 delayed action effects；
4. common-random-number paired simulation；
5. action substitution、action displacement 或 policy-induced substitution；
6. sequential/autoregressive joint action allocation；
7. action masking 下的反事实评估；
8. resource shadow price、opportunity cost 和 constrained MARL；
9. episode-return label validity、measurement bias 和 identifiability。

### 10.2 检索记录要求

每组检索必须记录：

- 数据库；
- 检索日期；
- 完整检索式；
- 时间范围；
- 语言和文献类型限制；
- 初筛数量；
- 纳入与排除理由；
- 最接近工作的原始论文链接或标识；
- 该论文在 Problem、Method、Evidence、Insight 四层与本项目的重叠和差异。

只引用实际阅读过的原始来源。综述可用于导航，但不能替代对关键优先权论文的
核验。

### 10.3 最接近工作矩阵

至少形成以下表格：

| 文献 | Problem | Method | Evidence | Insight | 与本项目重叠 | 剩余差异 | 定位判定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 待检索 | — | — | — | — | — | — | — |

### 10.4 文献门控 L

完成检索后执行唯一门控：

| 判定 | 条件 | 后续路径 |
| --- | --- | --- |
| L1：可形成独立论文 | 相邻工作未同时覆盖当前 Problem、完整账本 Method 和条件性 Insight | 进入 W1-B 至 W1-G |
| L2：适合作为方法论文组成部分 | Problem 已知，但完整联合动作分解或独立边界证据仍有明确差异 | 收窄标题和贡献，作为较大论文的测量章节 |
| L3：适合作为学位论文章节/技术报告 | 相邻工作已基本覆盖 Problem–Method–Insight，仅环境证据不同 | 停止独立投稿包装，保留为严谨章节 |
| L4：定位仍不清楚 | 关键原始文献无法取得或差异判断依赖未核验假设 | 不进入摘要和投稿表述，继续查新 |

若出现 L2 或 L3，不自动启动新实验，也不恢复旧算法路线。是否重新定义独立
算法问题，应另立任务并重新预注册。

## 11. W1-B：论文架构与故事压缩

### 11.1 一句话、三句话和一段话

在正式撰写章节前，分别完成：

1. 一句话：只包含中心洞见和边界；
2. 三句话：依次写 Problem、Method、Insight；
3. 一段话：加入最强定量证据，但不堆叠全部历史结果。

三种压缩尺度必须表达同一主张，不能随篇幅扩大适用范围。

### 11.2 推荐章节结构

```text
1. Introduction
2. Related Work
3. Problem Formulation and Evaluation Framework
4. Paired Counterfactual Cost Decomposition
5. Experimental Protocol
6. Results
   6.1 Local resource-credit ambiguity in sequential allocation
   6.2 Action substitution under paired counterfactual continuations
   6.3 Exact same-step and future cost decomposition
   6.4 Independent replication across new policy seeds
   6.5 Scenario and resource-type boundaries
   6.6 Resource-restoration opportunity as a negative boundary
7. Discussion
8. Limitations
9. Conclusion
```

若最终目标期刊不设置独立 Related Work，则将其按技术主题合并到 Introduction；
若 Methods 后置，则只改变排版位置，不改变证据先于解释的写作顺序。

### 11.3 主文取舍

主文保留：

- 局部信用测量问题；
- N/E 配对反事实设计；
- 完整成本恒等式；
- R1 机制发现；
- R2 独立确认；
- 场景与资源类型边界；
- 资源恢复负结果作为边界。

优先移入补充材料：

- 全部历史模型的逐项流水账；
- 与最终论证无直接关系的算法变体实现细节；
- smoke 结果；
- 重复的超参数表；
- 首轮无效账本的完整逐行结果；
- 不改变核心判断的次级指标。

## 12. W1-C：Results 写作任务

Results 必须先于 Introduction 和 Abstract 完成。每一小节采用：

```text
问题/检验目的 → 协议 → 观察结果 → 定量证据 → 本节边界
```

### 12.1 Results 1：局部资源信用歧义

任务：

- 简述动态掩码、无冲突序列动作和回合累计成本读出；
- 说明在结构合法后仍存在 engagement 与资源信用异常；
- 将问题收窄为“累计成本是否能代表当前动作局部成本”；
- 不把全部 BPCE 失败史写成主结果。

### 12.2 Results 2：成对反事实下的动作替代

任务：

- 定义 N/E 身份和共同随机带；
- 报告 R1 `time_pressure/resource` 的机制发现；
- 说明当前 engage 如何改变后续被测单元和其他单元射击；
- 区分采样不确定性与估计对象的结构性混叠。

必须使用的 R1 边界：

- 旧策略种子仅用于机制发现；
- 独立性结论必须由 R2 支持；
- 未来替代结论不能覆盖同一步后缀替代。

### 12.3 Results 3：完整成本分解

必须报告：

- 首轮 future-only 公式在 `287/7776` 条账本出现非零残差；
- 最大原始残差为 `2.0`；
- 原因是无冲突自回归后缀单元可在同一步发生动作替代；
- 修正后总账本和 probe/other 子分解最大误差均为 `8.88e-16`；
- 相同模型、上下文、随机带和门槛仅完整重跑一次；
- 首轮无效结果已归档，没有依据结果重新选样。

写作上必须把该过程表述为测量定义的透明修正，而不是隐藏为实现细节，也不得
把意外发现包装成预注册假设。

### 12.4 Results 4：跨新策略种子独立确认

必须报告：

```text
来源模型：9/9
新上下文：108/108
旧 hash 重叠：0
上下文—重复记录：3,456
目标成本账本：7,776
Actor 最大参数差：0
软件回归：264 passed
```

`time_pressure/resource` 核心门控：

```text
mean(Sub_shot)>0：13/18
lower95(Sub_shot)>0：13/18
正块级下界种子：3/3
符号掩盖率不低于 50% 的种子：2/3
非正累计成本差可解释：7/7
```

允许结论：

> 动作替代机制跨此前未用于该标签设计的新策略种子复现。

禁止结论：

> 该机制已经对所有策略、所有环境或所有 MARL 算法普遍成立。

### 12.5 Results 5：场景和资源类型边界

资源类型结果：

| 类型 | 上下文 | `Sub_shot` | 95% 下界 | `rho_sub` | 掩盖上下文 |
| --- | ---: | ---: | ---: | ---: | ---: |
| missile | 9 | 0.373 | 0.133 | 0.571 | 2 |
| laser | 9 | 0.726 | 0.497 | 1.175 | 5 |

场景结果：

| 场景 | `Sub_shot` | `Sub_cost_total` | `rho_sub` | 符号掩盖率 |
| --- | ---: | ---: | ---: | ---: |
| medium | 0.544 | 0.949 | 0.747 | 0.620 |
| time pressure | 0.550 | 0.864 | 0.873 | 0.589 |
| heterogeneity pressure | 0.876 | 1.435 | 0.972 | 0.865 |

本节必须明确区分：

- missile 和 laser 均存在正动作替代；
- missile 未通过的是“稳定改变成本标签符号”的门槛；
- laser 较低的直接成本更容易被替代成本完全抵消；
- 异质场景替代最强不等于已经证明一般因果规律。

### 12.6 Results 6：资源恢复负边界

任务：

- 说明扩大可行动集合不等同于最终安全改善；
- 报告可靠机会证据集中于少量上下文、单一种子或 missile；
- 否决通用机会成本 oracle；
- 将该结果作为“不能直接从测量诊断跳到在线辅助算法”的边界。

不得把负结果改写成一个成功的机会成本方法。

## 13. W1-D：Methods 写作任务

Methods 以可复现为目标，不承担结果解释。

### 13.1 AirDefense v1 与任务形式化

- 状态、动作、资源、目标和保护区；
- missile/laser 的成本与交战约束；
- 动态合法目标和目标占用；
- episode 终止条件；
- 奖励和成本分量；
- 当前结论适用的场景与规模。

### 13.2 来源策略与联合动作

- factorized joint PPO；
- engagement/target 因子化；
- 动态合法动作掩码；
- 无冲突自回归后缀；
- strict joint PPO fallback；
- 来源模型训练、保留和不筛选原则。

不得把来源策略描述成 W1 提出的新算法。

### 13.3 成对反事实轨迹

- 状态快照；
- N/E 当前步身份；
- 环境 CRN 随机带；
- 策略 uniform tape；
- 合法目标精确边缘化；
- stochastic continuation；
- Actor 冻结和参数一致性检查。

### 13.4 标签语义与前置审计

- A/B/C 标签定义；
- target 边缘化；
- deterministic 与 stochastic continuation；
- 完整回合与短视窗；
- ENGAGE/STOP/AMBIGUOUS；
- 前置审计如何将问题收窄到成本测量。

该节只保留对最终研究问题必要的前置证据。

### 13.5 成本账本

- 定义全部当前和未来成本字段；
- 给出完整主恒等式与 probe/other 子分解；
- 说明 `Sub_shot` 和 `Sub_cost_total` 的范围差异；
- 定义 `rho_sub` 和 cost-sign masking；
- 说明数值容差和账本完整性门槛。

### 13.6 独立确认协议

- seeds 17/18/19；
- medium、time pressure、heterogeneity pressure；
- 9 个来源模型和 108 个新上下文；
- observation hash 零重叠；
- 每块 missile/laser 配额；
- 每上下文 32 次重复；
- P-C1/P-C2/P-C3；
- 所有来源模型无条件保留。

### 13.7 账本修正与研究完整性

主文 Methods 至少包含一段简述，补充材料完整记录：

- 原 future-only 公式；
- 发现残差的门控；
- 根因定位；
- 修正后的公式；
- 未修改的模型、样本、随机带和判定门槛；
- 唯一一次完整重跑；
- 无效结果归档位置。

## 14. W1-E：Figure 与 Table 故事板

### Figure 1：测量问题与反事实分支

建议面板：

- 动态合法动作与自回归联合动作；
- N/E 当前步差异；
- 同一步后缀替代；
- 未来 probe/other 替代；
- 为什么正直接成本可对应零值或负累计差。

唯一结论：

> Episode-level cumulative cost can be a biased readout of local action cost.

### Figure 2：审计协议与完整成本恒等式

建议面板：

- 状态快照和 CRN；
- 合法目标精确边缘化；
- stochastic continuation；
- 四项成本分解；
- 账本残差验证。

### Figure 3：R1 机制发现与 R2 独立确认

建议面板：

- 旧种子和新种子的 `Sub_shot`；
- 块级 95% 下界；
- 非正 `Delta_C_episode` 的替代解释；
- R1 与 R2 的职责区分。

### Figure 4：同一步与未来替代组成

建议面板：

- `C_direct`、`Sub_cost_same` 和未来替代；
- time/resource 平均替代组成；
- 约 17% 同一步与 83% 未来替代；
- `rho_sub` 与符号掩盖关系。

### Figure 5：场景与资源类型边界

建议面板：

- medium/time/heterogeneity 的 `rho_sub`；
- missile/laser 的 `Sub_shot` 与下界；
- 两类资源的掩盖上下文；
- 明确标记 P-C3 未通过。

### Table 1：任务、策略和反事实协议

### Table 2：R1/R2 独立性与完整性

### Table 3：P-C1/P-C2/P-C3 门控结果

### Table 4：场景和资源类型边界

### Supplementary Table：前置标签审计与资源恢复负结果

每张主图只服务一个结论。若图中不同面板需要不同范围的主张，应拆图或在图注
中逐面板限制。

## 15. W1-F：Discussion、Introduction、Conclusion 与 Abstract

### 15.1 Discussion

推荐顺序：

1. 中心发现及其测量意义；
2. 同一步和未来替代为何产生结构性混叠；
3. 为什么更多 rollout 不能修复估计对象；
4. 与反事实信用、difference reward 和序列决策文献的关系；
5. 场景、资源类型、环境和冻结策略边界；
6. 对未来在线信用方法的设计约束。

必须讨论的竞争解释：

- 结果是否只是随机 rollout 方差；
- 是否只是 all-noop 策略的副作用；
- 是否只是 laser 直接成本较低；
- 是否由目标冲突处理或单元执行顺序造成；
- 是否依赖 AirDefense v1 的成本定义；
- CRN 是否引入不现实的耦合。

### 15.2 Introduction

采用“技术瓶颈”漏斗：

```text
动态资源分配需要局部信用
→ 回合累计回报常被用作反事实读出
→ 序列联合动作会改变同一步和未来动作
→ 现有读出可能混合直接成本和动作替代
→ 本研究进行可审计的成本分解与独立边界确认
```

Introduction 末段只写经文献门控后保留的贡献，不报告全部门控数字。

### 15.3 Conclusion

按以下顺序：

```text
贡献 → 决定性证据 → 测量意义 → 场景/资源边界
```

不得引入新实验、新机制或“将解决在线 PPO”的承诺。

### 15.4 Title

Results、文献定位和贡献列表稳定后，再生成 3–5 个候选标题。标题必须：

- 包含可搜索的对象和问题；
- 体现 cost measurement、action substitution 或 counterfactual credit；
- 不含 `universal`、`general`、`solves` 或未经核验的 `first`；
- 若独立论文定位不足，应避免把环境级证据包装为领域通用原理。

### 15.5 Abstract

最后撰写，使用：

```text
背景/问题 → 测量缺口 → 审计方法 → 最强结果 → 独立确认 → 条件边界
```

摘要至少包含一个决定性定量事实和一个明确边界，不得只写方法流程。

## 16. 中英文写作规则

### 16.1 中文证据稿

先将每条材料拆成：

```text
主张 / 证据 / 条件 / 比较 / 含义 / 局限
```

中文稿的作用是冻结科学意图，不追求逐句翻译成英文。

### 16.2 英文稿

- 先按段落功能重写，不按中文语序直译；
- Results 主要报告观察，不混入机制推测；
- Discussion 才解释可能原因和推广意义；
- `show`、`demonstrate` 只用于直接证据；
- `suggest`、`indicate` 用于间接或趋势性解释；
- 不使用无范围的 `robust`、`generalizable`、`comprehensive`；
- 每个场景和资源类型结论必须带适用条件；
- `action substitution` 的最终英文术语需由文献检索确认是否与既有术语冲突。

## 17. W1-G：完整稿整合与审稿压力测试

完成分节草稿后，必须先合并中文证据稿并检查全篇论证一致性，再按段落功能
重写英文完整初稿。不得在分节主张尚未对齐时直接逐段翻译。

整合后执行以下检查：

- 一句话论证、三项贡献、摘要和结论是否表达同一范围；
- 每个 Results 小节是否能回溯到 Claim ID 和正式数据；
- Introduction 提出的问题是否由 Results 实际回答；
- Discussion 是否解释证据而不是重复图表；
- Limitations 是否包含环境、策略、场景、资源类型和成本定义边界；
- 补充材料是否承接必要细节，而不是隐藏不利结果。

| 风险 | 可能质疑 | W1 处理 |
| --- | --- | --- |
| 贡献只是记账恒等式 | 公式显然，科学新意在哪里？ | 将贡献放在“序列动作替代足以改变局部信用读出”及独立边界证据，而不是代数形式本身。 |
| 负算法结果重新包装 | 是否只是 BPCE 失败后的故事转换？ | 明确记录创新演化；失败只负责暴露问题，最终贡献由独立 R1/R2 证据支持。 |
| 查新不足 | difference reward 或 temporal credit 是否已覆盖？ | 文献门控必须比较 Problem、Method、Evidence、Insight，不依赖关键词不同。 |
| 账本事后修正 | 是否看到结果后修改指标？ | 披露 287/7776 残差、根因、未变样本和门槛、无效结果归档及唯一一次重跑。 |
| 同环境过拟合 | 是否仅是 AirDefense v1 特例？ | 主张限定 AirDefense v1；把三场景和新种子写成环境内独立性，不冒充跨环境泛化。 |
| 资源类型外推 | missile/laser 是否都发生符号掩盖？ | 区分“替代为正”和“足以改变符号”；明确 P-C3 未通过。 |
| 种子独立性 | seeds 17/18/19 是否真的未被使用？ | 披露其历史用途与未参与动作替代标签设计/来源模型选择的事实，并说明 9/9 无条件保留。 |
| 缺少算法提升 | 没有新策略为何值得发表？ | 将稿件定位为测量有效性和诊断协议；由文献门控决定独立论文还是较大论文组成部分。 |
| 历史链条过长 | 是否像项目日志而不是论文？ | 主文只保留完成因果收敛所需结果，其余放入补充材料。 |
| 共同随机数合理性 | CRN 是否人为制造相关性？ | 说明 CRN 用于方差控制和成对比较，不改变分支边缘过程；完整记录随机带。 |
| 成本方向歧义 | E−N、直接成本与替代成本是否一致？ | 术语账本冻结方向，所有图表从同一字段生成并做恒等式检查。 |

## 18. 新实验触发规则

W1 默认零新实验。写作过程中发现问题时，先分类：

| 问题类型 | 处理 |
| --- | --- |
| 缺少已存在数字 | 回查正式结果文件，不重跑 |
| 图表需要已有数据重聚合 | 允许只读分析或确定性重绘，记录脚本和输入 |
| 术语或相关工作不清楚 | 继续文献检索 |
| 主张超过现有证据 | 收窄或删除主张 |
| 账本数字互相冲突 | 暂停写作，执行数据完整性审计 |
| 审稿定位需要跨环境证明 | 记录为新研究问题，不在 W1 自动扩展 |
| 必须形成算法创新 | 另立任务、重新定义问题和门控，不恢复失败路线 |

只有发现以下致命完整性问题，W1 才暂停并转入独立诊断：

- 正式报告数字不能从冻结结果重现；
- N/E 身份或成本方向在数据层不一致；
- 修正恒等式不能在冻结账本上成立；
- 新旧上下文实际发生重叠；
- Actor 冻结检查与正式报告冲突。

## 19. 交付目录与文件

W1 启动后创建：

```text
docs/manuscript/action_substitution_cost_identifiability/
```

### 19.1 P0 交付物

```text
terminology_ledger.md
literature_search_protocol.md
literature_evidence_matrix.md
manuscript_traceability_matrix.md
paper_positioning_decision.md
manuscript_outline.md
figure_table_plan.md
results_draft_zh.md
methods_draft_zh.md
```

### 19.2 P1 交付物

```text
discussion_draft_zh.md
introduction_draft_zh.md
limitations_draft_zh.md
conclusion_draft_zh.md
manuscript_draft_zh.md
manuscript_draft_en.md
```

### 19.3 P2 交付物

```text
supplementary_methods.md
supplementary_results.md
reviewer_pressure_test.md
submission_checklist.md
```

图表源文件和导出文件应放在该目录下的 `figures/`，并保持脚本、输入数据和
导出图之间可追溯。

## 20. 阶段门控

### G0：来源冻结

通过条件：

- 权威证据文件列表完整；
- 每个关键数字有唯一来源；
- W0 占位文本不进入正式稿；
- 现有 Claim–Evidence 矩阵保持唯一主张源。

### G1：术语与公式

通过条件：

- N/E 方向唯一；
- 完整三类替代成本进入公式；
- `Sub_shot` 与 `Sub_cost_total` 范围不混用；
- 全部主图和正文使用同一符号。

### G2：文献与定位

通过条件：

- 检索过程可复核；
- 最接近工作完成四层比较；
- 不存在未经核验的优先权主张；
- 已明确选择 L1、L2、L3 或 L4。

L4 不得进入最终摘要和投稿稿。

### G3：Results 与 Methods

通过条件：

- 每个 Results 小节有可追溯证据；
- Methods 足以复现 N/E、CRN 和成本账本；
- 首轮公式修正透明披露；
- 观察与解释分开；
- 负结果和失败门控未被隐藏。

### G4：图表

通过条件：

- 每张主图只有一个主要信息；
- 所有数字能回溯到冻结结果；
- 图注写明场景、资源类型、种子和统计单位；
- P-C3 失败在边界图中可见。

### G5：中文证据稿

通过条件：

- 全文章节完成；
- 每段只有一个功能；
- 核心主张、贡献和边界一致；
- 不存在项目流水账式章节；
- 不包含未经文献支持的优先权表述。

### G6：英文完整初稿

通过条件：

- 英文按论证重写而非逐句翻译；
- 摘要最后完成；
- 术语和符号跨章节一致；
- Results、Discussion 的语气与证据等级匹配；
- 标题不超过证据范围。

### G7：投稿前审计

通过条件：

- 每项贡献均有正文、图表和数据来源；
- 每个强动词都能回溯到直接证据；
- 局限性包含环境、策略、资源类型和测量定义；
- 目标期刊格式、篇幅和材料要求已经适配；
- 无“算法已成功”“跨资源通用”等越界表述。

## 21. 推荐执行顺序

```text
1. 创建 manuscript 工作目录
2. 冻结证据来源、术语和公式
3. 执行系统文献检索
4. 完成最接近工作矩阵
5. 通过文献与稿件定位门控 L
6. 完成一句话/三句话/一段话故事压缩
7. 建立稿件可追溯矩阵
8. 完成 Figure/Table 故事板
9. 撰写 Results 中文证据稿
10. 撰写 Methods 中文证据稿
11. 完成 Discussion 与 Limitations
12. 完成 Introduction 与 Related Work
13. 完成 Conclusion
14. 生成并选择标题
15. 最后撰写 Abstract
16. 合并中文完整稿
17. 按段落功能重写英文完整稿
18. 执行审稿压力测试
19. 确定期刊并完成格式适配
20. 执行投稿前主张—证据审计
```

步骤 3–5 是防止项目进入写作死路的首要门控。未完成该门控，不应投入大量
时间润色 Introduction、Title 或 Abstract。

## 22. W1 完成定义

W1 完成必须同时满足：

1. 系统查新能够支持当前稿件定位；
2. 第一创新主张未超过 R2 冻结边界；
3. 成本公式已经从 future-only 更新为完整联合动作替代分解；
4. 每项主张均映射到正式结果、图表和正文位置；
5. Results 与 Methods 先于 Introduction 和 Abstract 完成；
6. 首轮账本修正过程得到透明披露；
7. P-C3、机会成本和在线算法负结果均被保留；
8. 完成中文证据稿和英文完整初稿；
9. 完成一次独立的审稿压力测试；
10. 已根据目标期刊完成最后的结构和格式调整。

W1 完成不等于：

- 已经证明跨环境普适性；
- 已经形成新 PPO 算法；
- 已经解决 all-noop；
- 已经证明 GNN 有效；
- 已经可以删除失败和条件边界；
- 已经允许在没有查新依据时使用“首次”。

## 23. 下一阶段出口

W1 结束后只允许进入以下一种路径：

| 出口 | 条件 | 下一阶段 |
| --- | --- | --- |
| M1：独立论文投稿准备 | L1 且 G0–G7 全部通过 | 期刊适配、图表定稿和投稿材料 |
| M2：较大论文组成部分 | L2 且正文证据链完整 | 与新的独立算法问题或更大研究问题整合 |
| M3：学位论文章节/技术报告 | L3 | 保留严谨贡献，不继续包装独立优先权 |
| M4：写作暂停 | L4 或发现致命完整性冲突 | 完成缺失文献或独立数据诊断 |

不得因为担心成果“不像算法论文”而自动回到已经失败的 BPCE/MCH-PPO 路线。
若后续确实需要算法创新，应基于 W1 已识别的测量边界重新定义问题，而不是
继续在不可辨识标签上叠加模型复杂度。

## 24. 最终阶段判定

当前下一项工作已经不是继续寻找新的正实验结果，而是：

> **先通过系统查新确认贡献位置，再把已独立确认的动作替代测量失真、完整
> 成本分解和场景/资源类型边界组织成可追溯、可复核且不过度外推的论文。**

该路径同时保留两种诚实结果：

1. 文献支持其成为独立论文时，完成投稿级稿件；
2. 文献表明独立性不足时，及时收窄为方法组成部分或学位论文章节，避免在
   错误定位上继续投入。
