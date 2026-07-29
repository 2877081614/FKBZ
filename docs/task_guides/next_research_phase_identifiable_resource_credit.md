# 下一研究阶段：可辨识资源信用的新算法问题定义与预注册

更新时间：2026-07-28  
任务状态：可立即启动  
任务编号：N1  
任务优先级：P0  
任务性质：主线重立项、系统查新、候选机制筛选、离线语义验收与在线实验预注册  
默认实验权限：只允许只读复核和离线开发性诊断；未通过本任务出口门控前，不启动新的在线算法正式实验

## 1. 决策摘要

W1 已完成主张—证据冻结、双语整稿和对抗性审稿，阶段出口为 L2/M2。当前
动作替代成果应作为较大信用分配方法论文中的测量、诊断与资源信用分解模块，
而不是继续包装为独立在线算法。

历史 BPCE/MCH-PPO 路线已经给出明确负边界：

- 独立 engagement/target ratio 与 clipping 不能严格退化为已验证的 joint PPO；
- BPCE-PPO v0 虽满足 joint PPO fallback，但 10k 实验仍出现 all-noop 和高成本分叉；
- deterministic continuation、随机后续和短视窗标签都没有形成跨场景、跨种子的
  稳定双向 engagement 监督；
- 回合累计资源成本差混合当前直接消耗、同一步后缀动作替代和未来策略替代；
- R2 已确认动作替代跨新策略种子存在，但符号掩盖强度受场景和资源类型约束；
- 弹药恢复没有形成通用安全机会价值，不能恢复为机会成本 oracle。

因此，下一项主线任务不是继续修改 BPCE/MCH-PPO，也不是直接实现 GNN，而是：

> **重新定义一个以“局部资源信用可辨识性”为中心的新算法问题，比较能够保留
> 成本分量语义的候选机制，完成系统查新、离线证伪和正式在线实验预注册。**

本任务结束时必须做出“选择一个候选、停止当前候选或转向其他创新问题”的明确
决策，不能以“继续调参”作为出口。

## 2. 任务入口

### 2.1 权威状态材料

- [学术项目进度](../project/academic_project_progress.md)
- [研究创新路线图](../project/research_innovation_roadmap.md)
- [第一创新 Claim–Evidence 矩阵](../project/first_innovation_claim_evidence_matrix.md)
- [R2 动作替代独立确认](../experiments/air_defense_v1_action_substitution_confirmation.md)
- [R1 动作替代与弹药机会成本审计](../experiments/air_defense_v1_action_substitution_opportunity_cost_audit.md)
- [BPCE 标签语义审计](../experiments/air_defense_v1_bpce_label_semantics_audit.md)
- [BPCE 短视窗双分量审计](../experiments/air_defense_v1_bpce_short_horizon_label_audit.md)
- [BPCE-PPO 设计与失败边界](../algorithms/boundary_probed_counterfactual_engagement_ppo.md)
- [MCH-PPO 公式与失败边界](../algorithms/masked_counterfactual_hierarchical_ppo.md)

### 2.2 可复用软件入口

```text
rein_learning/envs/air_defense_v1/config.py
rein_learning/envs/air_defense_v1/centralized_env.py
rein_learning/envs/air_defense_v1/scenarios.py
rein_learning/envs/air_defense_v1/wrappers/conflict_free_joint_action.py

rein_learning/trainers/air_defense_v1_ppo.py
rein_learning/experiments/air_defense_v1_benchmark.py

rein_learning/common/action_substitution_confirmation.py
scripts/run_air_defense_v1_action_substitution_confirmation.py
tests/test_action_substitution_confirmation.py
```

上述代码是开发基础，不代表其中任一历史算法重新成为正式候选。

### 2.3 历史文档使用规则

`academic_project_progress.md` 和 `research_innovation_roadmap.md` 采用阶段追加方式
记录。较早章节中的“下一步”“Immediate Next Task”只代表当时决策，N1 必须以
2026-07-28 的 W1/T10 出口和本任务文档为准。

## 3. 主线创新组合

当前后续研究保留两个尚未完成的算法方向，但必须顺序进入。

### 3.1 主线算法创新 A：可辨识的资源约束信用

工作定义：

> 在动态掩码、无冲突、自回归联合分配中，避免把受动作替代影响的回合累计成本
> 差直接当作当前动作的 STOP/ENGAGE 标签；保留安全收益、当前直接成本和动作
> 替代分量的语义，形成能够安全接入 joint PPO 的局部信用或显式约束机制。

N1 只负责把该方向收敛为一个通过查新和离线语义门控的正式候选，并冻结后续
在线实验。N1 不预先宣称候选必然构成创新。

### 3.2 结构泛化创新 B：类型化二部图反事实 Critic

保留定义：

> 面向变规模防空资源分配，用类型化资源—目标—保护区关系图一次估计多个动态
> 合法反事实动作价值，并验证跨资源数、目标数和编组结构的泛化。

创新 B 不是普通 `PPO + GNN`。只有出现以下至少一种证据时才允许另立任务：

1. 创新 A 的非图信用语义和在线更新已经稳定，但多个合法反事实的逐项估值成本
   成为主要瓶颈；
2. 非图模型在冻结信用目标上通过同分布门控，却在资源数、目标数或编组结构变化
   时系统失败；
3. 关系类型、动态合法边或前缀目标占用造成可复现的表示瓶颈，并能与优化器失稳、
   标签不可辨识和数据不足区分。

在此之前，GNN、GAT、Transformer 和变规模正式训练继续冻结。

## 4. N1 的 Problem–Method–Insight 工作版本

以下表述用于立项和证伪，不是论文最终贡献。

| 层次 | N1 工作版本 |
| --- | --- |
| Problem | 在动态掩码序列资源分配中，回合累计资源成本差是合法的全局结果，但不是可直接辨识的当前动作局部信用，因为它同时包含当前直接消耗、同一步其他单元替代和未来策略替代。将该标量转成 ENGAGE/STOP 监督会产生场景、资源类型和策略依赖的符号混叠。 |
| Method | 比较“分量保持的资源信用”“显式预算/约束价值”和“受控 continuation 的差异回报”三类候选；所有候选必须区分安全收益、当前直接成本和替代分量，保持标准 factorized joint PPO 为安全主干，并在正式在线训练前通过标签语义、支持覆盖、数值退化和独立数据协议。 |
| Insight | 增加反事实重复数只能提高差值精度，不能消除局部信用的结构性混叠；可训练算法必须在决策聚合前保留成本分量或改用显式全局约束，不能先把受替代影响的累计成本压成二值局部标签。 |

### 4.1 一句话版本

> N1 研究如何在动态掩码序列分配中保留直接成本与动作替代的不同语义，使资源
> 约束信用可被安全用于 joint PPO，而不是继续提高一个结构性混叠标签的精度。

### 4.2 三句话版本

1. **Problem**：累计资源成本差会被同一步和未来动作替代抵消，不能稳定代表当前
   动作的局部资源信用。
2. **Method**：比较分量保持、显式约束和受控差异回报候选，并要求每个候选满足
   可辨识标签、独立支持和 joint PPO 严格 fallback。
3. **Insight**：资源信用的主要问题是测量语义而不是网络容量，算法应先修复信用
   接口，再讨论图表示和跨规模泛化。

## 5. 冻结的成本分解与符号

对状态 `s_t`、已生成的自回归前缀 `p_{t,i}` 和被测单元 `i`，定义：

- `N`：被测单元选择 no-op；
- `E`：被测单元交战，合法目标按冻结策略条件概率边缘化；
- `D_direct`：E 分支中被测单元当前动作的直接资源成本，方向为正；
- `S_same_other`：E 在同一步替代后缀其他单元动作所减少的成本；
- `S_future_probe`：E 替代被测单元未来动作所减少的成本；
- `S_future_other`：E 替代其他单元未来动作所减少的成本。

冻结恒等式：

```text
S_total
= S_same_other
  + S_future_probe
  + S_future_other
```

```text
Delta_C_episode
= C_episode(E) - C_episode(N)
= D_direct - S_total
```

其中 `Delta_C_episode <= 0` 不等价于“当前交战没有成本”，也不能自动生成 STOP
或 ENGAGE 标签。N1 中任何候选若改变上述方向、分支语义或目标边缘化规则，必须
建立新任务，不能继续使用 R1/R2 证据作为直接支持。

## 6. 候选机制空间

N1 必须比较候选，不得先选模型名称再寻找问题。

### 6.1 候选 A：分量保持的约束信用

核心思路：

- 安全收益、直接成本和三类替代成本保持为向量，不在标签生成阶段压成单一符号；
- `D_direct` 使用环境可观测的确定值，不重复用神经网络拟合已知量；
- 对不可在决策时直接获得的安全收益和替代分量分别估值；
- 决策层使用冻结预算、对偶变量或明确效用函数聚合；
- 对 Actor 的辅助或约束关闭时，更新必须严格退化为 factorized joint PPO。

暂定工作名可使用：

```text
Substitution-Decomposed Resource Credit, SDRC
```

该名称只用于工程沟通。完成系统查新前不得写入论文标题或宣称为新算法。

### 6.2 候选 B：显式全局资源预算与约束价值

核心思路：

- 不把回合成本差转换为局部二值标签；
- 将资源消耗保留为 episode/CMDP 级约束；
- 分别估计安全价值和约束成本价值；
- 使用预注册预算或对偶更新控制资源消耗；
- 局部直接成本只作为已知即时成本和诊断量。

该候选是必要对照。若它在更少机制和相同预算下达到相同稳定性，候选 A 的额外
分解不能自动构成贡献。

### 6.3 候选 C：受控 continuation 的差异回报

核心思路：

- 通过固定或匹配 continuation 减少策略替代对局部差值的影响；
- 明确区分“评估真实策略后果”和“估计当前动作直接贡献”；
- 检查受控 continuation 是否引入不可达轨迹、支持外动作或新的偏差。

该候选风险最高。若 continuation 不对应可执行策略分布，或其标签只能依赖人为
固定后缀，则不得作为主要在线信用机制。

### 6.4 暂不进入的候选

- 恢复 BPCE 的边界半径、重复数、类别平衡或辅助剂量搜索；
- 恢复 MCH/RG-MCH/SA-RG-MCH 的独立 ratio/clipping；
- 重新训练机会成本 oracle；
- 用更大 MLP、GNN 或 Transformer 直接拟合原累计成本标签；
- 仅通过修改奖励权重消除 all-noop；
- 挑选没有塌缩的种子形成正面结果。

## 7. 候选选择标准

每个候选按以下六层比较，缺一不可：

| 维度 | 必答问题 | 最低要求 |
| --- | --- | --- |
| Problem | 它解决的是局部信用不可辨识，还是只改变优化器？ | 必须直接对应冻结测量边界 |
| Method | 新机制中真正不可替代的干预是什么？ | 核心机制不超过一个，其他均为对照或保障 |
| Label | 每个训练目标是否可观测、可重构或可独立验证？ | 禁止把混叠标量直接二值化 |
| Fallback | 信号失效时是否回到已验证 joint PPO？ | loss、梯度和一次更新数值等价 |
| Evidence | 什么结果会支持或否决机制？ | 必须有独立数据、消融和停止规则 |
| Insight | 即使最终增益有限，能留下什么可复用认识？ | 不能只写“平均奖励更高” |

选择优先级：

1. 标签语义和可辨识性；
2. 与冻结问题的因果对应；
3. 严格安全退化；
4. 跨种子、跨场景可验证性；
5. 计算成本；
6. 网络容量和最终平均奖励。

## 8. 系统查新任务

N1 必须在冻结正式候选前完成定向查新。当前已知入口包括 COMA、difference
reward、H-PPO、HAPPO/HATRPO、CAPO、约束/CMDP 强化学习、资源影子价格和
GNN-WTA，但不得把现有列表视为完整查新。

### 8.1 检索问题

至少回答：

1. 是否已有工作区分当前直接成本与由策略响应产生的未来替代成本？
2. difference reward、counterfactual baseline 和 marginal contribution 如何处理
   continuation 政策改变？
3. CMDP/拉格朗日 Actor-Critic 如何把 episode 级资源约束分配到结构化离散动作？
4. 顺序/自回归联合动作中，前缀占用与后缀动作替代是否已有正式处理？
5. 是否已有方法在动态合法动作集内批量估计 engage/no-op 与 conditional target
   反事实？
6. 现有图 WTA 工作解决的是表示与泛化，还是已经包含相同的资源信用分解？

### 8.2 文献差异矩阵

每篇最接近工作至少比较：

| 层次 | 比较内容 |
| --- | --- |
| Problem | 是否研究局部信用被动作替代混叠 |
| Method | 是否显式分离直接成本、同一步替代和未来替代 |
| Action structure | 是否包含动态掩码、无冲突前缀和条件目标 |
| Constraint | 资源成本是奖励项、全局约束还是局部标签 |
| Evidence | 是否有跨种子、跨场景和独立数据门控 |
| Insight | 是否得到“精度不能修复测量语义”的同类结论 |

输出只能判定为：

```text
already covered
adjacent but technically distinct
promising but requires experiment
insufficient evidence
```

在查新完成前禁止使用“首次”“首个”“填补空白”和等价表述。

## 9. 可证伪命题

### 9.1 N1 阶段命题

| 编号 | 命题 | 支持条件 | 否决条件 | 所需证据 |
| --- | --- | --- | --- | --- |
| N1-P1 | 新候选使用的每个资源信用分量具有明确、可重构的语义 | 标签可由冻结账本或环境定义独立核对，方向与恒等式一致 | 仍依赖无法区分直接成本和替代成本的累计标量 | 标签数据字典、恒等式测试和人工构造轨迹 |
| N1-P2 | 候选机制在问题层和洞见层区别于现有反事实信用或约束 RL | 至少与 3–5 篇最接近工作形成可辩护差异 | 差异仅为换网络、换任务或组合已知模块 | 系统查新记录和五层差异矩阵 |
| N1-P3 | 候选关闭新增信号时严格恢复 factorized joint PPO | loss、梯度、采样动作和一次参数更新在容差内等价 | gate/系数为零时仍存在不同 ratio、clip 或参数更新 | 确定性回归测试 |
| N1-P4 | 现有数据足以完成开发性语义诊断，但不足以充当新算法独立确认 | R1/R2只用于机制设计和兼容性复核，正式协议另留新策略种子与状态 | 把已经查看过的 R2 数据重新称为独立 test | 数据用途表和 seed/hash 审计 |
| N1-P5 | 能在正式训练前写出明确的在线支持与停止门槛 | 指标、非劣界、种子、场景、预算和失败出口全部冻结 | 需要看到在线结果后才能决定成功定义 | 机器可读预注册配置和任务评审 |

N1-P1 至 N1-P5 必须全部通过，才允许创建在线实现任务。

### 9.2 后续在线阶段待预注册命题

N1 只负责冻结，不执行以下命题：

| 编号 | 待验证命题 | 必要对照 |
| --- | --- | --- |
| ON-P1 | 新机制减少跨种子的 all-noop/高成本两极分叉 | factorized joint PPO、等预算约束基线 |
| ON-P2 | 安全改善不是简单增加交战率 | 仅正标签/无预算或等交战率对照 |
| ON-P3 | 资源成本受控且不以高威胁突防恶化为代价 | 安全—成本 Pareto、逐场景结果 |
| ON-P4 | 分量保持本身有用，而不是增加参数或仿真预算有用 | 等参数、等采样、分量合并消融 |
| ON-P5 | 机制结论可在新的独立策略种子和状态中复现 | 预注册新种子、零 hash 重叠 |

## 10. 数据使用与独立性

### 10.1 现有数据的允许用途

R1、R2、BPCE 和 Task14 数据均已被查看，允许用于：

- 公式和方向核对；
- 失败模式复盘；
- 开发性特征分布检查；
- 软件回归和人工构造样例；
- 估计正式实验所需样本量和计算预算。

它们不得用于：

- 声称新候选已经获得独立验证；
- 同时选择模型、阈值并报告 test；
- 根据优势种子选择正式来源模型；
- 重新命名旧结果形成新正面主张。

### 10.2 新数据协议

若 N1 通过并进入在线实现任务，必须：

1. 先生成全项目 seed 使用审计；
2. 在查看新候选结果前选择连续且未用于候选设计的策略种子；
3. 无条件保留所有预注册种子，不按 all-noop、奖励或资源行为替换；
4. 冻结训练、校准、机制 test 和最终独立确认的用途；
5. 记录 observation/context hash，避免开发数据与确认数据重叠；
6. 将重复随机 rollout 视为同一上下文内重复，不伪装成独立样本量；
7. 将场景、资源类型和策略种子作为分块统计单位。

## 11. N1 执行阶段

### N1-01：失败边界与术语冻结

目标：

- 把 R1/R2、BPCE、MCH/RG/SA-RG 和 Task14 的正负证据整理为一张问题地图；
- 冻结 `direct cost`、`same-step substitution`、`future substitution`、
  `local resource credit`、`global episode constraint` 等术语；
- 明确哪些旧结论可以进入新算法动机，哪些只能作为失败边界。

验收：

- 每项候选训练信号都能追溯到环境量、账本量或明确定义的估计量；
- 不出现把“回合全局成本有效”与“局部信用有效”混为一谈的表述；
- 不恢复已否决的机会成本和二值标签主张。

### N1-02：系统查新与差异定位

目标：

- 完成检索协议、检索日志、纳入排除记录和最接近工作矩阵；
- 对三个候选分别给出 `already covered / adjacent / promising / insufficient`
  判定；
- 选择最值得进入离线证伪的一个主候选和一个必要对照。

验收：

- 至少覆盖反事实信用、difference reward、约束 RL、顺序动作和图 WTA 五类边界；
- 至少 3–5 篇最接近工作完成 Problem–Method–Detail–Evidence–Insight 比较；
- 主候选的差异不只来自防空场景迁移、模块堆叠或指标提升。

### N1-03：候选公式、接口与人工轨迹验收

目标：

- 冻结候选输入、输出、损失、约束、fallback 和计算复杂度；
- 构造能够分别出现零替代、同一步替代、未来替代和完全符号掩盖的最小轨迹；
- 检查候选在各类轨迹上的方向是否符合定义。

验收：

- 成本恒等式误差不超过 `1e-6`；
- 已知 `D_direct` 不由模型重复估计；
- 所有二值或排序标签都能说明为何不会重现旧混叠；
- 新信号关闭时的 loss、梯度和一次更新与基线在预注册容差内一致。

### N1-04：离线开发性证伪

目标：

- 使用现有数据检验候选分量的支持覆盖、可预测性、跨场景方向和计算成本；
- 与累计成本标量、全局约束基线和等容量模型比较；
- 判断失败来自标签语义、数据支持、模型容量还是跨批次漂移。

规则：

- 本阶段结果只能用于选择或否决候选，不能写成独立确认；
- 不得通过增加网络深度、GNN、更多随机重复或挑选种子掩盖标签失败；
- 如果核心分量无法在非图模型上形成可靠支持，应先否决或修改信用语义，
  不能自动触发 GNN。

### N1-05：在线实验预注册与出口评审

目标：

- 冻结正式候选、必要对照、消融、场景、种子、训练预算和统计单位；
- 为安全、资源成本、all-noop、wasteful-engage、计算成本和跨种子稳定性设置
  数值门槛；
- 建立 smoke、10k、30k 和 100k 的条件进入关系；
- 形成机器可读配置，确保结果出现后不能改变主门槛。

最低预算结构：

```text
软件与语义 smoke
        只验证运行、方向、fallback和成本
                    ↓ 全部门控通过
10k × 3 seeds × 至少2个冻结压力场景
        机制证伪，不选择优势种子
                    ↓ 全部门控通过
30k × 5 seeds 消融
                    ↓ 全部门控通过
100k正式实验 + 第二结构化分配任务
```

N1 不执行上述在线训练，只冻结下一任务可直接使用的协议。

## 12. 在线门控必须包含的指标

具体数值在 N1-05 根据冻结基线分布一次性确定，但指标集合不得删减：

### 12.1 策略稳定性

- `all_noop_episode_rate`
- `engagement_rate`
- `actionable_engagement_rate`
- deterministic/stochastic engagement gap
- 跨种子方差和最差种子

### 12.2 安全效果

- `high_threat_leak_rate`
- `leak_rate`
- `avg_total_damage`
- `avg_zone_weighted_damage`
- `intercept_rate`

### 12.3 资源效果

- `avg_resource_cost`
- `avg_ammo_used`
- `avg_shots`
- `damage_reduction_per_ammo`
- `wasteful_engage`

### 12.4 机制与计算

- 各信用分量覆盖率和可靠率；
- Actor 主梯度与新增梯度的夹角/范数比；
- gate 激活率；
- fallback 数值误差；
- 训练时间和额外 transition 比；
- 单次决策时间。

正式门控必须同时约束安全和资源，不能只要求平均奖励改善。

## 13. 必要对照与消融

| 类型 | 对照/消融 | 回答的问题 |
| --- | --- | --- |
| 安全主干 | factorized joint PPO | 新机制是否优于同一动作结构下的稳定主干 |
| 简单约束 | 全局资源预算/对偶基线 | 分量分解是否比更简单的 CMDP 约束有必要 |
| 标签对照 | 原累计成本标量 | 改善是否确实来自避免混叠 |
| 容量对照 | 等参数非分解模型 | 改善是否只是参数更多 |
| 预算对照 | 等 transition/等运行时间 | 改善是否只是额外仿真 |
| 分量消融 | 合并或移除替代分量 | 哪一类分量承担机制作用 |
| fallback | 新信号完全关闭 | 是否严格恢复 joint PPO |
| 行为对照 | 等交战率策略 | 安全改善是否只是多开火 |

历史 BPCE/MCH/RG/SA-RG 可作为失败参照，不作为需要重新训练和调优的主要基线。

## 14. 预注册决策规则

### 14.1 通过

只有 N1-P1 至 N1-P5 全部通过，才：

1. 冻结一个主候选和一个简单必要对照；
2. 创建新的在线算法实现任务；
3. 允许新增算法文件、训练入口和 10k 正式机制实验；
4. 继续冻结 GNN 和变规模任务。

### 14.2 条件通过

如果标签语义和 fallback 通过，但查新显示方法层差异较弱：

- 将候选定位为工程基线或论文方法组件；
- 不使用独立算法名称和优先权表述；
- 只有它能为后续图 Critic 提供必要、稳定的信用接口时才继续实现。

### 14.3 否决

出现任一情况即否决当前候选：

- 仍依赖原累计成本标量生成局部 STOP/ENGAGE 标签；
- 只能在挑选的种子、场景或资源类型上形成方向；
- 新信号关闭时不能严格恢复 joint PPO；
- 简单全局约束基线以更低复杂度达到相同机制效果；
- 文献已经在相同 Problem、Method 和 Insight 层覆盖；
- 需要先查看在线结果才能定义成功门槛。

否决后允许返回 N1-02 选择另一个预先列出的候选一次。不得在同一候选上连续
增加模块、标签规则、阈值网格或随机种子。

### 14.4 创新 B 的进入

N1 通过不等于 GNN 自动进入。创新 B 必须等待创新 A 的在线机制至少完成 10k
三种子门控，并另行证明关系估值效率或跨规模泛化已经成为主要瓶颈。

## 15. 伪创新风险检查

| 风险 | 当前易犯错误 | N1 修复要求 |
| --- | --- | --- |
| 缺少动机 | “换成多头网络后更稳” | 每个结构必须对应直接成本或替代分量的明确语义 |
| 模块堆叠 | PPO + dual + auxiliary + uncertainty + GNN | N1 只冻结一个核心干预，其他作为对照或后续任务 |
| 跨领域搬运 | 把 CMDP/COMA 直接用于防空 | 说明动态掩码、前缀占用和动作替代使直接搬运哪里失效 |
| 指标叙事 | 只报告平均奖励增加 | 把可辨识性和跨种子稳定作为命题，性能作为证据 |
| “首次”脆弱 | 未完成查新就声称新算法 | 查新前禁止优先权表述 |
| GNN 捷径 | 用图网络拟合原混叠标签 | 先通过非图信用语义门控，再判断关系表示瓶颈 |

## 16. 审稿人压力测试

| 压力点 | 审稿人可能提问 | 必须准备的回答或实验 |
| --- | --- | --- |
| 与 difference reward 重叠 | 这是否只是已有差异回报换到防空任务？ | 比较 continuation、动态合法集、同一步后缀替代和资源约束语义 |
| 与 CMDP 重叠 | 为什么不用普通 PPO-Lagrangian？ | 将其设为必要强基线；若同样有效则收窄贡献 |
| 分解恒等式显然 | 贡献是否只是成本记账？ | 证明分解如何改变可训练信用接口，并用在线消融验证 |
| 旧数据泄漏 | R2 已被查看，何来独立验证？ | 明确 R1/R2 仅开发使用，正式确认采用新种子和零重叠状态 |
| 多开火换安全 | 是否牺牲资源效率降低突防？ | 等交战率对照和安全—成本 Pareto |
| 模型容量 | 是否只是更多 head 和参数？ | 等参数非分解对照 |
| 计算开销 | 成对反事实是否不可部署？ | 报告 transition、训练时间和决策时间，并设置上限 |
| 只在 AirDefense v1 有效 | 是否为环境特例？ | 30k 后才进入第二结构化任务，不提前外推 |
| GNN 缺席 | 为什么不用更强关系模型？ | 当前命题是信用语义；GNN 等关系瓶颈证据后进入 |

## 17. 任务产物

N1 完成时建议形成：

```text
docs/task_guides/next_research_phase_identifiable_resource_credit.md

docs/literature/n1_identifiable_resource_credit_search_protocol.md
docs/literature/n1_identifiable_resource_credit_novelty_review.md

docs/algorithms/identifiable_resource_credit_candidate_matrix.md
docs/algorithms/substitution_decomposed_resource_credit.md

docs/experiments/air_defense_v1_n1_offline_semantic_audit.md

configs/air_defense_v1/n1_online_preregistration.json
results/air_defense_v1/n1_offline_semantic_audit/
  experiment_config.json
  label_dictionary.json
  seed_usage_audit.json
  support_summary.csv
  candidate_comparison.csv
  gate_summary.json
```

候选未通过时，不要求创建 `substitution_decomposed_resource_credit.md`；应改为
失败报告并保留候选矩阵。

## 18. 实施顺序

```text
1. 冻结 N1 范围、成本方向和历史失败边界
2. 建立 Problem–Method–Insight 与术语表
3. 执行系统查新和最接近工作差异矩阵
4. 比较候选 A/B/C，冻结一个主候选和一个简单对照
5. 定义候选公式、软件接口和严格 fallback
6. 构造最小人工轨迹并完成方向/恒等式测试
7. 使用现有数据执行离线开发性证伪
8. 按 N1-P1 至 N1-P5 判定
9. 通过时冻结新种子、场景、预算、指标和数值门槛
10. 生成机器可读在线预注册配置
11. 创建后续在线实现任务；N1 本身停止
```

## 19. N1 验收清单

- [x] N1 范围没有恢复 BPCE/MCH-PPO 调参；
- [x] 成本恒等式、方向和术语与 R2 一致；
- [x] 三个候选均完成问题—方法—洞见比较；
- [x] 系统查新覆盖五类相邻工作；
- [x] 至少 3–5 篇最接近工作完成五层差异矩阵；
- [x] 未将多个机制拼成主候选；N1-P2 失败后不选择主候选；
- [x] 各候选目标语义和未解决的规范性选择已明确；
- [x] 人工轨迹覆盖零替代、同一步替代、未来替代和符号掩盖；
- [x] 新信号关闭时严格恢复 factorized joint PPO loss 与梯度；
- [x] 现有数据只被标记为开发数据；
- [x] 因 no-go 未放行正式实验，新种子和新状态保持未分配；
- [x] 因 no-go 未放行正式实验，安全与资源数值门槛保持未分配；
- [x] 10k、30k、100k 均被机器可读预注册明确禁止；
- [x] 失败和停止路径已经执行；
- [x] GNN 进入条件保持冻结；
- [x] 输出明确的 N1-E4 否决结论。

## 20. 创新演化日志

| 版本 | 当前洞见 | 新证据 | 修订原因 | 下一证伪测试 |
| --- | --- | --- | --- | --- |
| MCH-PPO | engagement/target 分层信用可能修复联合 advantage 混叠 | 10k 多种子出现 all-noop 和安全退化 | 独立层级更新与 Critic 误差共同失稳 | 恢复 joint PPO 主干 |
| BPCE-PPO | 当前策略边界的成对探测可作为 joint PPO 辅助 | 2/6 塌缩，异质场景成本 1.928 倍 | 正负标签覆盖和资源成本随种子分叉 | 审计标签语义 |
| BPCE A/A2 | 随机后续或短视窗可能形成双向可靠标签 | 随机后续可靠 25/72；短视窗可操作 31/72 | time/resource 缺少稳定 STOP 标签 | 审计动作替代 |
| R1/R2 | 累计成本标签受动作替代结构性混叠 | P-C1/P-C2 通过，P-C3 失败 | 精度不是测量有效性；资源类型存在边界 | 冻结测量贡献并完成 W1 |
| N1 | 回合成本差是全局后果但不是唯一局部信用；四分量账本只能解决读出语义 | 恒等式误差 `8.88e-16`，标量含混率 60.84%；N1-P2 失败 | A 仅是组件，B 是强基线，C 被否决；规范目标仍未定义 | 重定义全局预算—局部责任的可证伪关系 |

## 21. 阶段出口

N1 只能进入以下一种出口：

| 出口 | 条件 | 下一阶段 |
| --- | --- | --- |
| N1-E1：算法候选通过 | N1-P1 至 N1-P5 全部通过 | 创建新的在线算法实现与 10k 机制实验任务 |
| N1-E2：方法组件通过 | 语义/fallback通过，但创新差异不足 | 作为更大方法的组件，不独立命名；评估是否为创新 B 提供接口 |
| N1-E3：候选否决 | 标签、查新或 fallback 任一核心门控失败 | 停止该候选，最多返回候选矩阵选择另一路一次 |
| N1-E4：问题路线否决 | 三类候选均不能形成可辨识且有差异的机制 | 重新评估其他主线创新，如分层编组、持续压制或安全任务建模 |

无论出口为何，N1 都不得直接跳转到 GNN、30k 或 100k 正式实验。

## 22. 执行结果

更新时间：2026-07-28。  
任务状态：已完成；阶段出口为 **N1-E4**；在线训练未授权。

### 22.1 已完成工作

- 完成五类相邻工作的系统查新和最近工作五层差异矩阵；
- 比较候选 A“分量保持的约束信用”、候选 B“全局 CMDP 约束”和候选 C
  “受控延续差异回报”；
- 实现四分量资源信用语义接口、严格零系数 fallback 和人工轨迹测试；
- 对冻结 R2 数据执行零新增 rollout 的离线语义审计；
- 生成机器可读 no-go 预注册，明确禁止直接启动在线训练。

### 22.2 审计事实

108 个 context、7,776 条目标账本上的扩展恒等式最大误差为
`8.881784197001252e-16`，直接成本最大误差为 `0.0`。4,731 条账本满足
“直接成本为正但回合成本差非正”，含混率为 60.84%；72.22% 的 context
中多数账本受到符号掩盖。软件与既有动作替代测试合计 `12 passed`。

### 22.3 候选判决

| 候选 | 判决 | 原因 |
| --- | --- | --- |
| A：分量保持的约束信用 | 方法组件 | 账本可辨识，但局部直接成本是否应独立惩罚仍是未冻结的规范目标；分解本身与已有因果效应分解相邻 |
| B：全局 CMDP 约束 | 强基线 | 保持全局预算语义，但 CPO 和安全 MARL 已直接覆盖 |
| C：受控延续差异回报 | 否决 | 与 CCA/DAE/COCOA 相邻，且 continuation 干预可能引入策略偏差 |

N1-P1、P3、P4、P5 通过，N1-P2 失败。因此没有候选满足 N1-E1；由于
三条路线均不能同时形成目标一致且有足够差异的机制，最终出口为 N1-E4，
而不是把候选 A 包装成独立算法的 N1-E2。

### 22.4 正式产物

```text
docs/literature/n1_identifiable_resource_credit_search_protocol.md
docs/literature/n1_identifiable_resource_credit_novelty_review.md
docs/algorithms/identifiable_resource_credit_candidate_matrix.md
docs/experiments/air_defense_v1_n1_offline_semantic_audit.md
rein_learning/common/identifiable_resource_credit.py
tests/test_identifiable_resource_credit.py
scripts/analyze_air_defense_v1_n1_offline_semantic_audit.py
configs/air_defense_v1/n1_online_preregistration.json
results/air_defense_v1/n1_offline_semantic_audit/
```

按任务约定，没有创建
`docs/algorithms/substitution_decomposed_resource_credit.md`，因为不存在通过
门控的已成立算法。

### 22.5 下一主线入口

下一项主线不是在线实现，而是重新定义规范性算法问题：

> 新方法究竟保持全局资源预算、刻画局部责任，还是建立二者之间可证伪的
> 双层关系？

只有形成新的 Problem–Method–Insight、完成公式级查新并冻结独立在线协议
后，才能新建实现任务。GNN、BPCE-PPO 和 MCH-PPO 继续冻结。
