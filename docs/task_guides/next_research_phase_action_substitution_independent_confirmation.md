# 下一项工作建议：动作替代导致资源成本测量失真的独立确认与适用边界审计

更新时间：2026-07-23  
建议状态：已完成；P-C1/P-C2通过，P-C3未通过  
任务编号：R2  
任务优先级：P0  
任务性质：一次性独立机制确认与论文贡献冻结  
路线关系：不运行 E-R，不训练机会成本 oracle，不恢复 BPCE/MCH-PPO，不进入GNN

## 0. 执行冻结记录

启动时间：2026-07-23。

正式执行前冻结以下工程参数，后续不得根据模型行为或N/E结果修改：

- 来源策略种子保持`17、18、19`；它们虽在单元测试和Task14预测器训练中
  出现过，但未用于动作替代、BPCE成本标签设计或factorized来源策略选择；
- 三个训练场景为`medium/time_pressure/heterogeneity_pressure`；
- 每个来源模型训练10k steps，`n_steps=256`、`batch_size=64`、
  `n_epochs=2`、CUDA，所有9个模型无条件保留；
- 新上下文环境基准种子为`1,283,000`，每个场景—策略种子使用24个候选
  回合，选择前排除旧正式数据observation hash；
- 每块选择6个safety上下文；resource按状态分数分别选择
  `3 missile + 3 laser`，不读取N/E结果；
- N/E共同随机带基准种子为`1,293,000`，每上下文32次重复；
- 目标条件概率精确边缘化，概率重建容差`1e-12`；
- transition预算、P-C1/P-C2/P-C3及决策分支均保持本文第11至14节不变；
- 正式成本账本同时报告当前其他单元成本差。协议主恒等式仍按本文
  `Delta_C_episode=C_direct-Sub_cost`判定；另提供包含当前其他单元差的
  扩展恒等式作为诊断，不用它替换或放宽P-C1。

### 0.1 首轮P-C1账本修正记录

首轮正式执行完整性门控全部通过，但原始主恒等式在287/7776条目标账本中
出现非零残差，最大值为2.0。扩展恒等式和未来probe/other子分解的最大
误差均为`8.88e-16`。逐条诊断确认：

- 282条记录中，E分支被测单元占用目标后，无冲突自回归后缀单元在同一步
  少执行一次交战；
- 5条记录中，同一步其他单元成本增加；
- 原公式只计入未来替代，遗漏了当前联合动作内部的其他单元替代。

按第14.4节只修复成本账本，不修改任何科学样本或门槛。修正后的总替代
成本定义为：

```text
Sub_cost_total
= same_step_other_unit_substitution
 + future_probe_substitution
 + future_other_unit_substitution
```

其中：

```text
same_step_other_unit_substitution
= current_other_cost(N) - current_other_cost(E)
```

原始`future_sub_cost`和`future_only_residual`继续单独保存。P-C1主恒等式
改为对完整联合动作替代记账：

```text
Delta_C_episode = probe_direct_cost - Sub_cost_total
```

`Sub_shot`仍只统计当前步之后的未来射击，不修改P-C2/P-C3、上下文、
种子、重复、目标概率或阈值。首轮结果保存为`pre_ledger_correction_*`，
随后只按原配置完整重跑一次。

### 0.2 正式确认结论

账本修正后的唯一完整重跑于2026-07-23完成：

- 9/9个新来源模型全部保留，108/108个新上下文完成；
- 与旧正式数据observation hash重叠为0；
- resource槽每块`3 missile + 3 laser`，总计各27个；
- 目标概率最大重建误差为0，Actor参数差为0；
- 3456条上下文—重复记录、7776条目标账本和157,485个额外transition；
- 总替代成本恒等式、扩展恒等式和probe/other子分解最大误差均为
  `8.88e-16`。

P-C2独立确认通过：

- `time_pressure/resource`中13/18个上下文的`mean(Sub_shot)>0`且
  95%下界大于0；
- seeds17/18/19三个块的下界均大于0；
- seeds17和19的符号掩盖率超过50%；
- 7个非正累计成本差上下文全部具有正替代成本。

P-C3未通过。time/resource中missile和laser的聚合`Sub_shot`下界及三个
种子块方向均为正，但missile只有2个`cost_sign_masked`上下文，低于3个
门槛；laser有5个。按第14.2节冻结**资源类型条件确认**，不写成跨资源
类型通用结论，不追加种子或实验。

正式报告见
[动作替代测量失真独立确认](../experiments/air_defense_v1_action_substitution_confirmation.md)；
贡献决策见
[第一创新claim–evidence矩阵](../project/first_innovation_claim_evidence_matrix.md)。

## 1. 决策摘要

动作替代与弹药机会成本审计已经形成明确分叉：

- P-R1 强通过：`time_pressure/resource` 的18/18个上下文均有可靠正 `Sub_shot`，11个非正累计成本差全部可由未来成本替代解释；
- P-R2/P-R3 失败：可靠资源机会价值仅为 `time 5/18`、`heterogeneity 2/18`，异质场景只由seed9贡献，且只覆盖missile；
- 通用机会成本 oracle、BPCE/MCH-PPO 接入和GNN扩展均已停止。

因此，下一项工作不再尝试“修复”资源监督，而是对唯一成立的正机制进行一次最终独立确认：

> **在全新策略种子、全新状态和三个预注册场景中，确认当前射击是否会通过替代未来射击而系统性掩盖累计资源成本，并刻画该失真在哪些场景和资源类型中成立。**

本任务完成后：

- 若独立确认通过，停止继续扩展机制实验，转入贡献冻结与论文写作；
- 若只在部分场景或资源类型通过，收窄主张后转入写作；
- 若独立确认失败，将原 P-R1 降级为旧种子和旧上下文中的条件性发现；
- 无论结果如何，都不重新启动机会成本 oracle 或在线辅助路线。

依据材料：

- [动作替代与弹药机会成本正式审计](../experiments/air_defense_v1_action_substitution_opportunity_cost_audit.md)
- [动作替代与弹药机会成本任务建议](./next_research_phase_action_substitution_resource_opportunity_cost_audit.md)
- [BPCE 短视窗安全—资源双分量标签正式审计](../experiments/air_defense_v1_bpce_short_horizon_label_audit.md)
- [状态条件资源预算与显式双价值实验](../experiments/air_defense_v1_task14_state_conditioned_value.md)
- [项目创新路线](../project/research_innovation_roadmap.md)

## 2. 当前证据与尚未解决的问题

### 2.1 已成立的机制证据

原 R1 在 `time_pressure/resource` 中得到：

```text
mean(Sub_shot) > 0：18/18
lower95(Sub_shot) > 0：18/18
非正累计成本差：11个
可由未来成本替代解释：11/11
平均 Sub_shot：0.990
平均 Sub_cost：1.995
首次替代时刻：当前动作后2.86步
成本重构最大误差：4.00e-15
```

这支持：

> 当前强制交战平均替代约一次未来射击，直接成本因此被后续少射击抵消；累计资源成本差并不稳定表示当前动作消耗的资源。

### 2.2 已被否决的路线

R1 同时表明：

- 恢复弹药虽然扩大合法动作集合，但不稳定改善最终安全结果；
- 可靠机会价值跨场景、跨种子和跨资源类型覆盖不足；
- 机会安全价值只在少数missile上下文中可靠；
- 不能把 E-R 结果构造成通用训练标签。

本任务不得再次测试：

- E-R 弹药恢复标签；
- 新的安全—资源加权分数；
- 机会成本预测网络；
- BPCE 或 MCH-PPO 辅助剂量；
- seed9专用机制；
- missile专用在线算法。

### 2.3 剩余证据缺口

原 P-R1 仍使用了反复参与机制设计的：

```text
策略种子：8、9、10
场景：time_pressure、heterogeneity_pressure
上下文：阶段A/A2原72个
```

在将其写成论文主要贡献前，仍需回答：

1. 动作替代是否能在完全未参与设计的策略种子上复现？
2. 该机制是否只在 `time_pressure` 成立，还是在 `medium` 和 `heterogeneity_pressure` 中也存在？
3. 该机制是否只来自高成本missile，还是低成本高弹药laser也会发生？
4. 哪个可解释边界决定累计成本差何时被部分抵消、完全抵消或反向？

## 3. 本阶段研究目标

本任务只完成三件事：

1. **独立确认**：用新策略种子和新上下文复验 N/E 成本分解；
2. **边界刻画**：比较三个场景及 missile/laser 的替代比率；
3. **贡献冻结**：根据确认结果决定第一创新能否收窄为测量与可辨识性贡献。

本任务不以获得算法性能提升为目标。

## 4. Problem–Method–Insight

| 层次 | 本阶段表述 |
| --- | --- |
| Problem | 回合累计资源成本差无法区分当前直接消耗与后续策略少执行的替代射击，因此可能把一个确定有成本的当前动作标为零成本甚至负增量成本。 |
| Method | 使用全新策略种子、全新状态和共同随机数 N/E 分支，建立逐时刻直接成本—未来替代成本恒等分解，并按场景与资源类型报告替代比率。 |
| Insight | 当未来替代成本接近或超过当前直接成本时，回合累计成本标签会系统性掩盖当前动作的资源消耗；这种失真必须作为反事实资源信用的适用边界，而不是通过增加样本量处理。 |

该 Insight 只有在独立确认达到预注册门控后才能提升为论文主张。

## 5. 可证伪命题

| 命题 | 支持证据 | 否决证据 | 所需测试 |
| --- | --- | --- | --- |
| P-C1：N/E成本可以被直接成本和未来替代成本精确分解 | 所有重复、目标和上下文的分解残差低于冻结误差 | 存在系统性非零残差，说明成本账本或分支定义不完整 | 逐步成本日志和恒等重构 |
| P-C2：time/resource 的动作替代可在全新策略种子上复现 | 至少2/3新种子具有可靠正块级 `Sub_shot`，且多数当前成本被显著抵消 | 只有单一种子出现替代，或块级方向不稳定 | 三个新策略种子的独立N/E审计 |
| P-C3：替代强度具有可解释的场景与资源类型边界 | `rho_sub`在场景和missile/laser分层中形成稳定差异，结论不依赖合并统计 | 分层结果方向随机或只由单一模型种子决定 | 三场景、资源类型配额与分块报告 |

P-C1 是完整性命题；P-C2 是主要确认命题；P-C3 决定主张可以写成通用、场景条件还是资源类型条件。

## 6. 独立性协议

### 6.1 策略种子

冻结新的 factorized joint PPO 策略种子：

```text
17、18、19
```

当前项目文档和现有 R1/BPCE 正式结果中未使用这三个种子进行机制设计。

正式准备前生成 `seed_usage_audit.json`，检索：

- 项目实验配置；
- 结果目录；
- 模型文件名；
- 项目进度与创新路线；
- 已完成任务文档。

如果发现 `17、18、19` 中任一种子已经用于动作替代或成本标签设计，则在查看任何本任务结果前整体改为下一个连续的三个未使用种子，并在预注册配置中记录一次性替换原因。不得逐个替换或根据模型行为换种子。

### 6.2 来源策略准备

每个场景为三个种子训练或加载完全相同配置的：

```text
factorized_engagement_ar_ppo_order_012
```

冻结配置：

| 项目 | 配置 |
| --- | --- |
| 训练场景 | `medium`、`time_pressure`、`heterogeneity_pressure` |
| 训练种子 | `17、18、19` |
| 训练预算 | 10k steps/model |
| PPO epochs | 2 |
| 环境与奖励 | 保持现有 AirDefense v1 配置 |
| 模型选择 | 禁止；每个预注册种子全部进入审计 |
| BPCE/MCH辅助 | 禁止 |

如果已有完全相同配置的模型，可以直接复用；否则训练9个来源模型。来源模型只用于生成状态和冻结策略延续，不是本任务的候选算法。

以下行为不能成为排除模型的理由：

- all-noop；
- 高交战；
- 低奖励；
- 资源成本异常；
- 与旧种子行为差异较大。

这些都是策略种子外部有效性的一部分。

### 6.3 状态独立性

新上下文必须与以下正式数据 observation hash 重叠为0：

```text
bpce_label_semantics_audit
bpce_short_horizon_label_audit
action_substitution_opportunity_cost_audit
```

新环境种子范围在运行前冻结，不得因上下文方向不理想而重新生成。

## 7. 上下文采样

每个“场景×策略种子”采集12个上下文：

```text
6个 safety 上下文
6个 resource 上下文
```

总量：

```text
3场景 × 3策略种子 × 12上下文 = 108上下文
```

### 7.1 Safety 槽

沿用既有不读取未来结果的安全评分：

```text
threat × payload × protected-zone value / time-to-impact
```

每块选取6个合法且可执行的上下文。

### 7.2 Resource 槽

沿用弹药稀缺度、单位成本和替代单元覆盖率评分，但增加预注册资源类型配额：

```text
3个 missile 上下文
3个 laser 上下文
```

在各资源类型内部按冻结 resource score 排序，不读取 N/E 结果。

如果某一块无法形成3个合法laser或3个合法missile上下文：

- 记录 `type_quota_unavailable`；
- 不得从另一资源类型补齐并伪装为类型平衡；
- 该块的数据完整性门控失败；
- 不得更换策略种子或环境种子。

### 7.3 去重

同一上下文不得重复以下组合：

```text
observation_hash
unit_index
environment_step
legal_targets
```

同一状态可以为不同单元提供上下文，但必须分别记录，不得在统计中视为独立状态而遗漏聚类关系。置信区间以状态或上下文组为聚类单元。

## 8. 冻结 N/E 分支

本任务只运行两个分支。

### 8.1 N：当前 no-op

```text
N:
  被测单元当前强制 no-op
  其他单元保持冻结联合动作构造语义
  后续采用冻结策略 stochastic continuation
```

### 8.2 E：当前 engage

```text
E:
  被测单元当前强制 engage 合法目标
  正常扣弹、记录成本和设置cooldown
  正常命中、目标推进和奖励计算
  后续采用冻结策略 stochastic continuation
```

### 8.3 随机控制

N/E 必须共享：

- 当前步和后续环境命中随机带；
- 后续策略 uniform tape；
- 同一回合终止和截断规则；
- 同一上下文前缀；
- 相同的其他单元动作构造语义。

每个合法目标分别计算 E，再按冻结策略的目标条件概率精确边缘化。

本任务禁止：

- masked-argmax target；
- deterministic continuation；
- E-R 弹药恢复；
- 新的奖励或资源惩罚；
- Actor更新。

## 9. 逐步成本账本

对 N/E 每个分支分别记录：

```text
current_probe_cost
current_other_unit_cost
future_probe_cost
future_other_unit_cost
future_total_cost
future_probe_shots
future_other_unit_shots
future_total_shots
first_future_shot_step
first_substitution_step
```

当前被测单元 engage 的直接成本：

```text
C_direct
= current_probe_cost(E)
 - current_probe_cost(N)
```

未来替代成本：

```text
Sub_cost
= future_total_cost(N)
 - future_total_cost(E)
```

未来替代射击：

```text
Sub_shot
= future_total_shots(N)
 - future_total_shots(E)
```

回合累计成本差：

```text
Delta_C_episode
= total_cost(E)
 - total_cost(N)
```

必须满足恒等式：

```text
Delta_C_episode
= C_direct - Sub_cost
```

同时分解：

```text
Sub_cost
= Sub_cost_probe
 + Sub_cost_other_units
```

用于判断当前交战替代的是被测单元自己的未来射击，还是改变了其他单元的后续分工。

## 10. 替代比率与失真类型

合法 engage 的 `C_direct` 必须大于0。定义：

```text
rho_sub
= Sub_cost / C_direct
```

冻结解释：

| 条件 | 解释 |
| --- | --- |
| `rho_sub <= 0` | 未观测到正的未来成本替代 |
| `0 < rho_sub < 1` | 当前直接成本被部分抵消 |
| `rho_sub >= 1` | 当前直接成本被完全抵消，累计成本差为0或负 |

符号掩盖事件：

```text
cost_sign_masked
= (C_direct > 0)
  and (Delta_C_episode <= 0)
```

主要报告：

- `rho_sub` 均值、中位数、四分位数和95%置信区间；
- `cost_sign_masked` 比例；
- `Sub_shot` 和 `Sub_cost` 的上下文可靠性；
- 第一次替代发生时刻；
- missile/laser；
- safety/resource；
- medium/time/heterogeneity；
- 策略种子17/18/19；
- 被测单元替代与其他单元替代的比例。

## 11. 冻结实验协议

| 项目 | 配置 |
| --- | --- |
| 来源策略 | factorized joint PPO 10k |
| 策略种子 | `17、18、19` |
| 场景 | `medium`、`time_pressure`、`heterogeneity_pressure` |
| 上下文 | 108 |
| 每块 safety/resource | 6/6 |
| resource类型配额 | missile 3 / laser 3 |
| 每上下文重复 | 32 |
| 目标处理 | 全部合法目标精确概率边缘化 |
| 后续 | stochastic continuation |
| 环境随机控制 | N/E共享 |
| 策略随机控制 | N/E共享uniform tape |
| Actor更新 | 禁止 |
| E-R | 禁止 |
| 正式探测transition上限 | 266,198 |

如果预算投影超过266,198，只允许在查看正式结果前使用固定顺序早停：

- 某上下文已经达到可靠方向，且剩余重复不能改变其可靠性判定；
- 即使剩余重复全部同向也无法达到可靠门槛。

节省的预算不得动态用于搜索更多上下文。

## 12. 统计协议

### 12.1 重复层

每个上下文32次 N/E 配对差值计算：

- 均值；
- 样本标准误；
- 95%均值置信区间；
- `Sub_shot > 0` 方向；
- `Sub_cost > 0` 方向；
- `rho_sub >= 1` 比例。

### 12.2 块级

以“场景×策略种子×槽位×资源类型”为主要分层。

块级置信区间必须以上下文为聚类单元，不能把32次重复直接当作32个独立状态。

### 12.3 场景边界

`medium` 和 `heterogeneity_pressure` 是边界刻画场景，不要求复制
`time_pressure` 的同等替代强度。必须报告差异，但不得根据结果调整场景门槛。

### 12.4 多重比较

主要门控只使用：

- `time_pressure/resource` 的 P-C2；
- 成本恒等式 P-C1；
- missile/laser 的 P-C3。

其他场景、槽位、时刻和单元分解均为预注册次要分析，不用于事后改变总门控。

## 13. 正式门控

### 13.1 完整性与独立性门控

| 门控 | 通过条件 |
| --- | --- |
| 来源策略 | 9/9个预注册模型完成或加载 |
| 模型选择 | 17/18/19全部保留，无行为筛选 |
| 新上下文 | 108/108完成 |
| 旧数据重叠 | observation hash重叠为0 |
| 类型配额 | 每个resource块missile 3 / laser 3 |
| 重复 | 每上下文32次完整或符合冻结早停 |
| 目标概率 | 最大重建误差不超过 `1e-12` |
| Actor冻结 | 最大参数差为 `0.0` |
| transition预算 | 不超过266,198 |
| 软件回归 | 新增测试和项目相关回归全部通过 |

### 13.2 P-C1：成本分解门控

同时满足：

- 每条重复记录都可以重构 `Delta_C_episode`；
- 最大绝对重构误差不超过 `1e-6`；
- probe/other-units 子分解最大误差不超过 `1e-6`；
- 当前步和未来步成本不存在重复计入；
- 所有合法 engage 的 `C_direct > 0`。

P-C1 失败时，正式结果无效，只允许修复日志和分支语义。

### 13.3 P-C2：独立动作替代确认

在全新 `time_pressure/resource` 的18个上下文中同时满足：

- 至少12/18个上下文 `mean(Sub_shot) > 0`；
- 至少6/18个上下文 `lower95(Sub_shot) > 0`；
- 至少2/3策略种子的 resource 块均值满足 `lower95(Sub_shot) > 0`；
- 至少2/3策略种子的 `cost_sign_masked` 比例不低于50%；
- 非正累计成本差上下文中至少80%具有 `mean(Sub_cost) > 0`；
- 结果不能只由单一策略种子形成。

P-C2 是本阶段主要通过条件。

### 13.4 P-C3：资源类型边界

在 `time_pressure/resource` 中分别对missile和laser统计。

**跨资源类型主张通过**需同时满足：

- missile聚合 `lower95(Sub_shot) > 0`；
- laser聚合 `lower95(Sub_shot) > 0`；
- 两种资源类型均至少2/3种子块的 `mean(Sub_shot) > 0`；
- 两种资源类型均至少存在3个 `cost_sign_masked` 上下文；
- 任一类型的结论不能只来自单一种子。

如果 P-C2 通过但 P-C3 失败，阶段仍可形成**资源类型条件确认**，但不得写成跨资源类型通用结论。

### 13.5 场景适用边界

对 `medium`、`time_pressure`、`heterogeneity_pressure` 分别报告：

- `Sub_shot`；
- `Sub_cost`；
- `rho_sub`；
- `cost_sign_masked`；
- 首次替代时刻。

不设置“所有场景必须同方向”的总门槛。场景差异本身是边界结果，但必须满足：

- 每个场景全部三个种子均报告；
- 不合并场景掩盖方向冲突；
- 不选择替代最强的场景作为唯一主结果。

## 14. 决策规则

### 14.1 P-C1、P-C2、P-C3全部通过

冻结以下论文候选结论：

> 在 AirDefense v1 的动态掩码序列分配中，未来动作替代可以跨策略种子和资源类型系统性抵消当前动作成本，使回合累计成本差成为有偏的局部资源信用读出。

随后：

- 停止新的机制与算法实验；
- 编写 claim–evidence 矩阵；
- 开始论文问题定义、方法、负结果和边界章节；
- 开展针对反事实信用、资源影子价格和动作替代测量的系统文献检索；
- 不恢复 BPCE/MCH-PPO。

### 14.2 P-C1、P-C2通过，P-C3失败

冻结场景或资源类型条件结论，例如：

> 动作替代导致的累计成本失真在高时压missile资源分配中可独立复现，但不能外推到laser。

随后转入论文写作，不追加资源恢复、机会成本网络或更多策略种子。

### 14.3 P-C1通过，P-C2失败

将原 R1 的动作替代结论降级为：

> 在种子8/9/10和原72上下文中观察到的条件性机制。

不得作为第一创新的主要普适主张。下一工作应是重新定义第一创新问题，而不是继续补充确认样本。

### 14.4 P-C1失败

正式结果无效。只修复：

- 逐步成本账本；
- 当前/未来分界；
- N/E分支构造；
- 目标边缘化；
- 随机带对齐。

修复后可按原配置完整重跑一次，不得修改科学门槛。

### 14.5 数据完整性失败

如果新来源策略无法提供预注册类型配额或上下文数量：

- 报告该策略种子下的支持缺失；
- 不替换为表现更有利的策略；
- 不把缺失块从分母中删除；
- 独立确认总门控判定为失败或适用范围不足。

## 15. 实施任务

1. 创建并冻结种子使用审计；
2. 训练或加载种子17/18/19的三个场景来源策略；
3. 保存来源模型配置、参数哈希和训练日志；
4. 实现与旧正式数据的 observation hash 去重；
5. 实现 safety/resource 上下文采样及资源类型配额；
6. 复用 N/E、目标边缘化和共同随机数分支；
7. 移除本任务中的 E-R 调用路径；
8. 增加当前步/未来步和 probe/other-units 成本账本；
9. 增加射击次数、首次替代时刻和单位类型日志；
10. 实现成本恒等式与子分解重构；
11. 实现 `rho_sub` 与 `cost_sign_masked`；
12. 增加独立性、类型配额和聚类统计测试；
13. 运行只验证软件语义的 smoke；
14. 一次性生成108个正式上下文并执行32次配对审计；
15. 按 P-C1→P-C2→P-C3 顺序判定；
16. 生成正式实验报告和第一创新 claim–evidence 决策。

## 16. 建议产物

```text
rein_learning/common/action_substitution_confirmation.py
scripts/run_air_defense_v1_action_substitution_confirmation.py
tests/test_action_substitution_confirmation.py

results/air_defense_v1/action_substitution_confirmation/
  experiment_config.json
  seed_usage_audit.json
  source_model_manifest.json
  context_identity_check.csv
  context_selection.csv
  repeat_cost_ledger.csv
  context_substitution_estimates.csv
  block_summary.csv
  resource_type_summary.csv
  scenario_boundary_summary.csv
  gate_summary.json

docs/experiments/air_defense_v1_action_substitution_confirmation.md
docs/project/first_innovation_claim_evidence_matrix.md
```

新产物不得覆盖：

```text
results/air_defense_v1/action_substitution_opportunity_cost_audit/
results/air_defense_v1/bpce_short_horizon_label_audit/
results/air_defense_v1/bpce_label_semantics_audit/
```

## 17. 审稿压力点与预防措施

| 风险 | 可能质疑 | 预防措施 |
| --- | --- | --- |
| 仍然使用项目内部数据 | “独立确认”是否只是同一批状态换种子？ | 使用新策略种子、新环境种子、零observation hash重叠和三个场景。 |
| 新种子行为异常 | 是否排除了all-noop或高交战种子？ | 所有预注册种子无条件保留，来源策略性能不作为筛选门槛。 |
| 成本恒等式过于显然 | 贡献是否只是记账公式？ | 公式本身不是主要贡献；关键证据是替代比率在实际策略轨迹中足以改变累计成本标签符号。 |
| 上下文选择制造替代 | resource槽是否事后选择了高替代状态？ | 选择规则只读取当前状态特征，并使用新环境种子；N/E结果不可见。 |
| missile主导结果 | 是否只验证了高成本低弹药资源？ | 预注册missile/laser各半配额并分别门控。 |
| 重复伪装样本量 | 32次随机重复是否被当作32个独立状态？ | 块级置信区间以上下文/状态为聚类单元。 |
| 负结果包装为算法 | 是否暗示已经改进PPO？ | 全文定位为测量与可辨识性贡献，不报告在线性能提升。 |
| 泛化过度 | 是否从AirDefense v1外推到所有MARL任务？ | 主张限定为动态掩码、序列资源分配和冻结策略反事实评估。 |
| 缺少相关工作定位 | 是否已有action replacement或shadow-price分解？ | 形成论文主张前单独执行系统文献检索；本报告不使用“首次”。 |

## 18. 论文贡献冻结边界

### 18.1 若独立确认通过

允许形成三个候选贡献：

1. 识别动态序列资源分配中，未来动作替代会使回合累计资源差偏离当前动作资源消耗；
2. 提供共同随机数 N/E 分支和逐时刻成本账本，将当前直接成本与未来替代成本精确分解；
3. 通过新策略种子、场景和资源类型确认该失真的适用边界。

不能宣称：

- 提出了已经优于 PPO 的新算法；
- 解决了 all-noop；
- 得到了通用机会成本 oracle；
- 解决了资源过度交战；
- GNN 可以修复该问题。

### 18.2 若仅条件通过

贡献必须包含明确限定词：

```text
在高时压场景中
在missile资源上
在冻结factorized PPO策略下
在AirDefense v1环境中
```

不得把场景条件结果写成通用规律。

### 18.3 学位或论文结构风险

该贡献属于：

- 测量定义；
- 反事实信用审计；
- 失败机制；
- 可辨识性边界。

它不能单独替代原计划中的完整算法创新。如果学位要求必须包含算法创新，应在本阶段结束并冻结结论后重新定义第一算法问题，不能继续把已失败的 BPCE/MCH-PPO 包装为成功算法。

## 19. 创新演化记录

| 版本 | 当前认识 | 新证据 | 修订原因 | 下一证伪测试 |
| --- | --- | --- | --- | --- |
| 状态条件双价值 | 增量成本可以与安全收益分别学习 | cost相关仅 `-0.04–0.13` | 累计成本监督跨场景不稳 | BPCE标签语义审计 |
| BPCE阶段A/A2 | 随机后续与短视窗可能恢复STOP | 短视窗仅31/72可操作，time/resource为0/0/18 | 完整回合与短窗成本均受后续动作影响 | R1动作替代审计 |
| R1 | 累计成本差可能被未来射击替代 | P-R1为18/18；P-R2/P-R3失败 | 保留替代机制，停止通用机会价值 | R2独立确认 |
| R2独立确认 | 替代比率决定累计成本标签何时失真 | P-C1/P-C2通过；新种子三块方向可靠，P-C3因missile掩盖上下文2个失败 | 机制跨新种子成立，但标签符号变化受资源类型与场景约束 | 冻结claim–evidence并转入论文写作 |

## 20. 建议执行顺序

```text
1. 冻结本报告、种子17/18/19和新环境种子范围
2. 生成seed_usage_audit并确认独立性
3. 训练或加载9个factorized 10k来源模型
4. 冻结模型哈希，不按行为筛选
5. 实现成本账本、类型配额和旧数据去重
6. 完成软件测试与最小smoke
7. 一次性采集108个新上下文
8. 执行N/E共同随机数审计
9. 按P-C1、P-C2、P-C3顺序判定
10. 生成正式报告与第一创新claim–evidence矩阵
11. 停止机制实验，进入论文写作或重新定义第一算法问题
```

当前只建议执行步骤1至10。无论结果如何，不追加 E-R、机会成本模型、BPCE/MCH-PPO 在线训练或 GNN。
