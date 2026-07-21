# 下一研究阶段：交战概率校准与反事实分层信用分配

更新时间：2026-07-19  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`  
阶段编号：任务十三  
阶段状态：诊断阶段已完成；完整 MCH-PPO 与 30k 未触发  
阶段主题：概率校准、联合 advantage 信用混叠、Critic 估值误差与创新查新

## 1. 阶段定位

任务十二已经完成 no-op 机理诊断和交战-目标因子化，但 30k 筛选仍出现不开火与高成本开火两极分化。任务十三不直接加入 GNN，也不立即实现完整反事实 PPO，而是先回答：

> 交战不稳定主要来自 deterministic 阈值失配、联合 advantage 信用混叠，还是 Critic 估值错误？

只有形成可证伪的机理结论并完成系统查新后，才能冻结第一创新候选。

## 2. 核心研究问题

1. `p_engage=0.5` 是否存在跨种子、跨场景的稳定决策意义？
2. 预测交战概率是否与实际收益、命中、毁伤降低和资源成本校准？
3. engage 与 no-op 样本的 advantage 均值、方差和符号是否系统不同？
4. 同一个联合 advantage 是否错误地同时更新 engagement 和 target 分支？
5. Critic 的 TD error 和 value bias 是否在塌缩发生前分叉？
6. 反事实分层 advantage 相对 COMA、H-PPO、HAPPO 和 CAPO 的方法差异是什么？

## 3. 阶段边界

### 3.1 允许内容

- 冻结模型的交战阈值扫描；
- Brier score、ECE、可靠性曲线和期望效用曲线；
- 按 engage/no-op、单元、目标和场景分组的 advantage/TD error；
- Actor engagement/target 分支梯度夹角与范数；
- Critic 校准和时序误差诊断；
- 反事实 action-value 原型和离线公式验证；
- 系统查新、相关方法矩阵和创新边界文档；
- 在诊断证据充分后实现一个最小 MCH-PPO 候选。

### 3.2 必须冻结

- 环境状态、转移、奖励和终止条件；
- 三个核心场景和任务十二固定探针；
- 角色条件 pair scorer 输入语义；
- 自回归顺序 `012` 和目标去重掩码；
- 首轮诊断中的 PPO 超参数；
- GNN、GAT、Transformer 和变规模环境；
- 未完成查新前的“算法创新已成立”表述。

## 4. 工作任务

### 任务十三·一：系统查新与方法差异矩阵

检索并整理：

- counterfactual credit assignment；
- hierarchical/parameterized-action PPO；
- multi-agent advantage decomposition；
- masked structured-action policy optimization；
- constrained resource allocation RL；
- graph-based WTA 与 anti-UAV allocation。

验收标准：

- 至少覆盖 COMA、H-PPO、HAPPO/HATRPO、CAPO 和 GNN-WTA；
- 形成“已有公式、动作结构、Critic、约束、实验任务、与本项目差异”矩阵；
- 明确哪些设计属于已知方法，哪些仍是待验证差异；
- 给出继续、收窄或放弃 MCH-PPO 命题的结论。

### 任务十三·二：冻结模型阈值与概率校准

对任务十二 6 个正式模型扫描：

```text
交战阈值：0.10 至 0.90，步长 0.05
场景：medium / time_pressure / heterogeneity_pressure
评估：相同成对环境种子
```

记录：

- actionable engagement；
- all-noop；
- 奖励、毁伤、高威胁泄漏和资源成本；
- Brier score、ECE 和可靠性分箱；
- 阈值-效用 Pareto 曲线。

验收标准：判断是否存在跨种子统一有效阈值。若仅靠阈值即可稳定解决，则不得把阈值调整包装为核心算法创新。

### 任务十三·三：engage/no-op 信用分配诊断

扩展 rollout 记录：

```text
unit_index, resource_type,
engage_probability, selected_engage,
target_probability, selected_target,
return, value, advantage, td_error,
engagement_log_prob, target_log_prob,
engagement_gradient_norm, target_gradient_norm
```

验收标准：

- 成功与塌缩种子可以在相同检查点比较；
- advantage 统计区分 engagement 与 target，不只报告联合均值；
- 定位最早可重复的 Critic/advantage 分叉；
- 至少形成一个能被后续干预否证的机理假设。

### 任务十三·四：反事实估值原型

在不更新策略的条件下，对固定状态计算：

```text
Q(s, observed joint action)
Q(s, replace unit i with no-op)
Q(s, replace unit i with each legal target)
```

比较环境精确分支、Monte Carlo 分支和学习 Critic 估值误差。

验收标准：

- 反事实动作保持其他单元动作和环境随机协议一致；
- 非法、重复和前缀冲突动作不进入反事实集合；
- 能分别构造 engagement advantage 与 conditional target advantage；
- 给出估值偏差、方差和计算成本。

### 任务十三·五：冻结最小算法候选

只有任务十三·一至四支持信用混叠假设时，才冻结 MCH-PPO 候选：

- engagement 与 target 独立 advantage；
- 独立 ratio 和 clip/KL 统计；
- 动态 mask 感知的反事实 baseline；
- 资源成本约束保持可解释；
- 不加入 GNN。

验收标准：候选公式、伪代码、理论命题、复杂度和消融方案在训练前完成，不根据结果事后修改主门槛。

### 任务十三·六：条件性 30k 筛选

若候选冻结，使用新的不重叠种子运行标准配对筛选。候选至少满足：

- collapsed seed 为 0；
- 每个场景 all-noop 不超过 2%；
- deterministic/stochastic 交战率差不超过 0.05；
- 不出现高成本交战种子；
- medium 奖励相对冻结对照下降不超过 5；
- 毁伤、高威胁泄漏和资源成本通过非劣效门槛；
- 至少 2/3 种子同向改善。

未通过时不运行 100k，也不进入 GNN。

## 5. GNN 进入条件

GNN 不是任务十三内容。只有以下条件全部满足后，才允许规划类型化图反事实 Critic：

- 第一创新候选通过独立种子稳定性筛选；
- 交战决策不再是主要失败来源；
- 剩余误差集中于关系匹配、反事实估值效率或跨规模泛化；
- 已建立可变资源数和目标数的独立测试协议；
- 普通 MLP/关系 scorer 形成稳定对照。

图阶段的目标是批量反事实估值和规模泛化，不是简单替换 Encoder。

## 6. 阶段交付物

```text
docs/literature/task13_counterfactual_credit_novelty_review.md
docs/algorithms/masked_counterfactual_hierarchical_ppo.md
docs/experiments/air_defense_v1_task13_credit_diagnostics.md
scripts/analyze_air_defense_v1_task13_calibration.py
scripts/analyze_air_defense_v1_task13_credit.py
tests/test_air_defense_v1_task13_diagnostics.py
results/air_defense_v1/task13_*/
```

算法文档和训练产物只有在对应任务实际发生后创建，以上路径是预期交付，不代表当前已经存在。

## 7. 阶段决策规则

```text
系统查新能否形成明确差异？
├─ 否：收窄或放弃 MCH-PPO 命题
└─ 是
   └─ 信用混叠是否被诊断证据支持？
      ├─ 否：研究阈值校准或其他主因，不实现反事实 PPO
      └─ 是
         └─ 最小候选是否通过 30k 稳定性和非劣效门槛？
            ├─ 否：记录负结果，不运行 100k/GNN
            └─ 是：独立 100k 确认，再规划图反事实 Critic
```

## 8. 阶段完成定义

任务十三完成不等于新算法必须成功。满足以下任一结果即可形成合格阶段结论：

- 证明阈值校准是主因，否决不必要的算法复杂化；
- 证明信用混叠是主因，并冻结可复现的 MCH-PPO 候选；
- 证明 Critic 估值误差是主因，转向更精确的 value/Q 建模；
- 系统查新否决当前创新命题，及时调整研究路线；
- 候选通过筛选，进入独立确认阶段。

## 9. 实际执行结论

任务十三已完成系统查新、6 个冻结模型的 17 阈值扫描、随机概率采样校准、代表性种子的逐单元信用诊断和共同随机数反事实原型。

实际进入以下决策分支：

```text
系统查新：宽泛组合不新颖，命题必须收窄
-> 统一阈值：不存在跨模型和场景的统一有效阈值
-> 信用诊断：联合 advantage 局部判别力弱，但反事实显著性不足
-> Critic 结构：现有 V(s) 无法估计动作反事实 Q
-> 冻结估计公式和软件接口
-> 不冻结完整 MCH-PPO，不运行 30k，不进入 GNN
```

下一步先建立非图结构的掩码条件动作价值 Critic，验证 Q 排序和 advantage 符号后再决定是否恢复 MCH-PPO 训练候选。详见[任务十三实验报告](../experiments/air_defense_v1_task13_credit_diagnostics.md)。
