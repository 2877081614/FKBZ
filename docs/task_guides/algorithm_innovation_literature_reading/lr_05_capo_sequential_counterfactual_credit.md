# LR-05：CAPO/COSAC 顺序反事实信用论文阅读任务

任务状态：`PASSED`  
优先级：P0  
建议用时：4 小时  
实验授权：否  
前置：LR-01、LR-04

## 1. 论文身份

当前 arXiv 标题：*CAPO: Counterfactual Credit Assignment in Sequential Cooperative Teams*  
作者：Shripad Deshmukh、Jayakumar Subramanian、Raghavendra Addanki、Nikos Vlassis  
来源：arXiv:2604.17693，2026 年预印本  
官方页面：<https://arxiv.org/abs/2604.17693>

版本警告：

- 当前页面正文中同时出现 CAPO、COSAC 和 Sequential Aristocrat Utility；
- 任务开始时必须记录 arXiv 版本号、更新时间和算法最终名称；
- 该论文尚按预印本处理，不得写成已同行评审定论。

## 2. 选择理由

该论文直接研究固定顺序、共享团队奖励、逐个更新的合作团队，提出前缀条件的
Sequential Aristocrat Utility，并给出 bias/variance 分析。它与项目的：

- 自回归单元顺序；
- 同一步后缀动作替代；
- MCH 分层反事实信用；
- HAPPO/HARL 顺序更新；
- 冻结 Critic 支持域不足

存在高度直接的公式竞争关系。

## 3. 核心阅读问题

1. Sequential learnability 如何定义？
2. SeqAU 为什么在前缀条件 baseline 类中具有唯一性？
3. direct effect 与 downstream indirect effect 如何分解？
4. additive reward decomposition 的偏差如何进入 advantage？
5. fictitious continuation 是否需要额外环境调用？
6. 偏差和方差如何随团队规模、顺序位置和交互强度增长？
7. critic-free 的含义是什么，是否真的不学习任何回报模型？
8. CAPO/COSAC 与 COMA、HAPPO、MCH-PPO 的不可约化差异是什么？

## 4. 必读部分

- sequential cooperative team 问题定义；
- Sequential Aristocrat Utility；
- upstream cancellation；
- direct/indirect advantage；
- additive reward fit 和 fictitious sampling；
- bias/variance theorems；
- sequential bandit 实验；
- HAPPO/HA-GRPO、COMA 类比较；
- appendix 中非加性残差、覆盖矩阵和位置方差分析。

## 5. 必须重建的公式

至少重写：

1. prefix-conditional baseline；
2. SeqAU；
3. upstream cancellation identity；
4. direct + indirect advantage；
5. 非加性残差导致的 bias bound；
6. variance 与下游单元数/覆盖的关系。

必须明确区分：

```text
动作生成顺序偏置       ≠ advantage 信用偏差
逐单元策略更新非平稳性 ≠ 环境时间上的长期信用
团队回报加性近似       ≠ 项目成本账本恒等式
虚拟后续采样           ≠ 环境快照反事实 rollout
```

## 6. 项目压力测试

至少对照：

- [Task 13 信用诊断](../../experiments/air_defense_v1_task13_credit_diagnostics.md)；
- [MCH-PPO](../../experiments/air_defense_v1_mch_ppo_mechanism_stress_test.md)；
- [RG-MCH-PPO](../../experiments/air_defense_v1_rg_mch_ppo_stress_test.md)；
- [BPCE-PPO](../../experiments/air_defense_v1_bpce_ppo_stress_test.md)；
- [动作替代独立确认](../../experiments/air_defense_v1_action_substitution_confirmation.md)。

必须完成五层差异表：

| 层 | CAPO/COSAC | MCH/BPCE | 剩余差异是否充分 |
| --- | --- | --- | --- |
| Problem |  |  |  |
| Advantage |  |  |  |
| Counterfactual |  |  |  |
| Constraints |  |  |  |
| Evidence |  |  |  |

必须回答：

- 现有 MCH 创新叙事被该论文覆盖到什么程度？
- 项目动态合法集和目标占用是否产生实质新数学问题？
- SeqAU 是否能避开冻结 Critic 的 OOD 支持问题？
- 它是否处理安全—资源多约束，还是只处理团队回报信用？
- 其非加性偏差界能否容纳 AirDefense 的交互回报？

## 7. 交付物

```text
docs/literature/algorithm_innovation_reading/lr_05_capo_sequential_counterfactual_credit.md
```

必须包含：

- 版本核验记录；
- SeqAU/CAPO/COSAC 公式卡；
- 与 COMA/HAPPO/MCH/BPCE 的差异矩阵；
- 理论假设适用性审计；
- 对现有第一算法候选的保留/重写/放弃建议；
- `BASELINE / ADAPT / AVOID / OPEN` 判决。

## 8. 通过条件

- 能从公式解释为什么共享 baseline、HA 类重要性采样和 SeqAU 方差不同；
- 能指出 additive reward approximation 的风险；
- 能明确动态 mask 是否只是工程差异；
- 不利用预印本的新颖性声明代替独立查新；
- 形成对 MCH/BPCE 创新边界的明确判决。

## 9. 禁止结论

- 不把 CAPO/COSAC 当作已同行评审事实；
- 不因为其 benchmark 是顺序 bandit/LLM 就忽视公式重合；
- 不把算法改名视为创新差异；
- 不直接实现或训练其算法；
- 不选择项目中的正结果种子与论文比较。

## 10. 移交

结果移交 LR-06，用于判断即使采用新的顺序信用估计，离线/批内模型接入在线策略
时还需要哪些分布漂移保护。

## 11. 执行记录

- [x] 核验 arXiv v1/v2 日期、标题和算法名称变化；
- [x] 以当前 v2 为主版本，核对正文、算法和全部理论/实验附录；
- [x] 重建 sequential learnability、SeqAU、upstream cancellation、
  direct/indirect advantage、bias 和 variance 公式；
- [x] 区分虚拟策略后缀、环境快照反事实 rollout 和解释性因果效应；
- [x] 完成 COMA/HAPPO/MCH/BPCE 差异矩阵；
- [x] 对 additive reward、coverage、自然更新顺序和 contextual-bandit
  假设完成 AirDefense 适用性审计；
- [x] 发现完整 one-hot Gram 的结构奇异性并收窄理论引用边界；
- [x] 给出现有第一算法候选的保留、重写和放弃建议；
- [x] 给出 `BASELINE / ADAPT / AVOID / OPEN` 判决；
- [x] 未修改或运行任何算法、环境和训练实验。
