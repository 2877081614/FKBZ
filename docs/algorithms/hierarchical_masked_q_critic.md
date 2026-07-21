# 显式交战与目标分层 Q-Critic

更新时间：2026-07-20  
对应阶段：任务十四·分层诊断  
实现状态：离线原型与正式门控已完成

## 1. 方法目的

`HierarchicalMaskedQCritic` 将单标量动作价值拆成：

```text
Q_engage(s,h_i,e_i)             # 是否交战
Q_target(s,h_i,target | e_i=1) # 交战后选择目标
```

它用于检验第一创新假设中的层级信用是否能在接入 PPO 前离线成立。模型不更新 Actor，不使用图网络。

## 2. 交战价值

交战头一次输出 `[Q_noop, Q_engage]`：

```text
Q_noop = Q(s,h_i,no-op)

Q_engage
= sum_target pi(target | engage,s,h_i)
  * Q(s,h_i,target)
```

目标分支按冻结策略的条件目标概率加权，不使用最大目标回报。每个 rollout 内先加权再与 no-op 配对，使交战 advantage 与冻结策略执行语义一致。

交战头输入：

```text
observation
+ unit one-hot
+ unit entity features
+ prefix occupancy
+ conditional legal mask
-> MLP(256,128,Tanh)
-> [Q_noop,Q_engage]
```

只有至少存在一个合法目标的 actionable 组进入监督；只有 no-op 的组属于环境不可行动性，不属于交战偏好。

## 3. 目标价值

目标头复用完整 `MaskedActionQCritic` 输入与网络：

```text
observation + unit/target relation + prefix + mask
-> MLP(256,128,Tanh)
-> Q_target
```

目标头拒绝 no-op，只比较当前自回归前缀下的合法目标。

## 4. 训练目标

```text
L_engage = MSE(Qe) + centered_MSE(Qe)

L_target = MSE(Qt)
         + centered_MSE(Qt)
         + 0.5 * reliability_weighted_pairwise(Qt)

L_total = L_engage + L_target
```

两层使用独立的训练均值和标准差，避免目标样本数量与尺度主导交战损失。验证分数汇总两层的绝对与中心化 MAE，正式测试不参与早停。

## 5. 当前实验结论

108 个全新状态、32-rollout 正式实验中：

- 目标排序达到 `0.83-0.87`，相对单标量基线平均提高 `0.057`；
- 目标 top-1 达到 `0.75-0.875`；
- 目标 MAE 相对基线恶化约 `17%-21%`；
- engage 符号为 `0.588-0.706`，相对单标量基线平均下降 `0.255`；
- engage 有效组为 17，未达到 30。

显式拆头改善了目标判别，却没有改善交战信用。当前模型不能接入 MCH-PPO。下一研究问题应从“是否拆头”转向“均值回报是否足以表达资源成本、突防风险和交战效用”。

## 6. 实现入口

```text
rein_learning/models/hierarchical_masked_q_critic.py
rein_learning/common/hierarchical_q_diagnostics.py
scripts/run_air_defense_v1_task14_hierarchical_q.py
docs/experiments/air_defense_v1_task14_hierarchical_q.md
```
