# 下一研究阶段：MCH-PPO 机制压力实验

更新时间：2026-07-22  
任务状态：已完成；机制门控失败，不进入 30k/100k

## 1. 任务目标

本任务不再增加新的外围前置诊断，直接实现最小可训练版本的 Mask-aware Counterfactual Hierarchical PPO（MCH-PPO），并在预先冻结的困难场景和失败种子上开展机制压力实验。

本阶段只回答一个问题：在完全相同的因子化策略结构下，引入掩码感知的反事实分层信用与交战/目标两层独立近端更新，能否减少 all-noop 与高成本交战分叉，并改善困难场景中的高威胁突防和资源效率。

该实验属于候选机制验证，不直接证明跨场景普遍优越性，也不作为论文最终主结果。

## 2. 冻结实验协议

为避免根据 MCH-PPO 结果事后选择有利样本，本任务在训练前冻结以下协议：

- 对照方法：`factorized_engagement_ar_ppo_order_012`；
- 候选方法：`mch_ppo_order_012`；
- 训练与测试场景：`time_pressure`、`heterogeneity_pressure`；
- 训练种子：`8、9、10`，覆盖既有实验中的低交战、高交战和不稳定分支；
- 单次筛选训练预算：`10k` steps；
- 每轮 rollout 的 PPO 更新轮数：`2`，两种方法完全一致；
- 最终确定性评估：每个模型、每个场景 `30` episodes；
- 策略网络、价值网络、动作顺序、PPO 超参数和环境配置保持一致；
- MCH-PPO 使用任务十四冻结的三个层级 Q-Critic 检查点 `seed14/15/16`，训练期间不更新 Critic。

不得仅报告表现最好的单个种子。所有冻结种子都必须进入配对汇总。

## 3. 最小 MCH-PPO 定义

策略保持既有两层分解：

```text
pi(a_i | s, h_i)
= pi_e(z_i | s, h_i)
  * pi_t(y_i | z_i=1, s, h_i)
```

对每个单元和实际动作前缀，冻结 Q-Critic 集成估计：

```text
A_i^engage(z_i)
= Q_e(z_i) - sum_z pi_e(z | s,h_i) Q_e(z)

A_i^target(y_i)
= Q_t(y_i) - sum_{y in L_i} pi_t(y | s,h_i) Q_t(y)
```

其中 `L_i` 由环境基础掩码和实际动作前缀占用共同确定。无合法目标时不产生交战层更新；选择 no-op 时不产生目标层更新。

交战层和目标层分别计算新旧策略概率比并独立裁剪：

```text
L_actor = L_clip_engage + L_clip_target
```

联合 GAE 继续用于训练状态价值函数，但不代替上述分层反事实 actor advantage。联合策略 KL 仍用于早停和稳定性监控。

## 4. 实现内容

1. 实现 `MaskedCounterfactualHierarchicalPPO`；
2. 冻结旧策略副本，精确计算 rollout 策略的两层旧 log-prob；
3. 接入层级 Q-Critic 集成与逐检查点反归一化；
4. 根据实际自回归前缀构造动态合法目标集合和占用向量；
5. 对交战层、目标层分别进行 advantage 标准化、ratio 计算和 PPO clipping；
6. 记录两层策略损失、clip fraction、有效样本率和联合 KL；
7. 接入统一 trainer、模型保存/加载和评估接口；
8. 运行因子化 PPO 与 MCH-PPO 的冻结种子配对实验；
9. 输出配置、逐运行指标、配对差值和门控结论。

## 5. 验收标准

### 5.1 软件验收

- MCH-PPO 能在 AirDefenseResourceAssignmentEnv v1.0 上完成训练；
- 交战与目标 log-prob 在首次更新前的新旧比值为 1；
- 非法目标和前缀已占用目标不进入反事实 baseline；
- no-op 样本不产生目标层策略梯度；
- Critic 参数在 PPO 训练期间保持冻结；
- 模型可以保存、加载并执行掩码推理；
- 新增测试和现有测试通过。

### 5.2 机制验收

候选进入后续正式 `30k/100k` 实验需同时满足：

- 两个困难场景均无结构违规：invalid action、assignment conflict、overkill 均为 0；
- 每个场景至少 `2/3` 个配对种子的 all-noop episode rate 不劣于对照；
- 六个候选场景种子中不得出现 all-noop episode rate `>= 0.98` 或 actionable engagement rate `< 0.01` 的绝对塌缩；
- 至少一个困难场景的高威胁突防率均值下降；
- 资源成本不得以超过 `10%` 的代价换取微小交战率提升；
- 平均奖励或总损伤不得出现灾难性退化；
- 结果必须报告全部三个种子及方差，不能以单个优势种子替代总体结论。

若门控失败，结论应是当前冻结 Critic 或独立层级更新机制不成立，而不是继续挑选种子证明优势。

## 6. 产物

```text
docs/task_guides/next_research_phase_mch_ppo_mechanism_stress_test.md
rein_learning/algorithms/policy_gradient/mch_ppo.py
scripts/run_air_defense_v1_mch_ppo_stress_test.py
tests/test_mch_ppo.py
results/air_defense_v1/mch_ppo_mechanism_stress_test/
docs/experiments/air_defense_v1_mch_ppo_mechanism_stress_test.md
```

## 7. 后续计划

若本轮通过机制门控，下一步扩大为 `30k x 5 seeds` 并增加 `medium` 与跨场景泛化矩阵，随后冻结论文主实验协议。若失败，优先根据两层 loss、clip 和信用符号诊断定位是 Critic 外推失效还是独立 clipping 失效，不回到无休止的外围前置任务。

## 8. 执行结论

已完成 `10k x 3 seeds x 2 train scenarios x 2 methods`，共训练12个模型并执行24个场景评估块。候选保持结构违规为零，但出现3/6个绝对塌缩场景种子；两个核心场景的高威胁突防率和损伤均值均退化，奖励/损伤安全门控失败，总机制门控为 false。

`time_pressure/seed9` 是唯一明显正向配对，但其他种子不复现，不能单独用于证明 MCH-PPO 优势。详细结果见[正式实验报告](../experiments/air_defense_v1_mch_ppo_mechanism_stress_test.md)。
