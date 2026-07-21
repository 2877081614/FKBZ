# 下一研究阶段：安全临界状态与类别平衡交战估值

更新时间：2026-07-21  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`  
阶段编号：任务十四·交战判别修订  
阶段状态：已完成；正类功效和总体判别通过，资源约束边界未通过  
阶段主题：定向状态采样、稀有 engage 功效与成对符号学习

## 1. 阶段定位

上一阶段证明风险/约束效用在独立 test 上具有正向信号：balanced accuracy 从 `0.713` 提高到 `0.926`，false-noop 从 `0.333` 降为 `0`，wasteful-engage 从 `0.241` 降为 `0.148`。但57个可靠 test 组只有3个 `oracle_engage`，连续回归 Critic 三种子全部失败。

因此本阶段不再修改效用公式，不再均匀增加随机状态，而是解决两个已定位的问题：

1. 训练与测试中必要交战样本过少；
2. 绝对效用回归被多数 no-op 和状态共同价值主导。

## 2. 核心研究问题

> 使用不依赖反事实结果的安全临界度定向采样，并将 engagement 学习改为类别平衡的成对符号目标后，能否在全新独立状态上同时降低 false-noop 与 wasteful-engage？

该阶段只验证离线 engagement estimator，不训练 PPO Actor，不实现 GNN。

## 3. 冻结内容

冻结上一阶段的：

- 风险效用：`cost=2.0, damage=30.0, high=0.0, beta=0.5, alpha=0.25`；
- 安全-资源 oracle：`harm=30*damage+20*high_threat_leaks`；
- 90% 成对置信规则和 ambiguous 定义；
- 每分支32次共同随机数 rollout；
- factorized policy seeds 8/10、顺序 `012` 和三个核心场景；
- AirDefense v1.0 环境、奖励和 conditional-target 层；
- 模型输入、隐藏层宽度、优化器范围与训练种子数量；
- 所有既有正式 test、门槛和结果。

不得根据本轮 test 修改临界度公式、类别权重、margin 或门槛。

## 4. 安全临界度

定向采样只能使用决策前可观测信息。对每个合法“单元-目标”关系：

```text
base_risk = target_damage_potential
            * hit_probability
            * (1 + target_threat)

urgency = 1 + 5 / (1 + time_to_impact)

criticality = base_risk * urgency
```

状态临界度取所有合法关系的最大值，同时记录最小到达时间、最大威胁、最大毁伤潜力和合法关系数。

每个“source seed × scenario”先收集候选池，再按以下规则选择24个状态：

- 80%来自临界度最高且同回合间隔不少于3步的状态；
- 20%从其余候选中按时间位置和临界度分布补充；
- 选择过程不读取 rollout 回报、oracle 或未来泄漏结果。

## 5. 数据协议

正式新增：

```text
2 source policies * 3 scenarios * 24 states = 144 states
```

新数据按 state 分层划分 `40% train / 20% validation / 40% test`。历史任务十四交战效用数据中：

- 原 train 和 validation 可加入训练语料；
- 原 test 及更早所有正式 test 全部排除；
- 新 test 与全部旧观测交集必须为0。

主门控只使用新 targeted test。历史 test 只允许在方法冻结后做附加泛化审计，不参与选择。

## 6. 模型对照

所有方法使用同一个 `RiskAwareEngagementCritic`，输出 `[z_noop,z_engage]`。

### 6.1 连续回归基线

`risk_regression`：继续回归冻结风险效用的两个绝对值，并使用组内中心化损失。

### 6.2 类别平衡 BCE

对可靠 oracle 组：

```text
logit = z_engage - z_noop
target = oracle_engage
```

正负类别分别赋予总权重 `0.5`，ambiguous 不参与分类损失。

### 6.3 类别平衡 BCE + margin

在 BCE 基础上加入：

```text
L_margin = weight * max(0, 1 - y_sign * logit)
```

用于直接扩大可靠 engage/no-op 的符号间隔。margin 权重冻结为 `0.5`。

## 7. 正式门槛

### 7.1 数据与功效

- 新增状态正好144个，每分支32 rollout；
- 新 test 与所有旧观测重叠为0；
- 同一状态不跨 split；
- 总回报分量重构误差不超过 `1e-4`；
- 新 test 可靠组不少于40；
- oracle engage 与 no-op 各不少于8；
- 每场景可靠组不少于8；
- 至少两个场景包含 oracle engage；
- targeted test 的 engage 比例相对上一阶段 `3/57` 提高至少 `0.03`。

### 7.2 模型门槛

- balanced accuracy 不低于 `0.70`；
- 相对同种子 `risk_regression` 提高至少 `0.10`；
- false-noop rate 不高于回归基线；
- wasteful-engage rate 不高于回归基线；
- 每场景 no-op recall 不低于 `0.65`；
- 对含 engage 的场景，engage recall 不低于 `0.60`；
- 三个训练种子至少两个整体通过；
- 推理耗时低于32-rollout Monte Carlo。

## 8. 决策规则

```text
定向采样是否达到正类功效？
├─ 否：审查任务场景是否缺少真正的交战分水岭，不训练 PPO
└─ 是
   └─ 类别平衡方法是否至少2/3 seeds通过？
      ├─ 是：冻结 engagement sign head，进入最小 MCH-PPO 30k
      └─ 否
         ├─ 标签可分但模型失败：进入分位数/分布式 Critic
         └─ 标签本身不可分：重新审查 oracle 与场景生成机制
```

任一失败分支均不自动进入 GNN。

## 9. 交付物

```text
docs/task_guides/next_research_phase_critical_state_balanced_engagement.md
rein_learning/common/critical_engagement_sampling.py
rein_learning/common/balanced_engagement_training.py
scripts/run_air_defense_v1_task14_balanced_engagement.py
tests/test_air_defense_v1_task14_balanced_engagement.py
docs/algorithms/balanced_engagement_sign_critic.md
docs/experiments/air_defense_v1_task14_balanced_engagement.md
results/air_defense_v1/task14_balanced_engagement/
```

## 10. 阶段完成定义

满足任一条件即可完成：

- 功效与至少2/3模型种子同时通过，解锁最小 MCH-PPO；
- 功效通过但分类模型失败，形成估值结构的可靠负结果；
- 定向采样仍无法获得足够 engage，形成环境/场景分水岭不足的证据；
- 两类错误不能同时非劣，停止 PPO 集成并重新审查目标语义。

## 11. 执行结果

正式实验完成144个全新 targeted 状态、196个上下文组和每分支32次 rollout。新 test 81组中74组具有可靠 oracle，包含28个 engage 和46个 no-op；engage 比例从历史随机 test 的 `5.3%` 提高到 `37.8%`，三个场景分别有 `9/11/8` 个 engage。全部数据与功效门槛通过。

validation 平均 balanced accuracy 为 `balanced_bce=0.695`、`balanced_bce_margin=0.721`，因此冻结 margin 候选。正式 test 三种子 BA 为 `0.758 / 0.711 / 0.708`，相对风险回归提高 `0.146 / 0.128 / 0.125`；false-noop 从 `0.429-0.464` 降到 `0.071-0.214`。

候选未控制高成本交战。seed20/21 wasteful-engage 分别由 `0.348/0.370` 恶化到 `0.413/0.435`；`time_pressure` no-op recall 只有 `0.455 / 0.182 / 0.273`。三个种子均因逐场景 no-op recall 失败，整体通过数 `0/3`。

因此不恢复 MCH-PPO，不进入 GNN。下一入口为对 oracle 监督后 engagement logit 的资源约束阈值/对偶校准，而不是继续增加表示容量。详见[正式实验报告](../experiments/air_defense_v1_task14_balanced_engagement.md)。
