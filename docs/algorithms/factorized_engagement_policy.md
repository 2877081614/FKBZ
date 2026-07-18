# 交战-目标因子化自回归策略

更新时间：2026-07-18  
对应阶段：任务十二  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`

## 1. 研究动机

任务十一将每个单元的所有目标和 no-op 放入同一个 categorical 分布。即使所有目标的总交战概率较高，概率也会被多个目标分摊，单个 no-op 仍可能成为最大概率动作。确定性 argmax 因此可能把健康的随机交战概率放大为 all-no-op。

任务十二保持环境、奖励、PPO 超参数、Critic、关系 scorer 输入语义和单元顺序 `012` 不变，只分离两个决策：

1. 当前单元是否交战；
2. 决定交战后选择哪个合法目标。

## 2. 概率定义

对单元 `i`，Actor 输出一个交战 logit `e_i` 和每个目标的关系 logit `z_ij`：

```text
p_engage_i = sigmoid(e_i)
p(no-op) = 1 - p_engage_i
p(target_j) = p_engage_i * softmax(z_ij over legal targets)
```

若不存在合法目标，则强制 `p(no-op)=1`。前序单元选择目标后，后续单元的条件目标 softmax 会移除已分配目标，从而保持无冲突联合动作。

联合 log-prob 为各条件动作 log-prob 之和。最终离散动作熵按完整概率精确计算，等价于：

```text
H(A_i) = H(Bernoulli(p_engage_i))
         + p_engage_i * H(Target_i | engage)
```

## 3. 确定性规则

确定性推理不再对最终 `num_targets + 1` 个动作直接取 argmax，而是：

```text
p_engage_i >= 0.5 -> 选择合法目标分布的 argmax
p_engage_i <  0.5 -> no-op
```

随机采样仍严格服从最终离散动作概率。该规则显式避免“总交战概率大于 no-op，但被多个目标分摊后每个目标都小于 no-op”的概率碎片化问题。

## 4. 工程实现

核心文件：

```text
rein_learning/models/factorized_engagement_action_head.py
rein_learning/algorithms/policy_gradient/factorized_engagement_ppo.py
rein_learning/common/policy_probe.py
rein_learning/common/ppo_training_diagnostics.py
```

统一方法名：

```text
factorized_engagement_ar_ppo_order_012
```

模型签名记录概率公式、精确熵和分层确定性规则。Actor 为 34,946 参数，与任务十一对照完全相同；Critic 均为 34,945 参数且结构不变。

## 5. 固定策略探针

探针语料包含三个核心场景各 256 个状态，共 768 个状态。状态来源覆盖 Hungarian、任务十 order `012` 和任务十一 role order `012`，并按初始、早期、中期和临近终止四阶段分层。

```text
SHA-256:
2a58e44b9054945fa8e12bc44c8956d43d898b68b2571b3dcf4d237b000be4ef
```

训练期间记录交战概率、no-op 概率、no-op margin、交战熵、条件目标熵、确定性交战率和 Critic value，同时记录 PPO loss、KL、clip fraction、advantage 和梯度范数。

## 6. 验证结果

- 概率归一化、手工 log-prob、精确熵和无合法目标强制 no-op 测试通过；
- 重复目标动作会被自回归掩码拒绝；
- 保存加载和模型签名检查通过；
- schema 8 统一实验、训练动态和探针动态闭环通过；
- `pytest tests -q` 共 148 项测试通过；
- 30k × 3 seeds 正式筛选中，非法动作、冲突和过度分配均为 0。

但该策略未消除种子级塌缩，且交战强度在不同种子之间出现“不开火”和“高成本开火”两极分化，因此不能进入 100k 确认。

## 7. 学术结论

因子化能够明确区分交战和目标匹配，并显著降低异质场景高威胁泄漏中的未分配占比，但仅改变动作分布不足以稳定 PPO。剩余瓶颈位于交战概率校准及 Actor-Critic 优化动态，而不是关系 scorer 是否能够表达目标匹配。
