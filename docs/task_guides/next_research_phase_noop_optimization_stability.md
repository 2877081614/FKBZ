# 下一研究阶段：no-op 塌缩机理与 PPO 优化稳定性

更新时间：2026-07-17  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`  
阶段编号：任务十二  
阶段状态：已完成，30k 筛选未通过，未运行 100k  
阶段主题：训练动态诊断、交战决策因子化与低交战吸引域消除

## 1. 阶段背景

任务十一已经实现参数匹配的角色条件关系动作头，并完成三顺序 30k × 3 种子筛选。工程与实验得到以下证据：

- Actor 参数为 34,946，相对任务十减少 5.90%，Critic 完全不变；
- 单元和目标置换等变性、动作掩码、保存加载和结构测试全部通过；
- 非法动作、联合分配冲突和过度分配均严格为 0；
- 主方法相对任务十改善了奖励、毁伤和资源成本；
- 主方法异质高威胁泄漏没有改善，反而增加 0.00335，仅 1/3 种子同向；
- 主方法三个种子累计出现 5 个塌缩单元，三顺序合计出现 14 个；
- 主方法种子 1 的三个单元全部选择 no-op，异质场景资源成本为 0；
- 主方法异质高威胁泄漏中 94.8% 属于 `unassigned`；
- 实际发生分配时，三个单元的匹配效率约为 0.939、0.967 和 0.946；
- 三顺序鲁棒性和决策耗时门槛均未通过，因此没有运行 100k。

这些结果说明共享资源-目标关系 scorer 已能找到较好的目标，但 PPO 仍会在部分训练种子中进入低交战或 all-no-op 吸引域。当前瓶颈不是“选哪个目标”，而是“是否选择交战”。

任务十二先诊断 no-op 概率在训练中何时、为何塌缩，再检验一个不修改环境奖励的交战-目标因子化策略。

## 2. 核心研究问题

本阶段回答：

1. all-no-op 是训练早期形成、后期形成，还是确定性评估才暴露？
2. no-op 塌缩是否伴随策略熵骤降、KL/clip 异常、advantage 符号偏移或 Critic 失真？
3. 相同模型在随机采样评估下是否仍低交战？
4. no-op logit 相对最佳合法目标 logit 的差距如何随训练演化？
5. 将“是否交战”和“交战后选哪个目标”显式因子化，能否消除低交战吸引域？
6. 消除塌缩后，能否同时守住高威胁泄漏、资源成本、毁伤和计算耗时？

## 3. 研究假设

### H1：no-op 是独立的优化吸引域

角色条件关系头对已选择目标具有较高匹配效率，但共享 no-op logit 与全部目标 logits 在同一个 categorical 中竞争。PPO 可能通过提高 no-op 相对优势快速降低风险和方差，随后因交战样本不足而难以恢复。

### H2：训练分叉早于最终性能分叉

成功种子和塌缩种子应在训练中期之前出现可检测差异，例如交战概率、no-op margin、策略熵、梯度范数或 Critic explained variance 分离。

### H3：确定性 argmax 会放大边缘 no-op 偏置

若 no-op 仅略高于最佳目标，确定性评估会表现为全 no-op，而随机策略仍可能保持交战。确定性与随机评估差异必须单独量化。

### H4：交战-目标因子化可以改善优化

显式建模二元交战概率，再在合法目标中条件选择目标，可将“是否交战”的梯度从多个目标 logits 的竞争中分离，降低 no-op 塌缩概率。

### H5：稳定化不能依赖无约束开火

消除 no-op 塌缩不能以大幅增加射击和资源成本为代价。候选策略必须同时通过资源与任务性能门槛。

## 4. 阶段边界

### 4.1 允许修改

- 训练回调和 PPO 训练动态记录；
- 固定策略探针状态集及其生成脚本；
- no-op/交战概率、logit margin、熵、KL、clip fraction、loss 和梯度诊断；
- 角色条件 Actor 中 no-op 的概率参数化；
- 自回归分布的交战-目标因子化；
- 模型签名、统一实验 schema 和诊断产物；
- 为降低决策时延所做的等价向量化、缓存和批处理优化；
- 任务十一冻结模型的只读回放。

### 4.2 必须保持不变

- AirDefense v1.0 状态、观察、转移、命中、毁伤与终止条件；
- 奖励函数及所有奖励权重；
- 三个核心场景和高威胁阈值 0.8；
- 角色条件 unit-target pair scorer 的输入语义；
- `MultiDiscrete([6,6,6])` 环境动作语义；
- 自回归重复目标掩码；
- PPO 学习率、batch、rollout、折扣、GAE 和 clip 超参数；
- Critic 结构；
- 最终评估种子公式和任务十/十一诊断指标。

### 4.3 暂缓内容

- GNN、图注意力和 Transformer；
- 固定顺序或动态顺序的新一轮优化；
- 奖励塑形、no-op 惩罚或资源成本权重调整；
- curriculum、模仿学习或规则策略预训练；
- AirDefense v1.5 和多智能体环境；
- 未完成诊断就扩大到 100k；
- 根据最终结果事后挑选有利评估模式。

## 5. 冻结诊断定义

### 5.1 交战与塌缩

| 指标 | 定义 |
| --- | --- |
| `engagement_rate` | 非 no-op 单元动作数 / 全部单元决策数 |
| `actionable_engagement_rate` | 存在合法目标时的非 no-op 数 / 可行动决策数 |
| `all_noop_episode` | 回合存在至少一个合法交战机会，但整回合射击数为 0 |
| `all_noop_episode_rate` | all-no-op 回合数 / 评估回合数 |
| `collapsed_unit` | 至少 100 个可行动决策且分配率低于 1% |
| `collapsed_seed` | 指定场景下至少存在一个 collapsed unit 的训练种子 |

### 5.2 概率与 logit

| 指标 | 定义 |
| --- | --- |
| `engage_probability` | 当前条件掩码下非 no-op 总概率 |
| `noop_probability` | 当前条件掩码下 no-op 概率 |
| `noop_margin` | no-op logit - 最大合法目标 logit |
| `deterministic_engagement` | argmax 动作是否为非 no-op |
| `stochastic_engagement_gap` | 随机评估交战率 - 确定性评估交战率 |
| `conditional_target_entropy` | 已决定交战后合法目标分布的熵 |
| `engagement_entropy` | 二元交战分布的熵 |

### 5.3 PPO 优化动态

每个训练检查点至少记录：

```text
timesteps, seed, method,
policy_loss, value_loss, entropy_loss,
approx_kl, clip_fraction, explained_variance,
advantage_mean, advantage_std,
positive_advantage_rate,
actor_gradient_norm, critic_gradient_norm,
engage_probability_mean,
noop_probability_mean,
noop_margin_mean,
deterministic_engagement_rate,
probe_value_mean
```

指标必须区分 rollout 统计、固定探针统计和最终评估统计，不能混用分母。

## 6. 固定策略探针集

### 6.1 目的

训练中的环境状态不断变化，直接比较 rollout 均值会把策略变化与状态分布变化混在一起。任务十二新增固定 probe corpus，在每个检查点对同一组观察和 action masks 计算概率与 value。

### 6.2 生成协议

探针在任何候选方法训练前一次性生成并冻结：

```text
场景：medium / time_pressure / heterogeneity_pressure
来源策略：Hungarian / task10 order_012 / task11 role order_012
环境种子：40,000 起的固定不重叠区间
状态数：每场景至少 256，总数至少 768
采样层次：初始、早期、中期、临近突防状态均覆盖
过滤：至少一个单元存在合法目标
```

输出：

```text
probe_states.npz
probe_manifest.json
probe_summary.csv
```

manifest 记录场景、策略、环境种子、观察布局、样本数和 SHA-256。候选方法不得修改探针。

## 7. 候选概率参数化

### 7.1 现有联合 categorical

任务十一每个单元直接在 `num_targets + 1` 个动作上形成 categorical：

```text
pi(a_i | s, prefix) = Categorical(target_0, ..., target_n, no-op)
```

### 7.2 交战-目标因子化

任务十二候选方法：

```text
p_engage_i = sigmoid(engage_logit_i)
p(no-op) = 1 - p_engage_i
p(target_j) = p_engage_i * softmax(target_logits over legal targets)_j
```

若没有合法目标，必须强制 `p(no-op)=1`。前序单元选择目标后，后续目标 softmax 使用更新后的条件掩码。

联合 log-prob：

```text
no-op:
log pi(a_i) = log(1 - p_engage_i)

target j:
log pi(a_i=j) = log(p_engage_i) + log pi_target(j | engage, prefix)

joint:
log pi(a|s) = sum_i log pi(a_i | s, prefix)
```

熵必须按最终离散动作分布精确计算，或使用等价分解：

```text
H(A_i) = H(Bernoulli(p_engage_i))
         + p_engage_i * H(Target_i | engage)
```

候选名称冻结为：

```text
factorized_engagement_ar_ppo_order_012
```

首轮只使用规范顺序 `012`，不同时引入顺序调度。

### 7.3 初始化与参数量

- 初始 `engage_logit` bias 固定为 0，即初始 `p_engage=0.5`；
- 不增加额外 engagement entropy 权重；
- 不修改 PPO 总 entropy 系数；
- Actor 参数量相对任务十一增减不超过 10%；
- Critic 参数量严格不变；
- 模型签名记录因子化公式和初始化先验。

## 8. 工作任务

### 任务十二·一：冻结探针与训练动态 schema

工作内容：

- 固定指标、分母和检查点；
- 固定 probe corpus 生成协议；
- 固定诊断、筛选和确认种子；
- 将任务十一结果目录标记为只读参考。

验收标准：

- 候选训练前生成 manifest 和哈希；
- 概率、logit、rollout 和 episode 指标来源明确；
- 不根据诊断结果回改主要筛选门槛。

### 任务十二·二：实现策略探针与训练动态记录

工作内容：

- 采集固定观察、mask 和状态元数据；
- 在指定检查点运行无环境推进的批量策略探针；
- 记录 PPO logger、advantage 和梯度统计；
- 输出 `training_dynamics.csv` 和 `probe_dynamics.csv`。

验收标准：

- 同一模型重复探针结果逐元素一致；
- 探针不参与梯度和训练；
- CPU/GPU 结果在数值误差内一致；
- 不显著改变训练采样轨迹。

### 任务十二·三：冻结任务十一模型诊断回放

加载任务十一 `012` 的种子 0/1/2 模型，执行：

```text
场景：三个核心场景
回合：100 个相同环境种子/模型/场景
评估：deterministic 与 stochastic 各一组
探针：同一 frozen probe corpus
```

验收标准：

- 重现种子 1 的 all-no-op；
- 报告确定性/随机交战差异；
- 给出 no-op margin 和 engage probability 分布；
- 明确塌缩是评估放大还是策略概率本身趋近 no-op。

### 任务十二·四：运行 10k 训练分叉诊断

```text
方法：任务十一 role-conditioned order_012
训练场景：medium
训练种子：3 / 4 / 5 / 6 / 7
训练步数：10,000
检查点：每 1,000 步
曲线评估：10 回合/checkpoint
```

验收标准：

- 至少定位首次稳定 no-op 分叉的检查点；
- 比较成功/塌缩种子的 entropy、KL、value、advantage 和 gradient；
- 若 10k 尚未形成分叉，只允许延长到 20k，不直接扩大到 100k；
- 形成可检验的塌缩机理结论。

### 任务十二·五：实现交战-目标因子化分布

工作内容：

- 实现二元 engagement 与条件 target 分布；
- 支持采样、确定性预测、动作回放、entropy 和 log-prob；
- 保持环境动作索引和自回归重复目标掩码；
- 保存概率分解和初始化签名；
- 记录参数量。

验收标准：

- 最终动作概率和为 1；
- 采样 log-prob 可由动作前缀精确重建；
- 无合法目标时 no-op 概率严格为 1；
- 重复目标、非法动作和空条件集合均不会出现；
- Actor 参数量满足 ±10%，Critic 不变。

### 任务十二·六：自动化测试与 Smoke Test

至少测试：

- engagement/target 概率归一化；
- no-op 与 target log-prob 手工公式；
- entropy 手工公式；
- 条件 mask 和目标去重；
- deterministic/stochastic 动作；
- probe corpus 哈希与重复性；
- 训练动态字段和检查点计数；
- 模型保存加载和签名拒绝；
- 任务十一模型回归；
- schema 8 产物。

Smoke 协议：

```text
方法：role-conditioned baseline / factorized engagement candidate
种子：8 / 9
训练：一个完整 rollout
评估：medium / heterogeneity_pressure，各 2 回合
```

验收标准：训练、保存、重新加载、探针和评估闭环正常；Smoke 不用于性能结论。

### 任务十二·七：运行 30k × 3 配对筛选

```text
训练场景：medium
测试场景：medium / time_pressure / heterogeneity_pressure
方法：role-conditioned order_012 / factorized engagement order_012
配对训练种子：8 / 9 / 10
训练步数：30,000
检查点：每 2,000 步；曲线 10k / 20k / 30k
最终评估：50 个成对回合/场景/种子
评估模式：deterministic 为主，stochastic 为诊断
```

任务十、任务十一正式结果继续作为冻结外部参考。

验收标准：

- 6 个模型、18 个场景运行块和 900 个主评估回合完整；
- baseline 与 candidate 使用相同训练和评估种子；
- 训练动态、probe、episode、decision 和 attribution 可关联；
- 报告成功与失败种子，不只报告均值。

### 任务十二·八：条件性 100k 确认

只有候选通过第 9 节全部门槛后运行：

```text
候选：factorized_engagement_ar_ppo_order_012
对照：role-conditioned order_012
训练种子：11 / 12 / 13 / 14 / 15
训练步数：100,000
最终评估：100 回合/场景/种子
```

确认种子与诊断、筛选种子严格分离。

## 9. 30k 筛选门槛

候选相对同轮配对 role-conditioned baseline 必须满足：

| 指标 | 门槛 |
| --- | --- |
| 非法动作、冲突、过度分配 | 全部严格为 0 |
| Actor 参数量 | 相对 baseline 在 ±10% 内 |
| 三核心场景 collapsed seed | 全部为 0 |
| all-no-op episode rate | 每个场景不超过 2% |
| heterogeneity `unassigned` 泄漏占比 | 绝对下降至少 0.15 |
| heterogeneity 高威胁泄漏率 | 平均下降至少 0.02，至少 2/3 种子同向 |
| medium 奖励 | 下降不超过 5 |
| medium 总毁伤 | 增加不超过 0.10 |
| time_pressure 资源成本 | 增加不超过 0.50 |
| heterogeneity 总毁伤 | 增加不超过 0.10 |
| 随机-确定性交战率差 | 不超过 0.05 |
| 决策耗时 | 相对任务十 `order_012` 增加不超过 25% |

外部非劣效仍需满足：

- 相对原始 Maskable PPO，medium/time_pressure 奖励下降不超过 5；
- medium/heterogeneity 总毁伤增加不超过 0.10；
- time_pressure 资源成本增加不超过 0.50；
- time_pressure 资源成本低于 `Discrete(136)`。

通过筛选不等于显著性结论，只用于决定是否运行独立种子 100k。

## 10. 决策规则

```text
冻结模型与 10k 诊断能否定位塌缩时间和概率机制？
├─ 否：补充 probe/梯度记录，不实现更多结构
└─ 是
   ├─ 塌缩只出现在 deterministic，stochastic 概率仍健康
   │  └─ 研究决策规则校准；不修改环境奖励
   ├─ engage probability 在训练中趋近 0，伴随熵/advantage 分叉
   │  └─ 运行交战-目标因子化筛选
   ├─ factorized 方法消除塌缩并守住全部性能/成本/耗时门槛
   │  └─ 使用独立种子运行 100k 确认
   ├─ factorized 方法消除塌缩，但成本或毁伤退化
   │  └─ 研究交战概率校准，不进入 GNN
   ├─ factorized 方法仍出现 no-op 塌缩
   │  └─ 瓶颈位于 PPO 信用分配/优化，研究 critic 或 advantage 稳定性
   └─ 塌缩消失后目标匹配仍是主要失败来源
      └─ 才重新评估 GNN/图注意力进入条件
```

## 11. 阶段总验收标准

### 工程验收

- 固定 probe corpus、manifest 和哈希可复现；
- PPO 动态、概率、logit、梯度和 value 诊断完整；
- factorized 分布概率、entropy 和 log-prob 有独立测试；
- 模型签名、参数量和 schema 8 可复现；
- 旧模型和历史结果兼容；
- `tests/` 全量通过。

### 实验验收

- 任务十一冻结模型 deterministic/stochastic 回放完成；
- 10k 多种子训练分叉诊断完成；
- candidate Smoke 完成；
- 30k × 3 配对筛选完成；
- 根据冻结门槛决定是否运行 100k；
- 失败种子和塌缩时间点完整留档。

### 学术验收

- 能区分评估 argmax 放大与真实概率塌缩；
- 能解释 no-op 吸引域与 PPO 训练动态之间的关系；
- 能验证交战因子化是否改善优化，而不是依赖奖励惩罚；
- 能区分“是否交战”与“交战后目标匹配”两个瓶颈；
- 为后续优化稳定性、顺序机制或 GNN 分支提供明确证据。

## 12. 预期交付物

```text
rein_learning/common/policy_probe.py
rein_learning/common/ppo_training_diagnostics.py
rein_learning/models/factorized_engagement_action_head.py
rein_learning/algorithms/policy_gradient/factorized_engagement_ppo.py
rein_learning/experiments/air_defense_v1_benchmark.py
scripts/build_air_defense_v1_probe_corpus.py
scripts/diagnose_air_defense_v1_noop_collapse.py
scripts/analyze_air_defense_v1_task12.py
tests/test_policy_probe.py
tests/test_factorized_engagement_distribution.py
tests/test_air_defense_v1_task12_experiments.py
docs/algorithms/factorized_engagement_policy.md
docs/experiments/air_defense_v1_task12_noop_stability.md
results/air_defense_v1/task12_*/
```

## 13. 本阶段完成后的研究位置

任务十二不是简单“修复 no-op”，而是把强化学习训练不稳定性转化为可观察、可复现和可干预的学术问题。

可能得到：

```text
结论 A：交战因子化消除塌缩并守住性能、成本和时延
        -> 独立 100k 确认，形成稳定动作机制主方法

结论 B：交战因子化消除塌缩，但更积极开火造成资源退化
        -> 研究概率校准和约束优化，不进入 GNN

结论 C：因子化后仍出现概率塌缩
        -> 研究 PPO critic、advantage 和优化稳定性

结论 D：塌缩消失且交战稳定后，关系匹配仍限制高威胁保护
        -> 重新开启 GNN/图注意力进入评估
```

只有在“是否交战”的优化问题被单独控制后，才能把剩余性能不足合理归因于关系表示能力。

## 14. 实际完成结果

任务十二已完成固定探针、冻结模型回放、10k × 5 seeds 分叉诊断、因子化策略实现、Smoke 和 30k × 3 seeds 配对筛选。

- 固定探针共 768 个状态，内容哈希已冻结；
- 冻结回放证明 Task 11 种子 1 的 all-noop 主要受到 deterministic argmax 放大；
- 10k 诊断中 5 个种子有 3 个塌缩，no-op margin 在早期发生分叉；
- 因子化候选保持 Actor/Critic 参数匹配及零非法动作、零冲突；
- 正式筛选完成 6 个模型、18 个场景块和 900 个主评估回合；
- 19 项门槛通过 6 项、失败 13 项；
- 候选仍存在种子 8 all-noop 和种子 10 高成本交战的两极分化；
- 按冻结规则不运行 100k。

任务十二的负结果具有明确价值：仅靠动作概率因子化不能解决 PPO 交战稳定性，下一阶段应转向交战概率校准、advantage/Critic 动态和约束优化，而不是直接进入 GNN。
