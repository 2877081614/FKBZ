# 下一研究阶段：顺序式/自回归无冲突联合动作生成

更新时间：2026-07-17  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`  
阶段编号：任务九  
阶段状态：工程实现与 30k 筛选完成，100k 正式实验暂缓  
阶段主题：条件联合决策、资源效率恢复与动作生成机制消融

## 1. 阶段背景

任务八实现了 `Discrete(136)` 无冲突联合动作机制，并完成 30k × 3 种子筛选。该方法取得了以下结果：

- 非法动作率、联合分配冲突率和过度分配率严格降为 0；
- `medium/time_pressure/heterogeneity_pressure` 的平均奖励和毁伤均有改善趋势；
- `heterogeneity_pressure` 高威胁突防率平均下降约 0.064，且 2/3 种子同向；
- `time_pressure` 平均资源成本相对原始 Maskable PPO 增加 3.49；
- 资源成本增量超过预先冻结的 `+0.50` 门槛，因此未进入 100k × 5 种子正式实验；
- 三种子主要配对差异的 95% CI 均跨越 0，当前结果只能解释为筛选趋势。

任务八说明显式一对一约束可以消除动作协调错误，但将 136 个联合动作视为互不相关的离散类别，会改变探索和开火行为，并削弱任务七中观察到的资源效率优势。

## 2. 核心研究问题

本阶段回答：

> 在保持 AirDefense v1.0 环境、奖励函数、状态表示和 PPO 主体不变的条件下，按防御单元依次生成动作并实时屏蔽已分配目标，能否同时实现零冲突、稳定拦截和资源效率保持？

需要区分三种动作机制：

```text
独立动作头：各单元并行选择，可能重复分配目标
联合枚举：从 136 个无冲突联合动作中一次性选择
自回归生成：按条件概率逐单元选择，并动态排除已分配目标
```

本阶段只研究动作生成机制，不把状态关系表示、环境升级或奖励重构混入实验。

## 3. 研究假设

### H1：结构合法性

自回归条件掩码能够保证所有执行动作满足一对一目标分配，使非法动作、冲突和过度分配严格为 0。

### H2：资源效率

相较 `Discrete(136)` 联合枚举，自回归分解能够降低动作类别竞争和过度积极开火，使 `time_pressure` 资源成本恢复至原始 Maskable PPO 附近。

### H3：任务性能

资源效率恢复不能以明显增加毁伤或降低奖励为代价。

### H4：异质目标优先级

条件决策至少应保留任务八在 `heterogeneity_pressure` 中观察到的高威胁突防改善趋势。

## 4. 阶段边界

### 4.1 允许修改

- Actor 的联合动作生成方式；
- 每个防御单元动作头的条件掩码；
- 联合动作 `log_prob` 和 entropy 的计算；
- rollout 中动作、旧概率和条件前缀的记录；
- trainer、模型保存加载和统一实验方法注册；
- 动作机制签名、测试、实验配置和文档。

### 4.2 保持不变

- `AirDefenseResourceAssignmentEnv v1.0` 状态转移；
- 观测空间、特征顺序和归一化方式；
- 奖励函数及全部权重；
- 目标生成、运动、命中和毁伤机制；
- 共享 MLP 状态编码器的层数与宽度；
- Critic 的价值函数输入；
- PPO 的学习率、折扣因子、GAE、clip、epoch 等核心超参数；
- 训练与测试场景、种子协议和诊断指标定义。

### 4.3 暂缓内容

- GNN、图注意力、Transformer 状态编码器；
- MAPPO、HAPPO、HATRPO 等多智能体算法；
- 奖励塑形或增加资源成本惩罚；
- AirDefense v1.5 环境升级；
- 改变资源数、目标数或观测维度；
- 学习单元排序、随机排序和多种解码顺序的大规模消融；
- 将三个资源选择改成三个真实环境时间步。

不得通过环境执行前的动作修复替换冲突动作。不得为了改善资源成本而修改奖励权重。

## 5. 推荐算法设计

### 5.1 单环境步联合决策

三个单元的选择都在一次策略调用中完成，最后只调用一次 `env.step(joint_action)`：

```text
observation s_t
    ↓
共享 MLP 编码 h_t
    ↓
单元 0 动作头 + 基础合法掩码
    ↓ 选择 a_0，并屏蔽其已选目标
单元 1 动作头 + 条件掩码
    ↓ 选择 a_1，并继续更新掩码
单元 2 动作头 + 条件掩码
    ↓
joint_action = [a_0, a_1, a_2]
    ↓
一次 env.step(joint_action)
```

固定初始解码顺序为 `unit 0 -> unit 1 -> unit 2`。固定顺序可能带来顺序偏置，但初始实验中不同时引入排序学习；顺序偏置通过诊断指标记录，后续再决定是否消融。

### 5.2 最小自回归动作头

保持现有共享 MLP 状态编码器不变。每个单元动作头都从同一状态隐变量输出六个原始 logits：

```text
logits_i = actor_head_i(h_t)
```

前序动作先通过条件掩码影响后续分布。第一版不增加 GRU、Transformer 或新的关系编码器，从而把变量控制在“动作概率分解和条件合法集合”上。

### 5.3 条件掩码

设基础环境对单元 `i` 的动作掩码为 `base_mask_i`，前序单元已经选择的非 `no-op` 目标集合为 `A_<i`：

```text
conditional_mask_i(target) =
    base_mask_i(target) and target not in A_<i

conditional_mask_i(no-op) = base_mask_i(no-op)
```

每一步条件分布都至少保留 `no-op`，不得出现空动作集合。

### 5.4 联合概率与 PPO 更新

联合策略按固定顺序分解：

```text
pi(a|s)
= pi(a_0|s)
* pi(a_1|s,a_0)
* pi(a_2|s,a_0,a_1)
```

联合对数概率为：

```text
log pi(a|s) = sum_i log pi(a_i|s,a_<i)
```

PPO rollout 必须保存完整联合动作和采样时的联合旧 `log_prob`。更新阶段使用保存的动作前缀重建相同条件掩码，重新计算新 `log_prob`。不能在更新时重新采样前序动作。

条件 entropy 可按采样前缀上的逐头 entropy 求和，作为联合 entropy 的 Monte Carlo 估计；其定义和实现必须在测试与实验配置中留档。

### 5.5 环境接口和模型签名

环境仍接收长度为 3 的原始联合动作，Gym 动作空间保持 `MultiDiscrete([6,6,6])`。但模型与配置必须额外记录：

```text
action_generator: autoregressive_conflict_free
unit_order: [0, 1, 2]
conditional_target_mask: true
joint_log_prob: sum_of_conditional_log_probs
```

原始 Maskable PPO 和自回归模型虽然动作空间形状相同，但动作分布语义不同。模型加载时必须校验动作生成机制签名，不能只校验 Gym action space。

## 6. 工作任务

### 任务九一：冻结设计与实验协议

工作内容：

- 固定研究假设、方法名和比较对象；
- 固定单元顺序为 `[0,1,2]`；
- 固定最小自回归动作头，不增加新状态编码器；
- 冻结筛选指标、阈值和失败后的决策规则；
- 将任务八结果设为只读结构参考。

建议方法名：

```text
autoregressive_maskable_ppo
```

验收标准：

- 实现前完成协议冻结；
- 不根据新结果事后修改主要指标或门槛；
- 明确当前实验检验的是动作生成机制，而不是 GNN 表示能力。

### 任务九二：实现条件动作分布

工作内容：

- 实现固定顺序的逐单元动作采样；
- 实现训练采样、确定性预测和随机预测；
- 实现动态屏蔽前序已分配目标；
- 实现联合 `log_prob`、entropy 和分布诊断；
- 保证完整动作在一次策略调用中生成。

验收标准：

- 任意采样联合动作均不存在重复非 `no-op` 目标；
- 所有单元动作都满足基础环境合法掩码；
- 确定性预测在相同状态下可重复；
- 手工小例子中的联合概率等于条件概率乘积；
- `log_prob` 等于逐条件对数概率之和；
- 更新阶段使用保存动作前缀重建掩码。

### 任务九三：接入 PPO 训练闭环

工作内容：

- 复用现有 MLP encoder 和 value head；
- 接入 rollout、GAE 和 PPO clipped objective；
- 支持模型保存、加载、继续训练和评估；
- 接入统一 trainer、benchmark 和 CLI；
- 为模型和实验增加动作生成机制签名。

验收标准：

- 最小环境上 PPO loss、value loss 和 entropy 均为有限值；
- 参数能够产生非零梯度并完成更新；
- 训练、保存、加载、评估闭环可执行；
- 错误机制签名的模型加载必须失败并给出明确错误；
- 原始 PPO、Maskable PPO 和 `Discrete(136)` 方法行为不变。

### 任务九四：补充自动化测试

至少覆盖：

- 基础掩码与条件目标掩码的组合；
- 弹药耗尽、冷却、超射程和目标失效；
- 前序单元选择目标后的实时屏蔽；
- 全 `no-op` 和仅剩一个合法目标；
- 联合概率、联合 `log_prob` 和 entropy；
- rollout 动作重放时条件掩码一致；
- 确定性预测和随机种子复现；
- 模型保存加载与机制签名校验；
- 固定回合中的非法动作、冲突和过度分配为 0；
- 全量原有测试无回归。

验收标准：

- 新增模块的正常路径和边界状态均有测试；
- 全量测试通过；
- 不以只检查最终冲突率代替概率和梯度正确性测试。

### 任务九五：运行 Smoke Test

建议协议：

```text
训练场景：medium
测试场景：medium / time_pressure
方法：Discrete(136) conflict-free / autoregressive conflict-free
种子：0 / 1
训练步数：至少一个完整 rollout 的最小整数倍
最终评估：2 回合/场景/种子
```

验收标准：

- 配置、模型、日志、原始回合和汇总表完整生成；
- 非法动作、冲突和过度分配均为 0；
- 动作机制签名、单元顺序和概率定义正确留档；
- smoke 只用于工程链路验证，不用于性能结论。

### 任务九六：运行 30k × 3 种子机制筛选

冻结协议：

```text
训练场景：medium
测试场景：medium / time_pressure / heterogeneity_pressure
学习方法：maskable_ppo /
          conflict_free_maskable_ppo /
          autoregressive_maskable_ppo
规则方法：greedy_damage / hungarian_damage
训练种子：0 / 1 / 2
训练步数：30,000
最终评估：50 个配对回合/场景/种子
曲线检查点：10,000 / 20,000 / 30,000
统计：配对 Student-t 95% CI
```

主要比较统一定义为：

```text
autoregressive_maskable_ppo - maskable_ppo
```

次要机制比较为：

```text
autoregressive_maskable_ppo - conflict_free_maskable_ppo
```

验收标准：

- 五种方法、三个测试场景和三个种子完整运行；
- 使用相同训练预算和配对评估场景块；
- 汇总指标可由原始回合重新计算；
- 报告主要比较和次要机制比较，不只选择有利结果；
- 给出成功、资源浪费和高威胁突防的典型种子与回合；
- 性能门槛未全部满足时不扩大到 100k。

### 任务九七：条件性 100k × 5 种子正式实验

仅在筛选门槛全部通过后运行：

```text
训练步数：100,000
训练种子：5 个
最终评估：100 回合/场景/种子
统计：配对 Student-t 95% CI
```

正式实验至少包含：

- 原始 Maskable PPO；
- `Discrete(136)` 无冲突 Maskable PPO；
- 自回归无冲突 PPO；
- Greedy 和 Hungarian 规则基线。

正式实验用于形成“独立动作、联合枚举、自回归生成”的动作机制消融，不在这一阶段加入 GNN。

## 7. 30k 筛选门槛

所有门槛在运行前冻结，主要比较均为“自回归方法减原始 Maskable PPO”：

| 指标                              | 进入正式实验的门槛                |
| ------------------------------- | ------------------------ |
| 非法动作率                           | 必须为 0                    |
| 冲突率                             | 必须为 0                    |
| 过度分配率                           | 必须为 0                    |
| `medium` 平均奖励                   | 下降不超过 5                  |
| `medium` 平均毁伤                   | 增加不超过 0.10               |
| `time_pressure` 平均奖励            | 下降不超过 5                  |
| `time_pressure` 资源成本            | 增加不超过 0.50               |
| `heterogeneity_pressure` 高威胁突防率 | 平均下降至少 0.02，且至少 2/3 种子同向 |
| `heterogeneity_pressure` 平均毁伤   | 增加不超过 0.10               |

附加机制判据：自回归方法在 `time_pressure` 中的资源成本应低于同协议重新训练的 `Discrete(136)` 方法。该判据用于解释动作分解是否缓解联合枚举的资源浪费，但不能替代相对原始 Maskable PPO 的 `+0.50` 主门槛。

以上是工程筛选阈值，不是统计显著性结论。筛选通过后，正式实验仍需使用配对置信区间检验非劣效性和方向性优势。

## 8. 决策规则

```text
零冲突是否实现？
├─ 否：检查条件掩码、动作重放或环境接口，返回工程修复
└─ 是
   ├─ time_pressure 资源成本恢复，性能门槛全部通过
   │  └─ 进入 100k × 5 种子正式动作机制消融
   ├─ 资源成本恢复，但高威胁突防无改善
   │  └─ 动作协调瓶颈基本隔离，可论证进入关系表示研究
   ├─ 高威胁突防改善，但资源成本仍超标
   │  └─ 检查 no-op 概率、entropy 和固定顺序偏置，暂不进入 GNN
   └─ 奖励或毁伤明显退化
      └─ 检查联合概率、PPO ratio 和条件 entropy 实现，停止扩展实验
```

不能以三种子差异不显著宣称算法等价，也不能用单一成功种子替代跨种子结论。

## 9. 阶段总验收标准

### 工程验收

- 自回归条件动作分布完成；
- 一个联合动作只触发一次环境状态转移；
- 条件掩码、概率、梯度、保存加载和机制签名均通过测试；
- 原始方法和任务八结构基线无回归；
- 全量测试通过。

### 实验验收

- Smoke Test 完整通过；
- 30k × 3 种子筛选完整运行；
- 配对种子、原始回合、学习曲线和置信区间齐全；
- 严格报告零冲突、资源成本、毁伤、高威胁突防和决策耗时；
- 根据预设门槛决定是否运行正式实验。

### 学术验收

- 能解释联合枚举与自回归分解对探索空间的差异；
- 能证明联合 `log_prob` 与 PPO ratio 的计算正确；
- 能区分资源节制、拦截能力和高威胁目标优先级；
- 能区分动作协调瓶颈与状态关系表示瓶颈；
- 在证据满足进入条件前不引入 GNN。

## 10. 预期交付物

```text
rein_learning/algorithms/policy_gradient/autoregressive_ppo.py
rein_learning/models/autoregressive_action_head.py
rein_learning/trainers/air_defense_v1_ppo.py
rein_learning/experiments/air_defense_v1_benchmark.py
scripts/compare_air_defense_v1_methods.py
tests/test_autoregressive_joint_action.py
tests/test_air_defense_v1_trainers.py
tests/test_air_defense_v1_experiments.py
docs/algorithms/autoregressive_conflict_free_policy.md
docs/experiments/air_defense_v1_task9_screening.md
results/air_defense_v1/task9_*/
```

实际文件可根据现有代码边界调整，但概率分布、模型、训练器和实验职责必须保持清晰。

## 11. 本阶段完成后的研究位置

任务九是对任务八结构基线的机制深化。它不是增加环境复杂度，而是检验“如何表达和生成联合动作”是否决定资源效率与异质目标分配表现。

本阶段最终需要回答：

> 防空资源分配中，条件联合决策能否同时解决动作冲突和资源浪费；若仍不能改善高威胁目标优先级，剩余瓶颈是否可以归因于资源-目标关系表示？

只有在动作机制经过独立动作、联合枚举和自回归生成三类消融后，项目才具备进入 GNN 或关系表示研究的充分依据。

## 12. 2026-07-17 执行记录

- 已实现自回归条件动作分布、联合 `log_prob` 和条件 entropy；
- 已复用 Maskable PPO 的 rollout、GAE、clipped loss、回调与日志；
- 模型保存加载和动作生成机制签名校验已完成；
- 统一实验升级为 schema 5；
- 概率、梯度、动作前缀重放、训练器和实验回归测试已通过；
- 真实三单元 Smoke Test 已通过；
- 30k × 3 种子筛选已完成，45 个运行汇总和 2,250 个原始回合完整生成；
- 自回归方法非法动作、冲突和过度分配严格为 0；
- `time_pressure` 资源成本相对原始方法下降 1.47，相对 `Discrete(136)` 下降 4.96；
- `heterogeneity_pressure` 高威胁突防平均仅下降 0.01483，未达到 0.02 门槛；
- 根据冻结协议，不运行任务九的 100k × 5 种子正式实验；
- 下一入口为单元顺序偏置与异质目标优先级诊断，暂不直接实现 GNN。

完整结果见：

```text
docs/algorithms/autoregressive_conflict_free_policy.md
docs/experiments/air_defense_v1_task9_screening.md
results/air_defense_v1/task9_autoregressive_screening_30k_3seeds/
```
