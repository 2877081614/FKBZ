# Hungarian 即时损伤规避分配基线

更新日期：2026-07-17  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`

## 1. 研究定位

`HungarianDamageReductionPolicy` 是不需要训练的集中式优化基线。它在每个环境时间步读取完整状态，构造“防御单元-目标”收益矩阵，并求解一对一最大权匹配。

该基线用于回答：当决策只关注当前时间步时，联合全局匹配相对逐项贪心分配能带来多少改进。它不是有限时域规划器，不显式建模未来目标运动、弹药机会成本或多步协同，因此不要求最终回合奖励一定高于强化学习策略。

## 2. 即时收益

对合法的防御单元 `i` 与目标 `j`，定义：

```text
score(i, j)
= p_hit(i, j) * damage_penalty_weight * target_damage_potential(j)
+ p_hit(i, j) * intercept_reward_weight * target_priority(j)
- resource_cost(i)
```

其中：

- `target_damage_potential = payload * threat * protected_zone.value`；
- `target_priority = threat * payload * protected_zone.value`；
- `p_hit` 同时考虑资源基础命中率、距离衰减和目标规避能力；
- 非法分配在收益矩阵中记为 `-inf`，不能进入匹配。

该定义与 `GreedyDamageReductionPolicy` 完全共用 `expected_damage_reduction_score`，保证两种方法只在分配求解方式上不同。

## 3. 匹配约束

设防御单元数为 `M`，目标数为 `N`。实际求解矩阵包含 `N + M` 列：

```text
前 N 列：真实目标
后 M 列：每个防御单元各自独立的 no-op 虚拟目标
```

约束如下：

1. 每个防御单元恰好匹配一个真实目标或自己的 `no-op`；
2. 每个真实目标最多匹配一个防御单元；
3. 非法动作、非正收益真实目标和其他单元的虚拟列均被禁用；
4. 自己的 `no-op` 收益为 0，因此收益不为正时保留资源；
5. 使用 SciPy `linear_sum_assignment` 求解 `cost = -score` 的最小费用匹配。

相同状态和相同矩阵下，求解顺序固定，策略输出具有确定性。

## 4. 代码位置

```text
rein_learning/baselines/air_defense_v1.py
rein_learning/baselines/__init__.py
rein_learning/experiments/air_defense_v1_benchmark.py
scripts/evaluate_air_defense_v1_baselines.py
scripts/compare_air_defense_v1_methods.py
tests/test_air_defense_v1_hungarian.py
```

统一实验注册名为 `hungarian_damage`。运行全部规则基线：

```powershell
conda run -n rein-learning python scripts\evaluate_air_defense_v1_baselines.py
```

或进入带留档、置信区间和 PPO 对比的统一实验：

```powershell
conda run -n rein-learning python scripts\compare_air_defense_v1_methods.py --rules-only
```

## 5. 正确性验证

测试覆盖：

- 固定小矩阵的已知全局最优分配；
- 非正收益时全部选择 `no-op`；
- 非法资源-目标边不进入匹配；
- 资源与目标的一对一约束；
- 相同状态重复调用输出相同动作；
- 小规模实际环境中与全部联合动作穷举最优值一致；
- 同一即时收益函数下，目标值不低于 `greedy_damage`。

定向测试结果：

```text
16 passed
```

## 6. 初步评估

使用默认 `medium` 场景、评估种子 `200-249`，每种策略运行 50 回合。关键结果如下：

| 方法 | 平均奖励 | 成功率 | 拦截率 | 平均损伤 | 非法动作 | 单步决策耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `greedy_damage` | -41.22 | 0.12 | 0.48 | 1.17 | 0.00 | 0.019 ms |
| `hungarian_damage` | -40.48 | 0.12 | 0.48 | 1.15 | 0.00 | 0.028 ms |

决策耗时只统计策略生成动作的时间，不包含 `env.step()` 的状态转移和奖励计算。对 Maskable PPO，动作掩码构造和 `model.predict()` 均计入决策耗时。

这组 50 回合结果仅用于实现验收。Hungarian 略优于 greedy 的均值差异尚未经过跨运行置信区间检验，不能视为正式统计结论。

## 7. 当前边界

- 只优化当前时间步，不进行多步规划；
- 每个目标最多分配一个防御单元，不支持多资源协同打击；
- 评分依赖环境奖励权重，不是独立于奖励设计的作战效能模型；
- 若后续需要多资源协同、时间窗、库存约束和未来机会成本，应新增 MILP 或滚动时域优化基线，而不是改变 Hungarian 的一对一语义。
