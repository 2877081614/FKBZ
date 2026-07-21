# 跨场景鲁棒交战价值与可靠成本差监督

更新时间：2026-07-21  
实现状态：正式实验完成，候选未通过  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`

## 1. 方法目的

上一阶段总体性能通过但逐场景召回不稳定。本方法冻结 state-budget 网络结构，只修改损失和模型选择，检验场景-类别等权和最差块惩罚能否消除局部边界漂移。

## 2. 鲁棒分类目标

对每个非空 `(scenario, oracle_class)` 块分别计算 BCE+margin：

```text
L_robust = mean(L_block) + 0.5 * max(L_block)
```

因此每个场景中的 engage/no-op 都拥有相同块权重，且当前最差块获得额外梯度。复制容易样本不会改变块均值。

## 3. 可靠成本监督

成本差可靠性定义为：

```text
r = |mean(cost_delta_samples)| / (1.645 * SE)
```

裁剪到 `[0.25,4]` 并归一化为均值1，用于加权 cost-delta SmoothL1。该权重旨在降低高方差成对成本标签的影响，不修改安全收益或 oracle。

## 4. OOF 选择结果

| seed | standard：可行/最差召回 | robust：可行/最差召回 | robust+cost：可行/最差召回 |
| ---: | --- | --- | --- |
| 20 | 否 / 0.634 | 是 / 0.675 | 是 / 0.675 |
| 21 | 是 / 0.700 | 否 / 0.537 | 否 / 0.634 |
| 22 | 是 / 0.706 | 是 / 0.675 | 否 / 0.575 |

可行种子数为 `2/3、2/3、1/3`。standard 的平均最差召回更高，因此交叉拟合按预注册规则保留 standard。鲁棒损失对 seed20 有效，但在 seed21/22 上产生新的退化，不能形成统一候选。

可靠成本版本的 OOF cost correlation 为 `0.145/0.187/0.188`，相对 standard 的 `0.160/0.196/0.288` 未改善。低方差成本差不能仅靠置信加权恢复排序关系。

## 5. 新批次失败边界

冻结 standard 在新72状态 test 上 BA 为 `0.758/0.662/0.634`。seed21/22 在 heterogeneity 的 engage recall 只有 `0.273/0.182`，与上一批局部过度交战的方向不同。安全收益符号准确率同步降至0.675，说明失败发生在安全收益表示和状态批次泛化，而不是固定资源价格。

当前结论是：

> 场景标签级 Group-DRO 不能替代同一场景内部的多批次覆盖；现有202组训练语料不足以证明 engagement Critic 对临界状态分布稳定泛化。

因此该鲁棒损失保留为负消融，不接入 PPO。下一步应构建多个独立采样批次，并进行 leave-one-batch-out 验证；只有批次外稳定性通过后才恢复 MCH-PPO。

## 6. 实现入口

```text
rein_learning/common/state_conditioned_value_training.py
scripts/run_air_defense_v1_task14_cross_scenario_robust_value.py
tests/test_air_defense_v1_task14_cross_scenario_robust_value.py
docs/experiments/air_defense_v1_task14_cross_scenario_robust_value.md
```
