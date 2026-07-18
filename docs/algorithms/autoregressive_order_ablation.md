# 自回归单元顺序参数化与决策诊断

更新时间：2026-07-17  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`  
对应阶段：任务十

## 1. 目的与边界

任务十在任务九自回归无冲突策略上增加显式 `unit_order`，用于检验固定单元顺序是否造成异质资源与目标之间的条件决策偏置。环境状态、动作语义、转移、奖励、共享 MLP、Critic 和 PPO 超参数均保持不变。

首轮只比较三个循环顺序：

```text
order_012 = [0, 1, 2]
order_120 = [1, 2, 0]
order_201 = [2, 0, 1]
```

环境动作始终按原始单元索引保存。`unit_order` 只决定条件采样、确定性预测、动作回放、熵和联合对数概率的计算顺序。

## 2. 条件联合动作

给定顺序 `o=(o_1,o_2,o_3)`，联合策略分解为：

```text
pi(a|s) = product_i pi(a[o_i] | s, a[o_1], ..., a[o_(i-1)])
log pi(a|s) = sum_i log pi(a[o_i] | s, prefix)
```

每个前序单元选中目标后，该目标从后续单元条件掩码中移除。采样完成后，动作按环境索引逆映射，因而环境仍将动作第 `i` 位解释为防御单元 `i`。

非法排列、重复索引、漏掉单元或长度不匹配会在模型创建阶段失败。模型保存时记录完整动作生成签名：

```json
{
  "type": "autoregressive_conflict_free",
  "unit_order": [2, 0, 1],
  "conditional_target_mask": true,
  "joint_log_prob": "sum_of_conditional_log_probs",
  "environment_steps_per_joint_action": 1
}
```

加载模型时，模型签名必须与策略签名一致。

## 3. 决策轨迹

统一实验 schema 升级为版本 6。启用 `--record-decisions` 后，最终评估额外生成：

```text
decisions.csv
decision_summary.csv
leak_attributions.csv
leak_attribution_summary.csv
```

每条决策记录一个环境步内一个防御单元的原始上下文、条件合法集合、所选动作、目标威胁、预计毁伤降低、资源匹配效率和执行结果。`no-op` 的目标字段使用空值，不使用数值 0。

核心聚合指标包括：

- `assignment_rate`：非 no-op 数 / 全部单元决策数；
- `avoidable_noop_rate`：存在条件合法目标时仍 no-op / 可行动决策数；
- `high_threat_assignment_rate`：高威胁分配数 / 高威胁合法机会数；
- `matching_efficiency`：所选匹配的预计毁伤降低 / 当时最优可选值；
- `prefix_denial_rate`：被前序占用的目标机会 / 基础合法目标机会。

训练 rollout 不记录逐决策轨迹，避免显著增加训练数据量；仅最终评估和指定诊断回放记录。

## 4. 高威胁泄漏归因

每个高威胁泄漏目标只进入一个类别，优先级冻结为：

```text
prefix_denied > mismatched_resource > attempted_miss
unassigned（存在合法机会但从未分配）
resource_exhausted（几何可达但资源不可用）
never_legal（整个回合从未形成合法机会）
```

分类规则和分母由合成单元测试覆盖，实验后不能根据结果修改类别或优先级。

## 5. 软件入口

主要实现：

```text
rein_learning/models/autoregressive_action_head.py
rein_learning/algorithms/policy_gradient/autoregressive_ppo.py
rein_learning/common/air_defense_v1_decision_metrics.py
rein_learning/trainers/air_defense_v1_ppo.py
rein_learning/experiments/air_defense_v1_benchmark.py
```

冻结模型诊断：

```powershell
conda run -n rein-learning python scripts\diagnose_air_defense_v1_task9_models.py
```

三顺序统一实验：

```powershell
conda run -n rein-learning python scripts\compare_air_defense_v1_methods.py `
  --methods autoregressive_ppo_order_012 autoregressive_ppo_order_120 autoregressive_ppo_order_201 `
  --train-scenario medium `
  --eval-scenarios medium time_pressure heterogeneity_pressure `
  --seeds 0 1 2 --timesteps 30000 --record-decisions
```

门槛判定与池化汇总：

```powershell
conda run -n rein-learning python scripts\analyze_air_defense_v1_task10.py
```

## 6. 验收结论

- 三种循环顺序均能训练、保存和评估；
- 环境动作索引与生成顺序位置映射正确；
- 非法动作、分配冲突和过度分配均严格为 0；
- 任务九旧模型可加载并恢复默认顺序 `[0,1,2]`；
- 逐决策指标和六类泄漏归因有独立单元测试；
- `tests/` 全量测试为 `132 passed`。

实验结果表明顺序会显著改变单元参与模式，但简单固定换序存在高威胁保护与资源成本之间的权衡。详细结果见[任务十实验报告](../experiments/air_defense_v1_task10_order_diagnostics.md)。
