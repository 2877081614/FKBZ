# 自回归无冲突联合动作策略

更新时间：2026-07-17  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`  
方法名：`autoregressive_maskable_ppo`

## 1. 设计目的

任务八的 `Discrete(136)` 方法通过枚举全部一对一联合动作消除了冲突，但在 `time_pressure` 中使平均资源成本增加 3.49。本方法不再把 136 个联合动作视为互不相关的类别，而是在一次策略调用内按防御单元依次生成动作。

环境、奖励、观测、共享 MLP、Critic 和 PPO 超参数保持不变。三个单元选择共同构成一次联合决策，最后只调用一次 `env.step()`。

## 2. 条件联合策略

默认单元顺序固定为 `[0,1,2]`：

```text
共享 MLP 状态编码
    -> unit 0 动作分布
    -> 屏蔽 unit 0 已选目标
    -> unit 1 条件动作分布
    -> 屏蔽前两个已选目标
    -> unit 2 条件动作分布
    -> 执行一个联合动作
```

每个动作头输出五个目标和一个 `no-op` 的原始 logits。对单元 `i`，条件合法掩码为：

```text
conditional_mask_i(target)
= base_mask_i(target) and target not in selected_targets_before_i
```

`no-op` 不加入已分配目标集合，并始终遵循基础环境掩码。

联合策略概率分解为：

```text
pi(a|s)
= pi(a_0|s)
* pi(a_1|s,a_0)
* pi(a_2|s,a_0,a_1)

log pi(a|s)
= sum_i log pi(a_i|s,a_<i)
```

rollout 保存完整联合动作和旧联合 `log_prob`。PPO 更新时使用保存的动作前缀重建相同条件掩码，不重新采样前序动作。逐条件 entropy 之和作为采样前缀下的联合 entropy 估计。

## 3. 软件结构

```text
rein_learning/models/autoregressive_action_head.py
    AutoregressiveMaskedMultiCategorical
    AutoregressiveActionEvaluation

rein_learning/algorithms/policy_gradient/autoregressive_ppo.py
    AutoregressiveMaskableActorCriticPolicy
    AutoregressiveMaskablePPO

rein_learning/trainers/air_defense_v1_ppo.py
    train_autoregressive_maskable_ppo
```

实现复用 `sb3-contrib` 的 Maskable PPO rollout buffer、GAE、PPO clipped objective、日志和回调，只替换动作分布与策略前向过程。

Gym 动作空间仍为 `MultiDiscrete([6,6,6])`。实验 schema 5 额外记录：

```json
{
  "type": "autoregressive_conflict_free",
  "unit_order": [0, 1, 2],
  "conditional_target_mask": true,
  "joint_log_prob": "sum_of_conditional_log_probs",
  "environment_steps_per_joint_action": 1
}
```

模型加载器同时校验自回归策略类型和动作生成机制签名。原始独立 Maskable PPO 虽然具有相同 Gym 空间，也不能通过自回归模型入口加载。

## 4. 数学与工程验证

自动化测试覆盖：

- 固定 logits 下的确定性无冲突选择；
- 基础合法掩码与前缀目标掩码组合；
- 手工条件概率乘积与联合 `log_prob` 一致；
- rollout 采样概率可由保存动作前缀精确重建；
- 重复目标和基础非法动作被拒绝；
- `log_prob` 和 entropy 产生有限非零梯度；
- 训练、确定性预测、保存、加载和错误机制拒绝；
- 统一实验的 schema 5 签名和零冲突指标。

## 5. 运行方式

```powershell
conda run -n rein-learning python scripts\compare_air_defense_v1_methods.py `
  --train-scenario medium `
  --eval-scenarios medium time_pressure heterogeneity_pressure `
  --methods maskable_ppo conflict_free_maskable_ppo autoregressive_maskable_ppo `
  --seeds 0 1 2 --timesteps 30000
```

## 6. 当前结论

30k × 3 种子筛选中，本方法将非法动作、冲突和过度分配严格降为 0，并使 `time_pressure` 资源成本相对原始 Maskable PPO 下降 1.47、相对 `Discrete(136)` 方法下降 4.96。

但 `heterogeneity_pressure` 高威胁突防率相对原始方法仅下降 0.01483，未达到预设 0.02 门槛。因此当前方法证明了自回归分解能够兼顾冲突约束和资源节制，但尚未证明能够稳定解决异质资源-目标匹配。

完整结果见 [air_defense_v1_task9_screening.md](../experiments/air_defense_v1_task9_screening.md)。
