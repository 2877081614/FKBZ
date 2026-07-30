# AirDefense-v1 DST-05 更新级诊断仪表与可重放性

任务状态：`PASSED`  
训练：`0`（仅执行一次不保存的冻结批次合成梯度等价性测试）  
策略或环境语义修改：`0`

## 1. 结论

更新级只读仪表已经实现并通过不干扰性验证。现有 Task12 日志记录了优化统计和
冻结 probe 聚合量，但每个种子只有最终权重，不能恢复相邻 PPO 更新。因此
`replay_insufficient=true`，DST-06 需要执行预注册的
`heterogeneity_pressure, 10k × seeds 8/9/10` 短跑。

这不是 P2 阳性或阴性结果；DST-05 只解决测量与可重放性。

## 2. 冻结 probe

- 原始状态：`768`，核心场景状态：
  `512`；
- 策略无关状态—前缀上下文：`5881`，其中 DS 合格上下文：
  `1752`；
- 位置覆盖：`{"0": 512, "1": 1240, "2": 4129}`；
- 合法动作数覆盖：`{"1": 1559, "2": 1254, "3": 1179, "4": 1110, "5": 620, "6": 159}`；
- 高威胁可达/不可达上下文：`2125/`
  `3756`；
- 历史 timestep-0 聚合中同时包含 margin 两侧的核心场景行：
  `4`，涉及种子
  `[9, 10]`。

probe 于 2026-07-18 已生成；本任务只按场景和所有可行前缀做确定性展开，没有
按历史塌缩位置、奖励、Q 或结果标签筛选。

## 3. 指标定义

`unweighted_prefix_flip_rate` 是 DS 合格唯一上下文上的 argmax 变化率。
`ds_weighted_flip_mass` 定义为
`mean[1(a_old != a_new) * r_old(a_new)]`，其中 `r_old` 使用 DST-01 冻结的
Jaccard 结构风险；同时额外记录完整概率质量形式的 `ds_policy_distance`。
`suffix_count_change` 是新旧 argmax 所选动作精确可行后缀数量之差的绝对值均值。
`update_id` 与 DST-01 字段字典中的 `ppo_update` 是同值别名，均指已经完成的
PPO update。`probe_all_noop_rate` 是三单元全 no-op 的状态比例，
`probe_high_engagement_rate` 是至少两个单元 engagement 的状态比例；
`probe_high_threat_unassigned_rate` 以存在合法高威胁目标的状态为分母。

最后位置只参与普通 margin/argmax 边界统计，不进入 DS 加权指标。所有聚合以唯一
`context_id` 为单位，不读取动作对表，因此动作对复制不能改变结果。

## 4. 不干扰性

- 初始参数一致：`true`；
- 仪表前后训练 RNG 一致：`true`；
- 合成 rollout actions 一致：`true`；
- loss 绝对差：`0`；
- 更新后参数 bitwise 一致：
  `true`；
- 两次 probe 重放离散事件一致：
  `true`；
- 连续指标最大误差：
  `0`；
- 环境 step 调用：`0`。

仪表是显式 opt-in callback；未附加时不建立 probe 网格、不加载 probe，也不写文件。

## 5. 历史重放判定

Task12 的 seeds 8/9/10 各有 16 行训练诊断，但每个种子只有一个 30,208 步最终
模型；日志不含权重、上下文级概率或 DS 指标。不同实验的最终模型没有被拼接为
伪时间序列。故现有证据不足以执行 P2，下一步是 DST-06 预注册短跑，而不是从
聚合曲线推断先行性。

## 6. 交付物

```text
results/air_defense_v1/dynamic_support_trust_region/dst_05_instrumentation/
  probe_manifest.json
  instrumentation_equivalence.json
  replay_feasibility.json
  sample_update_metrics.csv
```
