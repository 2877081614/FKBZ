# DST-06：DS-1 短跑与先行性硬门

任务状态：`NOT_STARTED`  
训练授权：条件授权，最多 3×10k  
前置任务：DST-05.5=`PASSED`

## 1. 唯一问题

> DS 加权的前缀翻转是否比普通 KL 和未加权翻转更早、更稳定地指示策略崩塌？

## 2. 数据路径

### 路径 A：现有 checkpoint 足够

直接重放，不新增训练。

### 路径 B：现有 checkpoint 不足

只运行：

```text
algorithm: frozen factorized engagement-target joint PPO
scenario: heterogeneity_pressure
budget: 10k
policy seeds: 8, 9, 10
```

不得修改学习率、entropy、clip、网络、奖励或动作顺序。此次运行是诊断重现，
不是 DS-TR 实验。

## 3. 事件定义

塌缩、all-noop、高交战和安全/资源门全部复用现有 Task12/BPCE 正式口径。
不得因为此次轨迹不同而新设有利阈值。

正式事件标签、50 回合 CRN 评估带和时间轴必须使用 DST-05.5 冻结版本：

- 窗口按连续 `rollout_update_index` 计算；
- `sb3_n_updates` 只作原始追溯，不作窗口下标；
- `probe_all_noop_rate` 不得替代 `all_noop_episode_rate`；
- 每个种子只使用首次 collapse onset；
- 训练前已塌缩的种子不算 event-bearing seed。

对每个事件，以更新为时间单位构造预警窗口，比较：

```text
K0 = KL + clip_fraction + entropy
K1 = K0 + unweighted_prefix_flip_rate
K2 = K1 + ds_weighted_flip_mass
```

## 4. 通过门

P2 只有在以下条件同时满足时通过：

1. `K2` 对事件预警的分组外 AUROC/BA 相对 `K1` 增量不低于 DST-01 冻结值；
2. DS 指标在事件发生前变化，而不是只在事件后响应；
3. 三个种子方向不矛盾，不能由单个塌缩种子独占；
4. 小 KL 与高 DS-weighted flip 至少在一个真实失败窗口共存；
5. 相同强度的未加权 flip 不能解释全部增量。

若 3 个种子均未产生任何可判定事件：

- 不追加种子；
- 结果记为 `INCONCLUSIVE`；
- 由正式报告决定是否因缺乏事件证据停止，而不是自动进入算法实现。

## 5. 交付物

```text
results/air_defense_v1/dynamic_support_trust_region/dst_06_ds1/
  update_metrics.csv
  precursor_windows.csv
  model_comparison.csv
  seed_event_summary.csv
  gate_summary.json
docs/experiments/air_defense_v1_ds1_support_churn_precursor_audit.md
```

## 6. 阶段出口

- `PASS`：授权 DST-07；
- `STOPPED`：DS 没有增量先行性，停止 DS-TR；
- `INCONCLUSIVE`：不自动加预算，由项目级审查决定；
- 不允许把相关性结果写成“DS 导致崩塌”。
