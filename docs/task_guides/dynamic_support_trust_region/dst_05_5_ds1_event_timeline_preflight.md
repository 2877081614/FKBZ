# DST-05.5：DS-1 事件时间轴冻结与真实 Callback 集成预检

任务状态：`PASSED`（2026-07-30）  
任务优先级：P0，DST-06 前置硬门  
训练授权：仅两路配对 `512-step` 集成 smoke，不形成 P2 证据  
前置任务：DST-05=`PASSED`  
后续任务：仅在本任务 `PASSED` 后解锁 DST-06

## 1. 决策摘要

DST-05 已证明 DS 更新级仪表可以确定性计算且历史 checkpoint 不足，但尚未
解决两个会直接影响 DST-06 有效性的问题：

1. 仪表输出的是固定 probe 上的 `probe_all_noop_rate`，而 P2 冻结事件使用
   正式评估口径的 `all_noop_episode_rate` 或
   `actionable_engagement_rate`；
2. callback 每个 rollout 训练周期输出一行，但 SB3 的 `_n_updates` 按 PPO
   epoch 累加；若直接使用其数值差构造 `t-3:t-1`，会把时间窗口解释错误。

因此，在 3×10k 短跑前新增本任务：

> **冻结正式事件的逐更新采集方式和唯一时间轴，并通过两次 rollout 的真实 PPO
> 集成 smoke，证明事件评估与 DS callback 不改变下一次训练 rollout。**

本任务不检验 DS 是否预警崩塌，不得产生 P2 的阳性、阴性或趋势结论。

## 2. 必须保持的上游定义

以下内容直接继承 DST-01，不得在本任务中修改：

```text
主场景：heterogeneity_pressure
正式策略种子：8、9、10
正式训练预算：requested 10k，n_steps=256，n_epochs=10
塌缩事件：
  all_noop_episode_rate >= 0.98
  OR actionable_engagement_rate < 0.01
预警范围：未来 1—3 个完成的 rollout 更新
小 KL：approx_kl <= 0.01
K0/K1/K2 特征和所有 P2 统计门
```

禁止把 `probe_all_noop_rate`、`probe_high_engagement_rate` 或
`probe_high_threat_unassigned_rate` 替换为正式事件。它们只作为同步诊断字段。

## 3. 唯一时间轴

### 3.1 时间单位

新增并冻结：

```text
rollout_update_index
```

其语义为一次完整的：

```text
collect n_steps=256 rollout
→ 完成 n_epochs=10 的 PPO train()
→ 记录更新后 probe 与正式事件评估
```

编号：

- `rollout_update_index=0`：训练前基线，只记录策略状态与正式事件，不产生
  KL、flip 或 P2 predictor；
- `rollout_update_index=1...40`：10k 请求预算在 SB3 中实际完成的 40 个
  rollout 训练周期；
- 实际训练步数固定记录为 `10,240`，不能把 SB3 向上取整隐藏为正好 10,000。

同时保留原始字段：

```text
sb3_n_updates = 0, 10, 20, ..., 400
num_timesteps = 0, 256, 512, ..., 10240
```

但所有 `t-6:t-4`、`t-3:t-1` 和“未来 1—3 更新”只能按
`rollout_update_index` 的连续行序号计算，不能按 `sb3_n_updates±1` 计算。

### 3.2 指标对齐

第 \(t\) 行的：

- `approx_kl / clip_fraction / entropy` 来自刚完成的第 \(t\) 次 rollout
  训练周期；
- flip、DS 和 suffix 指标描述策略 \(t-1\rightarrow t\)；
- 正式事件指标由更新后策略 \(\pi_t\) 在冻结评估带上计算；
- probe 状态指标也由 \(\pi_t\) 计算。

日志不得混用“更新前 KL”和“更新后策略事件”。

## 4. 正式事件评估协议

### 4.1 冻结评估带

对所有正式策略种子和所有更新时间点使用同一组共同随机数环境带：

```text
scenario: heterogeneity_pressure
policy action: deterministic
episodes per update: 50
evaluation episode seeds: 73000...73049
```

理由：

- 50 回合是精确分辨 `0.98=49/50` 的最小回合数；
- 同一评估带跨更新、跨策略种子复用，避免环境样本变化伪装成事件；
- 评估种子与训练种子 8/9/10 分离；
- 正式运行后不得根据事件稀疏程度增加回合或更换种子。

评估必须使用独立环境实例，不得推进、重置或修改训练环境。

### 4.2 每个更新时间点的正式字段

```text
rollout_update_index
sb3_n_updates
num_timesteps
policy_seed
evaluation_seed_bank_sha256
evaluation_episodes
all_noop_episodes
all_noop_episode_rate
actionable_decisions
actionable_engagements
actionable_engagement_rate
collapse_event_state
collapse_event_onset
```

其中：

```text
collapse_event_state =
    all_noop_episode_rate >= 0.98
    OR actionable_engagement_rate < 0.01

collapse_event_onset(t) =
    collapse_event_state(t)
    AND NOT collapse_event_state(t-1)
```

每个策略种子只把首次 `collapse_event_onset` 作为 P2 事件。恢复后再次塌缩只进入
描述性附录，不能增加阳性窗口数量。

如果 `rollout_update_index=0` 已处于塌缩状态，该种子记为
`initially_collapsed=true`，不能作为具有可判定前兆的 event-bearing seed。

### 4.3 前向标签

对 predictor 行 \(t\)：

```text
event_within_3_updates(t) = 1
iff 首次事件发生在 t+1、t+2 或 t+3
```

排除：

- 事件并发行；
- 事件后所有行；
- 最后 3 个无法观察完整未来窗口的行；
- `rollout_update_index=0`；
- 字段缺失或评估回合不完整的行。

事件前中位数门：

```text
baseline window: onset-6 ... onset-4
pre-event window: onset-3 ... onset-1
```

只有 onset 至少有 6 个完整 predictor 更新在前时才可用于该门。不得用
训练前基线行补齐 DS flip。

## 5. 训练随机性隔离

逐更新正式评估和 probe 评估都必须：

- 保存并恢复 Python、NumPy、Torch CPU 和 CUDA RNG；
- 使用独立评估环境；
- 不读取或改变训练环境内部 RNG；
- 恢复 policy 的 train/eval 状态；
- 不更新 observation normalization、optimizer、scheduler 或 replay 状态；
- 不保存评估期间生成的 policy 梯度；
- 不把评估 episode 加入训练 timestep。

若任何一项不能证明，DST-06 不得运行。

## 6. 真实 Callback 集成 smoke

### 6.1 为什么必须是两次 rollout

一次 rollout 只能证明当前更新相同，不能证明更新后的评估不会改变下一次训练
采样。因此 smoke 固定为两路各两个 rollout：

```text
scenario: heterogeneity_pressure
policy seed: 8
n_steps: 256
n_epochs: 10
requested timesteps per route: 512

Route A: 原 factorized joint PPO，不附加仪表
Route B: 相同初始权重与随机带，附加 DS 仪表和正式事件评估
```

两路都不保存模型，不进入 P2 数据，不作为性能实验。

### 6.2 必须逐项相等

- 初始参数 bitwise 一致；
- 两次 rollout 的训练动作轨迹 bitwise 一致；
- rewards、dones、advantages 和 returns 一致；
- 每个训练周期的 PPO loss、KL、clip fraction 和 entropy 一致；
- 第一次评估后，两路第二次 rollout 仍一致；
- 512 步后参数和 optimizer state 在冻结容差内一致；
- Route B 恰好产生 2 行更新日志和 3 个事件评估点（含 \(t=0\)）；
- `rollout_update_index=[1,2]`；
- `sb3_n_updates=[10,20]`；
- `num_timesteps=[256,512]`；
- 正式评估环境 step 不计入训练 timestep；
- 关闭 callback 时不加载 probe、不创建评估环境、不写日志。

推荐数值容差：

```text
离散量与 RNG：bitwise exact
float64 聚合：absolute error <= 1e-10
模型参数：bitwise exact；若底层算子不确定，再使用 max abs <= 1e-8
```

## 7. 事件逻辑单元测试

新建建议：

```text
rein_learning/common/ds1_event_timeline.py
tests/test_ds1_event_timeline.py
```

至少覆盖：

1. `49/50` all-noop 触发，`48/50` 不触发；
2. actionable engagement `0.009` 触发，`0.01` 不触发；
3. 初始已塌缩不形成可判定前兆；
4. 只选择首次 onset；
5. `t+1/t+2/t+3` 标签正确；
6. concurrent、post-event 和尾部删失正确；
7. `sb3_n_updates` 跳 10 不影响按 rollout 行构窗；
8. onset 前不足 6 个 predictor 更新时，不伪造 baseline window；
9. 评估回合数不是 50 时正式行被拒绝；
10. 评估种子带哈希变化时正式合并被拒绝。

## 8. 建议交付物

代码：

```text
rein_learning/common/ds1_event_timeline.py
tests/test_ds1_event_timeline.py
scripts/run_air_defense_v1_dst05_5_event_timeline_preflight.py
```

结果：

```text
results/air_defense_v1/dynamic_support_trust_region/
  dst_05_5_event_timeline_preflight/
    event_protocol.json
    update_timebase_manifest.json
    evaluation_seed_bank.json
    integration_equivalence.json
    sample_update_event_metrics.csv
    gate_summary.json
```

正式报告：

```text
docs/experiments/
  air_defense_v1_dst05_5_event_timeline_preflight.md
```

所有机器可读产物必须记录输入、代码、环境配置和 seed bank 的 SHA-256。

## 9. 硬门

DST-05.5 只有在以下条件全部满足时为 `PASSED`：

1. 正式事件字段与 probe 字段没有混用；
2. 50 回合 CRN 评估带、首次 onset 和前向标签规则已冻结；
3. 三种时间字段同时记录，窗口只使用 `rollout_update_index`；
4. 事件逻辑 10 类测试全部通过；
5. 两路 512-step smoke 的训练轨迹、损失、参数与 RNG 通过等价门；
6. callback 产生的行数、时间编号和字段完整；
7. smoke 明确标记 `formal_p2_evidence=false`；
8. 没有运行正式 10k，也没有实现 DS-TR。

失败处理：

- 事件语义或时间轴不唯一：`BLOCKED`；
- callback 改变训练轨迹：`BLOCKED`；
- 50 回合评估无法在独立随机带运行：`BLOCKED`；
- 不得通过改用 probe 事件、减少回合或放宽等价容差绕过。

## 10. 阶段出口

```text
PASSED
→ 解锁 DST-06：heterogeneity_pressure，10k×seeds 8/9/10

BLOCKED
→ DST-06 保持冻结，先修复事件或时间轴接口
```

即使本任务通过，也只能说明 DST-06 的数据接口有效，不能说明 P2 成立，更不能
进入 DST-07 的 DS-TR 算法实现。

## 11. 执行记录（2026-07-30）

- 冻结 `rollout_update_index` 为唯一分析时间轴，并保留
  `sb3_n_updates`、`num_timesteps` 作为追溯字段；
- 正式事件只读取独立环境的 50 回合 CRN 评估，固定 episode seeds
  `73000...73049`，不读取 probe 退化字段；
- 新增 11 项事件逻辑测试，全部通过；
- Route A/B 各完成获授权的 512-step smoke，观测时间轴分别为
  `(1,10,256)`、`(2,20,512)`；
- 两轮 actions、rewards、dones、advantages、returns 逐位一致，
  loss/KL/clip/entropy 误差为 0，最终参数与 optimizer state 逐位一致；
- 3 次正式事件评估均恢复 RNG、策略模式、梯度、优化器、训练环境和 scheduler
  状态，评估 step 未进入训练 timestep；
- 生成 6 件机器可读产物和正式实验报告，全部 smoke 行均标记
  `formal_p2_evidence=false`；
- 未执行正式 10k，未保存 smoke 模型，未实现 DS-TR。

阶段出口：`DST-05.5=PASSED`，解锁 DST-06；本结果不构成 P2 先行性证据。
