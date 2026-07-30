# AirDefense-v1 DST-05.5 事件时间轴与 Callback 预检

任务状态：`PASSED`  
正式 P2 证据：`false`  
正式 10k 运行：`0`

## 1. 结论

正式事件、rollout 时间轴与真实 SB3 callback 已通过预检。DST-06 只能使用
`rollout_update_index` 构造未来 1—3 更新和事件前窗口；`sb3_n_updates`
保留为 0/10/20… 的追溯字段，不能作为窗口下标。

probe 的 all-noop/high-engagement/high-threat 字段仍是同步诊断，正式塌缩事件只由
50 回合 CRN 环境评估的 `all_noop_episode_rate` 和
`actionable_engagement_rate` 产生。

## 2. 冻结协议

- 场景：`heterogeneity_pressure`；
- 正式种子：`8/9/10`；
- 评估回合：`50`，episode seeds=`73000...73049`；
- 事件：all-noop `>=0.98` 或 actionable engagement `<0.01`；
- 时间单位：一轮 256-step rollout 加 10 epochs PPO train；
- 正式 10k 请求将产生 40 个 rollout 更新和实际 `10,240` timesteps；
- 每个种子只使用首次 onset，训练前已塌缩种子不算 event-bearing。

## 3. 两路 512-step 真实集成 smoke

- 初始参数 bitwise 一致：
  `true`；
- 两次 rollout 的 actions/rewards/dones/advantages/returns 全部 bitwise 一致；
- loss/KL/clip/entropy 每轮绝对误差均不超过 `1e-10`；
- 第一次正式评估后，第二次 rollout 仍完全一致；
- 最终参数 bitwise 一致：
  `true`；
- optimizer state bitwise 一致：
  `true`；
- Route B 更新行/事件点：`2/`
  `3`；
- 时间轴：rollout=`[1, 2]`，
  SB3=`[10, 20]`，
  timesteps=`[256, 512]`；
- Route A 正式评估环境调用：`0`；Route B：`3`；
- 两路模型均未保存，smoke 不进入 P2 数据。

## 4. 事件逻辑

事件逻辑测试覆盖 49/50 边界、0.009/0.01 边界、初始塌缩、首次 onset、
t+1/t+2/t+3 标签、并发/事件后/尾部排除、SB3 跳 10、六更新窗口、
非 50 回合拒绝和 seed-bank hash 冲突拒绝。

## 5. 阶段出口

`DST-05.5=PASSED`。该结果只证明 DST-06 数据接口有效，不说明 DS
能够预警崩塌，也不授权 DS-TR。下一步是冻结的
`heterogeneity_pressure, requested 10k × seeds 8/9/10`。
