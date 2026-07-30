# DST-05：更新级诊断仪表与可重放性

任务状态：`PASSED`（2026-07-30）  
训练授权：无  
前置任务：DST-04=`PASSED`

## 1. 目标

建立不改变 actor 更新的只读仪表，判断能否用现有 checkpoint 或日志恢复
“更新前—更新后”的策略变化；本任务本身不启动 10k 训练。

## 2. 固定 probe corpus

优先复用：

```text
results/air_defense_v1/task12_probe_corpus/probe_states.npz
```

probe 必须在训练开始前冻结，覆盖：

- time-pressure 与 heterogeneity-pressure；
- engagement margin 两侧；
- 不同合法动作数量；
- 三个自回归位置；
- 高威胁目标可达与不可达状态。

主 DS 分析仍只使用有下游决策的位置；最后位置仅记录普通边界翻转。

## 3. 每次策略更新记录

```text
update_id
approx_kl
clip_fraction
entropy
engagement_margin_crossings
engage_to_noop_flips
noop_to_engage_flips
joint_argmax_flips
unweighted_prefix_flip_rate
ds_weighted_flip_mass
suffix_count_change
probe_all_noop_rate
probe_high_engagement_rate
probe_high_threat_unassigned_rate
```

`ds_weighted_flip_mass` 必须使用 DST-01 冻结的 DS 定义，不可按结果引入 Q 权重。

## 4. 不干扰性验证

在相同配置和随机种子下比较“仪表关闭”和“仪表开启”：

- 初始参数完全一致；
- rollout actions、loss 和更新后参数在冻结容差内一致；
- 仪表不消费训练 RNG；
- 不调用环境 step；
- 关闭时没有运行时副作用。

## 5. 重放决策

检查现有训练是否保存了足够密度的 checkpoint：

- 若足够，优先做离线 checkpoint 序列重放；
- 若只有最终模型或间隔过大，记录 `replay_insufficient=true`，由 DST-06 授权短跑；
- 不允许用不同实验的最终模型拼成伪时间序列。

## 6. 交付物

```text
results/air_defense_v1/dynamic_support_trust_region/dst_05_instrumentation/
  probe_manifest.json
  instrumentation_equivalence.json
  replay_feasibility.json
  sample_update_metrics.csv
```

相关代码和测试应保持为通用只读诊断，不嵌入 BPCE 或 MCH。

## 7. 验收

- 仪表对训练零干扰；
- 所有指标定义可复算；
- probe 与训练数据隔离；
- 明确 DST-06 是否需要新增训练；
- 没有根据历史塌缩位置筛选 probe。

## 8. 执行记录

- 新增通用只读模块
  `rein_learning/common/dynamic_support_instrumentation.py`，以显式 opt-in
  callback 在完成的 PPO update 后记录指标；关闭时不加载 probe、不建立上下文、
  不写文件；
- Task12 原始 768 状态中保留两个核心场景的 512 状态，并对 order 012 的全部
  可行前缀做策略无关展开，得到 5,881 个唯一上下文，其中 1,752 个属于
  DS 主分析位置；
- unit position 0/1/2 分别覆盖 512/1,240/4,129 个上下文，合法动作数覆盖
  1—6；高威胁目标可达/不可达上下文分别为 2,125/3,756；
- `ds_weighted_flip_mass` 在本任务中落实为
  `mean[1(a_old != a_new) * r_old(a_new)]`，`r_old` 严格使用 DST-01
  Jaccard 结构风险；另输出连续概率质量形式 `ds_policy_distance`；
- 不干扰性验证中，仪表开关两路的初始参数、RNG、合成 rollout actions、
  loss 和更新后参数完全一致；两次重放离散事件 bitwise 一致，连续指标最大
  误差为 0，环境 step 调用为 0；
- 历史可重放性为 `replay_insufficient=true`：Task12 seeds 8/9/10 虽各有
  16 行训练诊断，但每个种子只有一个 30,208 步最终权重，日志没有上下文级
  概率或 DS 指标；其他实验最终模型未被拼接成伪时间序列；
- 正式报告：
  [AirDefense-v1 DST-05 更新级诊断仪表与可重放性](../../experiments/air_defense_v1_ds_update_instrumentation_audit.md)；
- 阶段出口：`PASS`。审查发现正式事件标签和 rollout 时间轴仍需在短跑前
  操作化，因此下一项为
  [DST-05.5](dst_05_5_ds1_event_timeline_preflight.md)；其通过后才允许 DST-06
  按冻结预算运行 `heterogeneity_pressure, 10k × seeds 8/9/10`。
