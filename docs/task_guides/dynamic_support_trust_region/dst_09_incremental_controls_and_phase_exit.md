# DST-09：增量控制与阶段出口

任务状态：`NOT_STARTED`  
训练授权：分项授权，不自动开放  
前置任务：DST-08=`PASSED`

## 1. 目标

排除 DS-TR 的收益只是普通策略减速、通用 churn 抑制或初始化偏置修复，并决定
是否值得进入正式跨场景确认。

## 2. 控制 A：等强度普通 KL/churn

构造与 DS-TR 具有相近：

- 平均 KL；
- 更新幅度；
- flip rate 或 penalty scale；

的普通控制，但不使用后缀支持域信息。

第一批只允许 heterogeneity-pressure、种子 8/9/10、10k。若普通控制达到与
DS-TR 相同效果，则 DS 的结构增量不足，主张必须降级。

## 3. 控制 B：可行后缀计数均匀初始化

按：

\[
q_i^\sigma(x\mid s,h)=
\frac{N_\sigma(s,h\cup\{a_i=x\})}{N_\sigma(s,h)}
\]

构造初始联合可行动作近似均匀的控制。

先做零训练初始化审计：

- 完整合法联合动作初始概率是否近似均匀；
- 不同顺序的 engagement 边际偏置是否缩小；
- 与原初始化的 KL、entropy 和 no-op 边际比较。

只有初始化审计表明原偏置足以解释 DST-08 结果时，才授权 3×10k 训练控制。
该控制源于 PASPO 的可行支持去偏原则，只能作为基线，不能作为核心创新。

## 4. 最终四种结论

| 结果 | 阶段结论 |
|---|---|
| DS-TR 优于普通控制，且初始化不能解释收益 | 形成正式算法候选，授权跨场景确认与专项查新 |
| 普通 KL/churn 达到相同效果 | DS-TR 降级为工程变体，不作为核心算法创新 |
| 初始化控制达到相同效果 | 首要机制改判为初始化偏置；接受简单方案 |
| 所有控制均不稳定 | 结束算法主张，只保留动态支持诊断发现 |

## 5. 论文主张门

只有第一种结果允许使用条件式表述：

> 下游可行后缀结构提供了普通概率信赖域之外的增量信息，并可用于稳定当前
> AirDefense-v1 自回归策略更新。

此时仍然禁止：

- “首次提出”；
- “通用于所有受约束 MARL”；
- “解决 all-noop”；
- “优于所有 PPO 基线”。

正式定位前必须补充 constrained autoregressive policy、state-dependent
action sets、action masking、structured trust region 和 policy churn 专项查新。

## 6. 交付物

```text
results/air_defense_v1/dynamic_support_trust_region/dst_09_controls/
  matched_control_config.json
  initialization_audit.csv
  control_comparison.csv
  gate_summary.json
  novelty_evolution_log.md
docs/experiments/air_defense_v1_ds_tr_incremental_controls.md
```

## 7. 阶段出口报告必须回答

1. DS 是否提供普通 KL/churn 之外的增量？
2. 收益是否能被初始化偏置解释？
3. 哪些前缀位置和场景有效，哪些无效？
4. 新增训练总量是多少？
5. 当前可支持的问题—方法—洞见各是什么？
6. 下一步是跨场景正式确认、收缩为诊断贡献，还是停止？

