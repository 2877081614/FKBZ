# DST-03：冻结语料重建与完整性审计

任务状态：`PASSED`
训练授权：无  
前置任务：DST-02=`PASSED`

## 1. 目标

从已有诊断、probe corpus、模型和配置恢复 DS-0 所需的状态—前缀语料，不新增
策略训练，不以重新采样替代无法恢复的历史状态。

## 2. 数据源优先级

1. `results/air_defense_v1/task12_probe_corpus/`
2. `results/air_defense_v1/task12_task11_frozen_replay/`
3. `results/air_defense_v1/task10_frozen_model_diagnostics/`
4. 对应正式模型与 `experiment_config.json`

若现有 169,887 行诊断不能还原精确状态和前缀，可使用冻结模型在原配置下做
确定性诊断重放，但必须标记为“重放语料”，不能声称是原始历史记录。

## 3. 语料单位

一行代表：

```text
一个冻结状态
× 一个早期单元位置
× 一个事实前缀
× 一个合法动作对
```

建议字段：

```text
context_id, source_run, scenario, policy_seed, env_seed, step
state_hash, unit_order, unit_position, prefix
action_a, action_b, is_noop_a, is_noop_b
suffix_count_a, suffix_count_b, intersection_count, union_count, ds_jaccard
legal_action_count, prefix_engagement_count
candidate_target_threat_a, candidate_target_threat_b
downstream_argmax_a, downstream_argmax_b, downstream_argmax_changed
prefix_denied, high_threat_legal_but_unassigned, engagement_extreme_direction
```

## 4. 完整性检查

- 每个 `state_hash + prefix` 的动作对覆盖完整；
- 交换 `(a,b)` 后度量一致；
- 同一上下文的环境配置、unit order 和模型哈希唯一；
- 不把同一状态的多个动作对当作独立轨迹；
- 分场景、种子、位置报告覆盖；
- 明确多少历史行因缺少精确状态而无法使用；
- 不用 BPCE 标签或完整回合资源责任作为结果变量。

## 5. 交付物

```text
results/air_defense_v1/dynamic_support_trust_region/dst_03_frozen_corpus/
  ds0_action_pairs.parquet
  context_summary.csv
  exclusion_ledger.csv
  integrity_report.json
  source_manifest.json
```

如果项目环境不支持 Parquet，可使用压缩 CSV，但格式必须在 manifest 中冻结。

## 6. 硬门

通过条件：

- 两个核心场景均有可用语料；
- 至少覆盖 3 个策略种子；
- 所有主分析行可追溯到状态哈希和源模型；
- 环境掩码交叉检查零错误；
- 排除规则完全在分析前固定。

若精确状态无法从现有数据恢复，状态为 `BLOCKED`，先提交恢复方案；不得用缺少
状态语义的聚合 CSV 近似计算 DS。

## 7. 执行记录

执行日期：`2026-07-29`
执行结果：`PASSED`
训练与环境重新采样：`0`
正式 P1 统计门：未执行

恢复路径固定为：

```text
Task12 frozen probe observation + official action mask
× Task11 order-012 frozen policy seeds 0/1/2
→ deterministic diagnostic replay
```

该语料统一标记 `source_kind=replay`，不声称恢复了原始训练轨迹。冻结 probe
提供 768 个互异的精确策略输入状态；独立使用 observation 中的单位位置、射程、
可用性、目标位置和存活状态重算基础 mask，与保存的 768 个正式 mask 零不一致。
事实前缀由三个冻结模型确定性重放得到，没有重新采样环境。

主要结果：

- 主语料 19,073 个合法无序动作对；
- 2,432 个可分析状态—前缀上下文；
- `time_pressure` / `heterogeneity_pressure` 分别覆盖 811 / 789 个上下文；
- 策略种子为 0、1、2；
- 2,176 个上下文因合法动作少于 2 个而无动作对；
- 2,304 个最后位置按 `not_applicable` 排除；
- 原 169,887 行顺序诊断及其他无状态快照的聚合/决策文件仅进入排除台账。

完整性门全部通过：环境基础 mask、条件 mask、事实 argmax、动作对覆盖、Jaccard
公式与交换对称性、模型/配置唯一性、行追溯和重复 ID 均为零错误。Parquet、摘要
和排除台账连续两次重建的 SHA-256 完全一致。

验收产物：

```text
results/air_defense_v1/dynamic_support_trust_region/dst_03_frozen_corpus/
  ds0_action_pairs.parquet
  context_summary.csv
  exclusion_ledger.csv
  integrity_report.json
  source_manifest.json
```

DST-04 已解锁。
