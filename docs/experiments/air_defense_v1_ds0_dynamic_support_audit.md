# AirDefense-v1 DS-0 动态支持增量机制审计

任务：`DST-04`  
阶段结论：`PASS`  
训练与策略修改：`0`

## 1. 结论

至少一个预注册失败结果同时通过增量、bootstrap、跨场景/种子方向和
分层 max-T 置换门，因此 P1 在当前冻结重放语料上成立。该结论只授权
进入更新级先行性审计，不证明 DS 导致崩塌，也不证明 DS-TR 有效。

## 2. 非退化检查

- pooled DS IQR：`0.333333`；
- IQR 达到 0.05 的场景—种子组：`6/6`；
- 合格分层中 DS 极差达到 0.10 的比例：`1.000`；
- 非退化门：`true`。

## 3. 共同主要结果

| 结果 | AUROC 增量 | BA 增量 | 非负场景 | 非负块 | 判定 |
|---|---:|---:|---:|---:|:---:|
| `high_threat_legal_but_unassigned_changed` | 0.066528 | 0.055062 | 2/2 | 6/6 | `true` |
| `prefix_denied_changed` | 0.087856 | 0.100790 | 2/2 | 6/6 | `true` |
| `engagement_extreme_direction_nonzero` | 0.002842 | 0.000000 | 2/2 | 2/6 | `false` |

### 3.1 正式统计硬门

| 结果 | 指标 | 增量 | 95% bootstrap CI | 置换中位数 | max-T FWER p | 通过 |
|---|---|---:|---:|---:|---:|:---:|
| `high_threat_legal_but_unassigned_changed` | `auroc` | 0.066528 | [0.039309, 0.086710] | -0.000171 | 0.000999 | `true` |
| `high_threat_legal_but_unassigned_changed` | `balanced_accuracy` | 0.055062 | [0.039724, 0.070812] | 0.000217 | 0.000999 | `true` |
| `prefix_denied_changed` | `auroc` | 0.087856 | [0.074691, 0.099042] | -0.000237 | 0.000999 | `true` |
| `prefix_denied_changed` | `balanced_accuracy` | 0.100790 | [0.040567, 0.159485] | -0.000227 | 0.000999 | `true` |
| `engagement_extreme_direction_nonzero` | `auroc` | 0.002842 | [-0.000284, 0.013263] | -0.000029 | 0.019980 | `false` |
| `engagement_extreme_direction_nonzero` | `balanced_accuracy` | 0.000000 | [0.000000, 0.000000] | 0.000000 | 0.999001 | `false` |

两个通过结果在两项指标上均超过 0.02，bootstrap 下界大于 0，
且 max-T FWER p 均为 0.000999。第三个共同主要结果未通过，
因此 P1 只支持已通过的两类结构失败，不能外推为所有退化模式。

## 4. 防伪创新检查

- DS 与 `noop_pair_indicator` 的 Spearman 相关：`-0.391335`。
- DS 与 `legal_action_count` 的 Spearman 相关：`-0.013103`。
- DS 与 `one_minus_completion_count_ratio` 的 Spearman 相关：`0.242061`。
- 普通 downstream argmax flip 已作为 `M0 + flip` 独立对照，在两个通过结果上的 AUROC/BA 最大增量仅为 `0.012044`，明显低于 DS 的正式增量；机械翻转本身不能解释 P1。
- M0 已包含 noop pair、合法动作数量、位置、威胁、前缀交战数、场景和
  策略种子；DS 的主增量是在这些变量之外计算。
- 高威胁结果在 engage-engage 与 noop-engage 子集的 AUROC 增量分别为
  `0.073378` 和 `0.058465`，因此通过结论不只来自 no-op 动作对。
- 高威胁结果在 position 0/1 的 AUROC 增量分别为 `0.029214` 和
  `0.099903`，且通过结果均有 `6/6` 块 log-loss 非负；不是只由
  position 0 或单一策略种子贡献。
- 前缀阻断结果的增量主要来自 noop-engage 子集，engage-engage 子集
  不呈正增量；该边界已保留，不能把它单独包装成普遍机制。

## 5. 创新演化记录

| 版本 | 当前洞见 | 新证据 | 修订原因 | 下一证伪测试 |
|---|---|---|---|---|
| DS-v1 | 动态后缀差异在当前冻结重放语料中提供基础变量之外的增量信息 | 通过结果：high_threat_legal_but_unassigned_changed, prefix_denied_changed | P1 仅为解释性门，尚无时间先行性或算法收益 | DST-05/06 更新级先行性 |

## 6. 文件

```text
results/air_defense_v1/dynamic_support_trust_region/dst_04_ds0_audit/
  distribution_summary.csv
  incremental_metrics.csv
  block_results.csv
  negative_controls.csv
  gate_summary.json
```
