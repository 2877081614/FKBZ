# DST-04：DS-0 增量机制审计与硬门

任务状态：`PASSED`（2026-07-29）  
训练授权：无  
前置任务：DST-03=`PASSED`

## 1. 唯一问题

> 动态支持域扰动是否提供超过动作类型、位置、威胁和合法动作数量的增量解释？

本任务不评价 DS-TR 性能，不实现算法。

## 2. 分析顺序

### A. 非退化检查

报告：

- DS 的零值率、均值、中位数和分位数；
- 分场景、种子、位置、动作对类型的分布；
- completion count 与 Jaccard DS 的关系；
- DS 是否几乎等价于 `engage↔no-op` 指示量。

如果 DS 近似常数或完全由单个基础变量确定，直接 `STOPPED`。

### B. 描述性机制检查

比较 DS 分层下的：

- 后续 deterministic suffix 改变；
- `prefix_denied`；
- 高威胁合法但未分配；
- engagement 数量向极端方向移动。

描述性相关不能单独通过门控。

### C. 增量预测

建立：

```text
M0 = 预注册基础变量
M1 = M0 + ds_jaccard
```

按“场景×策略种子”做留组验证。至少报告：

- 平衡准确率或 AUROC 的 M1−M0；
- 对数损失差；
- 分块方向；
- paired/group bootstrap 区间；
- 置换 DS 后的负对照。

主通过建议门：

1. 至少一个预注册主结果的分组外 BA/AUROC 增量不低于 `0.02`；
2. 两个核心场景方向均非负；
3. 至少 5/6 场景×种子块方向非负；
4. pooled 分组 bootstrap 区间不跨 0；
5. DS 置换后增量消失。

这些阈值必须在 DST-01 中冻结；DST-04 不得修改。

## 3. 防伪创新检查

报告必须回答：

- DS 是否只是 no-op 指示量？
- 是否只是合法动作数量的重表达？
- 是否只在 unit position 0 有效？
- 是否由单一策略种子贡献？
- 换成普通动作翻转指示后是否相同？

任何“仅因为集合定义必然相关”的结果不得作为 P1 证据。

## 4. 交付物

```text
results/air_defense_v1/dynamic_support_trust_region/dst_04_ds0_audit/
  distribution_summary.csv
  incremental_metrics.csv
  block_results.csv
  negative_controls.csv
  gate_summary.json
docs/experiments/air_defense_v1_ds0_dynamic_support_audit.md
```

## 5. 阶段出口

- `PASS`：全部主门通过，授权 DST-05；
- `STOPPED`：任一核心门失败，DS-TR 路线结束；
- `BLOCKED`：数据完整性不足，不能把未知写成阴性。

报告必须保留阴性结果，不得以“趋势合理”替代门控。

## 6. 执行记录

- 输入：DST-03 冻结语料 `ds0_action_pairs.parquet`，核心语料 12,511 行、
  1,600 个上下文、6 个场景—策略种子组；
- 非退化：pooled DS IQR=`0.333333`，6/6 组 IQR≥0.05，18/18 个合格
  分层的 DS 极差≥0.10；
- 正式检验：10,000 次层级 bootstrap，1,000 次冻结分层内 DS 置换，
  3 个共同主要结果×2 个指标做 max-T FWER 控制；
- 通过结果：`high_threat_legal_but_unassigned_changed` 与
  `prefix_denied_changed`；二者在 AUROC 和 BA 上均通过全部硬门；
- 阴性边界：`engagement_extreme_direction_nonzero` 未通过增量和
  分块方向门，不得写成普遍退化机制；
- 防伪检查：高威胁结果在 engage-engage/noop-engage、position 0/1
  子集中均保留正 AUROC 增量，普通 downstream flip 对照不能复现主增量；
- 正式报告：
  [AirDefense-v1 DS-0 动态支持增量机制审计](../../experiments/air_defense_v1_ds0_dynamic_support_audit.md)；
- 阶段出口：`PASS`，仅授权 DST-05 的零训练更新级仪表与可重放性审计。
