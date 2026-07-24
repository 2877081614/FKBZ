# AirDefense v1 BPCE 标签语义审计

更新时间：2026-07-23  
实验状态：已完成  
阶段结论：阶段 A 未通过；阶段 B/C 未启动

## 1. 实验目的

BPCE-PPO v0 使用“masked-argmax target + deterministic continuation”的
成对回报差作为 engagement 辅助标签。本实验在不训练 Actor 的条件下，
检验该标签能否稳定代表动态合法目标集合下的 engagement 条件价值方向。

审计比较三种标签：

```text
A：argmax target + deterministic continuation
B：合法目标精确概率边缘化 + deterministic continuation
C：合法目标精确概率边缘化 + stochastic continuation
```

只有标签语义、功效和双向覆盖通过冻结门槛后，才允许进入辅助剂量审计。

## 2. 冻结协议

| 项目 | 配置 |
| --- | --- |
| 场景 | `time_pressure`、`heterogeneity_pressure` |
| 冻结策略 | factorized joint PPO 10k |
| 策略种子 | `8、9、10` |
| 上下文 | 每个“场景×种子”12个 |
| 槽位 | 6个安全临界 + 6个资源临界 |
| 总上下文 | 72 |
| 每个分支重复 | 32次 |
| 目标边缘化 | 全部动态合法目标精确求和 |
| 环境随机性 | engage/no-op 共用命中随机带 |
| 策略随机性 | 标签 C 共用预生成 uniform tape |
| 可靠标签 | `|mean delta| >= 1.0` 且95%均值区间不跨0 |
| Actor 更新 | 禁止 |

上下文选择不读取反事实回报。安全槽按当前威胁、载荷、保护区价值和
time-to-impact 排序；资源槽按弹药稀缺度、相对成本和替代单元覆盖率排序。

## 3. 数据完整性

正式运行生成：

- 72条上下文聚合记录；
- 2304条上下文-重复记录；
- 169条上下文-合法目标记录；
- 266,198个额外环境 transition；
- 运行时间约17.0分钟；
- Actor 最大参数差为 `0.0`。
- BPCE语义定向测试5项通过，项目完整回归247项通过。

因此，本实验是冻结策略的只读机制审计，不包含训练性能混叠。

## 4. 标签功效

| 标签 | 可靠上下文 | 比例 |
| --- | ---: | ---: |
| A：argmax-det | 29/72 | 0.403 |
| B：target-marginal-det | 32/72 | 0.444 |
| C：target-marginal-stochastic | 25/72 | 0.347 |

标签 C 的块级功效如下：

| 场景 | seed8 | seed9 | seed10 |
| --- | ---: | ---: | ---: |
| time pressure | 6 | 0 | 4 |
| heterogeneity pressure | 6 | 2 | 7 |

总体只得到25个可靠 C 标签，低于48个门槛；`time_pressure/seed9` 和
`heterogeneity_pressure/seed9` 也明显低于每块6个的门槛。

## 5. 三种语义对照

### 5.1 目标 argmax 混叠

在全部非零均值上下文中，A/B 符号一致率为：

```text
总体：64/71 = 0.901
time_pressure：29/35 = 0.829
heterogeneity_pressure：35/36 = 0.972
```

该门控通过。在双方均可靠的24个上下文中，一致率为1.0，可靠符号反转为
`0/24`。

argmax target 并不总是收益最高：72个上下文中24个存在正反事实 regret，
regret 均值为3.485、最大值为25.239。但这些目标质量差异没有形成可靠的
engagement 方向反转。因此，当前数据不支持“target argmax 混叠是 v0
塌缩主因”。

### 5.2 deterministic continuation 混叠

B/C 符号一致率为：

```text
总体：56/72 = 0.778
time_pressure：26/36 = 0.722
heterogeneity_pressure：30/36 = 0.833
```

总体低于冻结的0.80门槛，因此失败。16个冲突上下文中，time pressure
占10个、heterogeneity pressure 占6个；冲突主要出现在标准误较大的低功效
上下文。双方均可靠的20个上下文仍保持同号，说明高功效方向相对稳定，
但 deterministic continuation 不能覆盖大量临界、低信噪比上下文。

结论是：标签 A/B 可用于复现既有机制，但不能继续作为下一在线候选的默认
engagement 标签。下一候选必须采用随机后续或预注册的短视窗估值。

## 6. 双向覆盖

可靠 C 标签方向：

| 场景 | 正标签 | 负标签 | 门槛 |
| --- | ---: | ---: | ---: |
| time pressure | 10 | 0 | 每类至少6 |
| heterogeneity pressure | 14 | 1 | 每类至少6 |

按槽位进一步分解：

| 场景/槽位 | 可靠标签 | 正 | 负 |
| --- | ---: | ---: | ---: |
| time/safety | 9 | 9 | 0 |
| time/resource | 1 | 1 | 0 |
| heterogeneity/safety | 12 | 11 | 1 |
| heterogeneity/resource | 3 | 3 | 0 |

资源槽没有产生任何可靠负标签。这否决了“先对现有标签做类别平衡即可修复
BPCE”的假设：当前不是损失函数忽略了负类，而是冻结数据中几乎没有可供
平衡的可靠负类证据。

24个可靠正标签的损伤与高威胁突防分量一致率为1.0，说明正方向收益没有
依赖明显增加毁伤或高威胁突防。但该结果不能弥补负标签缺失。

## 7. 冻结门控

| 门控 | 结果 | 状态 |
| --- | ---: | --- |
| 72个上下文完整 | 72/72 | 通过 |
| 总体可靠标签至少48 | 25 | 失败 |
| 每块可靠标签至少6 | 最差0 | 失败 |
| A/B总体及最差场景一致 | 0.901 / 0.829 | 通过 |
| B/C总体及最差场景一致 | 0.778 / 0.722 | 失败 |
| 可靠target符号反转不超过20% | 0/24 | 通过 |
| 每场景正负标签各至少6 | 10/0；14/1 | 失败 |
| 正标签分量一致率至少0.80 | 1.0 | 通过 |
| Actor保持冻结 | 最大差0.0 | 通过 |

阶段 A 总门控失败。

## 8. 研究结论

本实验把 BPCE v0 的失败原因进一步收窄为：

1. masked-argmax target 虽存在收益 regret，但不是当前可靠方向错误的主要
   来源；
2. deterministic continuation 在低功效临界上下文中不能稳定代表冻结旧
   策略下的条件期望；
3. 全回报 engagement 标签在当前两类槽位中严重偏向正方向，无法提供稳定
   的资源停止证据；
4. 因此，固定辅助系数和类别不平衡不是当前最先应修的问题。

按照预注册顺序，本轮不进入阶段 B 辅助剂量审计，也不进入阶段 C 选点
比较，更不运行修订版10k。直接实现 coverage-balanced loss 会稳定放大
尚未成立的标签语义。

## 9. 后续边界

下一项机制工作应单独预注册，并只比较：

- 标签 C 的随机后续条件期望；
- 短视窗安全收益与资源成本分量；
- 能否在不增加总分支预算的情况下形成跨种子双向可靠标签。

若随机后续或短视窗分量仍不能形成双向覆盖，应暂停 BPCE 在线辅助主线，
把现有成果收敛为 joint PPO 严格 fallback、成对反事实探测基础设施和
局部全回报标签不可辨识的失败机制证据。

## 10. 产物

```text
rein_learning/common/bpce_label_semantics.py
scripts/run_air_defense_v1_bpce_label_semantics_audit.py
tests/test_bpce_label_semantics.py
results/air_defense_v1/bpce_label_semantics_audit/
  experiment_config.json
  context_labels.csv
  repeat_deltas.csv
  target_outcomes.csv
  block_summary.csv
  gate_summary.json
```
