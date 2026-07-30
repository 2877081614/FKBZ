# DST-02：精确后缀枚举器与 DS 度量验证

任务状态：`PASSED`
训练授权：无
前置任务：DST-01=`PASSED`

## 1. 目标

实现一个与策略网络无关的精确合法后缀枚举器，并证明 DS 度量反映环境真实
掩码语义，而不是离线近似或重复实现偏差。

## 2. 建议实现

主模块：

```text
rein_learning/common/dynamic_support_distance.py
```

建议最小接口：

```python
enumerate_feasible_suffixes(state, prefix, unit_order)
dynamic_support_jaccard(state, prefix, action_a, action_b, unit_order)
suffix_count(state, prefix, action, unit_order)
```

优先复用：

- `rein_learning/common/masked_context_support.py`
- `rein_learning/common/air_defense_v1_decision_metrics.py`
- `rein_learning/envs/air_defense_v1/wrappers/conflict_free_joint_action.py`

不得复制另一套合法性规则。

## 3. 必须测试的性质

### 3.1 集合正确性

- 枚举出的每个完整后缀经环境掩码逐步验证为合法；
- 所有合法分支均被枚举，无遗漏、无重复；
- 与 `Discrete(136)` 联合动作枚举在可比较状态上一致；
- 不同 unit order 使用各自真实前缀语义。

### 3.2 度量性质

```text
0 ≤ c_DS ≤ 1
c_DS(a, a) = 0
c_DS(a, b) = c_DS(b, a)
相同后缀集合 → c_DS = 0
不相交非空集合 → c_DS = 1
```

### 3.3 边界

- 最后位置标记为 `not_applicable`，不得填成有解释意义的 0；
- 非法当前动作拒绝计算；
- 空 union 触发显式错误或预注册的 `not_applicable`；
- no-op 与 engage 走同一接口；
- 冷却、弹药、存活、射程、目标占用均进入合法性判断。

## 4. 测试集

新建：

```text
tests/test_dynamic_support_distance.py
```

至少包含：

- 人工构造小状态的手算后缀集合；
- no-op 保留目标与 engage 占用目标；
- 单元不可用；
- 目标全部不可达；
- 不同资源异质性；
- 三种已有 unit order；
- 与现有环境逐步 mask 的随机状态交叉测试。

## 5. 交付物

```text
rein_learning/common/dynamic_support_distance.py
tests/test_dynamic_support_distance.py
results/air_defense_v1/dynamic_support_trust_region/dst_02_metric_validation/
  validation_summary.json
  enumerator_crosscheck.csv
```

## 6. 硬门

通过条件：

- 所有性质测试通过；
- 环境掩码交叉检查零不一致；
- 枚举结果确定性复现；
- 未修改环境合法性规则。

任一不一致无法解释时状态为 `BLOCKED`，不得进入 DST-03。

## 7. 执行记录

执行日期：`2026-07-29`
执行结果：`PASSED`
训练与策略修改：`0`

已完成：

- 新增 `rein_learning/common/dynamic_support_distance.py`，直接读取环境正式
  `action_mask()`，并按现有自回归头的前缀目标占用规则枚举后缀；
- 实现精确后缀枚举、suffix count、Jaccard 动作对距离、完整动作代价矩阵、
  旧策略结构风险和策略级 DS 距离；
- 最后位置显式抛出 `not_applicable`，非法前缀、非法当前动作、空后缀和空并集均
  有独立错误类型；
- 新增 14 项定向测试，覆盖手算集合、no-op/engage、不可用单元、不可达目标、
  资源异质性、三种 unit order、Jaccard 性质以及策略级公式；
- 连同环境、决策跟踪和 conflict-free codec 回归测试，共 36 项全部通过；
- 在 3 个场景、5 个环境种子、3 种 unit order、60 个动态状态上完成 720 个
  状态—前缀交叉检查，与 `Discrete(136)` 真值的对称差、重复和掩码副作用均为 0。

验收产物：

```text
results/air_defense_v1/dynamic_support_trust_region/dst_02_metric_validation/
  validation_summary.json
  enumerator_crosscheck.csv
```

环境合法性规则未修改，DST-03 已解锁。
