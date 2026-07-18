# AirDefense v1.0 诊断指标定义与聚合规范

更新日期：2026-07-17  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`  
实验数据模式：`schema_version = 2`

## 1. 目的

基础奖励、拦截率和总损伤只能说明策略结果，难以解释策略为什么失败。本规范补充高威胁突防、联合分配冲突、资源浪费和资源效率指标，并固定逐回合与跨回合的唯一计算口径。

这些指标只用于评价，不改变 v1.0 的状态空间、动作空间、状态转移、奖励函数或终止条件。

## 2. 高威胁目标突防率

默认高威胁阈值为：

```text
high_threat_threshold = 0.8
高威胁目标 = target.threat >= high_threat_threshold
```

阈值可通过统一实验参数 `--high-threat-threshold` 修改，并进入 `experiment_config.json`。

逐回合定义：

```text
high_threat_leak_rate
= num_high_threat_leaked / num_high_threat_targets
```

没有高威胁目标时记为 `0`。跨回合时先汇总突防数和目标数，再计算比例，不对逐回合比例作简单平均。

## 3. 区域价值加权损伤

单个突防目标的损伤为：

```text
target_damage
= target.payload * target.threat * protected_zone.value
```

逐回合指标：

```text
zone_weighted_damage = sum(target_damage for leaked targets)
```

v1.0 的 `total_damage` 已包含 `protected_zone.value`，因此当前 `zone_weighted_damage` 与 `total_damage` 数值相等。新字段用于明确评价语义和兼容未来损伤模型，不能再次乘以区域价值，否则会重复加权。

跨回合字段为 `avg_zone_weighted_damage`。

## 4. 分配冲突率

设时间步 `k` 中分配给目标 `j` 的合法防御单元数为 `n(k,j)`。定义：

```text
engaged_target_events
= sum 1[n(k,j) >= 1]

conflict_target_events
= sum 1[n(k,j) >= 2]

assignment_conflict_rate
= conflict_target_events / engaged_target_events
```

该指标回答“被交战的目标事件中，有多少发生了多资源冲突分配”。非法动作和 `no-op` 不进入计数。没有合法交战事件时记为 `0`。

## 5. 过度分配率

冲突描述发生频率，过度分配描述浪费的资源份额。定义重复投入数：

```text
overkill_assignments
= sum max(0, n(k,j) - 1)

overkill_rate
= overkill_assignments / legal_shots
```

例如三个资源同时分配给同一目标，产生一个冲突目标事件和两个过度分配。没有合法射击时记为 `0`。

这里的“过度分配”是联合动作层面的重复投入，不表示目标已经被先前射击确定摧毁。若后续研究概率阈值或序贯射击下的战果过度，应新增独立指标，不能静默改变本定义。

## 6. 单位弹药损伤降低

被拦截目标的潜在损伤视为本回合已避免损伤：

```text
intercepted_damage_potential
= sum(payload * threat * zone.value for intercepted targets)

damage_reduction_per_ammo
= intercepted_damage_potential / ammo_used
```

没有消耗弹药时记为 `0`。跨回合时使用全部回合的潜在损伤总和除以弹药消耗总和。

该指标衡量结果层面的资源效率，不是单次射击的严格因果边际贡献。

## 7. 资源成本

逐回合累计所有合法射击所使用防御单元的成本：

```text
resource_cost = sum(unit.cost for every legal shot)
```

非法动作和 `no-op` 不产生资源成本。跨回合字段为 `avg_resource_cost`。

## 8. 决策耗时

逐回合字段 `decision_time_ms` 为该回合平均单步动作生成时间；跨回合字段 `avg_decision_time_ms` 按全部决策步加权。

计时范围：

- 规则策略：`policy.select_action(env)`；
- PPO：`model.predict()`；
- Maskable PPO：动作掩码构造和 `model.predict()`；
- 不包含 `env.step()` 的状态转移、命中采样、目标运动和奖励计算。

耗时依赖硬件和系统负载，正式比较应在相同设备和进程设置下重复运行。

## 9. 数据层级

统一实验生成三个评价层级：

| 文件 | 层级 | 内容 |
| --- | --- | --- |
| `episodes.csv` | 方法 × 运行 × 回合 | 原始计数、逐回合结果和逐回合派生指标 |
| `runs.csv` | 方法 × 运行 | 从该运行全部原始回合重聚合的指标 |
| `summary.csv` | 方法 × 指标 | 跨随机种子均值、标准差、标准误和 Student-t 95% CI |

比例指标在 `runs.csv` 中均由原始分子、分母池化计算。`summary.csv` 再对独立运行的比例进行跨种子统计。

## 10. 兼容策略

- 原有字段名称和公式保持不变；
- 新诊断字段采用追加方式进入 CSV；
- 实验配置模式从版本 1 升级为版本 2；
- 旧逐回合数据缺少诊断计数时，旧指标仍可按原语义聚合；
- 对旧数据，`avg_zone_weighted_damage` 回退到 `avg_total_damage`，其他无法恢复的新增诊断指标记为 `0`，不能伪造历史诊断数据。

## 11. 固定场景验收

测试场景包含两个目标和三个资源，一步执行 `[T0, T0, T1]`，命中率均为 0，两个目标均突防。人工可得：

```text
high_threat_leak_rate     = 1 / 1 = 1.0
zone_weighted_damage      = 0.9*2*2 + 0.5*1*2 = 4.6
assignment_conflict_rate  = 1 / 2 = 0.5
overkill_rate             = 1 / 3
damage_reduction_per_ammo = 0 / 3 = 0.0
resource_cost             = 1 + 2 + 3 = 6.0
```

对应测试：`tests/test_air_defense_v1_diagnostics.py`。
