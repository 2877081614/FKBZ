# 未来可覆盖性责任证书（FCRC）

更新时间：2026-07-29。  
状态：通过静态开发门控的算法候选；尚未通过预测性验证。

## 1. 目的

FCRC 回答：

> 当前单元攻击当前目标后，除该目标之外的其他活跃威胁损失了多少最大加权
> 一次覆盖能力？

它不回答当前动作最终造成多少回报，也不把 N1 替代成本转换成新标签。

## 2. 软件接口

实现：
`rein_learning/common/future_coverability.py`

核心类型：

```python
ThreatDemand(
    target_index,
    deadline,
    weight,
    position,
    velocity,
    evasion,
)

ShotOpportunity(
    unit_index,
    time,
    position,
    max_range,
    base_hit_probability,
)
```

核心函数：

```python
maximum_weight_coverability(opportunities, threats)
future_coverability_externality(
    snapshot,
    unit_index=...,
    target_index=...,
)
```

## 3. 精确求解

AirDefense v1 最多只有少量活跃目标，因此使用目标子集动态规划：

```text
state = 已覆盖目标的 bit mask
transition = 当前 shot opportunity 不使用
          或分配给一个尚未覆盖且在截止前、射程内的目标
objective = 最大加权命中价值
```

复杂度为：

\[
O(|\mathcal{K}|\,2^{|\mathcal{T}|}\,|\mathcal{T}|).
\]

指数项只依赖目标数；当前环境目标数为 5，该实现优先保证可审计性，不使用
近似网络。

## 4. 责任公式

对合法动作 \(i\rightarrow j\)，令其他威胁集合
\(\mathcal{U}=\mathcal{T}\setminus\{j\}\)。计算：

\[
E_{i,j}=
\max\{0,V(s,\mathcal{U})-V(s\ominus(i\rightarrow j),\mathcal{U})\}.
\]

当前目标同时从两个比较分支中排除，因此 \(E_{i,j}\) 不把“完成当前任务”
本身算成责任损失。正值只来自该资源消耗对其他任务的外部性。

## 5. 冻结建模假设

- 只考虑当前 alive targets；
- 目标按当前速度线性外推；
- deadline 为 `ceil(time_to_impact)`；
- 每个 shot opportunity 最多覆盖一个目标；
- 每个目标只需要一次覆盖机会；
- 边权为目标损伤权重乘环境命中概率；
- 当前动作只确定性消耗一发资源并更新下一可用时间；
- 不采样命中，不调用 Actor continuation。

这些假设使证书可辨识，但它只是 one-attempt current-wave certificate：

- 不是击毁概率保证；
- 不是未来波次安全保证；
- 不是不确定信念状态下的 robust certificate；
- 不是新的 reward。

## 6. 人工轨迹

`tests/test_future_coverability.py` 共 9 项测试，覆盖：

- 单威胁零外部性；
- 完全可替代单元零外部性；
- 灵活单元抢占专业目标产生正外部性；
- 灵活单元承担不可替代远目标时不产生额外损失；
- 冷却导致临近截止任务失去覆盖；
- 增加机会不降低可覆盖值；
- 非法 deadline、权重、时间和射程被拒绝。

结果：`9 passed`。

## 7. 静态门控结果

| 指标 | 结果 | 门槛 |
| --- | ---: | ---: |
| 合法动作 | 243 | — |
| 正外部性动作 | 86（35.39%） | ≥15% |
| 有同单元目标跨度的 context | 34/108 | ≥30 |
| 与单元成本 Spearman | 0.466 | \|ρ\|<0.90 |
| 与目标损伤权重 Spearman | −0.128 | \|ρ\|<0.90 |
| 与 N1 总替代量 Spearman | 0.479 | 仅报告 |
| 平均耗时 | 1.02 ms/context | ≤5 ms |
| 最大耗时 | 5.47 ms/context | ≤25 ms |

按场景，存在责任跨度的 context 为：

- medium：9/36；
- time pressure：11/36；
- heterogeneity pressure：14/36。

missile 的正外部性更常见，尤其 heterogeneity pressure 中为 36/58；
这与异质资源灵活性解释一致，但当前仍是开发性观察。

## 8. 使用限制

当前禁止：

```python
reward -= alpha * fcrc_externality
```

也禁止直接用阈值屏蔽动作。N3 已使用新的 paired continuation 验证
\(E_{i,j}\) 是否预测“其他威胁覆盖下降/条件损伤增加”。结果未通过因果
方向、增量预测和安全一致性门槛，因此 FCRC 只保留为静态解释组件。

## 9. 后续最小对照

预测性验证至少比较：

1. 高 FCRC 与低 FCRC 的同状态、同单元动作；
2. 原始目标损伤权重；
3. 单元直接成本；
4. 原始最大匹配边权；
5. 二元“是否仍可完全匹配”shield；
6. N1 总替代量。

N3 中加入 FCRC 前后的留一区组 CV MAE 均为 0.137041，增量约为 0；
high-low 主结局的单侧符号翻转 `p=0.3511`，次结局方向为负。在线约束接口
未获授权，FCRC 算法化路线按 N3-E3 停止。
