# AirDefense v1 BPCE-PPO v0 机制压力实验

更新时间：2026-07-23

实验状态：已完成；软件验收通过，10k机制门控失败

## 1. 研究问题

本阶段验证以下核心命题：

> 在保持 factorized joint PPO 完整 surrogate 的条件下，当前策略
> engagement 决策边界上的成对共同随机数反事实标签，能否减少
> all-noop/高成本交战种子分叉，并优于等预算随机上下文探测。

BPCE-PPO 不修改环境、奖励、动作掩码、value head、GAE、目标条件策略或
单元顺序。反事实标签只形成 engagement logit 排序辅助。

## 2. 实现与软件验收

本阶段实现：

- AirDefense v1 完整状态快照与恢复；
- 按“环境步×目标”索引的命中随机带；
- 保持实际动作前缀的 no-op/engage 双分支；
- 冻结旧策略 masked-argmax 目标和确定性 continuation；
- margin top-K 边界选择与等预算随机选择；
- 稀疏方向可靠性门控；
- BPCE maskable rollout buffer；
- joint PPO + engagement ranking 辅助损失；
- trainer、统一 benchmark、保存加载和压力实验脚本。

严格退化测试验证 `probe_budget=0` 和全部标签被拒绝时，BPCE 与
factorized PPO 的一次训练参数更新最大差不超过 `1e-6`。最终项目完整
回归为 `242 passed`，BPCE 定向回归为 `14 passed`。

## 3. Smoke 与预算冻结

初始 `K=4、B=8、interval=2` 的256步和1024步匹配时间比分别为
`2.72x/2.48x`，超过2倍门槛。正式性能结果产生前将 `K` 降为2，最终
1024步时间比为 `1.61x-1.86x`。

初始 `7/8` 方向一致规则在离散命中回报中把零差值当作反证，4个上下文
接受率为0。诊断显示每个上下文平均有2.5个非零差值、1.0个反向差值，
平均绝对效应为5.98。正式门控因此冻结为：

```text
至少2个非零 paired delta
均值方向票数严格多于反方向
反方向最多1个
abs(mean_delta) >= 1.0
```

最终1024步 smoke 的4个上下文有1个标签通过，训练时间比为 `1.61x`，
证明正式配置能够产生辅助更新且满足成本门槛。

## 4. 正式协议

| 项目 | 配置 |
| --- | --- |
| 候选 | `bpce_ppo_order_012` |
| 探测对照 | `bpce_random_probe_ppo_order_012` |
| 安全主干 | `factorized_engagement_ar_ppo_order_012` |
| 场景 | `time_pressure`、`heterogeneity_pressure` |
| 种子 | `8、9、10` |
| 训练预算 | `10k steps/model` |
| 新训练模型 | 12 |
| PPO epochs | 2 |
| 评估 | 每场景30回合，完整交叉评估 |
| BPCE预算 | `K=2、B=8、interval=2` |
| 辅助系数 | `0.05` |

factorized PPO 与 MCH 系列复用相同协议的历史冻结结果。正式运行没有修改
种子、场景、门控或辅助系数。

## 5. 主要结果

### 5.1 相对 factorized PPO

| 场景 | 奖励差 | 损伤差 | 高威胁突防差 | 成本比 | all-noop不劣种子 |
| --- | ---: | ---: | ---: | ---: | ---: |
| time pressure | -24.953 | +0.587 | +0.153 | 0.597 | 1/3 |
| heterogeneity pressure | +21.686 | -0.509 | -0.129 | 1.928 | 2/3 |

time-pressure 的平均安全结果明显退化。异质场景平均奖励、损伤和高威胁
突防显著改善，但资源成本达到 baseline 的1.93倍，不能视为完整通过。

### 5.2 种子分叉

| 场景 | seed | all-noop | 奖励差 vs baseline | 损伤差 | 标签 正/负 |
| --- | ---: | :---: | ---: | ---: | ---: |
| time pressure | 8 | 否 | +0.996 | -0.030 | 4/6 |
| time pressure | 9 | 是 | -58.961 | +1.384 | 0/1 |
| time pressure | 10 | 否 | -16.894 | +0.408 | 6/10 |
| heterogeneity | 8 | 否 | +39.015 | -1.080 | 8/10 |
| heterogeneity | 9 | 是 | -36.964 | +1.023 | 0/5 |
| heterogeneity | 10 | 否 | +63.008 | -1.469 | 7/6 |

两个塌缩运行均来自 seed9。time/heterogeneity 的标签接受数只有1/5，
且全部为负方向；辅助训练调用只有1/3次。BPCE 没有获得恢复必要交战的
正方向证据，最终两个场景均退化为绝对 all-noop。

异质场景 seeds 8/10 获得更多正标签并显著改善安全指标，但 seed8 的
actionable engagement rate 达到0.959，说明另一侧仍可能进入高成本交战。

### 5.3 相对等预算随机探测

| 场景 | 奖励差 | 损伤差 | 判定 |
| --- | ---: | ---: | --- |
| time pressure | +13.620 | -0.314 | 边界探测更好 |
| heterogeneity pressure | -13.563 | +0.340 | 边界探测更差 |

边界选择没有稳定优于随机上下文。异质场景的差异主要来自 seed9：随机探测
获得3个正标签并避免塌缩，而边界探测没有正标签并发生 all-noop。

## 6. 探测与计算诊断

每个候选运行平均执行20轮探测、选择40个上下文，接受10.5个标签，平均
接受率为0.2625；正/负标签分别为4.17/6.33。平均每模型增加17369.5个
反事实 transition，训练时间比为 `1.940x`，刚好通过2倍成本门槛。

平均绝对回报差为7.286，效应门槛通过率0.75；平均非零重复数5.47，
反方向重复数2.33。标签并非普遍无效，但不同种子的正负覆盖差异很大。

## 7. 机制门控

| 门控 | 结果 |
| --- | --- |
| 结构非法/冲突/overkill为0 | 通过 |
| 6个同场景运行无塌缩 | 失败，2/6塌缩 |
| all-noop不劣于baseline | 失败 |
| 至少一个场景高威胁突防改善 | 通过 |
| 奖励/损伤安全非劣 | 失败 |
| 资源成本不超过110% | 失败 |
| 边界探测优于随机探测 | 失败 |
| 训练时间不超过2x | 通过，1.940x |

总机制门控：`false`。

## 8. 研究结论

BPCE-PPO v0 取得三项正证据：

1. joint PPO 严格 fallback 在软件和训练更新层面成立；
2. on-policy 成对探测能够在2倍时间内生成非零局部监督；
3. 异质场景两个种子出现显著安全收益，说明局部反事实方向具有条件性价值。

但当前不能形成论文主算法结论：

1. 仍有2/6绝对 all-noop；
2. 边界探测没有稳定优于随机探测；
3. 正负标签覆盖随种子分叉；
4. 安全改善可能伴随1.93倍资源成本；
5. v0 只验证了 AirDefense v1，尚无第二任务证据。

因此不进入30k/100k，不恢复 target 辅助，不进入GNN。

## 9. 下一机制修订

下一版应保持 joint PPO、环境和 paired rollout 完全冻结，只修正辅助监督
接入规则：

1. **双向证据覆盖门控**：时间窗口内同时存在正、负标签才启用辅助更新；
   单边全负或全正时退化为 joint PPO；
2. **类别平衡辅助损失**：正负标签分别归一化，避免少量负标签持续压低
   engagement margin；
3. **辅助更新剂量预算**：限制单 rollout 和累计辅助梯度相对 joint PPO
   的范数，防止稀疏标签被多 epoch 放大；
4. **安全/资源分层选点**：在不读取未来回报的前提下，分别保留高威胁
   临界上下文和资源紧张临界上下文，避免 margin top-K 只覆盖一种方向；
5. 继续保留等预算随机探测，验证修订后收益是否真正来自边界信息。

该修订可称为 **coverage-balanced BPCE-PPO** 候选。在它通过新的10k
机制实验前，不扩大预算。

## 10. 产物

```text
rein_learning/common/boundary_counterfactual_probe.py
rein_learning/algorithms/policy_gradient/bpce_ppo.py
scripts/run_air_defense_v1_bpce_ppo_stress_test.py
tests/test_bpce_ppo.py
results/air_defense_v1/bpce_ppo_mechanism_stress_test/
```
