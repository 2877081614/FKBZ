# AirDefense v1 SA-RG-MCH-PPO 机制压力实验

更新时间：2026-07-23

实验状态：已完成

结论：支持感知与初始策略 KL 锚点未通过，总门控失败；发现独立层级 clipping 的安全退化语义缺陷

## 1. 实验目的

RG-MCH-PPO 虽然在两个场景均优于 MCH v0，但 ensemble agreement 对分布外共同错误过于乐观，并仍有2/6个同场景运行塌缩。本阶段实现 Support-Anchored RG-MCH-PPO（SA-RG-MCH-PPO），检验：

1. Critic 训练上下文支持度能否过滤分布外反事实信用；
2. 相对初始 actor 的累计 engagement KL 能否阻止极端策略漂移。

## 2. 冻结机制

支持域严格只读取 Q-Critic 数据集的338条 train split：

```text
engagement context
= observation + unit one-hot + prefix occupancy + legal mask

target context
= engagement context + selected action one-hot
```

特征标准化后使用最近邻 RMS 距离，并以训练支持点 leave-one-out 最近邻距离的95%分位数校准：

```text
support = exp(-ln(2) * (distance / d95)^2)
combined reliability = ensemble agreement * support
```

累计约束冻结为：

```text
KL_anchor = KL(Bernoulli(p_initial) || Bernoulli(p_current))
L_anchor = mean(relu(KL_anchor - 0.10)^2)
```

融合系数、残差上限和其他 PPO 参数与 RG-MCH 相同，没有结果后调参。

## 3. 实验协议

| 项目 | 配置 |
| --- | --- |
| 对照 | factorized PPO、MCH v0、RG-MCH |
| 候选 | SA-RG-MCH |
| 场景 | time_pressure、heterogeneity_pressure |
| 种子 | 8、9、10 |
| 新训练 | 10k steps × 6 models |
| 评估 | 每场景30回合，12个交叉场景评估块 |
| 支持数据 | task14 Q-Critic dataset train split |

## 4. 同场景结果

### 4.1 time_pressure

| seed | 塌缩 | all-noop差 | 突防差 | 奖励差 | 损伤差 | 成本差 |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 8 | true | +1.0000 | +0.2833 | -41.6401 | +1.0409 | -17.0000 |
| 9 | false | 0.0000 | +0.0192 | +3.3790 | -0.0794 | 0.0000 |
| 10 | true | +1.0000 | +0.3485 | -54.5268 | +1.2736 | -16.9500 |
| 均值 | - | +0.6667 | +0.2170 | -30.9293 | +0.7451 | -11.3167 |

三个种子中只有seed9保持有效交战；seed8和seed10均变为全 no-op。相对RG-MCH，奖励下降29.14、损伤增加0.622。

### 4.2 heterogeneity_pressure

| seed | 塌缩 | all-noop差 | 突防差 | 奖励差 | 损伤差 | 成本差 |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 8 | true | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 9 | true | +1.0000 | +0.3125 | -36.9640 | +1.0230 | -15.4000 |
| 10 | true | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 均值 | - | +0.3333 | +0.1042 | -12.3213 | +0.3410 | -5.1333 |

三个种子全部塌缩。相对RG-MCH，奖励下降26.81、损伤增加0.661。

## 5. 信用与约束诊断

| 指标 | 均值 |
| --- | ---: |
| combined engagement reliability | 0.1139 |
| combined target reliability | 0.0140 |
| engagement context support | 0.1244 |
| target context support | 0.0218 |
| engagement residual absolute mean | 0.0494 |
| target residual absolute mean | 0.0076 |
| initial-anchor engagement KL | 0.0171 |
| anchor penalty | 0.0000 |
| anchor excess rate | 0.0000 |
| 训练时间 / factorized PPO | 1.3711x |

支持门控把反事实残差压缩到接近零，说明当前在线状态-前缀大部分不在原始 Critic 训练支持域。这一诊断本身是可信的：原 Critic 只有338条训练候选，覆盖不足以支持在线 actor 的新分布。

累计 KL 从未超过0.10，anchor 约束完全没有产生梯度。然而 deterministic engagement 只需概率从阈值0.5一侧跨到另一侧就会形成 all-noop，可能只对应很小的 Bernoulli KL。因此平滑 KL 预算不能检测 argmax 决策边界塌缩。

## 6. 新发现：错误的安全退化语义

SA-RG 在低支持度时几乎关闭反事实残差，但它没有退化到 factorized PPO。当前训练仍然：

- 对 engagement 与 target 使用同一个 joint GAE；
- 分别计算 engagement ratio 和 target ratio；
- 分别执行 clipping 并相加。

这与 factorized PPO 的“joint log-prob ratio + 单个 joint clipping”不是同一优化器。5/6塌缩表明：

> 当反事实信用不可信时，独立层级 clipping 不是安全 fallback；它本身可能放大 engagement 分支的确定性阈值分叉。

这是本阶段最重要的核心算法结论。问题已从“如何继续调可靠度”收敛到“如何保证门控关闭时严格恢复标准 PPO”。

## 7. 门控结果

| 门控 | 结果 |
| --- | :---: |
| 结构违规为零 | 通过 |
| 候选绝对无塌缩 | 失败，5/6塌缩 |
| all-noop两场景至少2/3非劣 | 失败 |
| 奖励/损伤安全非劣 | 失败 |
| 高威胁突防改善 | 失败 |
| 资源成本不超过110% | 表面通过，但来源是不交战 |
| 两场景优于MCH v0 | 失败 |
| 相比RG-MCH减少塌缩 | 失败，2增加到5 |
| 总门控 | **失败** |

## 8. 科学结论与下一入口

SA-RG-MCH 不成立，不能进入30k/100k。当前结果否决两项假设：

1. 仅使用最近邻支持度乘法即可稳定在线反事实优化；
2. 初始策略 Bernoulli KL=0.10 能阻止 deterministic no-op 塌缩。

但本阶段成功隔离出下一项核心修改：

```text
joint PPO surrogate 作为严格安全主干
        +
支持感知的反事实辅助目标
        +
直接监控 deterministic engagement margin/coverage
```

当反事实门控为零时，新算法必须在数值上严格退化为现有 factorized PPO，而不是层级独立 clipping。GNN仍不进入当前步骤。

## 9. 产物

```text
rein_learning/common/masked_context_support.py
rein_learning/algorithms/policy_gradient/mch_ppo.py
scripts/run_air_defense_v1_sa_rg_mch_ppo_stress_test.py
tests/test_sa_rg_mch_ppo.py
results/air_defense_v1/sa_rg_mch_ppo_mechanism_stress_test/
results/air_defense_v1/sa_rg_mch_ppo_mechanism_stress_test/runs.csv
results/air_defense_v1/sa_rg_mch_ppo_mechanism_stress_test/sa_rg_mch_stress_summary.json
```

## 10. 验证

- SA-RG-MCH及相关算法回归：`24 passed`；
- 项目 `tests/` 完整回归：`228 passed`；
- 正式实验配置状态：`completed`；
- 结果包含6个模型文件和12行交叉场景运行记录。
