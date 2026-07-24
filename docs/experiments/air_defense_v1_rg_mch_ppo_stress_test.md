# AirDefense v1 RG-MCH-PPO 核心机制压力实验

更新时间：2026-07-22

实验状态：已完成

结论：GAE 锚定与可靠度残差方向有效，但稳定性总门控失败，暂不进入 30k/100k

## 1. 实验问题

上一阶段的 MCH-PPO v0 完全使用冻结 Q-Critic 反事实 advantage 训练 actor，导致3/6个同场景运行发生 no-op 塌缩。本阶段实现 Reliability-Gated MCH-PPO（RG-MCH-PPO），检验以下假设：

> 保留 on-policy GAE 作为全局主信用，只将 Critic 集成一致的反事实信用作为幅度受限残差，能否修复 MCH v0 的训练分叉。

## 2. 算法实现

RG-MCH 使用与 factorized PPO、MCH v0 相同的策略网络、价值网络、动作顺序和动态掩码。新增机制为：

```text
r_i = |mean(A_i^critic)| / (mean(|A_i^critic|) + eps)

delta_i = clip(0.5 * r_i * normalize(A_i^cf), -0.5, 0.5)

A_i^RG = normalize_valid(A^GAE + delta_i)
```

engagement 与 target 分别计算可靠度、残差和独立 PPO ratio。Critic 始终冻结；零可靠度时算法退化为层级 GAE。训练记录可靠度、残差幅度和门控激活率。

## 3. 冻结协议

| 项目 | 配置 |
| --- | --- |
| baseline | `factorized_engagement_ar_ppo_order_012` |
| 失败机制参考 | `mch_ppo_order_012` |
| 候选 | `rg_mch_ppo_order_012` |
| 场景 | `time_pressure`、`heterogeneity_pressure` |
| 种子 | `8、9、10` |
| 训练预算 | 10k steps/model |
| PPO epochs | 2 |
| 评估 | 每场景30回合，完整交叉评估 |
| 新训练模型 | 6 |
| 新评估块 | 12 |

baseline 与 MCH v0 直接复用上一阶段相同协议的冻结结果，没有重复训练或更换种子。本阶段没有搜索融合系数和残差上限。

## 4. 相对 factorized PPO 的同场景结果

### 4.1 time_pressure

| seed | 塌缩 | all-noop差 | 高威胁突防差 | 奖励差 | 损伤差 | 成本差 |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 8 | false | 0.0000 | -0.1500 | +37.1050 | -0.7021 | -0.5167 |
| 9 | false | 0.0000 | +0.0192 | +0.1293 | +0.0064 | 0.0000 |
| 10 | true | +0.8333 | +0.2879 | -42.6153 | +1.0655 | -16.7833 |
| 均值 | - | +0.2778 | +0.0524 | -1.7937 | +0.1233 | -5.7667 |

seed8 从 MCH v0 的全 no-op 失败中恢复，并显著优于 baseline；seed9 基本保持 baseline；seed10 则新发生塌缩。平均安全指标通过非劣门槛，但种子稳定性仍不成立。

### 4.2 heterogeneity_pressure

| seed | 塌缩 | all-noop差 | 高威胁突防差 | 奖励差 | 损伤差 | 成本差 |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 8 | true | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 9 | false | 0.0000 | -0.0208 | +4.8411 | -0.1723 | 0.0000 |
| 10 | false | -0.9333 | -0.2222 | +38.6273 | -0.7879 | +3.9900 |
| 均值 | - | -0.3111 | -0.0810 | +14.4895 | -0.3201 | +1.3300 |

RG-MCH 在异质场景取得明确平均改进，并将 seed10 从 baseline/MCH v0 的全 no-op 中恢复。seed8 仍未恢复，因此绝对无塌缩门控失败。成本比为 `1.259`，超过冻结的 `1.10` 门槛；其中一部分成本增长来自恢复交战，但门槛不能在读取结果后修改。

## 5. 相对 MCH-PPO v0

| 场景 | 奖励均值差 | 损伤均值差 | 是否同时改善 |
| --- | ---: | ---: | :---: |
| time_pressure | +8.2976 | -0.1711 | true |
| heterogeneity_pressure | +23.0283 | -0.6056 | true |

GAE 锚定和残差式信用在两个核心场景均显著优于“纯反事实 actor advantage”。这证明上一阶段定位的信用替代问题真实存在，也证明 RG-MCH 的核心修改方向有效。

## 6. 信用诊断

六个同场景训练模型的最后训练更新平均值为：

| 指标 | 数值 |
| --- | ---: |
| engagement reliability | 0.8836 |
| target reliability | 0.5745 |
| engagement residual absolute mean | 0.2952 |
| target residual absolute mean | 0.2070 |
| engagement gate active rate | 0.8876 |
| target gate active rate | 0.5787 |
| 训练时间 / baseline | 1.2496x |

engagement 可靠度和激活率接近0.9，说明当前门控并不稀疏。三个 Critic 可能在分布外状态上共同犯错，因此“集成方向一致”不能等同于“反事实信用正确”。这能够解释 seed8/10 在不同场景中仍出现互换式塌缩。

## 7. 门控结果

| 门控 | 结果 |
| --- | :---: |
| 结构违规为零 | 通过 |
| 六个候选绝对无塌缩 | 失败，2/6塌缩 |
| 两场景 all-noop 至少2/3非劣 | 通过 |
| 至少一个场景高威胁突防均值改善 | 通过 |
| 两场景奖励/损伤安全非劣 | 通过 |
| 资源成本不超过110% | 失败，异质场景为125.9% |
| 两场景均优于 MCH v0 | 通过 |
| 总门控 | **失败** |

## 8. 科学结论

本阶段取得的是**部分核心机制成功**，不是算法最终成立：

1. 保留 GAE 的 RG-MCH 在两个场景均优于 MCH v0，证明反事实信用不能完全替代 on-policy 全局信用；
2. 异质场景出现有意义的平均任务收益，说明反事实局部修正并非完全无效；
3. 仍有2/6个运行塌缩，说明当前可靠度无法识别 Critic 集成的共同错误；
4. 不能选择 time_pressure/seed8 或 heterogeneity_pressure/seed10 单独宣称算法优势；
5. 不进入30k/100k，也不转入GNN；下一版应增加分布支持或行为锚定可靠度，并显式约束 engagement policy 的累计漂移。

## 9. 产物

```text
rein_learning/algorithms/policy_gradient/mch_ppo.py
scripts/run_air_defense_v1_rg_mch_ppo_stress_test.py
tests/test_rg_mch_ppo.py
results/air_defense_v1/rg_mch_ppo_mechanism_stress_test/
results/air_defense_v1/rg_mch_ppo_mechanism_stress_test/runs.csv
results/air_defense_v1/rg_mch_ppo_mechanism_stress_test/rg_mch_stress_summary.json
```

## 10. 验证

- RG-MCH、MCH、factorized policy、trainer 与统一实验相关回归：`29 passed`；
- 项目 `tests/` 完整回归：`224 passed`；
- 实验配置状态：`completed`；
- 结果包含6个模型文件和12行跨场景运行记录。
