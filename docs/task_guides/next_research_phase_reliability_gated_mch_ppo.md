# 下一研究阶段：可靠度门控 MCH-PPO 核心机制验证

更新时间：2026-07-22

任务状态：已完成；部分机制成立，总门控失败

## 1. 研究目标

上一阶段已经证明：完全使用冻结 Q-Critic 的反事实 advantage 替代 on-policy GAE，会使 MCH-PPO 在不同随机种子下继续分叉到 all-noop 或高成本交战。下一阶段不再增加外围数据诊断，而是直接修改 actor 信用接入机制，形成 **Reliability-Gated MCH-PPO（RG-MCH-PPO）**。

本阶段核心假设是：

> on-policy GAE 应保留全局任务收益方向；反事实信用只在 Critic 集成一致时作为幅度受限的局部残差，修正 engagement 与 target 两层信用。

## 2. 冻结算法定义

对 rollout 的联合 GAE 先做标准化，得到 `A^GAE`。冻结 Critic 集成分别产生每个单元的反事实 advantage：

```text
A_i^cf,engage
A_i^cf,target
```

对同一候选的多个 Critic advantage，定义无量纲一致性可靠度：

```text
r_i = |mean(A_i^critic)| / (mean(|A_i^critic|) + eps)
```

当 Critic 完全同号时 `r_i` 接近 1；方向互相抵消时接近 0。融合信用为：

```text
delta_i = clip(lambda * r_i * normalize(A_i^cf), -c, c)

A_i^RG = normalize_valid(A^GAE + delta_i)
```

冻结默认参数：

- `lambda_engage = 0.5`；
- `lambda_target = 0.5`；
- `residual_clip = 0.5`；
- Critic 不可靠或反事实 advantage 为零时，残差自动趋近零；
- engagement 与 target 继续使用独立 ratio 和 clipping；
- no-op 不产生 target loss；
- 联合 GAE 继续训练状态价值函数。

该设计不是把 Critic 置信度解释为统计置信区间，而是把集成方向一致性作为保守的算法门控量。

## 3. 实现任务

1. 重构 MCH advantage 计算，使其返回集成均值、逐因子可靠度和合法样本掩码；
2. 保持 MCH-PPO v0 行为不变，作为历史算法对照；
3. 实现 `ReliabilityGatedMCHPPO`；
4. 实现 GAE 主信用、可靠度门控反事实残差和残差幅度裁剪；
5. 记录 engagement/target 平均可靠度、有效残差幅度和门控激活率；
6. 接入 trainer、统一 benchmark、模型保存加载和掩码评估；
7. 增加单元测试和训练 smoke test；
8. 运行冻结三种子、两个核心场景的 10k 筛选实验；
9. 复用上一阶段完全相同协议的 factorized PPO 与 MCH-PPO v0 结果，生成三方法配对报告。

## 4. 冻结实验协议

| 项目 | 配置 |
| --- | --- |
| 主对照 | `factorized_engagement_ar_ppo_order_012` |
| 失败机制参考 | `mch_ppo_order_012` |
| 新候选 | `rg_mch_ppo_order_012` |
| 场景 | `time_pressure`、`heterogeneity_pressure` |
| 训练种子 | `8、9、10` |
| 训练预算 | 每个新候选模型 10k steps |
| PPO epochs | 2 |
| 最终评估 | 每场景 30 episodes，并执行完整交叉评估 |
| 新训练模型数 | 6 |
| 历史对照来源 | `mch_ppo_mechanism_stress_test/runs.csv` |

不得根据 RG-MCH 结果替换种子、场景、融合系数或 residual clip。本阶段不搜索超参数。

## 5. 验收标准

### 5.1 软件门控

- Critic 参数保持冻结；
- 可靠度严格位于 `[0,1]`；
- 零可靠度时 actor advantage 退化为层级 GAE；
- 反事实残差绝对值不超过 `0.5`；
- no-op 不产生 target actor 梯度；
- 保存、加载和掩码推理正常；
- 结构违规保持为零；
- 新增测试与相关回归测试通过。

### 5.2 机制门控

RG-MCH 进入 30k 扩大实验必须同时满足：

- 六个同场景候选运行均不发生绝对 no-op 塌缩；
- 两个场景均至少 `2/3` 种子的 all-noop 不劣于 factorized PPO；
- 至少一个场景的高威胁突防率均值低于 factorized PPO；
- 两个场景的平均奖励差均不低于 `-10`；
- 两个场景的平均损伤差均不高于 `+0.20`；
- 资源成本不超过 factorized PPO 的 `110%`；
- 两个场景的奖励和损伤均优于 MCH-PPO v0；
- 不允许以单个最佳种子替代完整三种子结论。

## 6. 产物

```text
docs/task_guides/next_research_phase_reliability_gated_mch_ppo.md
rein_learning/algorithms/policy_gradient/mch_ppo.py
scripts/run_air_defense_v1_rg_mch_ppo_stress_test.py
tests/test_rg_mch_ppo.py
docs/experiments/air_defense_v1_rg_mch_ppo_stress_test.md
results/air_defense_v1/rg_mch_ppo_mechanism_stress_test/
```

## 7. 决策规则

门控通过后，才进入 `30k x 5 seeds` 和消融实验。门控失败时，应根据 GAE-only、可靠度和反事实残差诊断修改算法结构；不得继续挑选有利种子，也不得直接转入 GNN 掩盖信用机制问题。

## 8. 执行结论

已完成6个 RG-MCH 新模型和12个交叉场景评估块。候选在两个场景的奖励、损伤均优于 MCH v0；异质场景相对 factorized PPO 的平均奖励提高14.4895、损伤下降0.3201、高威胁突防率下降0.0810，证明 GAE 主信用加可靠度残差的方向有效。

总门控仍失败：2/6个同场景候选发生绝对 no-op 塌缩，异质场景资源成本比为1.259。engagement 平均可靠度0.8836、门控激活率0.8876，表明当前集成一致性门控过于乐观，无法识别 Critic 的共同错误。详细结果见[实验报告](../experiments/air_defense_v1_rg_mch_ppo_stress_test.md)。

软件验收通过：相关回归29项全部通过，项目 `tests/` 完整回归224项全部通过；正式实验状态为 `completed`，包含6个模型和12行跨场景运行记录。
