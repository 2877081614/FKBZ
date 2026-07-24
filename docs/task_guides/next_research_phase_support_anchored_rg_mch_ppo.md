# 下一研究阶段：支持感知与累计漂移约束 RG-MCH-PPO

更新时间：2026-07-23

任务状态：已完成；总门控失败，发现独立层级 clipping 的安全退化缺陷

## 1. 研究目标

RG-MCH-PPO 已证明“on-policy GAE 主信用 + 有界反事实残差”优于纯反事实 MCH-PPO v0，但仍有2/6个同场景运行发生 no-op 塌缩。训练诊断显示 engagement 的 Critic 集成一致性可靠度和激活率高达 `0.884/0.888`，说明多个 Critic 在分布外状态上可能一致犯错。

本阶段直接实现 **Support-Anchored Reliability-Gated MCH-PPO（SA-RG-MCH-PPO）**，解决两个核心问题：

1. 区分“Critic 彼此一致”和“当前状态-前缀上下文位于 Critic 训练支持域”；
2. 限制 engagement 策略相对初始 actor 的累计漂移，避免多个局部 PPO 更新逐步滑向极端策略。

## 2. 支持感知可靠度

支持域只使用 Q-Critic 实际训练数据：

```text
results/air_defense_v1/task14_q_critic/dataset.npz
split == train
```

数据包含338个候选动作行。支持特征冻结为：

```text
engagement context
= observation
  + unit one-hot
  + prefix occupancy
  + legal action mask

target context
= engagement context
  + selected target one-hot
```

各维使用 train split 均值和标准差标准化，最小标准差截断为 `0.05`。支持距离为标准化特征的最近邻 RMS 距离。尺度使用训练支持点 leave-one-out 最近邻距离的95%分位数 `d95`：

```text
support(x) = exp(-ln(2) * (d_min(x) / d95)^2)
```

因此位于训练支持点时接近1，在训练支持域95%边界处约为0.5，远离支持域时趋近0。

组合可靠度为：

```text
r_combined = r_ensemble * r_support
```

反事实残差仍保持：

```text
delta = clip(0.5 * r_combined * normalize(A_cf), -0.5, 0.5)
A_SA = normalize_valid(A_GAE + delta)
```

本阶段不搜索距离度量、分位数、融合系数或残差上限。

## 3. engagement 累计漂移约束

训练开始时冻结一份初始 factorized actor 作为 engagement anchor。对每个当前 rollout 状态和实际动作前缀，计算初始 actor 与当前 actor 的 Bernoulli engagement KL：

```text
KL_anchor
= KL(Bernoulli(p_anchor) || Bernoulli(p_current))
```

只在存在合法目标的单元上生效。冻结约束为：

```text
L_anchor
= 1.0 * mean(relu(KL_anchor - 0.10)^2)
```

该约束不要求当前策略保持初始随机策略，只惩罚超过累计预算的极端 engagement 分布漂移。target policy 不受该锚点约束。

## 4. 实现任务

1. 实现可复用的掩码上下文支持索引；
2. 严格只读取 Critic train split，构建 engagement/target 两类支持域；
3. 将支持度接入 MCH 反事实 advantage 批次；
4. 实现 `SupportAnchoredRGMCHPPO`；
5. 组合 ensemble reliability 与 context support；
6. 冻结初始 actor，加入累计 engagement KL 预算；
7. 记录支持度、组合可靠度、锚点 KL、锚点惩罚和残差幅度；
8. 接入 trainer、统一 benchmark、保存加载和掩码评估；
9. 增加支持索引、可靠度组合、锚点约束和训练 smoke tests；
10. 运行10k、三种子、两个核心场景正式筛选，并复用既有三方法结果。

## 5. 冻结实验协议

| 项目 | 配置 |
| --- | --- |
| baseline | factorized PPO |
| 第一失败参考 | MCH-PPO v0 |
| 第二参考 | RG-MCH-PPO |
| 新候选 | SA-RG-MCH-PPO |
| 场景 | `time_pressure`、`heterogeneity_pressure` |
| 种子 | `8、9、10` |
| 新训练预算 | 10k steps/model，共6个模型 |
| PPO epochs | 2 |
| 评估 | 每场景30回合，完整交叉评估 |
| 支持数据 | task14 Q-Critic dataset 的 train split |

不得根据 SA-RG-MCH 结果修改支持分位数、KL预算、惩罚系数、种子或场景。

## 6. 验收标准

### 6.1 软件门控

- 支持索引不读取 validation/test 行；
- 支持分数严格位于 `[0,1]`；
- 精确训练上下文的支持分数接近1；
- 分布外扰动上下文的支持分数显著下降；
- 组合可靠度不高于 ensemble reliability；
- 累计 anchor actor 参数冻结；
- KL预算内无惩罚，超过预算产生正惩罚；
- Critic 参数冻结；
- 保存、加载、掩码推理和统一实验正常；
- 完整回归测试通过。

### 6.2 机制门控

进入30k扩大实验需同时满足：

- 六个同场景候选均无绝对 no-op 塌缩；
- 两个场景均至少2/3种子的 all-noop 不劣于 factorized PPO；
- 两个场景的奖励差均不低于 `-10`、损伤差均不高于 `+0.20`；
- 至少一个场景高威胁突防率均值改善；
- 资源成本不超过 factorized PPO 的 `110%`；
- 两个场景的奖励和损伤均优于 MCH v0；
- 相比 RG-MCH，塌缩数必须从2下降，且两个场景不得同时出现奖励/损伤退化；
- 不能使用单个优势种子替代完整结论。

## 7. 产物

```text
docs/task_guides/next_research_phase_support_anchored_rg_mch_ppo.md
rein_learning/common/masked_context_support.py
rein_learning/algorithms/policy_gradient/mch_ppo.py
scripts/run_air_defense_v1_sa_rg_mch_ppo_stress_test.py
tests/test_sa_rg_mch_ppo.py
docs/experiments/air_defense_v1_sa_rg_mch_ppo_stress_test.md
results/air_defense_v1/sa_rg_mch_ppo_mechanism_stress_test/
```

## 8. 决策规则

通过后进入30k × 5 seeds消融；失败时根据支持度与 anchor KL 区分“支持估计失效”和“策略约束失效”。本阶段仍不进入GNN，也不通过挑选种子或事后修改门槛证明优势。

## 9. 执行结论

已完成6个新模型与12个交叉场景评估块。SA-RG-MCH 有5/6个同场景运行发生绝对 no-op 塌缩，总门控失败。time_pressure 相对 baseline 的平均奖励下降30.93、损伤增加0.745；heterogeneity_pressure 奖励下降12.32、损伤增加0.341。

支持机制将 engagement/target 组合可靠度降至0.114/0.014，残差降至0.049/0.008；初始 actor KL 均值仅0.017，anchor penalty和excess rate均为0。这说明支持覆盖不足是真实问题，但KL=0.10无法感知deterministic阈值跨越。

更关键的是，反事实残差关闭后算法退化为“joint GAE + 两层独立ratio/clipping”，并不等价于factorized PPO的joint ratio/clipping。5/6塌缩表明安全fallback语义错误。下一核心入口应使用标准joint PPO surrogate作为不可替代主干，将反事实信用降为支持感知辅助项，并直接约束deterministic engagement margin。详见[实验报告](../experiments/air_defense_v1_sa_rg_mch_ppo_stress_test.md)。

软件验收通过：相关回归24项、项目 `tests/` 完整回归228项全部通过；正式实验状态为 `completed`，包含6个模型与12行交叉场景记录。
