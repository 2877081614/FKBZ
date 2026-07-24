# 下一研究阶段：BPCE-PPO v0 语义实现与机制证伪

更新时间：2026-07-23

任务状态：已完成；软件验收通过，10k机制门控失败

## 1. 研究目标

SA-RG-MCH-PPO 在反事实残差接近关闭时仍有5/6个同场景运行发生绝对
all-noop，证明“joint GAE + engagement/target 独立 ratio/clipping”
不是 factorized joint PPO 的安全退化路径。

本阶段直接实现 **Boundary-Probed Counterfactual Engagement Auxiliary
PPO（BPCE-PPO）**：

1. 标准 factorized joint PPO 继续承担唯一 PPO 主信用；
2. 只在 on-policy rollout 的 engagement 决策边界附近执行稀疏反事实探测；
3. 使用成对共同随机数比较当前单元 engage 与 no-op 的剩余任务回报；
4. 可靠标签只形成 engagement logit 排序辅助，不替换 GAE；
5. 辅助机制关闭时严格退化为现有 factorized joint PPO。

## 2. 冻结算法语义

### 2.1 Joint PPO 安全主干

完整联合动作继续使用：

```text
r_joint = exp(log pi_theta(a|s) - log pi_old(a|s))

L_joint
= -mean(min(
    r_joint * A_GAE,
    clip(r_joint, 1-epsilon, 1+epsilon) * A_GAE
))
```

禁止重新引入 engagement/target 独立 ratio、独立 clipping 或反事实
advantage 替换。

### 2.2 边界上下文

对 rollout 中每个存在合法目标的单元，沿实际动作前缀计算：

```text
margin = logit(engage) - logit(noop)
```

冻结选择规则：

```text
abs(margin) <= 0.62
按 abs(margin) 从小到大取 top-K
```

`0.62` 约对应 `p_engage` 位于 `[0.35, 0.65]`。选择过程只读取当前
observation、动作前缀、合法掩码和旧策略概率，不读取未来回报。

### 2.3 反事实分支

对上下文 `(s,h_i,i)`：

```text
branch_noop:
    保持 h_i，当前单元强制 no-op

branch_engage:
    保持 h_i，当前单元强制选择
    masked argmax(pi_target_old)
```

同一步尚未发生的后缀单元以及后续时刻均由冻结旧策略确定性补全。该标签
明确解释为“当前冻结确定性目标上的 engage 相对 no-op 的方向”，不声称
边际化全部随机 target 行为。

### 2.4 共同随机带

环境快照必须包含：

- 当前步数；
- 保护区、单元和目标完整状态；
- NumPy RNG 状态；
- 动态合法掩码可重建所需状态。

每次 paired repeat 预生成：

```text
u[environment_step, target_index] ~ Uniform(0, 1)
```

两个分支按相同 `(step,target)` 索引读取命中随机数。禁止仅通过相同 seed
实现配对，因为分支动作不同会改变普通 RNG 的消费顺序。

### 2.5 可靠性门控与辅助损失

正式 v0 使用适配离散命中回报的稀疏方向门控：

```text
B = 8
至少2个 paired delta 非零
均值方向的非零票数严格多于反方向
反方向非零票数最多1个
abs(mean_delta) >= 1.0
```

通过门控后：

```text
d_i = sign(mean_delta)
L_BPCE = mean(softplus(-d_i * margin_i))
L_total = L_PPO + 0.05 * L_BPCE
```

零差值视为无信息而不是反方向。SNR、均值、标准差、非零数和方向一致率
只作为诊断，不把小样本门控表述为严格置信区间。

## 3. 探测预算

正式 v0 冻结为：

| 参数 | 数值 |
| --- | ---: |
| 每次探测最大上下文 `K` | 2 |
| 每上下文 paired repeats `B` | 8 |
| 探测间隔 | 每2个 PPO rollout 探测1次 |
| 边界半径 | `abs(margin) <= 0.62` |
| 最小回报效应 | `1.0` |
| 辅助系数 | `0.05` |

256步 smoke 的首个 rollout 允许执行探测。正式实验必须记录普通训练
transition、反事实 transition 和墙钟时间；若训练时间超过 factorized
PPO 的 `2.0x`，不得扩大实验。

成本修订记录：初始 `K=4` 在256步和1024步运行中的匹配训练时间比分别为
`2.72x` 和 `2.48x`，未满足冻结成本门槛。正式性能结果产生前按停止规则
将 `K` 减为2，保持 `B=8` 和每2轮一次的时间覆盖，不修改标签门控。

门控修订记录：初始 `7/8` 规则在1024步 smoke 的4个边界上下文上接受率
为0；这些上下文平均有2.5个非零差值和1.0个反方向差值，平均绝对效应
为5.98。原规则将大量零差值当作反证，不适合离散命中回报。正式性能结果
产生前改为上述“至少2个非零、方向多数、最多1个反向”规则，效应门槛保持
1.0。

## 4. 实现任务

1. 为 AirDefense v1 增加可测试的状态快照与恢复接口；
2. 增加按环境步和目标索引的命中随机带；
3. 实现边界候选选择、固定前缀双分支和确定性旧策略 continuation；
4. 实现 paired return、方向一致性和最小效应门控；
5. 实现携带 BPCE 标签的 maskable rollout buffer；
6. 实现 `BoundaryProbedCounterfactualEngagementPPO`；
7. 保留标准 joint PPO loss，只增加 engagement ranking loss；
8. 接入 trainer、算法导出、保存加载和统一 benchmark；
9. 增加探测覆盖、标签通过率、正负标签、额外 transition 和训练损失日志；
10. 实现 smoke/正式压力实验与结果汇总脚本。

## 5. 软件验收

- 快照恢复后的 observation、mask、状态实体和 RNG 完全一致；
- 相同随机带和相同分支可重复得到相同回报；
- 两个分支的已发生前缀完全相同；
- engage 目标在对应动态掩码下合法；
- 后缀不产生目标冲突；
- 候选选择不读取未来标签；
- 非可靠标签不产生辅助梯度；
- 负标签推动 margin 下降，正标签推动 margin 上升；
- `probe_budget=0`、`lambda_cf=0`、全部 gate 关闭时均严格退化；
- loss 绝对差不超过 `1e-7`；
- 最大梯度差和单次参数更新差不超过 `1e-6`；
- 模型保存、加载、mask 推理和统一评估正常；
- 项目完整回归测试通过。

## 6. 冻结实验协议

| 项目 | 配置 |
| --- | --- |
| 安全主干 | `factorized_engagement_ar_ppo_order_012` |
| 候选 | BPCE-PPO v0 |
| 探测对照 | 等预算随机上下文探测 |
| 历史失败参考 | MCH-PPO、RG-MCH-PPO、SA-RG-MCH-PPO |
| 场景 | `time_pressure`、`heterogeneity_pressure` |
| 种子 | `8、9、10` |
| 训练预算 | `10k steps/model` |
| PPO epochs | 2 |
| 评估 | 每场景30回合，完整交叉评估 |
| 环境与奖励 | 完全冻结 |

额外仿真分支必须计入总 environment transitions。随机探测与边界探测使用
相同 `K/B/interval`。不得根据正式结果修改种子、场景、边界半径、标签
门控或辅助系数。

## 7. 机制门控

进入30k消融需同时满足：

- 六个同场景候选均无绝对 all-noop 塌缩；
- 两个场景均至少2/3种子的 all-noop 不劣于 factorized PPO；
- 两个场景平均奖励差均不低于 `-10`；
- 两个场景平均损伤差均不高于 `+0.20`；
- 至少一个场景的高威胁突防率均值改善；
- 资源成本不超过 factorized PPO 的 `110%`；
- 边界探测优于等预算随机探测；
- 不允许用单个优势种子替代完整结论；
- 训练墙钟时间不超过 factorized PPO 的 `2.0x`。

在解释性能前还必须报告：边界候选覆盖、探测标签接受率、正负标签比例、
margin 跨零率、辅助/联合梯度范数比和额外仿真 transition。若标签几乎
全部被拒绝，结论只能是机制没有获得足够干预信息，不能宣称算法有效。

## 8. 产物

```text
docs/task_guides/next_research_phase_bpce_ppo_v0.md
docs/algorithms/boundary_probed_counterfactual_engagement_ppo.md
docs/experiments/air_defense_v1_bpce_ppo_stress_test.md
rein_learning/common/boundary_counterfactual_probe.py
rein_learning/algorithms/policy_gradient/bpce_ppo.py
scripts/run_air_defense_v1_bpce_ppo_stress_test.py
tests/test_bpce_ppo.py
results/air_defense_v1/bpce_ppo_mechanism_stress_test/
```

## 9. 停止规则

- 严格退化测试失败：停止实验，先修复 joint PPO 安全语义；
- 共同随机带不能复现：停止实验，不生成正式标签；
- smoke 超过 `2.0x` 时间预算：先降低探测频率，不读取正式性能后调参；
- 10k 仍出现两个以上绝对塌缩：不进入30k，判定 v0 稳定机制不足；
- 边界探测不优于随机探测：不能把主动边界选择作为论文贡献；
- v0 通过后才允许 target 辅助、30k/100k 和第二任务验证；
- GNN 继续作为后续关系表示与跨规模泛化方向，不与本阶段同时实现。

## 10. 执行结论

软件实现和验收完成，最终完整回归达到 `242 passed`，BPCE 定向测试为
`14 passed`。最终 `K=2` 配置的
1024步 smoke 时间比为1.61x，并产生1个可靠负标签。

正式10k三种子双场景实验训练12个新模型。BPCE v0 有2/6个同场景运行
绝对all-noop，均为seed9；time-pressure 相对baseline的奖励、损伤和高威胁
突防分别变化 `-24.953/+0.587/+0.153`。heterogeneity-pressure 分别改善
`+21.686/-0.509/-0.129`，但资源成本达到baseline的1.928倍。

边界探测在time-pressure优于等预算随机探测，在heterogeneity-pressure
更差，总门控失败。每模型平均接受10.5/40个标签，训练时间比1.940x。
seed9两个场景分别只有1/5个标签通过且全部为负，表明下一瓶颈是正负证据
覆盖和辅助更新剂量，而不是joint PPO fallback。

因此不进入30k/100k、target辅助或GNN。下一候选只允许在冻结joint PPO和
paired rollout的基础上增加双向覆盖门控、类别平衡辅助损失和辅助梯度预算。
详见[实验报告](../experiments/air_defense_v1_bpce_ppo_stress_test.md)。
