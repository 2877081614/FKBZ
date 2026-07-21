# 掩码感知反事实分层 PPO：诊断公式与候选边界

更新时间：2026-07-19  
对应阶段：任务十三  
当前状态：反事实估计公式与软件接口已冻结；完整 MCH-PPO 训练算法未冻结、未实现

## 1. 当前结论

任务十三没有把“掩码感知反事实分层 PPO”确认为已经成立的创新算法。当前只冻结一个可验证的估计器接口，用于下一阶段判断动作条件 `Q` 是否能稳定提供比联合 `V(s)` advantage 更有判别力的信用信号。

没有运行 30k 候选训练，原因是 16 次共同随机数反事实分支仍具有较高方差，26 个目标分支中没有单个分支达到 `|mean / SE| >= 1.96`。在该证据强度下直接实现 PPO 候选会把估值噪声、Critic 结构和优化效果混在一起。

## 2. 冻结动作分解

单元 `i` 在状态 `s` 和动作前缀 `h_i` 下先决定交战变量 `z_i`，再在动态合法目标集合 `L_i(s,h_i)` 中选择目标 `y_i`：

```text
pi(a_i | s,h_i)
= pi_e(z_i | s,h_i) * pi_t(y_i | z_i=1,s,h_i)
```

`L_i` 必须同时排除：

- 环境基础非法动作；
- 前序单元已经占用的目标；
- 构造“保持其他动作不变”的离线反事实时，由其他单元占用的目标。

`no-op` 始终合法，不进入目标占用集合。

## 3. 冻结反事实公式

在固定 `s,h_i` 下，目标条件价值和交战价值定义为：

```text
Q_e(1) = sum(y in L_i) pi_t(y | s,h_i) Q(s,h_i,1,y)
Q_e(0) = Q(s,h_i,no-op)

V_i^cf = pi_e(1) Q_e(1) + pi_e(0) Q_e(0)
```

分层 advantage 为：

```text
A_i^engage(z_i) = Q_e(z_i) - V_i^cf
A_i^target(y_i) = Q(s,h_i,1,y_i) - Q_e(1)
```

实际交战动作满足可加和关系：

```text
A_i^engage(1) + A_i^target(y_i)
= Q(s,h_i,1,y_i) - V_i^cf
```

对 `no-op`，target advantage 定义为 0。

## 4. 已验证的软件性质

`rein_learning.common.hierarchical_counterfactual_advantages` 已验证：

- 目标概率只在动态合法集合上重新归一化；
- engagement advantage 在二元策略下期望为 0；
- conditional-target advantage 在合法目标策略下期望为 0；
- 两层 advantage 对实际动作可加和；
- 非法目标的 target advantage 不参与计算；
- 无合法目标时强制退化为 `no-op`。

自回归分布已新增：

- `sample_with_engagement_threshold`；
- `conditional_probabilities`；
- 实际动作前缀上的层级概率和 log-prob 诊断。

## 5. 理论命题边界

若 `Q(s,h_i,a_i)` 为真实动作价值，且 baseline 在给定 `s,h_i` 后不依赖当前采样动作，则上述两层 baseline 分别具有零期望，未裁剪策略梯度保持无偏。动态合法集合可以依赖状态和已观测前缀，但在当前因子的边缘化过程中必须固定。

当前尚未证明：

- 使用学习 Q 后的有限样本偏差界；
- engagement 与 target 独立 clipping 的单调改进；
- 前序动作改变后续掩码时的整体 trust-region 界；
- 反事实 Q 的跨规模泛化性质。

因此不能把“独立 ratio + 独立 clip”直接宣称为理论保证。它只能在完成 Q-Critic 校准和预注册消融后成为候选优化器。

## 6. 下一验证步骤

下一步不是立即训练 MCH-PPO，而是建立非图结构的动作条件 Q-Critic 原型：

1. 输入冻结为 `observation + unit + prefix occupancy + candidate action`；
2. 使用共同随机数 Monte Carlo 分支回报监督合法候选 Q；
3. 在独立状态集报告 Q 偏差、排序准确率和 advantage 符号准确率；
4. 与状态价值 `V(s)`、一步回报和普通 joint GAE 对照；
5. 只有 Q 排序和符号达到门槛，才预注册独立 ratio/clip 的 MCH-PPO 训练实验。

GNN 仍不进入该步骤。先证明“动作条件反事实价值”本身有效，再研究图网络能否高效批量估值和跨规模泛化。
