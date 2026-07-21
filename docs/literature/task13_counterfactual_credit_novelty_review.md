# 任务十三：反事实分层信用分配查新与创新边界

更新时间：2026-07-19  
查新状态：第一轮公式级查新完成，后续需随投稿目标持续更新  
研究对象：动态合法动作约束下的自回归防空资源分配

## 1. 查新结论

“动作掩码 + 分层动作 + 反事实 advantage + PPO”作为组件组合不具备充分的新颖性。动作掩码、分层 PPO、反事实基线和顺序优势分解均已有直接先例。因此，MCH-PPO 不能以组件拼接作为创新主张。

当前仍值得验证的收窄命题是：

> 在前缀动作会改变后续可行集合的自回归匹配策略中，构造只在对应动态合法集合上边缘化、保持层级可加和的掩码条件反事实优势，并分别约束 engagement 与 conditional-target 的近端更新。

这一命题与现有工作的差异尚未被证明，只能记为“待验证创新假设”。任务十三必须先证明普通联合 GAE 在本环境中确实造成可重复的信用混叠，且新估计器能改善 no-op 塌缩而不诱发高成本开火，才允许将其冻结为第一创新候选。

## 2. 直接相关方法矩阵

| 方法 | 已有核心公式或机制 | 动作结构 | Critic / baseline | 约束处理 | 主要实验 | 与本项目的关键差异 |
| --- | --- | --- | --- | --- | --- | --- |
| PPO | clipped surrogate objective | 一般离散或连续动作 | 状态价值与 GAE | 无专门动态掩码理论 | 连续控制、Atari | 本项目不是改写基本 PPO，而是研究结构化动作的分层信用与独立近端约束 |
| Invalid Action Masking | 在 masked logits 上形成合法策略梯度 | 状态相关合法动作集 | 沿用原算法 value | 无效动作置零并重新归一化 | 复杂离散动作游戏 | 已证明“使用动作掩码”本身不是新贡献；本项目关注掩码随动作前缀变化时的反事实边缘化 |
| COMA | `A_i=Q(s,a)-sum(a_i') pi_i(a_i'|o_i)Q(s,(a_-i,a_i'))` | 多智能体联合动作 | 集中式动作价值 Critic | 未处理前缀占用导致的动态合法集 | StarCraft 多智能体协作 | COMA 替换一个并行 agent 的动作；本项目替换集中式策略内部的顺序决策因素，反事实集合依赖前缀 |
| Action-dependent factorized baseline | 对因子化策略构造无偏动作相关 baseline | 高维因子化动作 | 利用其他动作因子的 baseline | 无防空匹配掩码 | 高维控制、目标匹配 | 已覆盖因子化方差降低；本项目必须证明动态可行集和层级二元/条件目标分解带来新的估计问题 |
| H-PPO | 多子 Actor 与共享 Critic 的 hierarchical PPO | 参数化/分层动作 | 共享 Critic | 依任务结构选择子 Actor | 参数化动作任务 | 已覆盖“分层 PPO”；本项目差异只能来自掩码条件反事实估计与层级独立更新，而非双动作头 |
| HATRPO / HAPPO | 多智能体 advantage decomposition 与顺序策略更新 | 异质 agent 联合策略 | 联合 value/advantage | trust region 或 PPO 近似 | Multi-Agent MuJoCo、SMAC | 已覆盖顺序优势分解和单调改进；本项目是单个集中式自回归策略内部的决策因素，不可直接把单元等同独立 agent |
| CAPO | sequential-team counterfactual advantage，给出偏差与方差界 | 固定顺序协作 agent | critic-free reward decomposition | 未针对前缀匹配掩码 | 顺序 bandit、协作流水线 | 直接抬高“顺序反事实优化”的新颖性门槛；本项目必须在动态可行集、占用约束和层级匹配上形成不可约化差异 |
| GNN-WTA | 图表示、图动作或关系编码以提升扩展性 | 武器-目标分配 | 图网络 value/policy | 以图动作或动态掩码表达约束 | 动态 WTA | 已证明 `PPO/RL + GNN + WTA` 不是充分创新；后续图方法必须服务批量反事实估值和变规模泛化 |
| DT-GAT-MARL | GAT 增强的反无人机动态拦截分配 | 多智能体分层决策 | 图注意力 MARL | 动态目标与再分配 | 反无人机拦截 | 与本领域高度直接；本项目后续图创新必须与信用分配耦合，而不能只替换 encoder |

## 3. 已知内容与不可宣称内容

以下内容应视为工程基础或已有方法适配：

- 使用 Maskable PPO 排除非法动作；
- 把 `no-op/engage` 与目标选择拆成两个动作头；
- 使用固定顺序自回归生成无重复联合动作；
- 使用共享 Critic 和联合 GAE；
- 给 engagement 与 target 分支分别记录 entropy、KL 或 gradient norm；
- 使用 COMA 式“替换一个单元动作”的普通反事实 baseline；
- 使用 GNN/GAT 编码资源和目标关系。

不得把阈值从 `0.5` 调整到经验最优值包装为算法创新。若统一阈值已经消除塌缩，任务十三应转为概率校准研究，而不是继续增加反事实算法复杂度。

## 4. 待验证的技术差异

设单元按顺序 `o=(o_1,...,o_M)` 决策，前缀为 `h_i=(a_{o_1},...,a_{o_{i-1}})`，在状态 `s` 和前缀下的合法目标集合为 `L_i(s,h_i)`。动作分解为交战变量 `z_i` 和条件目标 `y_i`：

```text
pi(a_i | s,h_i)
= pi_e(z_i | s,h_i) * pi_t(y_i | z_i=1,s,h_i)
```

候选反事实值仅在 `L_i(s,h_i)` 上定义：

```text
Q_e(s,h_i,z_i=1)
= sum(y in L_i) pi_t(y | s,h_i) Q(s,h_i,z_i=1,y)

V_i^cf
= sum(z in {0,1}) pi_e(z | s,h_i) Q_e(s,h_i,z)

A_i^engage(z_i) = Q_e(s,h_i,z_i) - V_i^cf
A_i^target(y_i) = Q(s,h_i,1,y_i) - Q_e(s,h_i,1)
```

对实际交战动作，存在可检验的层级可加和关系：

```text
A_i^engage(1) + A_i^target(y_i)
= Q(s,h_i,1,y_i) - V_i^cf
```

真正需要验证的新问题是：当 `L_i` 随前缀占用变化时，使用合法集重归一化后的估计器是否保持所需的无偏或受控偏差性质，以及独立 PPO ratio/clip 是否比联合 ratio 更稳定。仅写出上述分解还不足以证明创新。

## 5. 理论与实验否证条件

MCH-PPO 命题应在任一条件成立时收窄或放弃：

1. `0.10-0.90` 阈值扫描找到跨种子、跨场景统一阈值，并消除主要失效。
2. 成功与塌缩种子的 engagement advantage、target advantage 和 Critic 误差没有可重复分叉。
3. 掩码条件反事实量退化为 COMA 或已有 factorized baseline 的直接实例，且没有新的理论性质。
4. 最小候选只增加射击率，却恶化资源成本、毁伤或高威胁泄漏。
5. 改进只存在于当前 `3 resource x 5 target` 环境，无法在第二个结构化分配基准复现。

## 6. 当前决策

当前结论为“继续，但收窄”：

- 保留 MCH-PPO 作为任务十三待验证候选名称；
- 创新核心从技术组合收窄为“动态可行集上的掩码条件反事实估计与层级独立近端更新”；
- 先执行阈值、advantage 和 Critic 诊断；
- 诊断支持信用混叠后，才预注册最小训练候选；
- 不进入 GNN，不运行 100k。

## 7. 主要资料

1. Schulman et al. Proximal Policy Optimization Algorithms. <https://arxiv.org/abs/1707.06347>
2. Huang and Ontanon. A Closer Look at Invalid Action Masking in Policy Gradient Algorithms. <https://arxiv.org/abs/2006.14171>
3. Foerster et al. Counterfactual Multi-Agent Policy Gradients. <https://arxiv.org/abs/1705.08926>
4. Wu et al. Variance Reduction for Policy Gradient with Action-Dependent Factorized Baselines. <https://arxiv.org/abs/1803.07246>
5. Fan et al. Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space. <https://arxiv.org/abs/1903.01344>
6. Kuba et al. Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning. <https://arxiv.org/abs/2109.11251>
7. Zhong et al. Heterogeneous-Agent Reinforcement Learning. <https://arxiv.org/abs/2304.09870>
8. Deshmukh et al. CAPO: Counterfactual Credit Assignment in Sequential Cooperative Teams. <https://arxiv.org/abs/2604.17693>
9. Oh et al. Artificial Intelligence in Combat Decision-Making: Weapon Target Assignment via Reinforcement Learning and Graph Neural Networks. <https://doi.org/10.1109/TCYB.2025.3610606>
10. Jia et al. Graph attention network-enhanced multi-agent reinforcement learning for dynamic interception task allocation in counter-drone defense. <https://doi.org/10.1038/s41598-026-55576-9>
