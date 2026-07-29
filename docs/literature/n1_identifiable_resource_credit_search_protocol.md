# N1 可辨识资源信用系统查新协议

更新时间：2026-07-28。

任务：N1-02。  
用途：在任何新在线训练之前，判断“分量保持的资源信用”“全局 CMDP
约束”和“受控延续差异回报”是否构成可辩护的新算法问题。

## 1. 检索边界

本次检索只回答三个问题：

1. 是否已有工作将一次动作的总反事实效应分解为后续智能体动作和状态路径效应；
2. 是否已有工作用差异回报、未来条件基线或贡献分配处理后续行为混叠；
3. 是否已有工作把累计成本作为显式约束，而非局部动作标签。

冻结排除项：

- 不为已完成的小论文补充引用；
- 不检索 GNN 结构以替代当前信用语义问题；
- 不把 AirDefense 场景换名、增加损失头或增加参数量视为创新；
- 不以摘要相似度代替公式、目标和干预语义比较。

## 2. 检索流程

采用“概念拆分—多源检索—主文核对—去重—最近工作矩阵”的流程。
学术检索连接器在本次运行中不可用，因此实际执行使用公开网页检索、出版方
页面和已下载原文；这一降级路径不改变“仅用一手来源确认关键主张”的规则。

来源优先级：

1. 会议或期刊官方页面、PMLR、NeurIPS Proceedings；
2. 作者公开原文或 arXiv 原文；
3. DOI/PubMed 只用于元数据交叉核对；
4. 二手综述只用于发现候选，不承担差异结论。

去重键依次为 DOI、规范化标题、标题—第一作者—年份。会议版和 arXiv
版本合并为同一工作，以正式出版版本为准。

## 3. 查询族

| 查询族 | 代表性检索式 | 目的 |
| --- | --- | --- |
| 反事实信用 | `multi-agent counterfactual credit assignment future actions baseline` | 找 COMA、CCA、COCOA 等 |
| 差异回报 | `difference rewards multi-agent policy gradient potential shaping` | 判断局部差值与策略偏差边界 |
| 因果分解 | `counterfactual effect decomposition multi-agent sequential decision making` | 检查“分解”是否已有直接先例 |
| 安全/约束 RL | `constrained policy optimization cumulative cost multi-agent` | 判断全局成本约束的新颖性 |
| 动态合法集 | `autoregressive multi-agent action masking counterfactual credit` | 查找与本项目同一步后缀替代的最近接口 |
| 关系结构 | `graph neural network resource allocation multi-agent reinforcement learning` | 仅确认 GNN 不是当前语义创新入口 |

中文补充词包括“多智能体 反事实 信用分配”“差异回报”“约束强化学习
累计成本”“自回归 动态动作掩码”。中文数据库需在形成投稿版综述前人工
补查；本次 N1 判决不依赖中文独占主张。

## 4. 纳入与排除

纳入条件：

- 给出可核对的训练目标、反事实干预、分解公式或约束形式；
- 与协作式多智能体、序贯决策、信用分配或累计成本至少一项直接相关；
- 能对候选 A/B/C 的创新距离或语义风险产生影响。

排除条件：

- 只有应用场景相似而没有方法对应；
- 只提出更强网络结构，不处理信用标签或约束目标；
- 只讨论解释性但无法明确其与训练接口的差异；
- 无法定位原始论文或正式元数据。

## 5. 纳入的核心工作

| 工作 | 原始来源 | 对 N1 的作用 |
| --- | --- | --- |
| Counterfactual Multi-Agent Policy Gradients, 2018 | [AAAI PDF](https://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/foersteraaai18.pdf) | 单智能体动作反事实基线的基础参照 |
| Difference Rewards Policy Gradients, 2018 | [NeurIPS](https://proceedings.neurips.cc/paper/2018/hash/94bb077f18daa6620efa5cf6e6f178d2-Abstract.html) | 差异回报与多智能体梯度 |
| Counterfactual Credit Assignment in Model-Free Reinforcement Learning, 2021 | [PMLR](https://proceedings.mlr.press/v139/mesnard21a.html) | 未来条件基线及偏差约束 |
| Difference Advantage Estimation for Multi-Agent Policy Gradients, 2022 | [PMLR](https://proceedings.mlr.press/v162/li22w.html) | 多步差异回报的信用—策略偏差权衡 |
| Counterfactual Credit Assignment with Hindsight, 2023 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d8bd445c2abe1343cce0e14b361b2fb3-Abstract-Conference.html) | 动作对未来奖励贡献的反事实估计 |
| Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making, 2025 | [PMLR](https://proceedings.mlr.press/v267/triantafyllou25a.html) | 对后续智能体动作和状态路径效应的直接分解 |
| Multi-level Advantage Credit Assignment, 2025 | [PMLR](https://proceedings.mlr.press/v258/zhao25c.html) | 多层级 advantage 信用的最近工作 |
| Constrained Policy Optimization, 2017 | [PMLR](https://proceedings.mlr.press/v70/achiam17a.html) | 累计成本约束基线 |
| Learning Pareto-Optimal Policies with Constraints, 2022 | [PMLR](https://proceedings.mlr.press/v164/huang22a.html) | 多目标与约束偏好学习 |
| Scalable Multi-Agent Reinforcement Learning for Safe Control, 2024 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fa76985f05e0a25c66528308dda33de0-Abstract-Conference.html) | 安全多智能体累计约束的直接邻域 |
| Counterfactual Influence in Markov Decision Processes, 2025 | [PMLR](https://proceedings.mlr.press/v275/kazemi25a.html) | 反事实路径漂移到干预分布的风险 |
| Modular Credit Assignment for Multi-Agent Reinforcement Learning, 2021 | [PMLR](https://proceedings.mlr.press/v139/chang21b.html) | 模块化信用的结构参照 |

关系图网络检索另发现空战任务中的 GNN-MARL 直接应用，说明“加入 GNN”
本身不能构成当前创新：

- [Graph Neural Network-Based Multi-Agent Reinforcement Learning for Air
  Combat](https://pubmed.ncbi.nlm.nih.gov/41115084/)
- [Graph reinforcement learning for cooperative air combat
  decision-making](https://www.nature.com/articles/s41598-026-55576-9)

## 6. 完整性与限制

- 检索截止到 2026-07-28，结论是阶段性查新，不等同于专利级全球穷尽检索。
- ICML 2025 因果分解论文已下载原文并核对核心定义与定理，不仅依据摘要。
- 对候选 B 的否决不是“方法无效”，而是其作为新颖核心已被约束 RL
  文献直接覆盖。
- 对候选 C 的否决同时来自创新距离和目标分布风险，不依赖单篇论文。
- 若后续提出新的 Problem–Method–Insight，必须基于新术语重新执行本协议，
  不能沿用本次“已查新”标签。

