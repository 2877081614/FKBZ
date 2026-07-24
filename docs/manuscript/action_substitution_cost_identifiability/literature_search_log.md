# W1-02 文献检索日志

检索日期：2026-07-24  
执行者：Codex  
协议：`literature_search_protocol.md`

## 1. 正式查询日志

每组 OpenAlex 查询均使用 `--limit 20 --year-from 1990 --sort relevance_score --compact`。
“直接相关”表示题名/摘要触及本任务至少一个核心层；它不是最终纳入篇数。

| ID | 完整检索式 | 结果位 | 题名/摘要直接相关 | 主要发现 |
| --- | --- | ---: | ---: | --- |
| Q1 | `multi-agent credit assignment global reward local contribution` | 20 | 8 | 召回 COMA、Shapley Q-value、全局奖励信用分配等 |
| Q2 | `counterfactual baseline difference rewards multi-agent policy gradient` | 20 | 5 | 召回 COMA、Difference Rewards Policy Gradients 等 |
| Q3 | `temporal credit assignment delayed action effects return decomposition reinforcement learning` | 20 | 3 | 宽检索精度低；后续精确补入 RUDDER、HCA、CCA、COCOA |
| Q4 | `common random numbers paired simulation counterfactual policy evaluation` | 20 | 0 | 前 20 条未准确召回经典 CRN；通过精确题名补入 |
| Q5 | `action substitution reinforcement learning cost credit assignment` | 20 | 0 | 未发现该词组作为本项目机制的规范信用分配术语 |
| Q6 | `sequential autoregressive joint action allocation multi-agent reinforcement learning` | 20 | 0 | 宽检索精度低；通过会议与精确题名补入 MAT、HAPPO、CAPO |
| Q7 | `invalid action masking counterfactual evaluation policy gradient` | 20 | 0 | 通过精确题名补入 invalid action masking 理论工作 |
| Q8 | `resource shadow price opportunity cost constrained multi-agent reinforcement learning` | 20 | 1 | 约束 MARL 研究预算可行性，但不等价于局部机会成本账本 |
| Q9 | `episode return local action credit measurement bias identifiability reinforcement learning` | 20 | 2 | 查询词过宽；通过引用链回到 CCA、COCOA 和差分回报偏差分析 |

合计：180 个查询结果位。由于同一论文可被多个检索式召回，且预印本与正式出版版并存，
本日志不把结果位误报为去重文献量。经合并和补充检索后，24 篇原始工作进入证据矩阵。

## 2. 关键补充检索日志

| 检索式/标题 | 数据源 | 核读结果 |
| --- | --- | --- |
| `"action substitution" reinforcement learning cost` | Web、OpenAlex | 主要含义是安全控制器替换动作或人类干预，不是本项目的成本信用机制 |
| `"action displacement" reinforcement learning credit assignment` | Web、OpenAlex | 未找到稳定的信用分配术语用法；部分结果指图结构位移 |
| `"policy-induced substitution" reinforcement learning` | Web、OpenAlex | 未形成规范术语簇 |
| `"downstream action substitution" multi-agent reinforcement learning` | Web、OpenAlex | 未发现同行评议论文以此作为固定方法名 |
| `Difference Rewards Policy Gradients` | Springer 原始页面 | 在 Dec-POMDP 中，未来局部历史依赖当前动作可破坏基线独立性；与本任务动机高度接近 |
| `Counterfactual Credit Assignment in Model-Free Reinforcement Learning` | PMLR、原始 PDF | 明确要求从外部因素和后续动作中分离当前动作影响 |
| `Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making` | PMLR、原始 PDF | 将总反事实效应拆为经后续智能体行为和状态转移传播的效应；是最近理论工作 |
| `Agent-Specific Effects` | PMLR/arXiv 原始记录 | 分析一个智能体动作通过其他智能体响应传播的因果效应 |
| `CAPO: Counterfactual Credit Assignment in Sequential Cooperative Teams` | arXiv:2604.17693 | 固定顺序协作团队、SeqAU 和逐智能体信用；是最近算法预印本 |
| `A Closer Look at Invalid Action Masking in Policy Gradient Algorithms` | arXiv:2006.14171 | 证明掩码策略梯度的有效性和实用性，但未进行下游成本分解 |
| `Some Guidelines and Guarantees for Common Random Numbers` | INFORMS/DOI | CRN 是系统比较的方差缩减工具，不能单独消除估计对象的结构性混合 |

## 3. 原始来源访问记录

| 来源 | 标识 | 阅读深度 | 用途 |
| --- | --- | --- | --- |
| Wolpert & Tumer, 2002 | DOI `10.1142/S0219525901000188` | B | 差分奖励/个体效用基础 |
| Foerster et al., 2018, COMA | DOI `10.1609/aaai.v32i1.11794` | A | 固定其他智能体动作的反事实基线 |
| Castellini et al., 2022 | DOI `10.1007/s00521-022-07960-5` | A | 未来历史依赖当前动作导致偏差 |
| Mesnard et al., 2021, CCA | PMLR 139 | A | 分离后续动作和外部随机性 |
| Harutyunyan et al., 2019, HCA | NeurIPS 2019 | B | 基于事后结果的时序信用 |
| Meulemans et al., 2023, COCOA | arXiv `2306.16803` | A | 贡献对象选择与虚假信用 |
| Arjona-Medina et al., 2019, RUDDER | NeurIPS 2019 | B | 长延迟回报重分配 |
| Wu et al., 2018 | arXiv `1803.07246` | B | 因子动作的动作依赖基线 |
| Tucker et al., 2018 | PMLR 80 | B | 动作依赖基线的偏差/收益边界 |
| Kuba et al., 2022, HAPPO/HATRPO | arXiv `2109.11251` | B | 顺序更新与多智能体优势分解 |
| Wen et al., 2022, MAT | arXiv `2205.14953` | B | 自回归多智能体联合动作 |
| Huang & Ontañón, 2020 | arXiv `2006.14171` | A | 动态非法动作掩码 |
| Glasserman & Yao, 1992 | DOI `10.1287/mnsc.38.6.884` | B | 配对 CRN 的能力边界 |
| Achiam et al., 2017, CPO | PMLR 70 | B | 约束策略优化 |
| Gu et al., 2021, MACPO | arXiv `2110.02793` | B | 多智能体约束策略优化 |
| Zhang et al., 2023, GRD | NeurIPS 2023 | B | 因果回报重分配与可辨识性 |
| Triantafyllou et al., 2024 | arXiv `2310.11334` | A | 经其他智能体响应传播的动作效应 |
| Triantafyllou et al., 2025 | PMLR 267 | A | 多智能体反事实效应路径分解 |
| Deshmukh et al., 2026, CAPO | arXiv `2604.17693` | A | 顺序协作团队反事实信用 |

## 4. 术语核验结论

截至 2026-07-24，在上述检索范围内：

- 未发现 `action substitution` 被稳定用作“当前动作改变后续合法动作/策略响应，进而替代资源消耗”的规范信用分配术语；
- 该词在安全强化学习中容易被理解为“安全控制器替换执行动作”；
- 当前机制与因果文献中的 `mediated effect`、`effect propagation through subsequent actions`
  更接近。

因此建议正文使用“后续动作介导的成本替代”
(`downstream action-mediated cost substitution`) 作为总称，并区分：

- 同一步后缀替代：`same-step suffix substitution`；
- 未来策略介导替代：`future policy-mediated substitution`。

该结论是术语变更建议，不直接覆盖 W1-01 冻结账本。

## 5. 可复核性说明

- 查询式、日期、范围、排序和截断数均已记录；
- 最近工作均回到原始出版页面或作者预印本；
- 宽检索未召回的论文通过精确标题和引用链补足；
- 未把综述、搜索摘要或二手网页当作优先权证据；
- 未以“检索不到”等价证明“绝对不存在”。
