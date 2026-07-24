# W1-02 文献证据矩阵

更新时间：2026-07-24  
纳入数量：24 篇原始工作  
判定对象：动作替代成本可辨识性，不代表对各论文整体价值的评价

## 1. 证据矩阵

| ID | 原始工作 | 主题 | 与本项目的重叠 | 未覆盖的关键边界 | 风险 | 阅读 |
| --- | --- | --- | --- | --- | --- | --- |
| E01 | [Wolpert & Tumer, 2002](https://doi.org/10.1142/S0219525901000188) | Difference rewards | 用反事实默认动作隔离成员对集体结果的边际贡献 | 未研究动态掩码、顺序后缀响应和资源成本账本 | 中 | B |
| E02 | [Foerster et al., 2018, COMA](https://doi.org/10.1609/aaai.v32i1.11794) | MARL 反事实基线 | 对单个智能体动作边缘化并固定其他动作 | 固定其他动作不能表达当前动作改变后缀合法集的情形 | 高 | A |
| E03 | [Wang et al., 2020, Shapley Q-value](https://doi.org/10.1609/aaai.v34i05.6220) | Shapley 信用 | 将全局价值分配给个体动作贡献 | 不分离同一步/未来替代，也不针对成本符号 | 中 | B |
| E04 | [Nguyen et al., 2018](https://papers.neurips.cc/paper_files/paper/2018/hash/94bb077f18daa6620efa5cf6e6f178d2-Abstract.html) | 全局奖励信用 | 从全局奖励估计个体贡献 | 未建立配对反事实成本恒等式 | 低 | B |
| E05 | [Castellini et al., 2022](https://doi.org/10.1007/s00521-022-07960-5) | Difference-return PG | 明确指出未来局部历史依赖当前动作时，差分回报基线可失去无偏性 | 未给动态掩码序列分配的精确资源账本和符号边界 | 很高 | A |
| E06 | [Mesnard et al., 2021, CCA](https://proceedings.mlr.press/v139/mesnard21a.html) | 反事实时序信用 | 分离当前动作、外部因素和后续动作对未来奖励的影响 | 目标是低方差策略梯度，不是资源成本替代恒等分解 | 很高 | A |
| E07 | [Harutyunyan et al., 2019, HCA](https://proceedings.neurips.cc/paper/2019/hash/195f15384c2a79cedf293e4a847ce85c-Abstract.html) | Hindsight credit | 用未来结果信息改善时序信用 | 未显式拆分后续行为介导的资源消耗 | 中 | B |
| E08 | [Meulemans et al., 2023, COCOA](https://arxiv.org/abs/2306.16803) | Counterfactual contribution | 指出按状态而非奖励对象归因可产生虚假贡献 | 未覆盖动态合法动作后缀和成本账本恒等式 | 高 | A |
| E09 | [Arjona-Medina et al., 2019, RUDDER](https://proceedings.neurips.cc/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html) | 延迟回报分解 | 将延迟回报重分配到关键动作 | 解决时间延迟，不区分行为介导替代 | 中 | B |
| E10 | [Wu et al., 2018](https://arxiv.org/abs/1803.07246) | 因子动作基线 | 对条件独立动作因子构造动作依赖基线 | 本项目因子经掩码和前序动作条件化，并非独立因子 | 中 | B |
| E11 | [Tucker et al., 2018](https://proceedings.mlr.press/v80/tucker18a.html) | 动作依赖基线审计 | 讨论动作依赖基线的偏差和实际收益边界 | 不涉及多智能体/自回归资源替代 | 低 | B |
| E12 | [Kuba et al., 2022, HAPPO/HATRPO](https://arxiv.org/abs/2109.11251) | 顺序策略更新 | 多智能体优势分解和顺序更新 | 不测量一个动作诱发的后续资源成本变化 | 中 | B |
| E13 | [Wen et al., 2022, MAT](https://arxiv.org/abs/2205.14953) | 自回归 MARL | 把联合动作建模为智能体动作序列 | 未讨论动态掩码下局部成本标签的可辨识性 | 中 | B |
| E14 | [Huang & Ontañón, 2020](https://arxiv.org/abs/2006.14171) | Invalid action masking | 分析掩码策略梯度的理论有效性与样本效率 | 不处理动作改变后续掩码后产生的反事实成本差 | 中 | A |
| E15 | [Glasserman & Yao, 1992](https://doi.org/10.1287/mnsc.38.6.884) | Common random numbers | 用共同随机数降低两个随机系统差值的方差 | CRN 不改变估计量混合了哪些因果路径 | 高 | B |
| E16 | [Achiam et al., 2017, CPO](https://proceedings.mlr.press/v70/achiam17a.html) | 约束 RL | 在期望成本约束下更新策略 | 预算可行性不等价于动作级机会成本归属 | 低 | B |
| E17 | [Gu et al., 2021, MACPO](https://arxiv.org/abs/2110.02793) | 约束 MARL | 多智能体约束和安全成本优化 | 未分解局部动作与后续策略响应的成本 | 中 | B |
| E18 | [Zhang et al., 2024](https://doi.org/10.52202/079017-4400) | 可扩展约束 MARL | 扩展多智能体安全约束优化 | 仍以约束满足为核心，不是信用估计量审计 | 低 | B |
| E19 | [Triantafyllou et al., 2024, Agent-Specific Effects](https://arxiv.org/abs/2310.11334) | 多智能体因果效应 | 研究动作效应如何经其他智能体响应传播 | 未针对动态合法动作和资源成本符号给出操作账本 | 很高 | A |
| E20 | [Triantafyllou et al., 2025](https://proceedings.mlr.press/v267/triantafyllou25a.html) | 反事实效应分解 | 将总效应拆为经后续智能体行为和状态转移传播的效应 | 未分离同一步掩码后缀、未来探针/其他单元成本及其符号边界 | 很高 | A |
| E21 | [Deshmukh et al., 2026, CAPO](https://arxiv.org/abs/2604.17693) | 顺序团队信用 | 固定顺序团队的 SeqAU、奖励分解和逐智能体反事实优势 | 预印本；未覆盖动态掩码、精确成本恒等式和场景/资源类型边界 | 很高 | A |
| E22 | [Li et al., 2026, CCPO](https://arxiv.org/abs/2603.21563) | 协作智能体信用 | 对顺序/投票协作构造动态反事实基线 | 面向 LLM 协作，未处理本项目成本测量对象 | 中 | B |
| E23 | [Zhang et al., 2023, GRD](https://proceedings.neurips.cc/paper_files/paper/2023/hash/402e12102d6ec3ea3df40ce1b23d423a-Abstract-Conference.html) | 因果回报重分配 | 识别潜在 Markov 奖励和因果结构，讨论回报分解的可辨识性 | 面向延迟奖励，不分离下游智能体行为介导的资源成本 | 中 | B |
| E24 | [Li et al., 2026, Counterfactual Shapley](https://arxiv.org/abs/2607.16999) | 因果 Shapley 信用 | 用反事实 Shapley 重分配时序信用 | 极新预印本；未提供本项目的顺序成本替代边界 | 中 | B |

## 2. 主题覆盖

| 检索主题 | 支撑文献 | 对当前定位的约束 |
| --- | --- | --- |
| 多智能体信用 | E01-E05 | “全局结果不能直接代表局部贡献”是已知问题 |
| 反事实/差分基线 | E01、E02、E05-E08 | 不能把反事实信用本身写成新意 |
| 时序与延迟效应 | E06-E09 | 后续动作与外部随机性混合已有明确讨论 |
| CRN 配对仿真 | E15 | CRN 只能降低方差，不能替代结构分解 |
| 顺序/自回归动作 | E12、E13、E21 | 顺序团队并非新设置；动态掩码资源账本仍有差异 |
| 动作掩码 | E14 | 掩码 PG 有先例，但“掩码改变反事实后缀”未被等价覆盖 |
| 资源与约束 | E16-E18 | 约束优化不等价于局部资源机会成本 |
| 因果路径分解 | E19、E20 | “后续智能体行为是中介路径”已有最接近理论先例 |
| 方法与算法前沿 | E21-E24 | 因果回报分解和顺序信用持续发展，定位必须保守 |

## 3. 证据结论

1. **已知部分**：全局/回合结果对局部动作贡献的混合、后续动作引入的信用困难、
   反事实基线和因果路径分解均已有直接先例。
2. **仍有实质差异**：当前项目把动态合法掩码下的同一步后缀替代、未来探针替代和
   未来其他单元替代写成精确资源成本恒等式，并审计符号掩盖和资源类型边界。
3. **不能支撑的定位**：不能声称提出一般意义上的动作介导反事实效应、反事实信用或
   顺序团队信用分配。
4. **可支撑的定位**：可作为更大方法论文中的测量诊断与资源信用分解模块，当前对应 L2。
