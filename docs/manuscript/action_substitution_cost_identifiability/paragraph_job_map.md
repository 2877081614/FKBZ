# W1-03 段落工作表

更新时间：2026-07-24  
规则：每个 Paragraph ID 只有一个主要功能；完整正文尚未在本任务中撰写

## 1. Introduction 与 Related Work

| Paragraph ID | Section | 单一功能 | Claim | Evidence | Figure/Table | 边界句 |
| --- | --- | --- | --- | --- | --- | --- |
| I01 | Introduction | context | — | — | — | 只建立动态资源分配的重要性，不宣称方法优势 |
| I02 | Introduction | gap | C1 | E01-E09、E19、E20 | — | 团队回报到局部信用的问题本身不是本文首创 |
| I03 | Introduction | gap | C1、C2 | E12-E14、E21 | Fig. 1 | 缺口限定为动态掩码后缀和资源成本操作账本 |
| I04 | Introduction | approach | C2 | T04-T18、EV-R2-06 | Fig. 1、Fig. 2 | “精确”仅指代数恒等式 |
| I05 | Introduction | result | C3、C4 | EV-R2-08 至 EV-R2-13 | Fig. 3、Fig. 5 | 复现不等于跨环境泛化，边界结果必须并列 |
| I06 | Introduction | limitation | C6-C8 | BD-01、BD-02、EV-BPCE-02 | — | 不主张在线 PPO 改进或 GNN 修复 |
| RW01 | Related Work | comparison | C1 | E01-E05 | — | difference reward/COMA 是直接先例 |
| RW02 | Related Work | comparison | C1 | E06-E09、E19、E20、E23 | — | 后续动作介导效应不是新的普遍概念 |
| RW03 | Related Work | comparison | C2 | E12-E14、E21、E22 | — | 自回归与掩码本身不是贡献 |
| RW04a | Related Work | comparison | C5 | E16-E18 | Table S2 | 预算约束与局部成本归属不是同一问题 |
| RW04b | Related Work | comparison | C2 | E15 | Fig. 2 | CRN 只降低方差，不提供结构可辨识性 |
| RW05 | Related Work | implication | C2、C4、C7 | W1-02 L2 decision | — | 当前差异是操作化账本与条件边界，不是一般算法优先权 |

## 2. Problem Formulation、Method 与 Protocol

| Paragraph ID | Section | 单一功能 | Claim | Evidence | Figure/Table | 边界句 |
| --- | --- | --- | --- | --- | --- | --- |
| PF01 | 3.1 Environment | context | — | AirDefense v1 design | Fig. 1 | 仅保留理解资源成本所需环境定义 |
| PF02 | 3.2 Frozen policy | method | C1 | T01-T03 | Fig. 1 | factorized 是条件分解，不是假设单元独立 |
| PF03 | 3.3 Estimand | method | C1 | T04-T15 | — | 估计量限定冻结策略和局部 N/E 干预 |
| PF04 | 3.4 Scope | limitation | C7、C8 | BD-01、BD-02 | — | 不评估在线算法和 GNN 修复 |
| M01 | 4.1 N/E intervention | method | C1、C2 | T04-T08 | Fig. 1 | N/E 身份和方向不可反转 |
| M02 | 4.2 CRN | method | C2 | T05、E15 | Fig. 2a | CRN 只降低方差，不提供结构可辨识性 |
| M03 | 4.2 Target marginalization | method | C2 | EV-R2-03 | Fig. 2b | 只消除目标采样误差 |
| M04 | 4.3 Direct cost | method | C2 | T08 | Fig. 2c、Fig. 4a | 只含当前探针单元直接成本 |
| M05 | 4.3 Same-step substitution | method | C2 | T09、EV-R2-04、EV-R2-05 | Fig. 2c-d、Fig. 4a-b | 不得并入 future-only 项 |
| M06 | 4.3 Future substitution | method | C2 | T10-T13 | Fig. 2c、Fig. 4a-b | future probe 与 future other 分开定义 |
| M07 | 4.4 Cost identity | method | C2 | EV-R2-06 | Fig. 2c-d | 数值恒等式不自动证明统计无偏 |
| M08 | 4.4 Ratio and masking | method | C4 | T16、T17 | Fig. 4c、Fig. 5 | \(\rho_{\mathrm{sub}}\) 不是概率 |
| M09 | 4.5 Identifiability | limitation | C1-C4 | T18、E19、E20 | — | 只对给定干预、观测和冻结策略成立 |
| P01 | 5.1 Discovery/confirmation | method | C1、C3 | EV-R1-01、EV-R2-01 | Table 2 | R1 发现与 R2 确认职责不可混用 |
| P02 | 5.2 Independence | method | C3 | EV-R2-01、EV-R2-02 | Table 1、Table 2 | hash 零重叠不等于分布外泛化 |
| P03 | 5.3 Sampling units | method | C2、C3 | EV-R2-03 | Table 1、Table 2 | ledger row 不作为独立 context |
| P04 | 5.4 Integrity | method | C2、C3 | EV-R2-06、EV-R2-07 | Table 2、Table 3 | Actor 冻结只覆盖确认过程 |
| P05 | 5.4 Gates | method | C3-C6 | 正式预注册门控 | Table 3 | 失败门控不因写作重新定义 |

## 3. Results

| Paragraph ID | Section | 单一功能 | Claim | Evidence | Figure/Table | 边界句 |
| --- | --- | --- | --- | --- | --- | --- |
| RES-6.1-01 | 6.1 | result | C5 | EV-BPCE-01 | Table S1 | 仅针对冻结短视窗标签协议 |
| RES-6.1-02 | 6.1 | limitation | C7 | BD-01 | — | 问题收窄不等于算法性能失败 |
| RES-6.2-01 | 6.2 | method | C1 | EV-R1-01 | Fig. 3a | R1 仅使用旧策略种子 |
| RES-6.2-02 | 6.2 | result | C1 | EV-R1-01 | Fig. 3a | R1 成本量是未来替代成本 |
| RES-6.2-03 | 6.2 | result | C1 | EV-R1-02 | Fig. 3d | 独立性由 R2 提供 |
| RES-6.3-01 | 6.3 | result | C2 | EV-R2-04、EV-R2-05 | Fig. 2d | 首轮遗漏不是预注册正面发现 |
| RES-6.3-02 | 6.3 | result | C2 | EV-R2-13、账本字段 | Fig. 4a-b | 0.864/0.147/0.718 为 context 等权聚合 |
| RES-6.3-03 | 6.3 | result | C2 | EV-R2-06、EV-R2-07 | Fig. 2d、Table 3 | “精确”只指逐账本恒等式 |
| RES-6.4-01 | 6.4 | result | C3 | EV-R2-01 至 EV-R2-03、EV-R2-07 | Table 1、Table 2 | 新模型/上下文不等于新环境 |
| RES-6.4-02 | 6.4 | result | C3 | EV-R2-08 | Fig. 3b | 13/18 只限 time/resource contexts |
| RES-6.4-03 | 6.4 | result | C3 | EV-R2-09 | Fig. 3c | 三个 seed block 不外推到任意种子 |
| RES-6.4-04 | 6.4 | result | C1、C3 | EV-R2-10 | Fig. 3d | 机制一致性不是总体发生率 |
| RES-6.5-01 | 6.5 | result | C4 | EV-R2-13 | Fig. 5a、Table 4 | 场景聚合不能替代 seed 不确定性 |
| RES-6.5-02 | 6.5 | result | C4 | EV-R2-11、EV-R2-12 | Fig. 5b-c、Table 4 | 两种资源都必须报告 |
| RES-6.5-03 | 6.5 | result | C4 | BD-03 | Fig. 5d、Table 3 | P-C3 明确为失败门控 |
| RES-6.6-01 | 6.6 | method | C5 | EV-R1-03 | Table S2 | E/E-R 不改变当前交战结果 |
| RES-6.6-02 | 6.6 | result | C5 | EV-R1-03 | Table S2 | 只否决通用机会价值 |
| RES-6.6-03 | 6.6 | result | C5 | EV-R1-03、EV-BPCE-01 | Table S1、Table S2 | 行动集合扩大不等于安全收益 |
| RES-6.6-04 | 6.6 | limitation | C5、C7 | C5、BD-01 | Table S2 | 不外推为所有弹药无价值 |

## 4. Discussion、Limitations 与 Conclusion

| Paragraph ID | Section | 单一功能 | Claim | Evidence | Figure/Table | 边界句 |
| --- | --- | --- | --- | --- | --- | --- |
| D00 | 7.1 | implication | C1-C4 | EV-R2-06、EV-R2-09、BD-03 | Fig. 1-Fig. 5 | 中心推进是带边界的测量诊断，不是恒等式或算法胜出 |
| D03 | 7.2 | mechanism | C2 | EV-R2-04、EV-R2-05、E14 | Fig. 1、Fig. 2 | 动态掩码解释同一步后缀为何不能固定；不排除顺序效应 |
| D01 | 7.2 | mechanism | C1、C2 | EV-R2-06、E15 | Fig. 2、Fig. 4 | 区分结构混合与 Monte Carlo 方差 |
| D02a | 7.3 | comparison | C1、C2 | E01-E04 | — | 不替代 difference rewards、COMA 或 Shapley 信用 |
| D02b | 7.3 | comparison | C1、C2 | E05-E09、E19、E20、E23 | — | 与一般时序/因果效应分解互补 |
| D02c | 7.3 | comparison | C2 | E12-E14、E21、E22 | — | 自回归、掩码和顺序本身不是贡献 |
| D02d | 7.3 | comparison | C5 | E16-E18、EV-R1-03 | Table S2 | 预算约束与局部资源归属分开 |
| D02e | 7.3 | comparison | C2 | E15 | — | CRN 只负责方差控制 |
| D04 | 7.4 | mechanism | C4 | EV-R2-11、EV-R2-12 | Fig. 5 | 资源类型解释是条件机制，不是普遍定律 |
| D05 | 7.5 | implication | C6-C8 | BD-01、BD-02、EV-BPCE-02 | — | 只提出未来方法设计要求 |
| L01 | 8 | limitation | C3 | EV-R2-01、EV-R2-02 | — | 单环境、三策略种子、冻结策略 |
| L02 | 8 | limitation | C1、C2 | T18、E19、E20 | — | 不声称识别全部因果路径 |
| L03 | 8 | limitation | C5、C6、C7 | EV-R1-03、EV-BPCE-02、BD-01 | Table S2；C6 留正文 | 机会价值和在线改进均未通过 |
| L04 | 8 | limitation | C8 | BD-02 | — | GNN 继续冻结且不作为已验证修复 |
| C01 | 9 | implication | C1-C4、C6-C8 | EV-R2-06、EV-R2-09、EV-R2-11、BD-01 至 BD-03 | — | 结论止于测量、确认和条件边界 |

## 5. 延后项目

| 项目 | 状态 | 负责人 |
| --- | --- | --- |
| Title | 延后至 W1-08 | W1-08 |
| Abstract | 延后至 W1-08，且在 Results/Methods/Discussion 后 | W1-08 |
| 正式段落文本 | 本任务不写 | W1-04、W1-05、W1-07、W1-08 |
| 正式图表 | 本任务只给计划 | W1-06 |
