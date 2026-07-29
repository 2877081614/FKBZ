# W1-03 稿件追溯矩阵

更新时间：2026-07-24  
主张权威：`docs/project/first_innovation_claim_evidence_matrix.md`  
数值权威：`evidence_source_index.md`  
规则：本矩阵只追踪去向，不重新定义 C1-C8

## 1. Claim 到正文、图表和证据

| Claim ID | 当前状态 | Evidence ID | 章节 | Paragraph ID | Figure/Table | 允许动词 | 禁止外推 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | 支持 | EV-R1-01、EV-R1-02、EV-R2-10 | §1、§3、§6.2、§7.1-§7.3、§9 | I02-I03、RW01-RW02、PF03、RES-6.2-01 至 RES-6.2-03、RES-6.4-04、D00、D01、D02a-D02b、C01 | Fig. 1、Fig. 3、Fig. 4 | `show`、`reveal mixing`、`is consistent with` | 不写任意环境/策略均发生；不把非正成本等同于所有符号掩盖 |
| C2 | 支持 | EV-R2-03 至 EV-R2-07、EV-R2-13 | §1-§5、§6.3、§7.1-§7.3、§9 | I03-I04、RW03、RW04b、RW05、M04-M07、P03-P04、RES-6.3-01 至 RES-6.3-03、D00、D01、D02a-D02e、D03、C01 | Fig. 2、Fig. 4、Table 3 | `establish the ledger identity`、`exactly reconstruct` | “精确”只指逐账本代数误差；不写统计无偏或因果完备 |
| C3 | 支持 | EV-R2-01、EV-R2-02、EV-R2-07 至 EV-R2-10 | §1、§5、§6.4、§7.1、§9 | I05、P01-P04、RES-6.4-01 至 RES-6.4-04、D00、C01 | Fig. 3、Table 1、Table 2 | `replicate`、`independently confirm`、`show across three new seeds` | 不写分布外泛化、任意种子、任意算法或新环境 |
| C4 | 否决普遍性 | EV-R2-11 至 EV-R2-13、BD-03 | §1、§6.5、§7.1、§7.4、§8、§9 | I05、RES-6.5-01 至 RES-6.5-03、D00、D04、L01、C01 | Fig. 5、Table 3、Table 4 | `differs by`、`is conditional on`、`does not support universality` | 不写 missile/laser 同强度；不删除 missile 未达门槛 |
| C5 | 否决 | EV-R1-03、EV-BPCE-01 | §6.1、§6.6、§7.3、§8 | RES-6.1-01、RES-6.6-01 至 RES-6.6-04、D02d、L03 | Table S1、Table S2 | `did not pass`、`does not support a universal label` | 不写所有弹药均无价值；不挑选 seed9 训练 oracle |
| C6 | 否决 | EV-BPCE-02 | §1、§7.5、§8、§9 | I06、D05、L03、C01 | Limitations 正文 | `failed the gate`、`remained unstable` | 不写已改进 PPO、已解决 all-noop 或已形成 MCH-PPO |
| C7 | 不支持 | BD-01 | §1、§6.1、§6.6、§7.5、§8、§9 | I06、RW05、RES-6.1-02、RES-6.6-04、D05、L03-L04、C01 | — | `is positioned as`、`provides a diagnostic basis` | 不写完整算法创新、性能优势或投稿级通用方法 |
| C8 | 未验证 | BD-02 | §1、§3.4、§7.5、§8、§9 | I06、PF04、D05、L04、C01 | — | `remains untested`、`requires separate validation` | 不写 GNN 能修复、提升信用或作为当前贡献 |

## 2. 文献定位追溯

| 论证 | 文献证据 | 使用位置 | 允许结论 |
| --- | --- | --- | --- |
| 全局结果到局部信用是已知问题 | E01-E05 | I02、RW01 | 当前 Problem 有先例 |
| 后续动作和行为传播已有研究 | E06-E09、E19、E20 | RW02、D02b | 不主张一般介导效应首创 |
| 顺序动作和动态掩码已有研究 | E12-E14、E21、E22 | I03、RW03、D02c、D03 | 差异在动态后缀成本账本 |
| CRN 只负责方差缩减 | E15 | RW04b、M02、D01、D02e | CRN 不解除结构混合 |
| 约束 MARL 不等于局部资源信用 | E16-E18 | RW04a、D02d | 预算约束与成本归属分开 |
| L2 定位 | W1-02 decision | I05-I06、RW05、D05 | 方法论文组成模块 |

## 3. 图表到权威数据

| Figure/Table | Evidence ID | 唯一文件/字段入口 | 复核责任 |
| --- | --- | --- | --- |
| Fig. 1 | T01-T08、BD-01 | 术语账本与公式冻结文件 | W1-05/W1-06 |
| Fig. 2 | EV-R2-03 至 EV-R2-07 | R2 `gate_summary.json`、首轮/正式账本 CSV、冻结公式 | W1-05/W1-06 |
| Fig. 3 | EV-R1-01、EV-R1-02、EV-R2-08 至 EV-R2-10 | R1/R2 context CSV、R1/R2 gate JSON | W1-04/W1-06 |
| Fig. 4 | EV-R2-10、EV-R2-13 | R2 `context_substitution_estimates.csv` | W1-04/W1-06 |
| Fig. 5 | EV-R2-11 至 EV-R2-13、BD-03 | `resource_type_summary.csv`、`scenario_boundary_summary.csv`、R2 gate JSON | W1-04/W1-06 |
| Table 1 | EV-R2-01 至 EV-R2-03 | R2 config JSON | W1-05/W1-06 |
| Table 2 | EV-R1-01、EV-R2-01、EV-R2-02、EV-R2-07 | R1/R2 gate JSON、R2 manifest | W1-05/W1-06 |
| Table 3 | EV-R2-06 至 EV-R2-11、BD-03 | R2 gate JSON、冻结代码判据 | W1-06 |
| Table 4 | EV-R2-11 至 EV-R2-13 | scenario/resource summary CSV | W1-06 |
| Table S1 | EV-BPCE-01 | label semantics/short-horizon gate JSON | W1-04/W1-06 |
| Table S2 | EV-R1-03 | R1 opportunity gate JSON | W1-04/W1-06 |
| Table S3 | EV-R2-04 至 EV-R2-07 | R2 gate/config JSON、预修正归档 | W1-05/W1-06 |

## 4. 下游更新权限

| 任务 | 可更新字段 | 禁止修改 |
| --- | --- | --- |
| W1-04 Results | 最终 Results 段落号、主文结果图表号 | Claim 状态、Evidence 数字、允许/禁止外推 |
| W1-05 Methods | 最终 Methods 段落号、方法表/补充方法号 | 公式方向、统计单位、N/E 身份 |
| W1-06 Figures | 最终图表文件名、panel、caption 追溯 | 科学主张、聚合口径、资源类型边界 |
| W1-07 Discussion | 最终 Discussion 段落号 | 将否决主张改为支持 |
| W1-08 Framing | Introduction/Related Work/Abstract 段落号 | L2 定位、优先权边界 |

## 5. 空缺处理

- 若正文需要一个不在 C1-C8 中的新科学主张，先返回 Claim–Evidence 矩阵评审；
- 若图表数字没有 Evidence ID，不得进入主文；
- 若 Evidence ID 指向多个冲突数字，按 W1-01 冲突处理流程暂停该段；
- 若目标期刊改变章节编号，只更新“章节/Paragraph ID”，不改变科学映射。

## 6. W1-05 Methods 与 Supplement 最终落点

| Paragraph ID | 最终章节 | 补充位置 | 主要复现材料 | Evidence/Freeze |
| --- | --- | --- | --- | --- |
| PF01 | §3.1 AirDefense v1 动态资源分配环境 | S1 | 环境配置、状态、动作、转移和奖励/成本分离 | AirDefense v1 design |
| PF02 | §3.2 冻结的因子化联合策略 | S2 | factorized action head、joint PPO、无冲突后缀 | T01-T03 |
| PF03 | §3.3 局部资源信用估计对象 | S4-S5 | N/E 身份、累计成本方向和冻结策略估计对象 | T04-T15 |
| PF04 | §3.4 研究范围与非主张 | — | 三场景边界、算法与 GNN 非主张 | BD-01、BD-02 |
| M01 | §4.1 N/E 成对干预 | S4 | 快照、前缀、N/E 干预和动态后缀 | T04-T08 |
| M02 | §4.2 共同随机数与目标精确边缘化 | S4 | 环境随机带、策略 uniform tape | T05、E15 |
| M03 | §4.2 共同随机数与目标精确边缘化 | S4-S5 | 合法目标条件概率精确边缘化 | EV-R2-03 |
| M04 | §4.3 同一步与未来成本账本 | S5 | 当前探针直接成本 | T08 |
| M05 | §4.3 同一步与未来成本账本 | S5；科研完整性披露 | 同一步其他单元替代 | T09、EV-R2-04、EV-R2-05 |
| M06 | §4.3 同一步与未来成本账本 | S5 | 未来 probe/other 与总替代成本 | T10-T15 |
| M07 | §4.4 回合成本恒等式与符号掩盖 | S5 | 完整恒等式与 \(10^{-6}\) 容限 | EV-R2-06 |
| M08 | §4.4 回合成本恒等式与符号掩盖 | S5 | \(\rho_{\mathrm{sub}}\)、\(I_{\mathrm{mask}}\) | T16、T17 |
| M09 | §4.5 可辨识边界 | S4 | 冻结 Actor、分支合法性与作用域 | T18、E19、E20 |
| P01 | §5.1 发现与独立确认 | S6 | A/B/C、短视窗和 R1/R2 职责 | EV-R1-01、EV-R2-01 |
| P02 | §5.2 来源模型与上下文独立性 | S2-S3 | 9 模型、108 上下文、旧 hash 零重叠 | EV-R2-01、EV-R2-02 |
| P03 | §5.3 配对采样与统计单位 | S4、S7 | 32 repeats、五级统计单位 | EV-R2-03 |
| P04 | §5.4 统计区间与确认门控 | S5、S7 | 完整性门控、Actor 冻结、容差 | EV-R2-06、EV-R2-07 |
| P05 | §5.4 统计区间与确认门控 | S7 | P-C1/P-C2/P-C3 完整判据 | 正式冻结门控 |

W1-05 复现材料总索引为 `reproducibility_map.md`；首轮账本修正的完整时间线与
不变项由 `research_integrity_disclosure.md` 管理。上述更新只填写方法落点，
未改变 Claim 状态、Evidence 数值、N/E 方向或统计单位。

## 7. W1-07 Discussion 与 Limitations 最终落点

| Paragraph ID | 最终章节 | 功能 | 证据/文献 | 强制边界 |
| --- | --- | --- | --- | --- |
| D00 | §7.1 | 中心推进 | C1-C4、EV-R2-06、EV-R2-09、BD-03 | 测量诊断，不是通用算法 |
| D03 | §7.2 | 同一步结构混合 | EV-R2-04、EV-R2-05、E14 | 不排除顺序/冲突规则 |
| D01 | §7.2 | 方差与估计对象 | EV-R2-06、E15 | CRN 不提供结构识别 |
| D02a | §7.3 | 反事实信用比较 | E01-E04 | 不替代 COMA/difference rewards |
| D02b | §7.3 | 时序/因果比较 | E05-E09、E19、E20、E23 | 不主张介导效应首创 |
| D02c | §7.3 | 顺序与掩码比较 | E12-E14、E21、E22 | 不主张顺序鲁棒性 |
| D02d | §7.3 | 约束与资源比较 | E16-E18、EV-R1-03 | 不支持通用 opportunity oracle |
| D02e | §7.3 | CRN 比较 | E15 | 只负责方差控制 |
| D04 | §7.4 | 条件机制 | EV-R2-11 至 EV-R2-13、BD-03 | 低直接成本不是唯一归因 |
| D05 | §7.5 | 后续方法含义 | BD-01、BD-02、EV-BPCE-02 | BPCE/GNN 不是已验证贡献 |
| L01 | §8 | 环境/资源边界 | EV-R2-01、EV-R2-11 至 EV-R2-13 | P-C3 失败可见 |
| L02 | §8 | 策略/干预边界 | T18、E14、E15、E19、E20 | 不识别全部中介路径 |
| L03 | §8 | 成本/算法负边界 | EV-R1-03、EV-BPCE-02、BD-01 | 不写所有资源无价值 |
| L04 | §8 | 泛化/GNN 边界 | BD-02 | 跨算法/环境未验证 |

竞争解释的逐项证据和排除等级由 `rival_explanations_matrix.md` 管理。W1-07
没有改变 Claim 状态、文献定位 L2 或任何实验数字。

## 8. W1-08 框架章节、标题与摘要最终落点

| 交付物 | Paragraph ID / 决策 | 覆盖主张 | 强制边界 |
| --- | --- | --- | --- |
| Introduction | I01-I06 | C1-C4、C6-C8 | 已知信用问题不主张首创；三场景不是跨环境泛化 |
| Related Work | RW01-RW05 | C1、C2、C4、C5、C7 | 按机制综合；L2 测量/诊断定位 |
| Conclusion | C01 | C1-C4、C6-C8 | 不引入新证据；停止于测量、复现和条件边界 |
| 主标题 | T-A | C1、C2 | 不使用优先权或通用算法表述 |
| Abstract | 单段中文冻结稿 | C1-C4、C7 | 含 P-C3 失败、单环境和非算法边界 |
| 最终贡献 | Contribution 1-3 | C1-C4 | 删除 opportunity oracle、在线 PPO 和 GNN 候选贡献 |

W1-08 只完成框架写作与范围冻结，没有改变 Claim 状态、Evidence 数字、L2
定位或实验结论。标题、摘要、结论和一句话论证均使用“动态掩码序列资源分配
中的成对反事实成本审计”这一范围。

## 9. W1-09 中英文整稿最终落点

| 集成对象 | 中文位置 | 英文位置 | 追溯状态 |
| --- | --- | --- | --- |
| Abstract | A01 | A01 | C1-C4、C7 与决定性数字一致 |
| Introduction | I01-I06 | I01-I06 | gap、方法、证据预告和三项贡献一致 |
| Related Work | RW01-RW05 | RW01-RW05 | E01-E24 主题定位一致 |
| Problem Formulation | PF01-PF04 | PF01-PF04 | 环境、策略、估计对象和范围一致 |
| Cost Decomposition | M01-M09 | M01-M09 | N/E 方向、三分量公式和可辨识边界一致 |
| Experimental Protocol | P01-P06 | P01-P06 | 发现/确认、统计单位、门控和修正披露一致 |
| Results | RES-6.1-01 至 RES-6.6-04 | 相同 ID | C1-C5 的支持/失败状态一致 |
| Discussion | D00、D01-D05 | 相同 ID | 解释强度和文献边界一致 |
| Limitations | L01-L04 | L01-L04 | 环境、策略、成本和泛化边界一致 |
| Conclusion | C01 | C01 | 贡献—证据—意义—边界一致 |

中英文稿各包含 66 个唯一 Paragraph ID。完整数字、公式、场景、资源、门控和
强动词核对见 `bilingual_consistency_audit.md`；所有非阻断投稿占位见
`unresolved_placeholders.md`。W1-09 没有改变 Claim 状态、Evidence 数字、
L2 定位或任何实验结论。

## 10. W1-10 终审与 M2 出口

| 终审对象 | 交付物 | 裁决 |
| --- | --- | --- |
| 三审稿人压力测试 | `reviewer_pressure_test.md` | 无未关闭 R2/R3/RX |
| 最终主张证据 | `claim_evidence_final_audit.md` | C1-C3 支持；C4-C6 失败边界；C7-C8 非主张 |
| 终稿修订 | `revision_log.md` | 只增加选择透明度、统计边界和可用性声明 |
| 双语终稿 | `final_manuscript_zh.md`、`final_manuscript_en.md` | 66 个 Paragraph ID、数字和公式一致 |
| 期刊适配 | `target_journal_fit.md` | 目标期刊未定，不宣称格式定稿 |
| 投稿与可用性 | `submission_checklist.md`、`data_code_availability_draft.md` | M2 可移交；立即外部投稿 NO-GO |
| 阶段出口 | W1-10 | **L2/M2** |

W1-10 未新增实验、未改变 Evidence 数字或门控，也未恢复 BPCE/MCH-PPO、
opportunity oracle 或 GNN 主张。
