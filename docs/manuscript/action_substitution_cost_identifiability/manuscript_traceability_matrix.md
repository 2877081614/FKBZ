# W1-03 稿件追溯矩阵

更新时间：2026-07-24  
主张权威：`docs/project/first_innovation_claim_evidence_matrix.md`  
数值权威：`evidence_source_index.md`  
规则：本矩阵只追踪去向，不重新定义 C1-C8

## 1. Claim 到正文、图表和证据

| Claim ID | 当前状态 | Evidence ID | 章节 | Paragraph ID | Figure/Table | 允许动词 | 禁止外推 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | 支持 | EV-R1-01、EV-R1-02、EV-R2-10 | §1、§3、§6.2、§7.1 | I02-I03、PF03、RES-6.2-01 至 RES-6.2-03、RES-6.4-04、D01 | Fig. 1、Fig. 2 | `show`、`reveal mixing`、`is consistent with` | 不写任意环境/策略均发生；不把非正成本等同于所有符号掩盖 |
| C2 | 支持 | EV-R2-03 至 EV-R2-07、EV-R2-13 | §4、§5、§6.3 | M04-M07、P03-P04、RES-6.3-01 至 RES-6.3-03 | Fig. 2、Table 1 | `establish the ledger identity`、`exactly reconstruct` | “精确”只指逐账本代数误差；不写统计无偏或因果完备 |
| C3 | 支持 | EV-R2-01、EV-R2-02、EV-R2-07 至 EV-R2-10 | §5、§6.4 | P01-P04、RES-6.4-01 至 RES-6.4-04 | Fig. 3、Table 1 | `replicate`、`independently confirm`、`show across three new seeds` | 不写分布外泛化、任意种子、任意算法或新环境 |
| C4 | 否决普遍性 | EV-R2-11 至 EV-R2-13、BD-03 | §6.5、§7.4、§8 | RES-6.5-01 至 RES-6.5-03、D04、L01 | Fig. 4、Table 2 | `differs by`、`is conditional on`、`does not support universality` | 不写 missile/laser 同强度；不删除 missile 未达门槛 |
| C5 | 否决 | EV-R1-03、EV-BPCE-01 | §6.1、§6.6、§8 | RES-6.1-01、RES-6.6-01 至 RES-6.6-04、L03 | Table 2、Table S6 | `did not pass`、`does not support a universal label` | 不写所有弹药均无价值；不挑选 seed9 训练 oracle |
| C6 | 否决 | EV-BPCE-02 | §8 | L03 | Table 2、Table S6 | `failed the gate`、`remained unstable` | 不写已改进 PPO、已解决 all-noop 或已形成 MCH-PPO |
| C7 | 不支持 | BD-01 | §1、§6.1、§6.6、§7.5、§8 | I06、RES-6.1-02、RES-6.6-04、D05、L03 | — | `is positioned as`、`provides a diagnostic basis` | 不写完整算法创新、性能优势或投稿级通用方法 |
| C8 | 未验证 | BD-02 | §3.4、§8 | PF04、L04 | — | `remains untested`、`requires separate validation` | 不写 GNN 能修复、提升信用或作为当前贡献 |

## 2. 文献定位追溯

| 论证 | 文献证据 | 使用位置 | 允许结论 |
| --- | --- | --- | --- |
| 全局结果到局部信用是已知问题 | E01-E05 | I02、RW01 | 当前 Problem 有先例 |
| 后续动作和行为传播已有研究 | E06-E09、E19、E20 | RW02、D02 | 不主张一般介导效应首创 |
| 顺序动作和动态掩码已有研究 | E12-E14、E21 | I03、RW03、D03 | 差异在动态后缀成本账本 |
| CRN 只负责方差缩减 | E15 | RW04、M02、D01 | CRN 不解除结构混合 |
| 约束 MARL 不等于局部资源信用 | E16-E18 | RW04 | 预算约束与成本归属分开 |
| L2 定位 | W1-02 decision | I05-I06、RW05、D05 | 方法论文组成模块 |

## 3. 图表到权威数据

| Figure/Table | Evidence ID | 唯一文件/字段入口 | 复核责任 |
| --- | --- | --- | --- |
| Fig. 1 | T01-T08、BD-01 | 术语账本与公式冻结文件 | W1-05/W1-06 |
| Fig. 2 | EV-R1-01、EV-R1-02、EV-R2-04 至 EV-R2-06 | R1/R2 `gate_summary.json`、首轮账本 CSV | W1-04/W1-06 |
| Fig. 3 | EV-R2-08 至 EV-R2-10 | R2 `gate_summary.json`、`block_summary.csv` | W1-04/W1-06 |
| Fig. 4 | EV-R2-11 至 EV-R2-13 | `resource_type_summary.csv`、`scenario_boundary_summary.csv` | W1-04/W1-06 |
| Table 1 | EV-R2-01 至 EV-R2-03、EV-R2-07 | R2 config/gate JSON | W1-05/W1-06 |
| Table 2 | EV-R1-03、EV-BPCE-01、EV-BPCE-02、BD-03 | R1/BPCE gate JSON 与 Claim 矩阵 | W1-04/W1-06 |

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
