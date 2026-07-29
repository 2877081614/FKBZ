# W1-10 最终 Claim-Evidence 审计

更新时间：2026-07-28  
主张权威：`docs/project/first_innovation_claim_evidence_matrix.md`  
数值权威：`evidence_source_index.md` 与冻结机器结果  
出口定位：L2/M2

## 1. 三项最终贡献

| 贡献 | Problem/Method/Insight | 正文 | 图表 | 数据来源 | 审稿裁决 |
| --- | --- | --- | --- | --- | --- |
| Contribution 1：局部资源成本测量问题 | Problem：累计成本混合直接成本与动作替代 | I02-I06、RES-6.2、D00-D01 | Fig. 1、Fig. 3、Fig. 4 | EV-R1-01、EV-R1-02、EV-R2-10 | SUPPORTED，限冻结策略 |
| Contribution 2：三分量成对反事实账本 | Method：N/E、CRN、目标边缘化与 same/future 三分量 | M01-M09、RES-6.3 | Fig. 2、Fig. 4、Table 3 | EV-R2-03 至 EV-R2-07、EV-R2-13 | SUPPORTED；exact 仅指代数闭合 |
| Contribution 3：独立确认与条件边界 | Insight：新 seeds 复现替代，但符号掩盖不跨资源普遍 | RES-6.4、RES-6.5、D04、L01 | Fig. 3、Fig. 5、Tables 2-4 | EV-R2-01、EV-R2-02、EV-R2-08 至 EV-R2-13、BD-03 | SUPPORTED WITH FAILED UNIVERSALITY GATE |

三项贡献均不依赖“首次”、在线 PPO 性能或 GNN 结果。

## 2. C1-C8 最终状态

| Claim | 状态 | 终稿落点 | 最强证据 | 禁止外推 | 最终裁决 |
| --- | --- | --- | --- | --- | --- |
| C1 累计成本混合直接消耗与替代 | 支持 | I02-I05、RES-6.2、D00-D01、C01 | R1 18/18；R2 7/7 非正成本 context 有正替代 | 任意策略/环境 | KEEP |
| C2 三分量账本逐行闭合 | 支持 | M04-M09、RES-6.3、C01 | 7,776 条；最大误差 \(8.88\times10^{-16}\) | 统计无偏/因果完备 | KEEP |
| C3 新策略种子复现 | 支持 | P01-P04、RES-6.4、C01 | seeds 17/18/19 block 下界均为正 | 外部复现/跨算法 | KEEP |
| C4 跨资源普遍符号掩盖 | 否决 | RES-6.5、D04、L01 | missile 2/9，laser 5/9；P-C3 FAIL | 两类型同强度 | KEEP AS NEGATIVE BOUNDARY |
| C5 通用安全机会价值 | 否决 | RES-6.6、D02d、L03 | time 5/18、heterogeneity 2/18，覆盖不足 | 所有弹药无价值 | KEEP AS NEGATIVE BOUNDARY |
| C6 BPCE/MCH-PPO 稳定改进 | 否决 | I06、D05、L03、C01 | 在线门控失败 | 已提升 PPO | KEEP AS NON-CLAIM |
| C7 完整算法创新 | 不支持 | I06、RW05、D05、L03 | 无独立性能门控 | 通用算法论文 | KEEP AS L2 POSITIONING |
| C8 GNN 修复 | 未验证 | I06、D05、L04、C01 | 无验证实验 | 已改善信用/性能 | KEEP AS FUTURE R4 |

## 3. 技术方向复核

| 项目 | 冻结表达 | 机器/正文复核 | 结果 |
| --- | --- | --- | --- |
| N | 当前探针 no-op | 中英文一致 | PASS |
| E | 当前探针合法 engage | 中英文一致 | PASS |
| 累计成本差 | \(C(E)-C(N)\) | 公式与图注一致 | PASS |
| 替代量 | \(N-E\) saving direction | 三分量一致 | PASS |
| `Sub_shot` | 严格未来射击 | 未与成本主量混用 | PASS |
| `Sub_cost_total` | same + future probe + future other | 恒等式唯一主量 | PASS |
| R1/R2 | 发现/独立确认 | 正文和图表分离 | PASS |
| 首轮修正 | 287/7,776，2.0，唯一重跑 | 主文、补充和披露一致 | PASS |

## 4. 冻结数据只读审计

W1-10 直接读取正式机器结果，未训练模型、未新增 rollout：

```text
R3_AUDIT=PASS
models=9; behavior_selected=0; model_files_present=9
contexts=108; repeats=3456; ledger_rows=7776
max_error=8.88178419700125E-16; actor_diff=0.0
P-C1=True; P-C2=True; P-C3=False
missile_masked=2/9; laser_masked=5/9
```

因此 RP-06 的 R3 核查关闭。

## 5. 强主张词扫描

### English

| 命中 | 用途 | 裁决 |
| --- | --- | --- |
| `First` | 三项贡献或 Limitations 的枚举词 | ALLOW，不是优先权 |
| `general observation/idea` | 明确说明既有研究已覆盖一般概念 | ALLOW，收窄主张 |
| `general counterfactual baselines` | 说明本文不替代一般方法 | ALLOW，否定外推 |
| `novel/universal/robust/state-of-the-art/solve/prove/always/never` | 无支持性命中 | PASS |

### 中文

| 命中 | 用途 | 裁决 |
| --- | --- | --- |
| `通用` | 均出现在“不支持通用算法/oracle/信号”的否定边界 | ALLOW |
| `解决/证明` | 均出现在“不声称解决/不证明”的否定边界 | ALLOW |
| `稳健/鲁棒性` | 均用于“不构成稳健性证据/不证明鲁棒性” | ALLOW |
| `首次/普适/始终` | 无支持性命中 | PASS |
| `完全` | 描述成本可被完全抵消的条件解释，不表示方法完备 | ALLOW |

## 6. R2/R3/RX 关闭状态

| 级别 | 数量 | 状态 |
| --- | ---: | --- |
| R2 | 3 | 全部通过收窄范围或降级动词关闭 |
| R3 | 1 | 冻结数据只读核查通过 |
| RX | 0 | 无致命完整性冲突 |
| R4 | 2 类 | 跨环境/算法/顺序和在线方法，登记但不执行 |

## 7. 最终判定

Claim-Evidence 链通过终审。当前稿件可以冻结为 **L2/M2 测量与诊断模块**。
它不是独立通用算法论文，不具备在线性能、跨环境泛化或 GNN 修复主张。
