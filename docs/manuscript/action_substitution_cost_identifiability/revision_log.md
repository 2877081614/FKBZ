# W1-10 终稿修订日志

更新时间：2026-07-28  
基线：W1-09 `manuscript_draft_zh.md` / `manuscript_draft_en.md`  
终稿：`final_manuscript_zh.md` / `final_manuscript_en.md`

## 1. 修订记录

| Revision ID | 来源问题 | 等级 | 修订 | 影响文件 | 科学结果是否改变 |
| --- | --- | --- | --- | --- | --- |
| REV-10-01 | context 选择可能被误解为结果筛选 | R1 | 在 P02 明确 safety/resource 选择不查看后续 N/E 成本结果，并指向 Supplement S3 | 中英文终稿 | 否 |
| REV-10-02 | 小 context 数正态近似区间可能被过度解释 | R2 | 在 L02 增加区间仅服务冻结门控、不代表总体高精度推断 | 中英文终稿 | 否 |
| REV-10-03 | Data/Code Availability 不得虚假承诺 | 投稿完整性 | 增加当前本地可审计、尚无公共标识和许可证的真实声明 | 中英文终稿、availability 文件 | 否 |
| REV-10-04 | “独立确认”可能被外推 | R2 | 保持“新来源策略 seeds/context、同算法/环境”限定 | 中英文终稿 | 否 |
| REV-10-05 | “精确”可能被外推 | R2 | 保持“逐账本代数重构”限定 | 中英文终稿 | 否 |
| REV-10-06 | 强主张词 | R0/R2 | 逐项扫描；支持性正文不保留优先权、通用性或算法胜出词 | 中英文终稿、Claim audit | 否 |
| REV-10-07 | 目标期刊未定 | 格式 | 明确 target journal fit 为 INCOMPLETE，采用 L2/M2 阶段出口 | journal fit、checklist | 否 |

## 2. 明确未做的修改

- 未新增环境、模型、策略种子、context、repeat 或 rollout；
- 未重新训练任何来源策略；
- 未改变 P-C1/P-C2/P-C3 判据或 PASS/FAIL；
- 未重新解释 missile 2/9 和 laser 5/9；
- 未恢复 opportunity oracle、BPCE/MCH-PPO 或 GNN 贡献；
- 未把 E-ID 占位伪装成正式期刊参考文献；
- 未生成虚假的公共 DOI、仓库地址或许可证。

## 3. 数字不变性

终稿继续使用：

```text
9 source models
108 contexts
3,456 context-repeat records
7,776 target-conditioned ledger records
287 affected future-only records
maximum future-only residual = 2.0
maximum complete-identity error = 8.88e-16
P-C1 = PASS
P-C2 = PASS
P-C3 = FAIL
missile masking = 2/9
laser masking = 5/9
```

## 4. 修订裁决

所有 R2、R3 和 RX 问题已关闭；R4 问题只登记为后续研究。终稿相较 W1-09
草稿只增加可复现性说明、统计边界和真实可用性状态，不改变科学结论。
