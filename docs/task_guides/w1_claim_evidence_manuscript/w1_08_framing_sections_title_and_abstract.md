# W1-08：框架章节、标题与摘要

更新时间：2026-07-24  
任务状态：NOT_STARTED  
前置任务：W1-02 非 L4；W1-04 通过 T04；W1-05 通过 T05；W1-07 通过 T07  
后续任务：W1-09  
允许并行：无  
任务性质：Introduction、Related Work、Conclusion、Title、Abstract

## 1. 目标

在 Results、Methods、文献定位和 Discussion 已稳定后，完成论文的框架性章节。
本任务不能修改科学结果来适应一个更响亮的标题或摘要。

## 2. 输入

- W1-02 的定位决策和文献矩阵；
- W1-03 的故事压缩和章节架构；
- W1-04 Results；
- W1-05 Methods；
- W1-07 Discussion 与 Limitations；
- 当前追溯矩阵。

## 3. 执行顺序

```text
Introduction
→ Related Work（或并入 Introduction）
→ Conclusion
→ Title 候选
→ Abstract（最后）
```

不得先写摘要再倒逼正文。

## 4. Introduction

采用技术瓶颈漏斗：

```text
动态资源分配需要局部信用
→ 回合累计回报常被用作反事实读出
→ 序列联合动作改变同一步和未来动作
→ 累计读出可能混合直接成本和动作替代
→ 本研究进行完整分解与独立边界确认
```

建议四段：

1. 研究对象与局部信用的重要性；
2. 当前测量方式及其潜在瓶颈；
3. 既有工作覆盖与剩余缺口；
4. 本研究做了什么及经查新保留的三项贡献。

末段不堆叠 Results 数字，也不使用未经核验的“首次”。

## 5. Related Work

按机制主题组织：

- multi-agent/counterfactual credit；
- temporal credit and delayed effects；
- sequential joint actions and masking；
- resource value and constrained MARL；
- measurement validity and paired simulation。

每个主题段落使用：

```text
主题范围 → 代表方法 → 已解决内容 → 剩余限制 → 本项目区别
```

若目标期刊不设独立 Related Work，将其合并到 Introduction，但保留主题综合，
不改成逐篇文献列表。

## 6. Conclusion

严格按：

```text
贡献 → 决定性证据 → 测量意义 → 边界
```

必须包含：

- 同一步和未来替代；
- 跨新种子复现；
- 场景/资源类型条件；
- 非算法改进边界。

不引入新数据、新机制或未验证的未来性能承诺。

## 7. Title

生成 3–5 个候选，覆盖：

- finding-led；
- method/measurement-led；
- object-and-consequence。

标题必须：

- 可搜索；
- 包含 cost measurement、action substitution 或 counterfactual credit 中的核心概念；
- 反映 AirDefense v1 或动态掩码序列分配范围；
- 不使用 `universal`、`general`、`solves`；
- `first` 只有 W1-02 明确允许时才可考虑。

记录每个候选标题对应的证据和过度承诺风险，再选择一个主标题。

## 8. Abstract

最后撰写，结构为：

```text
背景/问题
→ 测量缺口
→ 配对反事实与成本账本
→ 最强结果
→ 独立确认
→ 场景/资源边界
```

摘要至少包含：

- 一个决定性定量事实；
- 一个独立性事实；
- 一个明确失败或条件边界。

摘要不得：

- 只介绍流程；
- 声称优于 PPO；
- 省略 P-C3 所代表的边界；
- 把环境内三场景写成跨环境泛化。

## 9. 贡献列表审计

最终贡献以 3 项为上限：

| 贡献 | 必须证明 | 最弱环节 |
| --- | --- | --- |
| 测量混叠发现 | 替代解释和符号掩盖 | 文献是否已有相同洞见 |
| 完整审计协议 | 三类替代与恒等式 | 是否被视为常规记账 |
| 独立确认与边界 | 新种子、场景、资源类型 | 单环境范围 |

若 W1-02 为 L2/L3，按定位决策收窄，不恢复已删除贡献。

## 10. 交付物

```text
introduction_draft_zh.md
related_work_draft_zh.md
conclusion_draft_zh.md
title_candidates.md
abstract_draft_zh.md
final_contribution_list.md
```

## 11. 验收门控 T08

- Introduction 提出的问题由 Results 回答；
- Related Work 按主题综合；
- Conclusion 没有新证据；
- 标题不超过稿件定位；
- Abstract 最后完成并包含边界；
- 贡献列表与 W1-02 决策一致；
- 摘要、结论和一句话论证范围相同；
- 无未核验优先权表述。

## 12. 移交

通过 T08 后将全部文件、最终主标题和贡献列表移交 W1-09。

