# W1-09：中英文整稿集成

更新时间：2026-07-24  
任务状态：PASSED（T09）
前置任务：W1-04 至 W1-08 全部通过  
后续任务：W1-10  
允许并行：中文整合完成后可分节英文重写，但必须统一回收  
任务性质：章节整合、中文科学意图冻结和英文论证重写

## 1. 目标

将已验收的分节稿整合成一版论证一致的中文完整稿，再按段落功能重写为英文
完整初稿。

英文稿不是逐句翻译；中文稿也不是新的科学结果编辑入口。

## 2. 输入

- W1-04 Results；
- W1-05 Methods 和补充方法；
- W1-06 图表与图注；
- W1-07 Discussion/Limitations；
- W1-08 框架章节、标题、摘要和贡献；
- 术语账本与稿件追溯矩阵。

## 3. 中文整合

按目标期刊暂定结构合并：

```text
Title
Abstract
Introduction
Related Work
Problem Formulation / Methods
Results
Discussion
Limitations
Conclusion
References placeholders
```

合并时只允许：

- 删除重复背景；
- 修复跨节衔接；
- 统一术语、符号和数字格式；
- 调整段落位置；
- 根据追溯矩阵补齐引用和图表占位。

不允许：

- 增加新主张；
- 删除不利边界；
- 用摘要语言扩大正文结论；
- 重新解释正式数字。

## 4. 中文稿一致性检查

- 一句话论证、三项贡献、摘要和结论范围一致；
- Introduction 的每个 gap 在 Results 中有回答；
- Results 与 Discussion 观察/解释分开；
- Methods 的公式与图表一致；
- P-C3、机会成本和在线算法负结果保留；
- 每个段落只有一个功能；
- 所有 `[待证据]`、`[待引用]`、`[待图]` 可枚举。

## 5. 英文重写

每个中文段落先拆为：

```text
claim / evidence / condition / comparison / implication / limitation
```

再按英文段落功能重写。规则：

- Results 使用观察性动词；
- Discussion 使用与证据相称的解释性动词；
- `show/demonstrate` 只用于直接证据；
- `suggest/indicate` 用于间接解释；
- 不使用无范围的 `robust/generalizable/comprehensive`；
- 场景和资源类型结论必须带条件；
- 术语使用 W1-01 经 W1-02 核验后的最终版本；
- 句子不能因翻译而改变 N/E 方向或成本符号。

## 6. 反向提纲检查

为中英文稿分别生成：

| Paragraph ID | 首句 | 单一功能 | Claim ID | Evidence ID | 与上段关系 |
| --- | --- | --- | --- | --- | --- |
| 待填写 | — | — | — | — | cause/comparison/restriction/example |

若段落没有单一功能或与上下段无明确关系，先调整结构再润色语言。

## 7. 双语一致性检查

逐项核对：

- 数字；
- 单位；
- 种子和模型数量；
- 场景名称；
- missile/laser；
- `Sub_shot`/`Sub_cost_total`；
- P-C1/P-C2/P-C3；
- 强动词；
- 局限性；
- 贡献数量。

英文稿可以更精炼，但不能多出中文稿没有的科学主张。

## 8. 交付物

```text
manuscript_draft_zh.md
manuscript_draft_en.md
reverse_outline_zh.md
reverse_outline_en.md
bilingual_consistency_audit.md
unresolved_placeholders.md
```

## 9. 验收门控 T09

- 中文完整稿章节齐全；
- 英文为论证重写而非逐句翻译；
- 双语主张、数字、公式和边界一致；
- 摘要确为最后整合版本；
- 无未登记的占位符；
- 反向提纲显示每段单一功能；
- 追溯矩阵覆盖全部主要主张；
- 无新增科学结论。

## 10. 移交

通过 T09 后向 W1-10 提交六个文件、当前图表包、完整追溯矩阵和所有尚未解决
但不阻断审稿的问题。

## 11. 执行结果（2026-07-28）

中文科学意图稿先完成集成和反向提纲检查，随后按相同 Paragraph ID 重写英文
完整初稿。英文稿按段落功能组织，不采用逐句直译。

已生成文件：

- `manuscript_draft_zh.md`；
- `manuscript_draft_en.md`；
- `reverse_outline_zh.md`；
- `reverse_outline_en.md`；
- `bilingual_consistency_audit.md`；
- `unresolved_placeholders.md`。

T09 验收：

- [x] 中英文稿均包含 Title、Abstract、Introduction、Related Work、Problem
      Formulation、Methods、Results、Discussion、Limitations、Conclusion 和
      References；
- [x] 两稿共有 66 个相同且唯一的 Paragraph ID；
- [x] 双语数字、统计单位、场景、资源类型、公式方向和 P-C1/P-C2/P-C3 一致；
- [x] Abstract 使用整稿后的最终范围，并包含独立确认与 P-C3 失败边界；
- [x] 两份反向提纲均显示每段只有一个主要功能；
- [x] P-C3、机会成本、在线 BPCE/MCH-PPO 和 GNN 负边界完整保留；
- [x] 8 个期刊/投稿格式占位均登记，无未登记科学证据占位；
- [x] 英文稿未新增科学结论或未经核验的优先权表述。

移交信息：

```text
任务编号：W1-09
状态：PASSED（T09）
已生成文件：六项规定交付物
已通过门控：T09
未解决问题：目标期刊、参考文献格式、作者信息和投稿声明，均为非阻断项
禁止下游假设：不得借期刊适配扩大算法、跨环境、机会成本或 GNN 主张
下一接收任务：W1-10 审稿压力测试与投稿准备
```
