# 初稿状态、审查意见、内容导航与后续建议

更新时间：2026-07-28  
评估对象：`docs/manuscript/action_substitution_cost_identifiability/`  
工作边界：只服务当前稿件的发表流程，不推进项目研究主线

## 1. 执行摘要

当前产物已经超过通常意义上的“初稿”：中英文科学整稿、主图、主表、补充方法、
证据追溯和对抗性审稿均已完成，W1-01 至 W1-10 全部完成，T10 通过。科学内容
可以冻结为一个可复核的 **L2/M2 测量、诊断与资源信用分解模块**。

但当前材料还不是可立即上传投稿系统的投稿包，原因包括：

1. 尚未确定目标期刊和文章类型；
2. 24 条参考文献仍使用 E01-E24 占位标识；
3. 作者、单位、通讯作者、ORCID、基金、致谢、利益冲突和作者贡献未填写；
4. 代码和数据只有本地可审计状态，没有公共仓库、版本、DOI/accession 和许可证；
5. 尚未按目标期刊要求完成版式、字数、图表、补充材料和英文终校；
6. 现有证据支持单环境、冻结同源策略下的测量诊断，不支持把稿件包装为通用
   MARL 算法、在线 PPO 改进或跨环境泛化研究。

因此，当前最重要的不是继续泛化措辞或追加润色，而是先完成一次发表路线裁决：

> **推荐路线：维持 M2 定位，将其作为更大信用分配方法论文的核心测量模块。**
>
> **备选路线：以独立方法学短文试投，但必须接受“原创性和经验广度可能不足”
> 的较高编辑拒稿风险，并选择明确接收仿真测量、诊断和负边界贡献的期刊。**

如坚持独立投稿，不应把缺少的在线性能或跨环境证据通过写作放大来“补齐”。
若期刊策略确实要求新增实验，该需求已经超出本工作区边界，应交回项目主线另行
立项、预注册和执行。

## 2. 与项目主线的关系

项目总状态以
[Academic Project Progress](../../project/academic_project_progress.md)
和
[Research Innovation Roadmap](../../project/research_innovation_roadmap.md)
为准。两份文件均记录：

- W1 主张—证据冻结、双语整稿和对抗性审稿已经完成；
- 当前阶段出口为 L2/M2；
- 当前贡献不是独立通用算法论文；
- BPCE/MCH-PPO、GNN、跨环境、跨算法和跨顺序验证仍被冻结或登记为 R4；
- 目标期刊适配、公共发布标识和任何新在线算法属于后续独立任务。

本发表工作区据此只处理已有稿件的路线选择、期刊适配、编辑、材料打包、投稿和
返修，不改变上述研究裁决。

## 3. 当前科学稿的核心内容

### 3.1 研究问题

稿件研究动态掩码、自回归联合动作中的局部资源成本测量。在该结构下，一个当前
交战动作不仅发生直接资源消耗，还会改变：

- 同一步后缀单元的合法动作和资源消耗；
- 被测单元未来的动作；
- 其他单元未来的动作。

因此，回合累计成本差会混合直接成本和动作替代，不能无条件解释为当前动作的
纯局部资源信用。

### 3.2 方法

稿件使用冻结策略下的 no-engage/engage（N/E）成对反事实轨迹、共同随机数
（CRN）、合法目标精确边缘化和逐时刻成本账本，将总替代成本拆为：

1. 同一步其他单元替代；
2. 未来被测单元替代；
3. 未来其他单元替代。

冻结定义下的账本恒等式为：

\[
\Delta C_{\mathrm{episode}}
=C_{\mathrm{direct}}-Sub_{\mathrm{cost,total}}.
\]

这里“精确”只表示逐账本代数闭合，不表示统计无偏、因果完备或跨环境成立。

### 3.3 三项冻结贡献

1. 将已知反事实信用问题操作化为动态掩码序列分配中的局部资源成本测量问题；
2. 提供 N/E、CRN、目标边缘化和三分量账本组成的可复核审计协议；
3. 在新策略种子和新上下文上确认动作替代，同时保留场景与资源类型失败边界。

详细冻结表述见
[最终贡献列表](../../manuscript/action_substitution_cost_identifiability/final_contribution_list.md)
和
[最终 Claim-Evidence 审计](../../manuscript/action_substitution_cost_identifiability/claim_evidence_final_audit.md)。

### 3.4 关键证据

| 证据项 | 冻结结果 | 含义 |
| --- | ---: | --- |
| 新来源模型 | 9/9，无行为筛选 | 避免按结果挑选策略模型 |
| 新上下文 | 108，与旧正式观测 hash 零重叠 | 支持有限意义的独立确认 |
| context-repeat 记录 | 3,456 | 配对采样规模 |
| 目标账本记录 | 7,776 | 三分量账本审计规模 |
| 首轮受影响账本 | 287/7,776，最大残差 2.0 | 证明遗漏同一步后缀会产生实质错误 |
| 修正后最大恒等误差 | \(8.88\times10^{-16}\) | 冻结定义下代数闭合 |
| P-C1 | PASS | 完整成本恒等式成立 |
| P-C2 | PASS | 新策略种子上动作替代得到确认 |
| P-C3 | FAIL | 符号掩盖不具跨资源类型普遍性 |
| 资源类型边界 | missile 2/9；laser 5/9 | 必须保留差异，不能只报告有利资源类型 |
| 软件回归 | 264 passed | 冻结实现通过现有回归检查 |

### 3.5 明确不成立或未验证的主张

以下内容不能作为当前稿件贡献：

- 通用安全机会成本 oracle；
- BPCE/MCH-PPO 稳定优于 PPO；
- 完整、通用的反事实信用算法创新；
- GNN 已修复信用分配；
- 跨环境、跨算法或跨动作顺序泛化；
- missile 与 laser 具有相同强度的成本符号掩盖。

## 4. 稿件与产物导航

### 4.1 首选阅读路径

建议按以下顺序进入材料：

1. [中文科学终稿](../../manuscript/action_substitution_cost_identifiability/final_manuscript_zh.md)：
   最快理解完整故事、边界和数字；
2. [英文科学终稿](../../manuscript/action_substitution_cost_identifiability/final_manuscript_en.md)：
   后续期刊编辑的正文基线；
3. [稿件定位决策](../../manuscript/action_substitution_cost_identifiability/paper_positioning_decision.md)：
   理解 L2/M2 判定和禁止措辞；
4. [最终 Claim-Evidence 审计](../../manuscript/action_substitution_cost_identifiability/claim_evidence_final_audit.md)：
   核查每项主张、证据和外推边界；
5. [对抗性审稿压力测试](../../manuscript/action_substitution_cost_identifiability/reviewer_pressure_test.md)：
   查看技术、原创性和可读性风险；
6. [投稿与 M2 移交清单](../../manuscript/action_substitution_cost_identifiability/submission_checklist.md)：
   查看尚未完成的投稿工程；
7. [目标期刊适配状态](../../manuscript/action_substitution_cost_identifiability/target_journal_fit.md)：
   查看期刊筛选条件和当前 NO-GO 项。

### 4.2 正文结构

| 正文章节 | 主要任务 | 发表阶段注意点 |
| --- | --- | --- |
| Abstract | 问题、协议、关键结果与非算法定位 | 按期刊字数重写，但不能删除 P-C3 失败 |
| Introduction | 从局部资源信用引出动态后缀测量缺口 | 必须持续显式限定 AirDefense v1 和冻结策略 |
| Related Work | 反事实信用、时序信用、顺序 MARL、masking、资源约束、CRN | E01-E24 需要完整核验；结构随期刊调整 |
| Problem Formulation | 环境、冻结策略、估计对象与非主张 | 避免把估计对象写成因果效应全量 |
| Cost Decomposition | N/E、CRN、目标边缘化、三分量账本和恒等式 | “exact”只能修饰代数重构 |
| Experimental Protocol | R1 发现、R2 确认、统计单位和门控 | 首次定义 context/repeat/block/ledger row/seed |
| Results | 测量失真、账本修正、独立确认和条件边界 | P-C2 PASS 与 P-C3 FAIL 必须并列 |
| Discussion | 测量含义、结构混合和后续方法约束 | 不得改写为已提升策略性能 |
| Limitations | 单环境、固定顺序、同源算法、统计区间、在线方法和 GNN | 属于可信度核心，不宜过度压缩 |
| Conclusion | 测量诊断结论和使用边界 | 保持与摘要及三项贡献一致 |

### 4.3 图表与补充材料

现有清单包括：

- 5 张主图，已有 SVG、PDF、TIFF 和 PNG 预览；
- 4 张主表和 3 张补充表，已有 Markdown/CSV；
- 图表 source CSV/JSON 和 metadata；
- Supplementary Methods；
- Supplementary Results 结构；
- 图表数据追溯、QA、正文—补充材料分工和可复现映射。

图表 QA 当前全部通过。正式投稿时仍需按目标期刊重新核查尺寸、字体、颜色模式、
文件命名、图注位置和 source-data 要求。入口：

- [主文与补充材料计划](../../manuscript/action_substitution_cost_identifiability/main_vs_supplement_plan.md)
- [图表计划](../../manuscript/action_substitution_cost_identifiability/figure_table_plan.md)
- [图表 QA](../../manuscript/action_substitution_cost_identifiability/figure_qa_report.md)
- [图数据追溯](../../manuscript/action_substitution_cost_identifiability/figure_data_traceability.md)
- [补充方法](../../manuscript/action_substitution_cost_identifiability/supplementary_methods.md)
- [补充结果结构](../../manuscript/action_substitution_cost_identifiability/supplementary_results_outline.md)

### 4.4 证据、复现与科研完整性

| 任务 | 权威入口 |
| --- | --- |
| 数值来源和权威层级 | [Evidence Source Index](../../manuscript/action_substitution_cost_identifiability/evidence_source_index.md) |
| 主张到证据映射 | [Manuscript Traceability Matrix](../../manuscript/action_substitution_cost_identifiability/manuscript_traceability_matrix.md) |
| 方法到代码与输出映射 | [Reproducibility Map](../../manuscript/action_substitution_cost_identifiability/reproducibility_map.md) |
| 证据冲突及裁决 | [Evidence Conflict Log](../../manuscript/action_substitution_cost_identifiability/evidence_conflict_log.md) |
| 首轮账本修正披露 | [Research Integrity Disclosure](../../manuscript/action_substitution_cost_identifiability/research_integrity_disclosure.md) |
| 数据和代码现状 | [Data and Code Availability Draft](../../manuscript/action_substitution_cost_identifiability/data_code_availability_draft.md) |
| 中英文一致性 | [Bilingual Consistency Audit](../../manuscript/action_substitution_cost_identifiability/bilingual_consistency_audit.md) |

## 5. 已有审查意见归纳

### 5.1 共识优点

三位模拟审稿人的共识是：

- 冻结范围内的技术链条一致，账本修正过程透明；
- 9/9 模型无筛选、108 个新 context、Actor 冻结和旧 hash 零重叠，支持
  “新来源策略种子与上下文独立”的有限确认；
- 三项贡献不依赖“首次”、SOTA 或在线性能优势也能成立；
- P-C3、机会成本和在线算法的负结果提高了可证伪性和可信度；
- 中英文 66 个对应 Paragraph ID、图表数据和追溯矩阵为后续编辑提供了较好基础。

### 5.2 核心风险

| 风险 | 审查判断 | 当前处理 |
| --- | --- | --- |
| 恒等式被误读为因果无偏 | 高概率误读 | 已限定为冻结定义下的代数闭合，投稿版须继续保持 |
| 小 context 正态近似区间被过度解释 | 统计边界风险 | 已写为描述性确认门控，不作总体高精度推断 |
| “独立确认”被理解为外部复现 | 表述风险 | 只允许指新 seeds/context，同一算法与环境 |
| 单环境、固定动作顺序、同源 PPO | 科学广度风险 | 不能宣称通用性；属于 R4 |
| 缺少在线决策收益 | 定位风险 | 实际意义应写成防止错误监督、约束后续方法设计 |
| 标题较通用但证据场景具体 | 读者预期风险 | 摘要和引言必须持续给出 AirDefense v1 边界 |
| 缩写和统计层级密集 | 可读性风险 | 首次出现时展开并统一定义 |
| 参考文献和投稿元数据不完整 | 外投阻断项 | 必须在投稿前完成 |
| 公共代码与数据标识缺失 | 复现与期刊合规风险 | 需决定公开范围、许可证、版本和持久标识 |

### 5.3 三位审稿人的侧重点

- Reviewer 1：最关注技术正确性、统计边界、context 选择协议和可复现性；
- Reviewer 2：最关注原创性边界，反对把 L2 测量模块包装成完整算法创新；
- Reviewer 3：最关注非专门读者路径、缩写密度和统计单位定义。

### 5.4 已关闭与仍开放的问题

- R2：3 项，均通过收窄范围或降低动词关闭；
- R3：1 项，冻结数据只读核查通过；
- RX：0 项，未发现致命完整性冲突；
- R4：跨环境/算法/顺序及在线方法，保持开放但不阻断 W1，也不在本工作区执行；
- 投稿工程：目标期刊、公共标识和投稿元数据仍开放，直接阻断立即外投。

## 6. 当前成熟度判断

| 层面 | 状态 | 判断 |
| --- | --- | --- |
| 科学故事 | 已冻结 | 问题—方法—证据—边界完整 |
| Claim-Evidence | 通过 | R2/R3/RX 已关闭 |
| 中英文整稿 | 已完成 | 可作为期刊编辑基线，不等于语言终校完成 |
| 图表 | 已完成并通过内部 QA | 尚需按目标期刊规范复核 |
| 补充材料 | 结构和主要内容已具备 | 尚需按目标期刊打包 |
| 参考文献 | 未完成 | E01-E24 仍为占位，阻断外投 |
| 期刊定位 | 未完成 | 目前只有筛选判据，没有候选排序和最终选择 |
| 作者与声明 | 未完成 | 阻断外投 |
| 数据/代码发布 | 本地可审计 | 无公共标识、许可证和干净发布包 |
| 独立投稿科学适配 | 中低 | 原创性与经验广度是主要风险 |
| 大方法论文模块适配 | 高 | 与现有 L2/M2 裁决一致 |
| 立即外部投稿 | NO-GO | 投稿包未完成，且独立路线尚未裁决 |

## 7. 后续工作建议

### 阶段 A：先做发表路线裁决

在任何目标期刊格式化之前，形成 `publication_strategy_decision.md`，至少回答：

1. 当前目标是独立方法学短文，还是更大方法论文的一个模块？
2. 是否接受独立短文较高的 desk rejection/原创性风险？
3. 是否严格限定为“使用现有冻结证据投稿”，不要求项目主线补实验？
4. 若候选期刊普遍要求在线性能或多环境证据，是否停止独立投稿路线并回到 M2？

建议默认选择 M2。若选择独立短文，应把“测量有效性”和“precision is not
validity”作为主线，而不是伪装成新 PPO 算法。

### 阶段 B：期刊长名单与短名单

路线裁决后，再基于各期刊官网的当前信息进行检索。候选期刊首先按科学适配筛选，
再看影响力和周期。建议评分维度：

| 维度 | 建议权重 | 核查问题 |
| --- | ---: | --- |
| Scope 与贡献类型适配 | 30% | 是否接受测量、诊断、仿真方法和负边界 |
| 读者匹配 | 20% | 是否覆盖 MARL、序列决策、信用分配或资源约束 |
| 证据广度要求 | 15% | 单环境机制确认是否有现实机会 |
| 方法与补充材料容量 | 10% | 能否容纳 5 图、4 表和完整账本说明 |
| 数据/代码政策可满足性 | 10% | 是否强制公开，项目能否按时满足 |
| 周期与投稿风险 | 10% | 首轮周期、desk reject 风险、转投成本 |
| 费用与开放获取 | 5% | APC、版面费和资助条件 |

每个候选必须保存官网来源、访问日期、文章类型、长度/图表限制、数据政策、费用
和模板要求；不得依赖聚合网站或过期印象。最终宜保留 1 个主投、1 个同层备选和
1 个保守备选。

### 阶段 C：补齐外投阻断项

1. 逐条核验 E01-E24 的作者、题名、年份、期刊/会议、卷期页码、DOI；
2. 确定作者顺序、单位、通讯作者、ORCID 和 CRediT 贡献；
3. 填写基金、致谢、利益冲突和伦理/安全声明；
4. 决定代码、模型、账本和图表 source data 的公开范围；
5. 清理绝对路径、缓存、临时模型和无关历史结果；
6. 选择代码和数据许可证；
7. 生成版本化发布包，在干净环境复核安装、R3 审计和图表重建；
8. 获取真实 DOI/accession 后再更新 Data/Code Availability。

### 阶段 D：目标期刊化编辑

在不改变冻结科学结论的前提下：

1. 按 article type 重组 Related Work、Methods 和 Supplement；
2. 将英文约 5,236 词的当前稿压缩或扩展到期刊范围；
3. 降低缩写密度，首次明确 context/repeat/block/ledger row/seed；
4. 保持 P-C2 PASS 与 P-C3 FAIL 同时可见；
5. 保留首轮 287/7,776 账本遗漏和唯一重跑的完整性披露；
6. 调整图表尺寸、色彩、字体、命名和 source-data 包；
7. 完成专业英文终校及最终 PDF 视觉检查；
8. 准备 cover letter、highlights、graphical abstract 或 author summary（如要求）。

### 阶段 E：投稿与返修管理

- 提交前执行科学不变性、引用、数字、图表和声明五类终检；
- 保存提交版本、日期、稿件编号和系统字段快照；
- 收到编辑或审稿意见后建立逐点 response matrix；
- 区分“文字澄清”“补充分析”“新增实验”和“超出主线边界”；
- 任何要求改变冻结结论或新增研究主线实验的意见，先形成影响评估，再决定是否
  由项目主线执行；
- 每次返修都重新跑数字一致性和图表追溯检查。

## 8. 推荐的下一步

本聊天的下一项工作建议是：

> **先形成发表路线决策，再开展基于期刊官网当期信息的候选期刊长名单。**

原因是独立短文和 M2 整合稿面对的是不同期刊、文章类型、标题、摘要和证据期待。
在路线未定前直接格式化或大规模润色，返工概率很高。

如果维持“不推进项目主线”的约束，则后续可连续完成：路线决策、期刊检索与
评分、参考文献补全、投稿元数据模板、英文期刊化编辑、发布包规划、投稿材料和
返修管理；但不执行新训练、新 rollout、跨环境验证或新算法实现。

