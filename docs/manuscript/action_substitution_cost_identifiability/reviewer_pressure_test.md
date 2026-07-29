# W1-10 对抗性审稿压力测试

更新时间：2026-07-28  
输入范围：W1-09 中英文整稿、Fig. 1-Fig. 5、Table 1-Table 4、补充材料、
Claim-Evidence 矩阵、文献定位和冻结 R1/R2 结果  
审稿角色：三份报告共享同一事实基础，仅改变评估侧重点  
裁决边界：模拟审稿意见，不代表编辑决定或目标期刊接收判断

## Review setup

### Shared manuscript claim

稿件主张：在 AirDefense v1 冻结 factorized joint PPO 的动态掩码序列分配中，
回合累计成本会混合当前直接消耗、同一步后缀替代和未来策略介导替代；三分量
N/E 成对反事实账本可逐行重构这种混合，并在新策略种子上确认正动作替代，但
成本符号掩盖受场景和资源类型约束。

### Visible evidence base

- R1 使用 seeds 8/9/10 发现未来动作替代；
- R2 无条件保留 seeds 17/18/19 在三个场景下的 9 个来源模型；
- 108 个确认 context 与旧可核验观测 hash 零重叠；
- 3,456 条 context-repeat 记录和 7,776 条目标条件成本账本；
- 完整恒等式最大误差 \(8.88\times10^{-16}\)，Actor 参数差为 0；
- seeds 17/18/19 的 `time_pressure/resource` block 下界均为正；
- missile 和 laser 掩盖 context 为 2/9 和 5/9，P-C3 失败；
- 机会成本 oracle、在线 BPCE/MCH-PPO 和 GNN 均未作为正面贡献。

### Assessment boundary

可以评估稿件内部技术一致性、证据追溯、L2 新颖性和可读性。不能从现有材料
评估跨环境、跨算法、替代动作顺序、真实系统有效性或在线算法性能。目标期刊
尚未确定，因此只能评估稿件形态，不能裁决具体期刊格式或接收概率。

## Reviewer 1

侧重点：技术正确性与可复现性。

### Overall assessment

该工作对估计对象、干预方向、统计单位和负边界的处理较为严谨。最有说服力的
技术部分不是恒等式本身，而是预设完整性检查如何暴露 future-only 公式遗漏，
以及相同模型、context、随机带和门槛下的唯一重跑如何关闭账本。现有证据足以
支撑冻结策略、单环境内的测量诊断，但不足以支撑策略训练或跨环境因果结论。

### Who would be interested, and why

使用自回归联合动作、动态合法掩码、资源成本或反事实信用的 MARL 研究者会关心
这一结果，因为它指出“更精确地估计回合差值”与“差值是否对应局部直接成本”
不是同一问题。仿真评估和受约束资源分配研究者也可复用其完整性检查思路。

### Major strengths

1. N/E 身份、\(E-N\) 累计成本方向和 \(N-E\) 替代方向明确；
2. `Sub_shot` 与 \(Sub_cost_total\) 分离，避免统计对象偷换；
3. 同一步、未来探针和未来其他单元三分量均可追溯到字段；
4. 9/9 模型无行为筛选、Actor 冻结、旧 hash 零重叠和账本修正均透明；
5. P-C3 失败和机会价值负结果进入主文，没有只展示有利切片。

### Major concerns

1. “精确”容易被读成因果无偏，必须始终限定为冻结定义下的代数闭合；
2. block 和资源分层区间的 context 数较小，正态近似区间应被描述为确认门控
   的操作统计量，而不是总体参数的高精度推断；
3. context 从候选回合中按 safety/resource 分数选取，主文应明确选择不查看
   后续 N/E 成本结果，并把完整规则指向 Supplementary Methods S3；
4. 新 seeds 仍来自同一算法、环境和训练实现，“independent confirmation”
   必须限定为来源策略种子与 context 独立，而非独立研究复现。

### Technical failings requiring action

- 在终稿 Limitations 中增加小 context 数与正态近似区间的解释边界；
- 在 Methods 5.2 中补充 context 选择不查看 N/E 后续结果；
- Data/Code Availability 必须如实说明当前没有公共仓库标识和许可证；
- 跨顺序、跨算法和跨环境验证属于 R4，不得在本轮用推测性语言补齐。

### Nature-style criteria

| 维度 | 评估 |
| --- | --- |
| Originality | L2 层面成立：新意在特定测量对象的操作账本与边界，不在反事实信用概念本身 |
| Scientific importance | 对动态掩码 MARL 的测量有效性有明确方法意义，但当前仍是领域内模块 |
| Interdisciplinary interest | 对仿真、信用分配和资源约束研究有邻域价值，广泛影响尚未由多环境证据建立 |
| Technical soundness | 冻结范围内较强；小样本区间和 context 选择说明需加强 |
| Readability | 公式清楚，但统计层级和四类成本名词对非专门读者负担较高 |

### Recommendation posture

在完成上述文字和可用性修订后，可支持其作为较大方法论文的测量模块；现有证据
不支持将其作为已完成的通用算法论文。

## Reviewer 2

侧重点：原创性、科学意义与稿件定位。

### Overall assessment

稿件最可信的贡献是“测量对象被动作路径混合”这一受边界约束的实证诊断，而非
提出新反事实信用思想。文献矩阵已经识别 difference rewards、COMA、时序信用、
因果介导、顺序 MARL 和 masking 的直接先例，因此稿件不依赖优先权措辞仍可
成立。风险在于三项贡献若写得过宽，会让读者把一项 L2 模块与完整算法创新比较。

### Who would be interested, and why

研究 credit assignment、masked policies、multi-agent causal effects 和有限
资源决策的读者会关心，因为该稿件把抽象信用混合转化为可审计的物理成本账本，
并展示正替代不必跨资源类型达到相同符号后果。

### Major strengths

1. Problem-Method-Insight 三层差异清楚；
2. 同一步后缀项由完整性失败暴露，而非事后挑选有利结果；
3. 新 seeds 确认与资源类型失败边界共同出现，增强可证伪性；
4. 对“只是记账”的最佳回应是 287/7,776 条遗漏残差、符号掩盖和 P-C3 失败，
   而不是宣称恒等式本身具有理论普遍性；
5. 创新演化对失败算法、机会成本和 GNN 保持透明。

### Major concerns

1. 单一 AirDefense v1 和同源 PPO 限制该洞见作为独立论文的科学广度；
2. 当前没有在线决策收益，因此实际意义应表述为防止错误监督和约束后续设计，
   而不是已经提高策略；
3. 标题采用通用技术对象而不含 AirDefense，摘要和第一段必须持续给出场景边界；
4. 参考文献仍为 E-ID，占位状态阻止外部投稿，但不阻止 M2 章节冻结。

### Technical failings requiring action

- 保留 L2/M2 出口，不将稿件包装为独立通用算法；
- 贡献 1 必须写“操作化已知问题”，贡献 2 的 exact 只修饰代数重构；
- 贡献 3 必须同时呈现 P-C2 通过和 P-C3 失败；
- 跨环境、在线算法和 GNN 只登记为 R4 研究问题。

### Nature-style criteria

| 维度 | 评估 |
| --- | --- |
| Originality | 有限但可辩护，属于 Method/Insight 的组合差异 |
| Scientific importance | 能纠正一个具体测量实践；尚未达到由多系统支持的广泛方法结论 |
| Interdisciplinary interest | 对相邻计算领域可理解，但当前证据主要服务 MARL 与资源分配 |
| Technical soundness | 主张与证据匹配，前提是继续保持单环境和冻结策略限定 |
| Readability | 标题、摘要和主图能传达中心问题，Related Work 对定位很关键 |

### Recommendation posture

作为更大信用分配方法论文中的核心诊断模块具有价值；若孤立投稿，原创性与广度
可能被认为不足。建议保持 M2，而不是通过夸大措辞追求 L1。

## Reviewer 3

侧重点：跨领域可读性与论证清晰度。

### Overall assessment

稿件的中心直觉可概括为：一个动作花费资源，也可能阻止其他当前或未来动作继续
花费资源，因此回合总成本不等于该动作的直接成本。Fig. 1 和三分量恒等式能够
帮助非专门读者理解这一点。主要阅读障碍来自缩写、统计单位和多个历史审计阶段，
而不是中心逻辑本身。

### Who would be interested, and why

除 MARL 读者外，进行随机仿真对照、序列决策评估或资源消耗归因的研究者可能
关心“方差降低不等于测量有效性”这一结论。当前稿件尚不能证明该现象在这些
领域普遍发生，只能提供可迁移的审计问题。

### Major strengths

1. 标题、摘要、Fig. 1 和 Conclusion 使用相同范围；
2. Results 与 Discussion 的观察和解释基本分离；
3. missile/laser 和三场景边界采用表格和图形显式展示；
4. 负结果解释了为何项目没有直接进入在线算法宣传；
5. 双语 66 个段落 ID 为后续编辑提供清楚的结构控制。

### Major concerns

1. `context`、`repeat`、`block`、`ledger row` 和 `seed` 应在首次出现时保持
   一致定义，不能依靠补充材料才能区分；
2. R1/R2、P-C1/P-C2/P-C3、BPCE/MCH-PPO 等缩写密度较高；
3. Related Work 独立成节还是并入 Introduction 取决于目标期刊；
4. 当前没有作者信息、正式参考文献、Data/Code Availability 和投稿声明，
   因而不是格式定稿。

### Technical failings requiring action

- 保留 Table 1 的统计单位说明和 Table 3 的 PASS/FAIL 状态；
- 在最终投稿版首次出现时展开 CRN、N/E 和 R1/R2 角色；
- 目标期刊选定后再决定 Related Work、Methods 和补充材料位置；
- 不用“广泛影响”替代尚未完成的跨环境证据。

### Nature-style criteria

| 维度 | 评估 |
| --- | --- |
| Originality | 非专门读者可理解操作差异，但需依赖 Related Work 证明其边界 |
| Scientific importance | 测量有效性问题清楚，当前影响主要是方法论和设计约束 |
| Interdisciplinary interest | “precision is not validity” 有跨领域可读性，经验外推仍受限 |
| Technical soundness | 主文与补充材料形成可追溯链，未见未披露致命冲突 |
| Readability | 中心逻辑清楚；缩写和多层统计单位仍需目标期刊级编辑 |

### Recommendation posture

支持作为 M2 模块进入更大方法稿或学位论文方法章节；外部投稿前需要完成格式和
可用性材料，而不是新增宣传性主张。

## Cross-review synthesis

### Consensus strengths

- 三位审稿意见均认可冻结范围内的技术一致性和账本修正透明度；
- 三项贡献可以在不依赖“首次”或算法性能优势的条件下成立；
- P-C3、机会成本和在线算法负结果增强了边界可信度；
- 9/9 模型无筛选、108 context 和 Actor 冻结支持有限意义上的独立确认。

### Consensus technical risks

1. 代数恒等式可能被误解为统计或因果无偏；
2. 小 context 数的正态近似区间可能被过度解释；
3. “独立确认”可能被误读为跨算法或外部实验室复现；
4. 单环境和固定顺序限制外推；
5. 公共数据、代码标识和目标期刊格式尚未完成。

### Where emphasis differs

- Reviewer 1 最关注统计边界、选择协议和可复现性；
- Reviewer 2 最关注 L2 原创性是否被误包装为完整算法；
- Reviewer 3 最关注缩写密度、统计层级和非专门读者路径。

### Broad-interest and significance readout

稿件已建立领域内可发表的测量问题，但没有建立跨环境、跨算法的广泛科学结论。
最稳妥出口是 M2：作为较大信用分配方法论文中的核心诊断模块。将其单独包装为
通用算法会同时削弱原创性和技术可信度。

### Most important actions

1. 终稿补充 context 选择不查看后续 N/E 结果；
2. 终稿明确小样本正态近似区间的描述性边界；
3. 完成真实、不承诺未公开仓库的数据与代码声明；
4. 保持 P-C2 通过与 P-C3 失败并列；
5. 将跨环境、跨算法、顺序和在线收益全部留在 R4。

## Risk / unsupported claims

| ID | 风险 | 等级 | 当前裁决 | 状态 |
| --- | --- | --- | --- | --- |
| RP-01 | “只是记账” | R1 | 用动态后缀遗漏、符号掩盖和独立边界说明测量后果 | CLOSED |
| RP-02 | context 选择可能查看结果 | R1 | Supplement S3 明确选择不查看后续 N/E 结果；终稿补入 | CLOSED |
| RP-03 | 小样本区间被当作总体推断 | R2 | 终稿增加描述性门控和非总体泛化边界 | CLOSED |
| RP-04 | “独立确认”被外推为跨算法复现 | R2 | 限定新来源策略 seeds/context、同一算法和环境 | CLOSED |
| RP-05 | “精确”被外推为因果无偏 | R2 | 限定逐账本代数重构 | CLOSED |
| RP-06 | 冻结数据数字或门控不一致 | R3 | 直接回查 JSON/CSV/模型清单全部通过 | CLOSED |
| RP-07 | 跨环境、跨算法、跨顺序验证 | R4 | 记录为后续独立研究，不在 W1 执行 | OPEN_NONBLOCKING |
| RP-08 | 在线 PPO/GNN 性能 | R4 | 维持否决/未验证状态 | OPEN_NONBLOCKING |
| RP-09 | 目标期刊和公共仓库标识 | 投稿格式 | 不宣称格式定稿；移交后续投稿适配 | OPEN_NONBLOCKING |
| RP-10 | 致命证据或完整性冲突 | RX | 未发现 | NONE |

## Final reviewer verdict

不存在未关闭的 R2、R3 或 RX 问题。R4 仅记录未来独立研究问题，不进入 W1。
稿件可按 **L2/M2** 冻结为较大方法论文的测量与诊断模块；当前不具备指定期刊
格式定稿状态。
