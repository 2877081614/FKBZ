# W1-09 中文整稿反向提纲

更新时间：2026-07-28  
对应稿件：`manuscript_draft_zh.md`  
检查规则：每个 Paragraph ID 只有一个主要功能

## 1. Abstract、Introduction 与 Related Work

| Paragraph ID | 首句 | 单一功能 | Claim ID | Evidence ID | 与上段关系 |
| --- | --- | --- | --- | --- | --- |
| A01 | 动态资源分配需要把团队结果转化为当前动作的局部资源信用 | summary | C1-C4、C7 | EV-R2-01 至 EV-R2-13、BD-03 | 全稿压缩 |
| I01 | 动态资源分配要求策略在任务效果和有限资源消耗之间持续权衡 | context | — | AirDefense v1 | 展开研究对象 |
| I02 | 多智能体强化学习通常从团队回报或反事实回报差估计局部贡献 | known gap | C1 | E01-E09、E19、E20 | 从任务进入已知信用问题 |
| I03 | 这一测量问题在动态掩码自回归联合动作中更具体 | exact gap | C1、C2 | E12-E14、E21 | 收窄技术瓶颈 |
| I04 | 为回答这一问题，我们构造成对反事实审计 | approach | C2 | T04-T18、EV-R2-06 | 回应 gap |
| I05 | 该审计将机制发现与独立确认分开 | evidence preview | C3、C4 | EV-R2-01、EV-R2-08 至 EV-R2-13 | 预告证据与边界 |
| I06 | 据此，本文保留三项有边界的贡献 | contribution | C1-C4、C6-C8 | W1-02 L2 | 压缩贡献并停止外推 |
| RW01 | Difference rewards、COMA 与 Shapley 建立反事实局部信用 | comparison | C1 | E01-E05 | 说明直接先例 |
| RW02 | 时序信用方法处理后续行为介导效应 | comparison | C1、C2 | E06-E09、E19、E20、E23 | 扩展到时序先例 |
| RW03 | 顺序更新、自回归和动作掩码均有先例 | comparison | C2 | E12-E14、E21、E22 | 定位动态后缀差异 |
| RW04a | 约束 MARL 控制策略层累计成本 | comparison | C5 | E16-E18 | 区分预算与归属 |
| RW04b | 共同随机数是配对仿真的方差缩减工具 | comparison | C2 | E15 | 区分精度与识别 |
| RW05 | 现有研究已覆盖本研究所依赖的各个一般概念 | positioning | C2、C4、C7 | W1-02 L2 | 汇总为测量模块 |

## 2. Problem Formulation、Method 与 Protocol

| Paragraph ID | 首句 | 单一功能 | Claim ID | Evidence ID | 与上段关系 |
| --- | --- | --- | --- | --- | --- |
| PF01 | 我们在 AirDefense v1 中研究动态防空资源分配 | context | — | 环境设计、Table 1 | 从文献进入任务定义 |
| PF02 | 来源策略是 factorized joint PPO | method | C1 | T01-T03 | 定义条件联合动作 |
| PF03 | 我们冻结上下文并比较 N/E 两个局部分支 | estimand | C1 | T04-T15 | 定义测量对象 |
| PF04 | 正式确认覆盖同一环境族中的三个配置 | limitation | C7、C8 | BD-01、BD-02 | 限定作用域 |
| M01 | 每个上下文保存完整快照并重放 N/E 分支 | method | C1、C2 | T04-T08 | 实现实验干预 |
| M02 | 每次 repeat 为两分支生成共同随机数带 | variance control | C2 | T05、E15 | 控制随机差异 |
| M03 | 全部合法目标按条件概率精确边缘化 | integration | C2 | EV-R2-03 | 消除目标抽样误差 |
| M04 | 探针直接成本是当前步 E−N 成本 | definition | C2 | T08 | 定义直接项 |
| M05 | 同一步其他单元替代采用 N−E 方向 | definition | C2 | T09 | 增加同一步项 |
| M06 | 严格未来替代分为探针与其他单元 | definition | C2 | T10-T15 | 完成三分量账本 |
| M07 | 每条目标条件账本满足完整成本恒等式 | identity | C2 | EV-R2-06 | 汇总代数关系 |
| M08 | 替代比率和符号掩盖只在正直接成本下定义 | metric | C4 | T16、T17 | 定义边界指标 |
| M09 | 分解只识别冻结策略响应下的操作账本 | limitation | C1-C4 | T18、E19、E20 | 限定可辨识性 |
| P01 | R1 负责发现，R2 负责独立确认 | protocol | C1、C3 | EV-R1-01、EV-R2-01 | 分离证据职责 |
| P02 | R2 无条件保留 9 个新来源模型 | independence | C3 | EV-R2-01、EV-R2-02 | 定义独立性 |
| P03 | 每个上下文运行 32 次配对 repeat | sampling | C2、C3 | EV-R2-03 | 冻结统计单位 |
| P04 | 区间和完整性门控在确认前冻结 | statistics | C2、C3 | EV-R2-06、EV-R2-07 | 定义质量控制 |
| P05 | P-C1/P-C2/P-C3 分别检验恒等式、复现和资源边界 | gates | C3-C5 | 冻结门控 | 定义判定规则 |
| P06 | 首轮 future-only 公式遗漏同一步项 | integrity | C2 | EV-R2-04 至 EV-R2-07 | 披露测量修正 |

## 3. Results

| Paragraph ID | 首句 | 单一功能 | Claim ID | Evidence ID | 与上段关系 |
| --- | --- | --- | --- | --- | --- |
| RES-6.1-01 | 短视窗审计未产生可行动标签 | negative result | C5 | EV-BPCE-01 | 建立问题来源 |
| RES-6.1-02 | 后续结果只检验更窄的累计成本读出问题 | scope | C7 | BD-01 | 收窄研究问题 |
| RES-6.2-01 | R1 在旧种子上比较成对 N/E 分支 | setup | C1 | EV-R1-01 | 开始机制发现 |
| RES-6.2-02 | R1 的 18/18 个上下文均有正替代射击下界 | result | C1 | EV-R1-01 | 报告发现 |
| RES-6.2-03 | 非正累计成本上下文均有正未来替代 | result | C1 | EV-R1-02 | 连接替代与混合 |
| RES-6.3-01 | R2 首轮暴露 future-only 公式遗漏 | integrity result | C2 | EV-R2-04、EV-R2-05 | 引出完整分解 |
| RES-6.3-02 | 加入同一步项后得到完整三分量组成 | result | C2 | EV-R2-13 | 报告成本组成 |
| RES-6.3-03 | 修正后账本逐行闭合且 Actor 不变 | validation | C2 | EV-R2-06、EV-R2-07 | 验证完整性 |
| RES-6.4-01 | R2 使用全部新模型和新上下文 | independence | C3 | EV-R2-01 至 EV-R2-03 | 建立确认样本 |
| RES-6.4-02 | 新种子切片有 13/18 个正下界上下文 | result | C3 | EV-R2-08 | context 级确认 |
| RES-6.4-03 | 三个 seed block 的下界均为正 | result | C3 | EV-R2-09 | seed 级确认 |
| RES-6.4-04 | 7/7 个非正累计成本上下文具有正总替代 | result | C1、C3 | EV-R2-10 | 确认机制一致性 |
| RES-6.5-01 | 三个场景均有正替代但强度不同 | boundary result | C4 | EV-R2-13 | 扩展到场景边界 |
| RES-6.5-02 | 两类资源均有正替代但抵消比例不同 | boundary result | C4 | EV-R2-11、EV-R2-12 | 资源类型比较 |
| RES-6.5-03 | missile 未达到 P-C3 掩盖门槛 | failed gate | C4 | BD-03 | 否决跨类型普遍性 |
| RES-6.6-01 | 资源恢复分支分离当前交战与弹药恢复 | setup | C5 | EV-R1-03 | 定义机会价值审计 |
| RES-6.6-02 | 可靠正机会价值只出现在少数上下文 | negative result | C5 | EV-R1-03 | 报告覆盖不足 |
| RES-6.6-03 | 行动集合扩大未形成一致安全收益 | negative result | C5 | EV-R1-03、EV-BPCE-01 | 排除行动数替代 |
| RES-6.6-04 | 通用机会成本 oracle 路线停止 | decision | C5、C7 | BD-01 | 冻结负结论 |

## 4. Discussion、Limitations 与 Conclusion

| Paragraph ID | 首句 | 单一功能 | Claim ID | Evidence ID | 与上段关系 |
| --- | --- | --- | --- | --- | --- |
| D00 | 本研究把已知信用问题落实为可审计测量对象 | implication | C1-C4 | EV-R2-06、EV-R2-09、BD-03 | 解释中心推进 |
| D03 | 同一步替代来自联合动作内部条件依赖 | mechanism | C2 | EV-R2-04、EV-R2-05、E14 | 解释同一步项 |
| D01 | 未来替代反映后续状态和策略分工变化 | mechanism | C1、C2 | EV-R2-06、E15 | 区分结构与方差 |
| D02a | 一般反事实信用已有直接先例 | comparison | C1、C2 | E01-E04 | 限定方法关系 |
| D02b | 后续行为介导效应已有因果与时序先例 | comparison | C1、C2 | E05-E09、E19、E20、E23 | 限定理论新颖性 |
| D02c | 顺序策略和动作掩码已有先例 | comparison | C2 | E12-E14、E21、E22 | 限定动作结构新颖性 |
| D02d | 约束优化与局部成本归属是不同估计对象 | comparison | C5 | E16-E18、EV-R1-03 | 解释机会价值失败 |
| D02e | CRN 只控制配对差方差 | comparison | C2 | E15 | 排除方差控制混淆 |
| D04 | 资源分层只支持条件性解释 | mechanism boundary | C4 | EV-R2-11 至 EV-R2-13 | 解释 P-C3 失败 |
| D05 | 当前结果只提出在线方法设计要求 | implication | C6-C8 | BD-01、BD-02、EV-BPCE-02 | 停止算法外推 |
| L01 | 经验范围限于单一环境族 | limitation | C3、C4 | BD-03 | 环境和资源边界 |
| L02 | 可辨识范围限于当前策略和顺序 | limitation | C1、C2 | T18、E19、E20 | 干预结构边界 |
| L03 | 结论依赖当前资源成本定义 | limitation | C5-C7 | EV-R1-03、EV-BPCE-02 | 成本与算法边界 |
| L04 | 独立确认未覆盖跨算法或跨环境 | limitation | C8 | BD-02 | 泛化与 GNN 边界 |
| C01 | 本文将局部资源信用操作化为成对反事实审计 | conclusion | C1-C4、C6-C8 | EV-R2-06、EV-R2-09、BD-01 至 BD-03 | 贡献—证据—意义—边界 |

## 5. 检查结论

- 共 66 个 Paragraph ID，均具有唯一主要功能；
- Results 段落只报告观察、设置或门控决策，机制解释留在 Discussion；
- P-C3、机会成本、在线算法和 GNN 负边界均有独立段落；
- 没有无证据的新科学段落。
