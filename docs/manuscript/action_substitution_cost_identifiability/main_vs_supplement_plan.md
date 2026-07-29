# W1-03 主文与补充材料计划

更新时间：2026-07-24  
原则：决定主张成立或失败边界的证据进入主文；复核细节和历史流水进入补充材料

## 1. 主文保留项

| ID | 内容 | 主文位置 | 载体 | 保留理由 |
| --- | --- | --- | --- | --- |
| MAIN-01 | 动态掩码自回归动作与局部成本测量问题 | §1、§3 | Fig. 1 | 定义读者必须理解的 Problem |
| MAIN-02 | N/E 配对、CRN 和目标精确边缘化 | §4、§5 | Fig. 1、Table 1 | 定义估计协议及 CRN 能力边界 |
| MAIN-03 | 三类替代成本与主恒等式 | §4 | Fig. 2、Fig. 4 | 核心方法模块 |
| MAIN-04 | R1 的正替代和非正成本解释 | §6.1 | Fig. 3 | 建立测量混合的直接证据 |
| MAIN-05 | future-only 遗漏与完整账本修正 | §6.2 | Fig. 2、Fig. 4 | 证明同一步后缀项不可省略 |
| MAIN-06 | 新模型、新上下文和 Actor 冻结完整性 | §5、§6.3 | Table 1、Table 2 | 支撑“独立确认” |
| MAIN-07 | seeds 17/18/19 块级复现 | §6.3 | Fig. 3 | 支撑 C3 |
| MAIN-08 | 场景与资源类型边界 | §6.4 | Fig. 5、Table 3、Table 4 | C4 否决是核心科学边界 |
| MAIN-09 | 资源恢复通用机会价值失败 | §6.5 | 正文；Table S2 | 解释为何不进入 opportunity oracle |
| MAIN-10 | 在线 BPCE/MCH-PPO 总门控失败 | §8 | Limitations 正文 | 防止把测量贡献误写为算法胜出；不占用 Results 6.6 |
| MAIN-11 | 单环境、冻结策略和因果范围限制 | §8 | 正文 | 限定外推 |
| MAIN-12 | GNN 尚未验证 | §8 | 正文 | C8 的明确不使用去向 |

不利结果 MAIN-08 至 MAIN-10 不能仅因不利而移入补充材料。

## 2. 补充材料计划

| ID | 内容 | 补充位置 | 来源 | 主文引用方式 |
| --- | --- | --- | --- | --- |
| SUP-01 | AirDefense v1 完整状态、动作、奖励和终止定义 | Methods S1 | 环境设计文档/代码 | 主文只保留资源信用所需定义 |
| SUP-02 | 所有来源模型训练配置与模型清单 | Methods S2 | `source_model_manifest.json`、`experiment_config.json` | Table 1 引用汇总 |
| SUP-03 | 上下文选择、hash 身份与种子使用审计 | Methods S3 | `context_identity_check.csv`、`seed_usage_audit.json` | §5.2 引用 |
| SUP-04 | 完整 N/E 随机带、目标边缘化和统计计算步骤 | Methods S4 | 实验配置、实现代码 | §4.2/§5.3 引用 |
| SUP-05 | R1 全部上下文和 E/E-R 完整性表 | Results S1 | R1 CSV/JSON | §6.1、§6.5 引用 |
| SUP-06 | 首轮 future-only 逐行残差和修正前归档 | Audit S1 | `pre_ledger_correction/` | §6.2 报告 287/7,776 和最大 2.0 |
| SUP-07 | R2 全部 7,776 条目标账本复核摘要 | Results S2 | `repeat_cost_ledger.csv` | Fig. 2 只展示核心分布/误差 |
| SUP-08 | 全部 context、block 和 seed 区间 | Results S3 | `context_substitution_estimates.csv`、`block_summary.csv` | Fig. 3 展示预注册主切片 |
| SUP-09 | 全部场景、槽位和资源类型表 | Results S4 | `scenario_boundary_summary.csv`、`resource_type_summary.csv` | Fig. 4 展示决定性边界 |
| SUP-10 | 资源恢复完整标签审计 | Results S6 | R1 opportunity audit 结果 | Table 2 展示门控结论 |
| SUP-11 | BPCE/MCH-PPO 候选运行、smoke 和历史变体 | Limitations S1 | BPCE stress/历史报告 | 主文只在 Limitations 保留总门控失败 |
| SUP-12 | 完整软件版本、回归测试和字段映射 | Reproducibility S1 | 测试报告/代码版本 | Table 1 报告通过数量和关键完整性 |
| SUP-13 | 24 篇文献证据矩阵和检索协议 | Literature S1 | W1-02 交付件 | Related Work 只引用直接相关原始论文 |

## 3. 主图和主表冻结

| 编号 | 暂定内容 | 支撑主张 | 禁止承载 |
| --- | --- | --- | --- |
| Fig. 1 | 动态掩码序列分配与 N/E 配对评估框架 | C1、C2 | PPO 性能提升 |
| Fig. 2 | N/E 协议、三分量恒等式和账本完整性 | C1、C2 | 跨环境普遍性 |
| Fig. 3 | 新策略种子独立确认 | C3 | 分布外泛化 |
| Fig. 4 | 同一步/未来成本组成与符号掩盖 | C1、C2、C4 | 把 \(Sub_{\mathrm{shot}}\) 当总替代成本 |
| Fig. 5 | 场景和资源类型 \(\rho_{\mathrm{sub}}\) 与符号边界 | C4 | 所有资源类型同强度 |
| Table 1 | 任务、来源策略和反事实协议 | C2、C3 | 把 ledger row 当独立样本 |
| Table 2 | R1/R2 独立性与完整性 | C2、C3 | 把发现样本写成独立确认 |
| Table 3 | P-C1/P-C2/P-C3 门控 | C2-C4 | 隐藏 P-C3 失败 |
| Table 4 | 场景和资源类型边界 | C4 | 只报告 laser |

## 4. 图表与正文分工

- 正文先陈述问题或结果，图表提供对应证据，不让图注承担新主张。
- Methods 定义图中对象和统计单位；Results 只报告观察；Discussion 解释含义。
- Fig. 2 的恒等式由 W1-01 公式冻结文件提供，W1-06 不得重新推导方向。
- Fig. 4 必须同时展示 missile 和 laser，不能只展示更有利的 laser。
- Table 2 必须包含否决/未支持状态，不能仅列局部正例。
- 所有图表数字回指 `evidence_source_index.md` 的 Evidence ID，不从报告正文手工抄出新版本。

## 5. 不进入当前稿件的内容

| 内容 | 处理 | 原因 |
| --- | --- | --- |
| 未实现的 GNN 修复结果 | 不使用 | C8 未验证 |
| “MCH-PPO 已优于 PPO” | 不使用 | C6/C7 不支持 |
| 挑选 seed9 的机会成本正例 | 不单独使用 | 会绕过跨种子失败 |
| 只报告 laser 的成本符号掩盖 | 不使用 | 会隐藏资源类型边界 |
| 全部项目任务时间线 | 不使用 | 不是科学论证 |
| 重复的汇报稿数字 | 不作为来源 | 不是权威数值源 |

## 6. 完整性检查

| Claim | 去向 |
| --- | --- |
| C1 | 主文 MAIN-01 至 MAIN-04；细节 SUP-05 |
| C2 | 主文 MAIN-02、MAIN-03、MAIN-05；细节 SUP-06、SUP-07 |
| C3 | 主文 MAIN-06、MAIN-07；细节 SUP-02、SUP-03、SUP-08 |
| C4 | 主文 MAIN-08；细节 SUP-09 |
| C5 | 主文 MAIN-09；细节 SUP-05、SUP-10 |
| C6 | 主文 MAIN-10；细节 SUP-11 |
| C7 | 主文 §1/§7/§8 的非算法定位；不作为性能结果 |
| C8 | 主文 MAIN-12；明确不使用未验证结果 |
