# W1-04 Results 证据核查说明

更新时间：2026-07-24  
用途：逐段核对 Results 数字、动词和边界；不是新的数值权威

## 1. 段落—证据映射

| Paragraph ID | 主要观察 | Evidence ID | 唯一权威文件/字段 | 动词 | 边界 |
| --- | --- | --- | --- | --- | --- |
| RES-6.1-01 | time/resource 短标签全部歧义 | EV-BPCE-01 | `bpce_short_horizon_label_audit/gate_summary.json`：`slot_counts.time_pressure/resource` | 未产生、观察到 | 仅冻结短视窗协议 |
| RES-6.1-02 | 问题收窄到累计成本读出 | BD-01 | Claim 矩阵 C7 | 限定、检验 | 非算法性能结果 |
| RES-6.2-01 | R1 N/E 配对协议 | EV-R1-01 | R1 `experiment_config.json` 与正式报告 §2-§3 | 比较、定义 | 旧策略种子 8/9/10 |
| RES-6.2-02 | 18/18 正替代；平均 0.990、1.995 | EV-R1-01 | R1 `gate_summary.json` 的 P-R1；`context_opportunity_estimates.csv` 筛选 time/resource 后等权平均 `sub_shot_mean`、`sub_cost_mean` | 得到、观察到 | 1.995 是未来替代成本，不是 R2 总替代成本 |
| RES-6.2-03 | 11/11 非正成本可解释 | EV-R1-02 | R1 `gate_summary.json`：P-R1 非正成本字段 | 解释、重构 | 只负责机制发现 |
| RES-6.3-01 | 287/7,776、最大残差 2.0 | EV-R2-04、EV-R2-05 | `pre_ledger_correction/repeat_cost_ledger.csv`；R2 `gate_summary.json` | 暴露、出现 | 非预注册正面发现 |
| RES-6.3-02 | 总 0.864、同一步 0.147、未来 0.718、约 83% | EV-R2-13 + 账本字段 | `context_substitution_estimates.csv`：time/resource 三个 cost mean 字段 | 分解为、占 | context 等权聚合 |
| RES-6.3-03 | 最大误差 8.88e-16、Actor 差 0 | EV-R2-06、EV-R2-07 | R2 `gate_summary.json` 对应字段 | 满足、重构 | 精确只指代数恒等式 |
| RES-6.4-01 | 9 模型、108 上下文、3,456 重复、7,776 账本 | EV-R2-01 至 EV-R2-03、EV-R2-07 | R2 config/gate JSON | 完整使用、形成 | ledger row 不是独立 context |
| RES-6.4-02 | 13/18 均值和下界为正 | EV-R2-08 | R2 `gate_summary.json`：P-C2 | 超过门槛、为正 | 仅 time/resource |
| RES-6.4-03 | 三种子块区间与掩盖率 | EV-R2-09 | R2 `gate_summary.json`：`seed_block_intervals`、`seed_masked_rates` | 复现、为正 | 三个新标签设计种子 |
| RES-6.4-04 | 7/7 非正成本具有正总替代 | EV-R2-10 | R2 `gate_summary.json`：P-C2 非正字段 | 支持、观察到 | 不是总体发生率 |
| RES-6.5-01 | 三场景资源槽聚合 | EV-R2-13 | `scenario_boundary_summary.csv`：`slot=resource` | 不同、观察到 | 不外推场景排序 |
| RES-6.5-02 | missile/laser 分层 | EV-R2-11、EV-R2-12 | R2 P-C3；`resource_type_summary.csv` | 均存在、不同 | 仅 time/resource |
| RES-6.5-03 | P-C3 未通过 | BD-03 | Claim 矩阵 C4；R2 `mechanism_gates.P-C3` | 未通过、不支持普遍性 | 不等于 missile 无替代 |
| RES-6.6-01 | E/E-R 恢复干预 | EV-R1-03 | R1 配置与正式报告 §2-§3 | 构造、比较 | 当前交战结果保持不变 |
| RES-6.6-02 | 5/18、2/18、1/18；全为 missile | EV-R1-03 | R1 `gate_summary.json` 可靠上下文字段 | 只出现在、集中于 | 不否定个别上下文 |
| RES-6.6-03 | 复用/动作边扩大但安全门控不一致 | EV-R1-03、EV-BPCE-01 | R1 `context_opportunity_estimates.csv` 分场景/资源槽等权平均 `reuse_probe_mean`、`option_edge_mean`；short-horizon gate JSON | 扩大、未形成 | 行动集合不等于安全收益 |
| RES-6.6-04 | 通用机会成本 oracle 停止 | C5、BD-01 | Claim 矩阵 C5/C7 | 未支持、停止 | 不写所有弹药均无价值 |

## 2. 关键数字复核

| 数字 | 复核结果 | 来源 |
| --- | --- | --- |
| R1 18/18、11/11 | 一致 | R1 `gate_summary.json` |
| R2 287/7,776 | 一致；由首轮 CSV 确定性筛选 | EV-R2-04 |
| future-only 最大残差 2.0 | 一致 | R2 `gate_summary.json` |
| 完整账本最大误差 8.88e-16 | 一致 | R2 `gate_summary.json` |
| time/resource 0.864/0.147/0.718 | 一致 | R2 context 表等权平均 |
| 未来替代占比 83.05% | 正文按约 83% 报告 | 0.717850/0.864378 |
| 9 模型、108 上下文、3,456 重复、7,776 账本 | 一致 | R2 config/gate JSON |
| 13/18、13/18、3/3、2/3、7/7 | 一致 | R2 P-C2 |
| missile 2/9、laser 5/9 | 一致 | R2 P-C3 |
| 场景 \(\rho_{\mathrm{sub}}\) 0.747/0.873/0.972 | 一致 | scenario CSV |
| 机会价值 5/18、2/18，且全为 missile | 一致 | R1 gate/report |

## 3. 透明性记录

1. 首轮公式修正被写入 Results 6.3，而不是只放入补充材料。
2. `Sub_{\mathrm{shot}}` 始终表示未来替代射击，不与总替代成本互换。
3. R1 的 1.995 使用“未来替代成本”，没有升级为 R2 的
   \(Sub_{\mathrm{cost,total}}\)。
4. R2 的“独立”限定为新来源模型、新上下文和旧 hash 零重叠。
5. P-C3 明确报告为失败门控。
6. 资源恢复负结果保留在主文，不以 seed9 或 missile 正例替代总体判定。
7. 在线 BPCE/MCH-PPO 失败留给 Limitations；本 Results 不将其包装为方法结果。

## 4. Results 与 Discussion 边界

已从 Results 排除：

- 为什么 laser 更容易发生符号掩盖的机制解释；
- 动作替代与 COMA/CCA/因果效应传播的理论关系；
- CRN 为何不能解除结构混合的完整论证；
- 对未来在线信用估计器和 GNN 的设计建议；
- 对跨环境或任意策略稳健性的推断。

这些内容分别移交 W1-07 或明确保持未验证。
