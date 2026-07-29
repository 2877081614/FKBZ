# W1-04 补充结果提纲

更新时间：2026-07-24  
状态：供 W1-06/W1-09 使用  
原则：补充材料提高复核性，不隐藏决定性负结果

## Results S1：局部标签与 R1 机制发现

内容：

- 72 个 R1 上下文的场景、种子、槽位和资源类型；
- short/full 标签交叉表；
- `time_pressure/resource` 的 18 个短标签；
- R1 全部 \(Sub_{\mathrm{shot}}\)、未来替代成本和首次替代步；
- 11 个非正回合成本上下文的逐上下文重构。

权威来源：

```text
results/air_defense_v1/bpce_short_horizon_label_audit/gate_summary.json
results/air_defense_v1/action_substitution_opportunity_cost_audit/
```

主文对应：RES-6.1-01、RES-6.2-01 至 RES-6.2-03。

## Results S2：首轮成本账本修正审计

内容：

- 受影响 287 条账本的判定规则；
- future-only 残差分布与最大值 2.0；
- 同一步其他单元成本差；
- 修正前后字段映射；
- 首轮归档、唯一重跑和不变项清单。

权威来源：

```text
results/air_defense_v1/action_substitution_confirmation/
  pre_ledger_correction/repeat_cost_ledger.csv
  repeat_cost_ledger.csv
  gate_summary.json
```

主文对应：RES-6.3-01 至 RES-6.3-03。

## Results S3：完整三分量账本

内容：

- 7,776 条目标账本的残差摘要；
- \(Sub_{\mathrm{cost,same}}\)、
  \(Sub_{\mathrm{cost,future,probe}}\)、
  \(Sub_{\mathrm{cost,future,other}}\) 分布；
- context/repeat/target 条件概率层级说明；
- time/resource 0.864/0.147/0.718 的确定性聚合步骤；
- 每个场景和槽位的三分量表。

权威来源：

```text
repeat_cost_ledger.csv
repeat_marginal_metrics.csv
context_substitution_estimates.csv
scenario_boundary_summary.csv
```

主文对应：Fig. 2、Fig. 4、RES-6.3-02。

## Results S4：独立确认完整结果

内容：

- 9 个来源模型清单和训练种子；
- 108 个上下文与旧 hash 身份审计；
- 18 个 time/resource 上下文区间；
- seeds 17/18/19 块级均值、标准误和 95% 区间；
- 所有场景/槽位 block 表；
- 7 个非正回合成本上下文的总替代成本。

权威来源：

```text
source_model_manifest.json
context_identity_check.csv
context_substitution_estimates.csv
block_summary.csv
gate_summary.json
```

主文对应：Table 1、Table 2、Fig. 3、RES-6.4-01 至 RES-6.4-04。

## Results S5：场景与资源类型边界

内容：

- medium/time/heterogeneity 的场景聚合和区间；
- missile/laser 的全部 18 个 context；
- \(\rho_{\mathrm{sub}}\) 分布与符号掩盖计数；
- P-C3 门槛和失败判定；
- 不按有利资源类型筛选的完整表。

权威来源：

```text
scenario_boundary_summary.csv
resource_type_summary.csv
context_substitution_estimates.csv
gate_summary.json
```

主文对应：Fig. 5、Table 3、Table 4、RES-6.5-01 至 RES-6.5-03。

## Results S6：资源恢复与在线主张停止边界

内容：

- E/E-R 当前步一致性和终止步不可观测计数；
- Reuse、OptionEdge、Damage/Leak 安全分量；
- time/heterogeneity × safety/resource 全部可靠机会计数；
- 可靠上下文的资源类型和种子分布；
- BPCE/MCH-PPO 失败运行只作为算法主张边界，不与测量结果合并。

权威来源：

```text
results/air_defense_v1/action_substitution_opportunity_cost_audit/
results/air_defense_v1/bpce_ppo_mechanism_stress_test/
```

主文对应：Table S2、RES-6.6-01 至 RES-6.6-04；在线失败细节对应 Limitations。

## 补充图表占位

| 编号 | 内容 | 主张/边界 |
| --- | --- | --- |
| Fig. S1 | 18 个 R1 time/resource 上下文替代区间 | C1 |
| Fig. S2 | 修正前后账本残差 | C2 |
| Fig. S3 | 三分量成本分布 | C2 |
| Fig. S4 | 全部 seed/scenario/slot block 区间 | C3 |
| Fig. S5 | 资源类型逐上下文 \(\rho_{\mathrm{sub}}\) | C4 |
| Fig. S6 | E/E-R 行动集合与安全收益配对 | C5 否决 |
| Table S1 | 标签语义与短视窗前置审计 | C5 |
| Table S2 | 资源恢复机会价值负结果 | C5 |
| Table S3 | 首轮账本修正与完整性 | C2 |
| Table S4（PLANNED） | R1 全部上下文 | C1 |
| Table S5（PLANNED） | R2 完整性、全部块和统计单位 | C2/C3 |
| Table S6（PLANNED） | 场景/资源类型完整边界 | C4 |
| Table S7（PLANNED） | 在线算法停止门控 | C6 |
