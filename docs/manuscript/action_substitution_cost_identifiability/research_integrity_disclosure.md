# 科研完整性披露：R2 成本账本定义修正

更新时间：2026-07-28  
适用范围：动作替代独立确认 R2  
状态：W1-05 冻结披露

## 1. 事件时间线

1. 首轮正式确认按预定的 N/E、共同随机数、目标精确边缘化和冻结策略协议执行，
   但成本分解只包含严格晚于干预步的 future-only 替代项。
2. 完整性检查发现 7,776 条目标条件账本中有 287 条的 future-only 恒等式出现
   非零残差，最大绝对残差为 2.0，故首轮 P-C1 被判无效，未进入支持性结论。
3. 对分支动作和逐单元成本的核查显示，探针在当前步被强制 engage 后，会改变
   无冲突自回归掩码中的目标占用，因而使同一步后缀中的其他单元少执行或改写
   一次交战。原 future-only 公式遗漏了这一当前步其他单元成本差。
4. 成本账本增加
   \(Sub_{\mathrm{cost,same}}=C_{t,-i}(N)-C_{t,-i}(E)\)，并冻结完整恒等式：

\[
Sub_{\mathrm{cost,total}}
=Sub_{\mathrm{cost,same}}
+Sub_{\mathrm{cost,future,probe}}
+Sub_{\mathrm{cost,future,other}},
\]

\[
\Delta C_{\mathrm{episode}}
=C_{\mathrm{direct}}-Sub_{\mathrm{cost,total}}.
\]

5. 扩展恒等式在相同账本上的最大绝对误差为
   \(8.88\times10^{-16}\)，低于 \(10^{-6}\) 的冻结容限。
6. 首轮无效产物完整归档至
   `results/air_defense_v1/action_substitution_confirmation/pre_ledger_correction/`。
   随后只执行一次完整重跑。

## 2. 重跑中保持不变的项目

唯一重跑保持以下项目不变：

- 9 个来源模型及其参数；
- 108 个已选上下文、探针单元、槽位和合法目标；
- 每上下文 32 次重复；
- 环境命中随机带和策略均匀随机带；
- 目标精确边缘化概率；
- seeds 17/18/19 和三个场景；
- P-C1/P-C2/P-C3 阈值、完整性阈值和 transition 上限；
- 确认阶段 Actor 冻结；
- 9/9 来源模型无条件保留，不按结果重新筛选。

变更仅限成本账本的科学定义及与该定义对应的输出字段、断言和重算汇总。

## 3. 披露性质与措辞边界

该事件不应被描述为省略科学含义的“常规代码修复”。它修正了成本估计对象：
从仅统计未来策略介导替代，改为同时统计联合动作内部的同一步后缀替代和未来
替代。也不应声称研究在首轮执行前已预先假设“同一步其他单元替代”；这一项是
由预设完整性检查中的残差暴露后识别的。允许的表述是：

> 首轮 future-only 账本未通过预设代数完整性检查；残差定位出动态掩码下遗漏
> 的同一步后缀成本项。我们归档首轮无效结果，冻结扩展恒等式，并在不改变模型、
> 上下文、随机带和门槛的条件下执行唯一一次完整重跑。

扩展恒等式的数值闭合只证明成本字段之间的代数一致性，不证明任意因果效应
均已识别，也不把修正后的同一步分量转化为预注册正面假设。

## 4. 可审计材料

| 内容 | 文件 |
| --- | --- |
| 首轮 future-only 账本 | `results/air_defense_v1/action_substitution_confirmation/pre_ledger_correction/repeat_cost_ledger.csv` |
| 修正后完整账本 | `results/air_defense_v1/action_substitution_confirmation/repeat_cost_ledger.csv` |
| 最大残差与门控 | `results/air_defense_v1/action_substitution_confirmation/gate_summary.json` |
| 重跑标识与冻结配置 | `results/air_defense_v1/action_substitution_confirmation/experiment_config.json` |
| 来源模型哈希与无筛选字段 | `results/air_defense_v1/action_substitution_confirmation/source_model_manifest.json` |
| 正式过程报告 | `docs/experiments/air_defense_v1_action_substitution_confirmation.md` |
| 公式与方向冻结 | `docs/manuscript/action_substitution_cost_identifiability/formula_and_direction_freeze.md` |

