# 主表与补充表中文表注草稿

更新时间：2026-07-28

## Table 1 | AirDefense v1、来源策略与成对反事实协议

汇总三个正式场景、factorized joint PPO 来源策略、seeds 17/18/19、9 个无行为
筛选模型、108 个上下文、每块资源类型配额、每 context 32 次 N/E 配对、合法
目标精确边缘化和冻结 stochastic continuation。表中模型数、context 数和
repeat 数分别对应 model、context 和 repeat，不得把 7,776 条目标账本当作
独立样本。

## Table 2 | R1 发现与 R2 独立确认的样本和完整性边界

R1 seeds 8/9/10 仅承担动作替代机制发现；R2 seeds 17/18/19 承担独立确认。
R2 的 9/9 模型无条件保留，108 个 context 与可核验旧观测 hash 零重叠，
Actor 最大参数差为零。hash 零重叠表示未复用旧状态，不表示分布外环境泛化。

## Table 3 | P-C1、P-C2 与 P-C3 冻结门控

逐项报告成本恒等式、`time_pressure/resource` 独立替代确认和跨资源类型门控
的冻结判据、观察值与 PASS/FAIL。P-C3 只有 missile 和 laser 均满足正下界、
至少两个正 seed 和至少三个掩盖 context 时才通过；missile 的掩盖数为 2/9，
因此整体 P-C3 失败。失败状态不得被解释为 missile 不存在动作替代。

## Table 4 | 场景和资源类型边界

场景行使用 R2 resource 槽，每场景 18 个 context；资源类型行限定为
`time_pressure/resource`，每类型 9 个 context。`Mean Sub_shot` 是严格晚于
干预步的替代射击数，`Mean rho_sub` 使用完整三分量总替代成本；二者不是同一
估计量。Masked rate 是 context 内 paired repeat 的符号掩盖率再作 context
等权聚合，不支持跨环境普遍外推。

## Supplementary Table 1 | 标签语义与短视窗前置审计

列出 A（argmax 目标 + deterministic continuation）、B（精确目标边缘化 +
deterministic continuation）、C（精确目标边缘化 + stochastic continuation）
的可靠 context 数，以及短视窗 ENGAGE/STOP/AMBIGUOUS 计数。该表只说明最终
研究问题如何由标签语义收窄到累计成本测量，不作为 R2 独立确认。

## Supplementary Table 2 | 资源恢复机会价值负结果

P-R1 动作替代门控通过，但 P-R2 通用机会价值和 P-R3 资源关键性门控失败，
因此冻结“保留替代机制、停止通用 opportunity 路线”的决策。局部可靠
context 不得被选作通用弹药价值或在线训练 oracle。

## Supplementary Table 3 | 首轮账本修正与科研完整性

首轮 future-only 恒等式影响 287/7,776 条目标账本，最大残差 2.0；加入同一步
其他单元项后完整恒等式最大误差为 \(8.88\times10^{-16}\)。首轮无效结果已
归档，模型、context、随机带和门槛保持不变，只执行一次完整重跑。该事件是
科学测量定义修正，不描述为省略科学含义的常规代码修复。

