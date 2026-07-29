# 补充方法

更新时间：2026-07-28  
对应正文：`methods_draft_zh.md`

## S1. AirDefense v1 环境实现

### S1.1 默认规模与实体

| 项目 | medium 配置 |
| --- | --- |
| 地图尺度 | 100 |
| 决策间隔 | 1 |
| 最大步数 | 50 |
| 保护区 | command：位置 (0,0)、半径 5、价值 1.0；radar：位置 (25,-10)、半径 4、价值 0.8 |
| 目标数 | 5 |
| 目标生成距离 | 60-100 |
| 目标速度 | 1-3 |
| 目标威胁度 | 0.5-1 |
| 目标载荷 | 0.6-1.5 |
| 单元 0/1 | missile；位置 (-12,0)/(12,0)；弹药 3；射程 85；命中率 0.88；成本 2.0；射后冷却 1 |
| 单元 2 | laser；位置 (3,12)；弹药 10；射程 55；命中率 0.68；成本 0.5；冷却 0 |

`time_pressure` 只把目标速度范围改为 2.0-3.5。
`heterogeneity_pressure` 将单元 0 的射程/命中率/成本改为 92/0.94/2.8，
单元 1 改为 72/0.78/1.5，laser 改为 45/0.50/0.25；弹药量保持不变。

### S1.2 状态、动作与转移

扁平观测由以下归一化特征拼接而成：

| 实体 | 每个实体特征数 | 内容 |
| --- | ---: | --- |
| 保护区 | 7 | 位置、半径、价值、损伤、优先级和类型 |
| 目标 | 15 | 位置、速度、距离、到达时间、威胁、载荷、规避、保护区、类别、置信度、关注区和存活 |
| 防御单元 | 15 | 位置、类型、弹药、能量、冷却、射程、命中率、成本和可用性 |
| 全局 | 8 | 当前步、存活/拦截/突防计数、累计损伤、可用单元和剩余弹药 |

medium 的观测维度为 \(2\times7+5\times15+3\times15+8=142\)。
每个单元动作是目标索引 0-4 或 no-op 索引 5。合法交战要求单元有弹药、
冷却归零、能量为正、目标存活且在射程内。目标每步朝指派保护区运动；进入
区域半径后突防，损伤为目标载荷、威胁度和区域价值的乘积。合法射击按预设
命中概率结算，并在发射时立即扣除成本。累计损伤达到 2.5 或无存活目标时
终止，50 步时截断。

### S1.3 奖励与成本分离

奖励分量为：拦截 \(+8\)、损伤惩罚权重 30、非法动作 \(-5\)、冲突惩罚
权重 1、过度分配惩罚权重 0.5、每步时间惩罚 \(-0.1\)，成功/失败终局奖励
分别为 \(+25/-25\)。本文的 \(C\) 只累计环境记录的 missile/laser 发射成本；
其他奖励分量不进入资源成本账本。

## S2. 来源策略和训练

联合动作按单元顺序 0、1、2 自回归生成。每选择一个目标，该目标从后续单元
条件掩码中移除；后续单元仍保留 no-op。每个单元先由 sigmoid 交战头产生
engage/no-op 概率，再由 masked softmax 目标头在合法目标中分配 engage 概率。
联合 log-probability 为条件 log-probability 之和，PPO 使用 joint ratio
执行单一 clipped surrogate 更新。

正式确认模型参数如下：

| 参数 | 值 |
| --- | ---: |
| 每场景策略种子 | 17、18、19 |
| 请求训练步数 | 10,000 |
| rollout steps | 256 |
| batch size | 64 |
| epochs | 2 |
| learning rate | \(3\times10^{-4}\) |
| gamma | 0.98 |
| GAE lambda | 0.95 |
| clip range | 0.2 |
| entropy coefficient | 0.01 |
| value coefficient | 0.5 |
| max gradient norm | 0.5 |

来源模型清单记录每个模型的 SHA-256、场景、seed 和是否从现有冻结文件加载。
全部 9 个模型的 `selected_by_behavior=false`。确认阶段设置 evaluation 模式，
禁止 Actor/Critic 更新。

## S3. 上下文构造与独立性

每个“场景 × seed”块运行 24 个候选回合。候选上下文必须包含至少一个合法
engage 目标；保存观测哈希、环境状态快照、环境步、探针单元、合法目标、
条件目标概率和来源动作分布。context identity 由观测哈希、单元索引、环境步
和合法目标共同校验，并排除前置审计中可获得的 60 个旧观测哈希。
候选回合的 context base seed 为 1,283,000；采集来源动作时使用 0.5 的
engagement threshold。

safety 分数以合法目标中“威胁 × 载荷 × 区域价值”除以截断后的到达时间构造；
resource 分数综合当前弹药缺口、单次成本及是否存在其他可交战单元。每块分别
选取 6 个 safety 和 6 个 resource 上下文；resource 槽进一步固定为 3 个
missile 和 3 个 laser。3 个场景、3 个 seed 和每块 12 个上下文构成 108 个
确认上下文。上下文选择不查看后续 N/E 成本结果。

## S4. N/E 回放算法

对每个上下文和 repeat，branch base seed 为 1,293,000：

1. 生成环境命中随机带，形状为“最大环境步 × 目标数”；生成策略均匀随机带，
   索引为“环境步 × 单元”。
2. 恢复同一状态快照和固定前缀。
3. 在 \(N\) 分支将探针设为 no-op。
4. 对探针每个合法目标建立一个 \(E\) 分支，按条件目标概率赋权。
5. 对探针之后的同一步单元，按各分支实时合法掩码和共享均匀数生成后缀。
6. 环境推进后，冻结策略继续使用同一套按步、单元和目标索引的随机带进行
   stochastic continuation，直到终止或截断。
7. 记录当前探针、当前其他单元、未来探针和未来其他单元的成本与射击数。
8. 先在目标维度精确加权，再形成 repeat、context 和 block 层汇总。

动作依赖的状态与掩码允许分叉；共同随机带只固定外生随机输入的索引，不强制
两个分支执行相同动作。

## S5. 成本账本字段与完整性

| 论文量 | 账本字段 |
| --- | --- |
| \(C_{t,i}(N/E)\) | `current_probe_cost_n/e` |
| \(C_{t,-i}(N/E)\) | `current_other_cost_n/e` |
| \(C_{>t,i}(N/E)\) | `future_probe_cost_n/e` |
| \(C_{>t,-i}(N/E)\) | `future_other_cost_n/e` |
| \(C_{\mathrm{direct}}\) | `direct_cost` |
| \(Sub_{\mathrm{cost,same}}\) | `same_step_other_sub_cost` |
| \(Sub_{\mathrm{cost,future,probe}}\) | `future_sub_cost_probe` |
| \(Sub_{\mathrm{cost,future,other}}\) | `future_sub_cost_other` |
| \(Sub_{\mathrm{cost,total}}\) | `sub_cost` |
| \(Sub_{\mathrm{shot}}\) | `sub_shot` |
| \(\Delta C_{\mathrm{episode}}\) | `episode_cost_delta` |
| \(\rho_{\mathrm{sub}}\) | `rho_sub` |
| \(I_{\mathrm{mask}}\) | `cost_sign_masked` |

对每条目标账本检查：

\[
\Delta C_{\mathrm{episode}}
-C_{\mathrm{direct}}
+Sub_{\mathrm{cost,total}}=0,
\]

\[
Sub_{\mathrm{cost,total}}
-Sub_{\mathrm{cost,same}}
-Sub_{\mathrm{cost,future,probe}}
-Sub_{\mathrm{cost,future,other}}=0.
\]

分解容限为 \(10^{-6}\)，目标概率和容限为 \(10^{-12}\)。
\(\rho_{\mathrm{sub}}\) 和 \(I_{\mathrm{mask}}\) 只在
\(C_{\mathrm{direct}}>0\) 的目标条件账本上计算。

## S6. 前置标签语义审计

前置标签审计用于选择最终测量问题，而非独立确认。A 使用 argmax 目标和
deterministic continuation；B 使用精确目标边缘化和 deterministic
continuation；C 同时使用精确目标边缘化、共同策略均匀随机带和 stochastic
continuation。完整回合差值与短视窗差值分别计算；短视窗终点取剩余回合长度
与 \(\lceil TTI\rceil+1\) 的较小值。依据安全分量和资源分量的区间关系，
上下文标为 ENGAGE、STOP 或 AMBIGUOUS。可靠差异要求绝对均值至少 1.0 且
95% 区间排除 0。该流程显示标签对目标语义、延续方式和视窗敏感，因此正式
确认不训练标签 oracle，而改为直接审计累计成本恒等式及其替代分量。

## S7. 统计层级和门控实现

所有区间使用 \(\bar{x}\pm1.96s/\sqrt{n}\)。context 区间的 \(n=32\) 个
repeat；block 和资源/场景分组区间以 context 均值为样本。目标账本行只是精确
边缘化的构成项。

| 门控 | 判据 |
| --- | --- |
| P-C1 | 协议残差与完整替代分解残差均不超过 \(10^{-6}\)，且所有直接成本为正 |
| P-C2 | time/resource 为 18 个 context；正均值至少 12；正 95% 下界至少 6；正 seed-block 下界至少 2；掩盖率至少 0.5 的 seed 至少 2；非正累计成本 context 的正替代解释比例至少 0.8 |
| P-C3 | missile、laser 各自满足聚合 95% 下界为正、正 seed 均值至少 2、掩盖 context 至少 3 |

独立性门控还要求 9 个模型、108 个上下文、旧 hash 零重叠、每块资源配额
3/3、3,456 条 context-repeat 记录、Actor 参数最大差为零、总额外 transition
不超过 266,198，以及软件回归通过。
