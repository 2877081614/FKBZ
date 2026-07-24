# 动作替代成本可辨识性：术语账本

更新时间：2026-07-24  
状态：W1-01 冻结  
适用范围：AirDefense v1、冻结策略下的 N/E 配对反事实评估及其论文写作

## 1. 使用规则

1. 下表是本论文方向的术语和符号唯一账本；下游正文、图表、图注和补充材料均应引用同一含义。
2. `action substitution` 暂定译为“动作替代”。其与既有文献术语的关系由 W1-02 系统检索确认，在确认前不得声称“首次提出”。
3. 成本均表示非负资源消耗；奖励中的负成本项不使用本账本的成本符号。
4. “反事实差”必须同时写明量、方向和统计单位，不得省略 N/E 方向。

## 2. 冻结术语

| ID | 中文术语 | English | 符号 | 冻结定义 | 使用边界 |
| --- | --- | --- | --- | --- | --- |
| T01 | 动态合法动作掩码 | dynamic legal-action masking | \(\mathcal M_t^i\) | 单元 \(i\) 在决策步 \(t\) 可选动作的合法集合，随状态和前序单元动作更新 | 仅描述合法性约束；不等同于策略偏好或奖励塑形 |
| T02 | 无冲突自回归联合动作 | conflict-free autoregressive joint action | \(\mathbf a_t=(a_t^1,\ldots,a_t^n)\) | 按固定单元顺序逐项采样，每项使用由前序动作更新后的掩码 | “无冲突”仅指环境定义的同一步资源/目标冲突 |
| T03 | 因子化联合 PPO | factorized joint PPO | \(\pi_\theta(\mathbf a_t\mid s_t)=\prod_i\pi_\theta(a_t^i\mid s_t,a_t^{<i},\mathcal M_t^i)\) | 使用因子条件概率构成一个联合动作概率，并基于联合概率比进行 PPO 更新 | 不得写成各单元互相独立；“因子化”是条件分解，不是独立假设 |
| T04 | 配对反事实轨迹 | paired counterfactual trajectories | \((\tau^N,\tau^E)\) | 从同一冻结上下文出发，仅改变探针单元当前动作，之后按同一冻结策略继续执行的两条轨迹 | 只支持冻结策略下的局部机制识别，不等同于在线训练收益 |
| T05 | 共同随机数 | common random numbers | \(\omega\) | N/E 分支共享环境随机流和可共享的采样随机性，以降低配对差方差 | 不能消除由动作改变引起的状态与合法动作集合分叉 |
| T06 | 不交战分支 | no-engage branch | \(N\) | 探针单元在当前决策位置执行 no-op，其余动作按冻结协议生成的分支 | \(N\) 固定表示 no-op，不得在图表中反转 |
| T07 | 交战分支 | engage branch | \(E\) | 探针单元在当前决策位置执行指定合法交战动作，其余动作按冻结协议生成的分支 | \(E\) 固定表示 engage；目标按条件概率精确边缘化 |
| T08 | 探针直接成本 | probe direct cost | \(C_{\mathrm{direct}}\) | 当前步 E 相对 N 由探针单元直接增加的资源成本 | 只含当前探针单元，不含其他单元同一步变化 |
| T09 | 同一步其他单元替代 | same-step other-unit substitution | \(Sub_{\mathrm{cost,same}}\) | 当前步中，N 相对 E 的其他单元总成本差 | 属于总替代成本；不属于未来替代射击数 |
| T10 | 未来探针替代 | future probe substitution | \(Sub_{\mathrm{cost,future,probe}}\) | 当前步之后，N 相对 E 的探针单元累计成本差 | 只统计严格晚于干预步的成本 |
| T11 | 未来其他单元替代 | future other-unit substitution | \(Sub_{\mathrm{cost,future,other}}\) | 当前步之后，N 相对 E 的其他单元累计成本差 | 只统计严格晚于干预步的成本 |
| T12 | 未来替代成本 | future substitution cost | \(Sub_{\mathrm{cost,future}}\) | T10 与 T11 之和 | 不含 T09；不得简称为最终的“总替代成本” |
| T13 | 总替代成本 | total substitution cost | \(Sub_{\mathrm{cost,total}}\) | 同一步其他单元替代、未来探针替代和未来其他单元替代之和 | 是成本恒等式中的唯一替代成本主量 |
| T14 | 未来替代射击数 | future substituted shots | \(Sub_{\mathrm{shot}}\) | 当前步之后 N 分支总射击数减 E 分支总射击数 | 不含同一步后缀动作，不能与 \(Sub_{\mathrm{cost,total}}\) 互换 |
| T15 | 回合累计成本差 | episode-level cumulative cost difference | \(\Delta C_{\mathrm{episode}}\) | E 分支回合总成本减 N 分支回合总成本 | 正值表示 E 的累计成本更高；不得改写为 N−E |
| T16 | 替代比率 | substitution ratio | \(\rho_{\mathrm{sub}}\) | \(Sub_{\mathrm{cost,total}}/C_{\mathrm{direct}}\) | 仅在 \(C_{\mathrm{direct}}>0\) 时定义；不是概率 |
| T17 | 成本符号掩盖 | cost-sign masking | \(I_{\mathrm{mask}}\) | 直接成本为正但 \(\Delta C_{\mathrm{episode}}\le 0\) 的事件 | 表示累计成本符号未显露当前直接消耗，不等同于策略动作错误 |
| T18 | 可辨识性边界 | identifiability boundary | \(\mathcal B_{\mathrm{id}}\) | 给定干预、观测量和冻结策略时，可从累计结果可靠识别局部信用的条件范围 | 当前边界受场景、资源类型和标签定义约束；不得外推到任意环境 |
| T19 | 动作替代 | action substitution | — | 当前交战改变同一步后缀动作或未来动作序列，从而替代原本会发生的资源消耗 | 工作术语；W1-02 完成前不作术语优先权主张 |
| T20 | 局部资源信用 | local resource credit | \(c_{\mathrm{local}}\) | 归属于当前探针动作的资源消耗或节省，而不是整回合总成本变化 | 本阶段识别测量问题，尚未形成通过门控的在线信用分配算法 |

## 3. 代码字段映射

| 冻结符号 | R2 账本字段 | 说明 |
| --- | --- | --- |
| \(C_{\mathrm{direct}}\) | `direct_cost` | 当前探针直接成本 |
| \(Sub_{\mathrm{cost,same}}\) | `same_step_other_sub_cost` | `-current_other_delta` |
| \(Sub_{\mathrm{cost,future,probe}}\) | `future_sub_cost_probe` | 未来探针成本 N−E |
| \(Sub_{\mathrm{cost,future,other}}\) | `future_sub_cost_other` | 未来其他单元成本 N−E |
| \(Sub_{\mathrm{cost,future}}\) | `future_sub_cost` | 两类未来替代成本之和 |
| \(Sub_{\mathrm{cost,total}}\) | `sub_cost` | 论文中不得沿用过短名称 `Sub_cost` |
| \(Sub_{\mathrm{shot}}\) | `sub_shot` | 仅含未来射击 N−E |
| \(\Delta C_{\mathrm{episode}}\) | `episode_cost_delta` | 回合累计成本 E−N |
| \(\rho_{\mathrm{sub}}\) | `rho_sub` | 总替代成本与直接成本之比 |
| \(I_{\mathrm{mask}}\) | `cost_sign_masked` | 布尔事件 |

## 4. 禁止混用

- 不得把 `Sub_shot` 解释为同一步加未来的动作数。
- 不得把 R1 的 future-only `sub_cost` 直接写成 R2 的 \(Sub_{\mathrm{cost,total}}\)。
- 不得把成本符号掩盖写成“所有资源类型均发生符号反转”。
- 不得把冻结策略反事实可辨识性写成“BPCE/MCH-PPO 已稳定提升 PPO”。
- 不得把关系表示或 GNN 写成已验证的因果修复方案。
