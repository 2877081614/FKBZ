# 下一研究阶段任务：未来可覆盖性责任证书与主线重定义

任务编号：N2  
更新时间：2026-07-29  
任务状态：已完成，出口为 N2-E1  
前置任务：N1 已完成，出口为 N1-E4  
在线训练授权：否

## 1. 任务定位

N1 已证明：

- 回合累计成本差可以忠实表示动作替换的全局政策后果；
- 该标量不是当前动作的唯一局部资源信用；
- 将直接成本、替代成本或全局 CMDP 简单组合，不能形成有充分差异的新算法。

N2 不再尝试“分解哪部分回报属于当前动作”，而是重新定义局部责任对象：

> 一个当前合法的资源—目标分配，在完成当前任务之外，额外破坏了多少对其他
> 活跃威胁的未来可覆盖能力？

该责任对象是动作对未来可行域的结构性外部性，不依赖随机 continuation
回报，也不要求把全局成本强行拆成局部奖励。

## 2. 三层创新命题

| 层 | N2 冻结表述 |
| --- | --- |
| Problem | 当前动态动作掩码只保证“此刻合法”，全局累计约束只保证“平均预算”；二者都不指出某个合法前缀是否消耗了其他威胁不可替代的未来覆盖能力 |
| Method | 构造前缀条件的未来可覆盖性责任证书，对每个合法资源—目标动作计算其对“其余威胁最大加权可覆盖值”的额外损失 |
| Insight | 对消耗型异质资源，局部责任可以定义为可验证的未来可行域损失，而不是实现回报的事后分解 |

候选名称：

```text
Future-Coverability Responsibility Certificate, FCRC
未来可覆盖性责任证书
```

N2 只审查 FCRC 能否成为后续算法机制，不预先声称其为已成立创新。

## 3. 核心定义

### 3.1 威胁与射击机会

在状态 \(s\) 中，活跃威胁集合为 \(\mathcal{T}(s)\)。每个威胁 \(j\)
具有截止时间 \(d_j=\lceil \mathrm{TTI}_j\rceil\) 和损伤权重：

\[
w_j = \mathrm{payload}_j
\cdot \mathrm{threat}_j
\cdot \mathrm{zone\_value}_j.
\]

每个防御单元 \(i\) 根据剩余弹药、当前冷却和射后冷却产生未来射击机会
\(\mathcal{K}_i(s)\)。机会 \(k\) 的时间为：

\[
\tau_{i,k}
= \mathrm{cooldown}_i
+ k\max(1,\mathrm{cooldownAfterFire}_i).
\]

若 \(\tau_{i,k}<d_j\)，且线性外推后的目标在单元射程内，则机会
\((i,k)\) 可以覆盖威胁 \(j\)。边权为：

\[
q_{i,k,j}=w_j p_{i,j}(\tau_{i,k}),
\]

其中 \(p_{i,j}\) 使用环境冻结的距离—命中概率公式。

### 3.2 最大加权可覆盖值

令 \(\mathcal{U}\subseteq\mathcal{T}(s)\) 为需要覆盖的威胁集合。
每个射击机会最多匹配一个威胁，每个威胁最多接受一次覆盖：

\[
\mathcal{V}(s,\mathcal{U})
=\max_{x}
\sum_{(i,k),j} q_{i,k,j}x_{i,k,j}.
\]

该量是“一次有效尝试”的结构证书，不等同于真实击毁率，也不替代环境回报。

### 3.3 当前动作的责任外部性

对当前合法动作 \(a=(i\rightarrow j)\)，先从比较集合中排除其当前目标
\(j\)，再比较消耗该动作前后对其他威胁的可覆盖值：

\[
\mathcal{E}_{\mathrm{FCRC}}(s,i,j)
=\max\left[
0,\,
\mathcal{V}(s,\mathcal{T}\setminus\{j\})
-\mathcal{V}(s\ominus(i\rightarrow j),\mathcal{T}\setminus\{j\})
\right].
\]

\(s\ominus(i\rightarrow j)\) 只执行确定性的资源占用：

- 单元 \(i\) 弹药减一；
- 其下一可用时间按射后冷却更新；
- 不采样命中结果；
- 不执行后续策略；
- 不改变其他单元动作。

因此，该外部性不复用 N1 被否决的 continuation 差异回报。

## 4. 与相邻路线的冻结差异

| 相邻路线 | 解决对象 | N2 必须保留的差异 |
| --- | --- | --- |
| CPO/安全 MARL | 累计成本或安全约束 | FCRC 是单个合法前缀对其余任务可行域的结构外部性 |
| PASPO/动作约束 RL | 当前动作位于可行多面体或合法集合 | FCRC 比较当前合法动作之间的未来可覆盖损失 |
| Reachability CRL/安全 shield | 是否能持续避开不安全状态 | FCRC 的可行对象是带 TTI 的异质资源—威胁匹配，不是一般状态危险集 |
| WTA/匹配优化 | 求一个高质量资源—目标分配 | FCRC 输出每个自回归前缀动作的可解释责任证书，供 RL 约束或消融使用 |
| 因果效应/差异回报 | 实现回报属于哪个动作 | FCRC 不分解实现回报，不采样后续政策 |
| 分层优先级 WTA | 学习射手顺序和目标顺序 | FCRC 不把顺序网络本身作为创新，证书必须对任意冻结顺序可计算 |

## 5. 本任务范围

N2 包含：

1. 系统检索未来可行域、动作约束、reachability、安全 shield、动态 WTA
   和自回归约束分配；
2. 实现纯函数形式的可覆盖值和责任证书；
3. 构造人工匹配轨迹，验证方向、单调性和替代资源语义；
4. 在冻结 R2 的 108 个开发 context 上重新生成快照并执行静态审计；
5. 比较 FCRC 与直接成本、目标威胁、合法目标数和 N1 替代成本；
6. 给出后续预测性验证是否值得执行的 go/no-go 判决。

N2 不包含：

- PPO、MCH-PPO、BPCE-PPO 或 GNN 在线训练；
- 新随机种子或正式独立状态；
- 通过调整权重改善现有模型结果；
- 把静态证书直接写入 reward；
- 30k/100k 正式实验。

## 6. 实施任务

### N2-01：任务与术语冻结

交付：

- 本任务指导文件；
- FCRC 标签字典；
- 与 N1 成本信用的禁止混用规则。

### N2-02：系统查新

至少覆盖：

- action-constrained RL 与动态 action masking；
- constrained/safe MARL；
- reachability、dead-end avoidance 与 shields；
- 自回归约束分配；
- 动态/多阶段 WTA 与资源机会价值；
- 分层射手—目标选择。

### N2-03：证书实现

软件接口至少包含：

```text
ThreatDemand
ShotOpportunity
maximum_weight_coverability(...)
future_coverability_externality(...)
```

禁止接口返回“新 reward”或直接修改 Actor。

### N2-04：人工轨迹

必须覆盖：

1. 单威胁时外部性为零；
2. 完全可替代资源时外部性为零；
3. 灵活单元抢占专业单元可覆盖目标时外部性为正；
4. 冷却延迟使临近截止威胁失去覆盖；
5. 增加弹药或增加可覆盖边不能降低最大可覆盖值；
6. 非法权重、时间和容量被拒绝。

### N2-05：冻结 R2 静态审计

只允许重建已经冻结的 R2 context，不采集新选择性状态。输出每个 context
中所有当前合法动作的：

- FCRC 外部性；
- 单元成本；
- 目标损伤权重；
- 命中概率；
- 合法目标数；
- 同 context 的外部性跨度；
- 与 N1 总替代量的开发性关联。

### N2-06：阶段判决

形成最近工作差异矩阵、伪创新审查、FCRC 适用边界和下一阶段入口。

## 7. 预冻结命题与门槛

| 命题 | 支持门槛 | 否决门槛 |
| --- | --- | --- |
| N2-P1：形式与实现一致 | 全部人工轨迹、单调性和恒等测试通过 | 任一方向或单调性失败 |
| N2-P2：信号非退化 | 至少 30/108 context 的合法动作外部性跨度大于 `1e-9`，且至少 15% 合法动作外部性为正 | 任一数量门槛未达到 |
| N2-P3：不是成本/威胁换名 | 与单元成本、目标损伤权重的绝对 Spearman 相关均小于 `0.90`，且至少 20 个 context 存在同一单元不同目标的外部性差异 | 任一门槛失败 |
| N2-P4：静态计算可用 | 108 context 平均审计时间不超过每 context `5 ms`，最大不超过 `25 ms` | 任一时间门槛失败 |
| N2-P5：创新距离可辩护 | 相对最近工作在 Problem、Method、Insight 中至少两层为强差异，且不存在公式等价工作 | 仅为 WTA 匹配分数、shield 或 PASPO 换名 |

所有门槛均为开发性门槛，只决定是否值得创建一次新的预测性验证任务。

## 8. 阶段出口

| 出口 | 条件 | 下一阶段 |
| --- | --- | --- |
| N2-E1：候选可进入预测验证 | N2-P1 至 P5 全部通过 | 新建冻结 paired-rollout 预测性验证任务，仍不直接训练 |
| N2-E2：诊断组件 | P1-P4 通过但 P5 不足 | 保留为解释/筛选工具，不独立命名算法 |
| N2-E3：信号否决 | P1-P4 任一失败 | 停止 FCRC，不追加新状态修补 |
| N2-E4：主线再次重定义 | 静态信号成立但无法连接可证伪性能机制 | 转向环境/任务建模创新 |

## 9. 预期产物

```text
docs/task_guides/next_research_phase_future_coverability_certificate.md
docs/literature/n2_future_coverability_novelty_review.md
docs/algorithms/future_coverability_responsibility_certificate.md
docs/experiments/air_defense_v1_n2_static_coverability_audit.md

rein_learning/common/future_coverability.py
tests/test_future_coverability.py
scripts/analyze_air_defense_v1_n2_static_coverability.py

configs/air_defense_v1/n2_stage_gate.json
results/air_defense_v1/n2_static_coverability_audit/
```

若 N2-P5 失败，算法文档必须改写为“候选/诊断组件”，不得声称 FCRC
已经成为论文创新。

## 10. 完成标准

- [x] 任务范围和禁止项冻结；
- [x] 至少六类相邻路线完成查新；
- [x] 公式、实现和人工轨迹一致；
- [x] R2 context 身份与原清单一致；
- [x] N2-P1 至 P5 均有机器可读判决；
- [x] 无新增反事实 rollout 和在线训练；
- [x] 项目进度、路线图和文档索引同步；
- [x] 输出明确的 N2-E1。

## 11. 执行结果

更新时间：2026-07-29。  
阶段出口：**N2-E1：进入冻结 paired predictive validation**。  
在线训练授权：否。

### 11.1 已完成

- 完成六类以上相邻路线的系统查新与五层差异审查；
- 实现精确目标子集动态规划和 FCRC 责任外部性；
- 完成 9 项人工轨迹、单调性和异常输入测试；
- 重放冻结 R2 的 108 个 context，身份 108/108 匹配；
- 审计 243 个原自回归前缀下的合法目标动作；
- 生成机器可读阶段门控。

### 11.2 核心结果

| 指标 | 结果 | 门槛 | 判定 |
| --- | ---: | ---: | --- |
| 正外部性动作率 | 35.39% | ≥15% | 通过 |
| 有同单元目标跨度的 context | 34/108 | ≥30 | 通过 |
| FCRC—单元成本 Spearman | 0.466 | \|ρ\|<0.90 | 通过 |
| FCRC—目标权重 Spearman | −0.128 | \|ρ\|<0.90 | 通过 |
| 平均耗时 | 1.02 ms/context | ≤5 ms | 通过 |
| 最大耗时 | 5.47 ms/context | ≤25 ms | 通过 |

N2-P1 至 P4 全部通过。系统查新未发现与“目标排除后的剩余威胁精确匹配
外部性”公式等价的工作；Problem 和 Insight 层差异较强，Method 层与
WTA 机会价值、reachability 和 shield 相邻。因此 N2-P5 仅作进入预测性
证伪的有条件通过。

### 11.3 边界

静态结果说明 FCRC 非退化、不是成本/威胁换名且计算可用，但尚未证明：

- 能预测其他威胁的未来覆盖下降；
- 能预测条件损伤；
- 加入策略更新后改善性能；
- 对未知目标波次或观测不确定性有效。

因此，不得把 N2-E1 解释为在线算法已经成立。

### 11.4 下一入口

下一项任务必须是一次冻结的成对预测性验证：

```text
同状态、同单元
    高FCRC合法目标 vs 低FCRC合法目标
                    ↓
共同随机数 continuation
                    ↓
其他威胁覆盖率 / 条件损伤
                    ↓
与成本、目标权重、原始匹配分数、二元shield、N1替代量比较增量预测价值
```

该验证通过前，FCRC 不进入 reward、loss、action mask 或 GNN。
