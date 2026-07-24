# W1-02 稿件定位决策

更新时间：2026-07-24  
门禁结论：**L2**  
决策状态：冻结，移交 W1-03/W1-07/W1-08

## 1. 唯一判定

> **L2：当前成果定位为较大方法论文中的测量、诊断与资源信用分解模块，不作为已经完成的独立通用算法论文。**

不选择 L1：反事实信用、后续动作介导效应和顺序团队信用均已有直接先例。  
不选择 L3：当前差异不只是 AirDefense 场景；动态掩码同一步后缀、三分量资源成本账本、
逐行恒等式和条件化符号边界属于 Method 与 Insight 层差异。  
不选择 L4：关键原始来源可取得，检索式、阅读记录和差异判断均可复核。

## 2. 判定依据

### 2.1 已被既有工作覆盖

- 全局/回合结果不能直接代表个体或当前动作的局部贡献；
- difference reward、COMA 和反事实基线；
- 后续动作与外部因素会混入当前动作的未来回报信用；
- 一个动作的总效应可经其他智能体行为和状态转移传播；
- 固定顺序协作团队中的逐智能体反事实信用；
- invalid action masking、自回归 MARL 和约束策略优化本身。

### 2.2 当前仍可保留

- 动态合法动作掩码下，当前单元动作改变同一步后缀单元可行动作的具体测量问题；
- 资源成本的三分量操作账本：
  \(Sub_{\mathrm{cost,same}}\)、
  \(Sub_{\mathrm{cost,future,probe}}\) 和
  \(Sub_{\mathrm{cost,future,other}}\)；
- 逐行验证
  \(\Delta C_{\mathrm{episode}}=C_{\mathrm{direct}}-Sub_{\mathrm{cost,total}}\)；
- 用新策略种子独立确认结构性替代，并保留场景和资源类型的符号边界；
- 明确区分“CRN 降低方差”和“账本定义估计对象”。

## 3. 三项贡献的处理

### 贡献 1：收窄后保留

**修订表述：**

> 将既有反事实/介导效应信用问题操作化到动态掩码的自回归资源分配：回合累计成本
> 同时包含当前直接消耗和后续动作介导的资源变化，因此不应直接解释为局部资源信用。

删除“发现一种全新的信用问题”“首次揭示后续动作污染”等表述。

### 贡献 2：保留为核心方法模块

**修订表述：**

> 给出冻结策略 N/E 配对、共同随机数和逐时刻成本账本结合的审计方法，将成本差分解为
> 同一步后缀、未来探针和未来其他单元三类替代成本，并逐行验证代数恒等式。

该贡献是特定测量对象的精确操作化，不宣称替代一般因果效应分解、COMA 或 CCA。

### 贡献 3：保留并强化边界

**修订表述：**

> 在与旧正式数据零重叠的新策略种子上复现动作介导的成本替代，同时证明成本符号掩盖
> 并不跨资源类型普遍成立，从而给出估计量的经验适用边界。

资源类型失败不是需要隐藏的负结果，而是可辨识性结论的一部分。

## 4. 优先权措辞

### 允许

- “我们将已知的反事实信用问题具体化到动态掩码序列资源分配”；
- “我们给出一个面向资源成本的操作账本”；
- “在所检索文献中，未发现同时覆盖以下五项条件的工作……”；
- “结果表明”“独立确认”“条件化成立”“在冻结策略评估范围内”；
- “与一般因果效应分解互补”。

### 禁止

- “首次发现动作替代”；
- “首次解决后续动作污染”；
- “提出通用反事实信用分配算法”；
- “证明回合回报/成本普遍有偏”；
- “适用于任意 MARL、任意资源类型或任意动态掩码环境”；
- “MCH-PPO/BPCE 已稳定提升 PPO”；
- “GNN 已解决该问题”。

## 5. 术语变更建议

W1-01 的 `action substitution` 保留为历史内部术语，本文件不直接修改冻结账本。
向后续术语评审提交以下正文候选：

| 层级 | 中文 | English |
| --- | --- | --- |
| 总称 | 后续动作介导的成本替代 | downstream action-mediated cost substitution |
| 同一步 | 同一步后缀替代 | same-step suffix substitution |
| 未来 | 未来策略介导替代 | future policy-mediated substitution |

理由：

1. `action substitution` 在安全 RL 中常表示控制器替换执行动作，存在语义碰撞；
2. `mediated` 与最近因果文献中“效应经后续智能体行为传播”的表述兼容；
3. 新术语不会把一般因果分解据为项目独有，只限定成本测量对象。

正式采用前，需按 W1-01 流程更新术语账本和冲突日志。

## 6. 稿件形态

### 当前推荐

一篇较大方法论文中的核心机制章节，暂定结构：

1. 动态掩码序列资源分配的局部成本估计问题；
2. 三分量配对反事实成本账本；
3. 结构性混合与 CRN 方差缩减的区分；
4. 跨策略种子的机制确认；
5. 场景和资源类型边界；
6. 在此诊断基础上另行设计并门控在线信用算法。

### 当前不推荐

- 只以 AirDefense 案例包装成“新 PPO”；
- 以失败的 BPCE/MCH-PPO 性能作为方法贡献；
- 在没有在线算法门控或更广环境验证时按通用 MARL 算法论文投稿。

L2 不触发新实验。W1 阶段继续完成论证、图表和写作；任何算法扩展必须另立任务。

## 7. 故事压缩

### 一句话

动态掩码序列资源分配中的回合成本会混合当前直接消耗与后续动作介导的成本替代，
三分量配对反事实账本能够分离这种混合并揭示其场景和资源类型边界。

### Problem–Method–Insight 三句话

**Problem：** 回合累计资源成本不是动态掩码自回归动作的纯局部成本读出。  
**Method：** 我们以冻结策略 N/E 配对、CRN 和逐时刻账本分离同一步后缀及两类未来替代。  
**Insight：** 增加 rollout 只能降低方差，符号是否被掩盖取决于直接成本与动作介导替代的相对强度。

### 120 词以内英文版本

In dynamically masked autoregressive resource allocation, an episode-level cost difference
mixes the focal action's direct consumption with cost changes mediated by same-step suffix
actions and future policy responses. We operationalize this known counterfactual-credit
problem with paired no-engage/engage rollouts, common random numbers, and an exact
three-component resource-cost ledger. Across independently trained policy seeds, the ledger
reconstructs every audited cost difference while showing that cost-sign masking is conditional
on scenario and resource type. The result is a measurement and diagnostic component for a
broader credit-assignment method, rather than a claim of a new general-purpose PPO algorithm.

## 8. 移交决定

- W1-03：使用 L2 组织 Problem–Method–Evidence–Boundary 论证；
- W1-07/W1-08：按“已知问题、操作化差异、边界证据”组织相关工作；
- 保留 F1-F3 支持与 F4-F6 否决；
- 不恢复已被否决的通用机会成本 oracle 或在线性能主张；
- 不因 L2 追加实验。

