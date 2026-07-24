# W1-02 系统文献检索协议

更新时间：2026-07-24  
状态：已执行  
用途：核验“动作替代成本可辨识性”工作的既有覆盖范围，并给出唯一论文定位

## 1. 研究问题

本轮检索不以堆积参考文献为目标，而回答以下问题：

1. 回合累计回报或成本混合当前动作直接效应、后续动作响应和环境随机性的现象是否已被研究；
2. 既有反事实信用分配是否已经分离同一步后缀动作与未来动作介导效应；
3. 动态合法动作掩码和自回归联合分配是否已有等价的资源成本账本；
4. `action substitution` 是否是该机制的规范术语；
5. 当前成果应定位为独立论文、较大方法论文组成部分，还是学位论文技术章节。

## 2. 检索范围

| 项目 | 冻结设置 |
| --- | --- |
| 检索日期 | 2026-07-24 |
| 时间范围 | 1990-01-01 至 2026-07-24 |
| 语言 | 英文为主；中文仅用于项目内部术语对照 |
| 文献类型 | 同行评议会议/期刊论文、正式论文集、与当前前沿直接相关的预印本 |
| 数据源 | OpenAlex；PMLR、AAAI、NeurIPS、ICLR/OpenReview、Springer、INFORMS 等原始出版页面；arXiv 原始记录 |
| 排序 | OpenAlex `relevance_score`；补充检索按精确标题、术语和引用链 |
| 每个主题初筛 | OpenAlex 前 20 条；精确术语和最近工作补充检索前 10 条高相关结果 |

## 3. 九组正式检索式

以下英文检索式保持原样，便于复核：

```text
Q1 multi-agent credit assignment global reward local contribution
Q2 counterfactual baseline difference rewards multi-agent policy gradient
Q3 temporal credit assignment delayed action effects return decomposition reinforcement learning
Q4 common random numbers paired simulation counterfactual policy evaluation
Q5 action substitution reinforcement learning cost credit assignment
Q6 sequential autoregressive joint action allocation multi-agent reinforcement learning
Q7 invalid action masking counterfactual evaluation policy gradient
Q8 resource shadow price opportunity cost constrained multi-agent reinforcement learning
Q9 episode return local action credit measurement bias identifiability reinforcement learning
```

OpenAlex 的可复现命令模板为：

```powershell
python -X utf8 C:\Users\Admin\.agents\skills\nature-academic-search\scripts\academic_search.py `
  "<QUERY>" --limit 20 --year-from 1990 --sort relevance_score --compact
```

## 4. 补充检索

为弥补宽检索对特定术语和最新论文召回不足，执行以下精确检索：

```text
"action substitution" reinforcement learning cost
"action displacement" reinforcement learning credit assignment
"policy-induced substitution" reinforcement learning
"downstream action substitution" multi-agent reinforcement learning
"difference rewards policy gradients"
"counterfactual effect decomposition" multi-agent sequential decision making
"agent-specific effects" multi-agent MDP
"counterfactual credit assignment" sequential cooperative teams
"invalid action masking" policy gradient
"common random numbers" stochastic system comparison
```

## 5. 纳入与排除规则

### 5.1 纳入规则

- 明确研究全局回报到局部动作或个体的信用归属；
- 明确讨论当前动作对未来回报、后续动作或状态路径的影响；
- 给出反事实基线、差分回报、因果效应分解或配对仿真方法；
- 研究顺序行动、自回归联合动作、合法动作掩码或约束资源分配；
- 对当前术语、测量对象、估计量偏差或可辨识性边界有直接约束；
- 最近工作必须能够取得原始论文、原始出版页或作者预印本。

### 5.2 排除规则

- “action substitution”仅指安全控制器替换危险动作、人类接管或动作重标记；
- “action displacement”仅指图结构中的位移矩阵或空间移动；
- 仅使用强化学习解决资源调度，但不讨论局部信用或测量混合；
- 仅为综述、新闻、二手解读，且不能追溯到原始论文；
- 只有环境名称相似，没有共享 Problem、Method 或 Insight；
- 重复的预印本和正式出版版本只保留信息更完整的版本，必要时记录版本演化。

## 6. 筛选和阅读流程

1. 对九组查询分别读取前 20 条题名和元数据，共形成 180 个“结果位”；
2. 按标题和摘要判断是否触及本任务的测量对象；
3. 以规范化标题、DOI 和 arXiv ID 合并重复出版版本；
4. 通过精确题名、引用链和官方会议页面补足宽检索漏召回；
5. 将 24 篇原始工作纳入证据矩阵；
6. 对最接近工作的原始摘要、方法核心段落和主要结论进行阅读；
7. 按 Problem、Method、Evidence、Insight 四层对照，而不是按环境名称判断差异；
8. 将同行评议工作与 2026 年预印本分层处理，预印本只用于前沿风险判断。

## 7. 证据等级

| 等级 | 含义 |
| --- | --- |
| A | 原始全文或原始方法/理论段落已核读 |
| B | 原始出版页、摘要和关键方法说明已核读 |
| C | 仅用于导航的元数据或摘要，不承担最近工作判定 |

最近工作比较仅使用 A/B 级来源；综述不作为优先权证据。

## 8. 局限

- OpenAlex 的相关性排序对 Q3、Q4、Q6-Q9 精度有限，因此加入精确题名和引用链检索；
- 2026 年最新预印本尚未完成同行评议，其结论不能与正式发表工作等权；
- 本轮检索能够支持“截至检索日、在所列数据源和检索式内未发现”，不能证明全世界不存在其他工作；
- 文献定位不会改变 W1-01 已冻结的实验事实和公式方向；术语变更只能以建议形式提交。

