# N2 未来可覆盖性责任证书创新性审查

更新时间：2026-07-29。  
判决：**有条件通过 N2-P5，可进入预测性证伪；尚不能称为已成立算法创新。**

## 1. 检索问题

N2 检索的不是一般“资源约束 RL”，而是以下公式是否已有直接等价工作：

> 对一个当前合法的自回归资源—目标前缀，排除其当前目标后，比较资源消耗
> 前后对其余带截止时间威胁的最大加权匹配值，并把差值作为局部责任证书。

检索覆盖：

- 累计约束和安全 MARL；
- 当前动作硬约束与自回归可行分配；
- reachability、dead-end avoidance 和安全 shield；
- 动态动作掩码；
- 动态/多阶段武器—目标分配；
- 分层射手—目标优先级；
- 机会价值和未来资源保留。

学术检索 MCP 本次不可用，实际使用 PMLR、NeurIPS、AAAI、期刊官网和 arXiv
等一手页面。检索截止为 2026-07-29；中文数据库仍需投稿前人工补查。

## 2. 最近工作矩阵

| 工作 | 已解决问题 | 与 FCRC 重叠 | 冻结差异 |
| --- | --- | --- | --- |
| [CPO](https://proceedings.mlr.press/v70/achiam17a.html) | 期望累计成本约束 | 全局资源预算 | 不产生合法前缀的任务外部性证书 |
| [MAFOCOPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7b64c47dcb067efd6be5eee854c14835-Abstract-Conference.html) | 多智能体安全约束更新 | 安全 MARL | 不计算剩余威胁匹配可行域 |
| [Scal-MAPPO-L](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fa76985f05e0a25c66528308dda33de0-Abstract-Conference.html) | 局部交互下的可扩展安全 MARL | 顺序局部策略更新、全局约束 | 局部目标是截断 advantage，不是资源机会损失 |
| [PASPO](https://arxiv.org/abs/2409.18735) | 在凸多面体内自回归生成硬约束分配 | 自回归、当前合法性 | 保证当前动作位于可行域，不比较合法动作的未来覆盖损失 |
| [Action-constrained RL via Frank–Wolfe](https://proceedings.mlr.press/v161/lin21b.html) | 动作约束和零梯度问题 | 当前可行动作 | 无时间窗匹配证书 |
| [Reachability CRL](https://proceedings.mlr.press/v162/yu22d.html) | 最大持续安全可行集 | 未来可行域 | 学习一般安全价值；FCRC 精确求解任务匹配外部性 |
| [Dead-end avoidance](https://arxiv.org/abs/2306.13944) | 避免不可恢复状态 | 当前动作导致未来失败 | 使用安全 critic/recovery policy，不给出资源—威胁责任差 |
| [Pure-Past Action Masking](https://ojs.aaai.org/index.php/AAAI/article/view/30163) | 非马尔可夫历史约束 | 动作掩码 | 约束来自历史规范，不衡量剩余覆盖机会 |
| [Shields for Safe RL](https://doi.org/10.1145/3715958) | 用形式模型过滤未来不安全动作 | 前瞻式合法性 | 二元安全判定；FCRC 是目标排除后的连续外部性 |
| [State-augmented multi-agent assignment](https://proceedings.mlr.press/v242/agorio24a.html) | 用对偶变量协调冲突任务 | 受约束多任务分配 | 对偶状态协调，不是匹配可行域的动作责任 |
| [Agent-priority WTA, 2026](https://doi.org/10.2514/1.I011676) | 学习射手选择和目标选择顺序 | 异质时间窗、自回归层级选择 | 顺序网络是方法核心；FCRC 对冻结顺序输出证书 |
| [Multi-stage DWTA, 2025](https://doi.org/10.13976/j.cnki.xk.2025.3302) | 多波次和阶段间资源协调 | 未来资源保留、动态掩码 | 以 pointer Actor 优化分配，没有公开等价的目标排除责任差 |
| [Counterfactual Effect Decomposition, 2025](https://proceedings.mlr.press/v267/triantafyllou25a.html) | 分解动作经后续智能体与状态的因果效应 | 局部责任 | FCRC 不分解实现回报、不干预后续政策 |

2026 年的新工作已经直接覆盖“射手优先级 + 目标优先级”的层级 WTA，因此
N2 不能把自回归顺序或层级选择本身作为创新。多阶段动态 WTA 也已覆盖一般
“为未来波次保留资源”的叙事。

## 3. 五层差异判定

| 层 | 判定 | 说明 |
| --- | --- | --- |
| Problem | 强差异 | 区分“当前合法”与“对其他任务保持未来可覆盖” |
| Method | 中等差异 | 目标排除后的两次精确匹配差尚未发现公式等价工作，但与匹配机会成本和 shield 相邻 |
| 技术细节 | 强差异 | 异质射击机会、TTI、冷却、线性位置外推和自回归合法前缀共同进入证书 |
| Evidence | 暂弱 | 当前只有人工轨迹和冻结 R2 静态非退化证据，没有预测性或性能证据 |
| Insight | 强差异 | 把局部责任定义为未来任务可行域损失，而非回报归属 |

Problem、技术细节和 Insight 三层形成可辩护差异，Method 层仍需通过最近
匹配/机会价值工作的进一步压力测试。因此 N2-P5 只作“进入证伪实验”的
有条件通过，不作优先权声明。

## 4. 伪创新审查

| 风险 | 审稿人攻击 | 当前处理 |
| --- | --- | --- |
| 只是 Hungarian/WTA 分数 | 为什么不直接求一次最优匹配？ | FCRC 比较同一前缀动作前后、排除当前目标后的剩余任务外部性；后续必须与原始匹配分数消融 |
| 只是安全 shield | reachability/shield 已考虑未来后果 | 当前只称责任证书；若作为 shield，必须证明连续外部性比二元可行性有额外预测价值 |
| 只是 reward shaping | 把匹配损失加到 reward 即可 | N2 禁止写入 reward；先验证预测性和约束语义 |
| 完全状态假设过强 | 实际 TTI、速度和威胁不确定 | 当前边界明确为 AirDefense v1 完全状态静态证书；后续需不确定性版本 |
| 一次覆盖过于乐观 | 击毁可能需要多次射击 | 当前仅是 one-attempt coverability，不称安全保证 |
| 未出现未来波次 | 只看当前 alive targets | 当前不声称跨波次 reserve；后续若扩展必须改变任务建模并重新查新 |

## 5. 可证伪命题

### N2-H1：非退化责任

同一单元面对多个当前合法目标时，FCRC 应在足够多状态中产生不同责任值。
冻结开发门槛为至少 30/108 个状态出现跨度，结果为 34/108。

### N2-H2：不是成本或威胁换名

FCRC 与单元直接成本、目标损伤权重的绝对 Spearman 相关均应小于 0.90。
结果分别为 0.466 和 0.128。

### N2-H3：预测未来脆弱性

在新冻结的 paired continuation 中，高 FCRC 动作相对同状态低 FCRC
动作应导致更低的其他威胁未来覆盖率或更高的条件损伤。该命题尚未测试；
失败将把 FCRC 降为静态解释指标。

## 6. 一句话、三句话与段落版本

一句话：

> 当前合法不等于未来负责；FCRC 用当前分配对其余限时威胁可覆盖能力的
> 额外损失定义消耗型资源的局部责任。

三句话：

1. 动态掩码只排除当前非法动作，全局成本约束又不能定位哪个合法动作消耗了
   不可替代的未来机会。
2. FCRC 排除当前目标，精确比较执行该动作前后其余威胁的最大加权可覆盖值。
3. 这把责任从事后回报归属改写为可验证的未来任务可行域损失。

当前证据段落：

> 在冻结 R2 的 108 个开发状态和 243 个合法前缀动作中，35.39% 的动作
> 具有正覆盖外部性，34 个状态出现同一单元不同目标的责任差异；该信号与
> 单元成本和目标损伤权重均未高度相关，平均计算时间为 1.02 ms/context。
> 这些结果支持非退化和可计算性，但尚不证明其能预测未来损伤或改善策略。

## 7. 最终判决

- N2-P5：有条件通过；
- 阶段出口：N2-E1；
- 获准动作：创建一次冻结的 paired predictive validation；
- 未获准动作：在线训练、reward shaping、GNN 扩展和性能创新声明。

