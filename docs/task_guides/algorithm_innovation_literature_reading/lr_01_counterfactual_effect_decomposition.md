# LR-01：反事实效应分解论文阅读任务

任务状态：`PASSED`  
优先级：P0  
建议用时：3–4 小时  
实验授权：否

## 1. 论文身份

标题：*Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making*  
作者：Stelios Triantafyllou、Aleksa Sukovic、Yasaman Zolfimoselo、Goran Radanovic  
来源：ICML 2025，PMLR 267  
官方页面：<https://proceedings.mlr.press/v267/triantafyllou25a.html>  
本地 PDF：[2025 ICML Counterfactual Effect Decomposition](../../../research_papers/02_innovation_references/2025_ICML_Counterfactual_Effect_Decomposition.pdf)

## 2. 选择理由

项目 R1/R2 已经发现：当前交战会经同一步后缀单元和未来策略动作改变累计资源
成本。该论文直接研究一个动作的总反事实效应如何经：

1. 后续智能体行为；
2. 环境状态变量和状态转移

两类路径传播。它是项目“动作替代测量模块”最接近的现有工作之一。

## 3. 核心阅读问题

1. 论文如何定义 total counterfactual effect？
2. agent-specific effect 与 state-specific effect 的干预语义分别是什么？
3. 为什么普通结果差不能直接解释为某个动作的责任？
4. structure-preserving intervention 保留了什么，又改变了什么？
5. Shapley 分解解决的是解释、问责还是策略优化？
6. 论文的识别假设在 AirDefense v1 完全状态模拟器中是否满足？
7. R2 的三分量成本账本是否只是该方法的领域实例，还是仍有不同问题层？

## 4. 必读部分

- Abstract、Introduction；
- 问题设置与结构因果模型；
- 总效应到后续智能体路径/状态路径的分解；
- agent-specific effect；
- structure-preserving intervention；
- 实验中怎样验证解释忠实性；
- limitations 和 appendix 中与识别假设相关的证明。

## 5. 必须重建的公式与图

阅读报告必须自行重写：

1. total counterfactual effect；
2. 后续智能体行为效应与状态路径效应的加和关系；
3. agent-specific effect 的定义；
4. Shapley 归因如何作用于路径效应。

另外绘制一张项目对应图：

```text
当前被测单元动作
├─ 同一步后缀单元动作变化
├─ 未来被测单元动作变化
├─ 未来其他单元动作变化
└─ 环境状态与命中路径变化
```

标注哪些分支已被 R2 账本直接测量，哪些仍未识别。

## 6. 项目压力测试

至少对照：

- [动作替代独立确认](../../experiments/air_defense_v1_action_substitution_confirmation.md)；
- [N1 可辨识资源信用审计](../../experiments/air_defense_v1_n1_offline_semantic_audit.md)。

必须回答：

| 检查项                      | 需要给出的判定       |
| ------------------------ | ------------- |
| R2 是否重复该论文的总效应分解         | 是/否/部分，并给公式理由 |
| R2 的同一步后缀替代是否被论文显式覆盖     | 是/否/依赖建模      |
| 论文效应能否直接作为 PPO advantage | 能/不能/需新增条件    |
| 测量解释与规范性优化是否被论文区分        | 如何区分          |
| 项目剩余差异                   | 不超过三条，且必须可验证  |

## 7. 交付物

建议生成：

```text
docs/literature/algorithm_innovation_reading/lr_01_counterfactual_effect_decomposition.md
```

报告至少包含：

- 一页公式卡；
- 一张路径效应图；
- R2/N1 对照矩阵；
- 已被覆盖的项目主张；
- 不可直接迁移点；
- `BASELINE / ADAPT / AVOID / OPEN` 判决。

## 8. 通过条件

- 能准确解释“后续行为路径”与“状态路径”的区别；
- 不把解释性归因自动写成优化目标；
- 明确项目测量贡献相对该论文的可辩护边界；
- 若差异不足，明确建议收窄主张，而非修改措辞掩盖重合。

## 9. 禁止结论

- 不因任务领域不同就声称方法创新；
- 不把账本分量数不同作为核心差异；
- 不把 Shapley 或因果术语直接接入 PPO；
- 不在未完成公式对照前声称“项目首次发现动作替代”。

## 10. 移交

本任务结果移交 LR-05，用于判断 CAPO/COSAC 的顺序信用与因果效应分解之间的
边界。

## 11. 执行结果

完成时间：2026-07-29。  
阅读报告：[LR-01 反事实效应分解与项目边界](../../literature/algorithm_innovation_reading/lr_01_counterfactual_effect_decomposition.md)

验收结果：

- [x] 重建 TCFE、tot-ASE、r-SSE、ASE 与 ASE-SV 公式；
- [x] 明确正确恒等式为 `TCFE = tot-ASE - r-SSE`；
- [x] 完成 structure-preserving intervention 与识别假设审计；
- [x] 绘制 AirDefense 路径效应图；
- [x] 完成 R2/N1 公式级对照；
- [x] 给出 `BASELINE / ADAPT / AVOID / OPEN` 判决；
- [x] 形成 LR-05 移交边界；
- [x] 未修改实验、reward、loss、mask、PPO、FCRC 或 GNN。

