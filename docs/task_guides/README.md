# Task Guides

本目录用于存放项目后续阶段的任务指导文档，重点记录：

- 当前阶段目标与研究问题；
- 工作任务、依赖关系和执行顺序；
- 代码、实验与文档交付物；
- 可检查、可复现的验收标准；
- 进入下一研究阶段的前置条件；
- 明确暂缓内容，控制环境与算法复杂度。

## 当前任务

- [下一研究阶段：场景难度、优化基线与泛化诊断](next_research_phase_difficulty_generalization.md)
- [下一研究阶段：无冲突联合动作机制](next_research_phase_conflict_free_joint_action.md)
- [下一研究阶段：顺序式/自回归无冲突联合动作生成](next_research_phase_autoregressive_joint_action.md)
- [下一研究阶段：单元顺序偏置与异质目标优先级诊断](next_research_phase_order_bias_diagnostics.md)
- [下一研究阶段：角色条件化关系动作头与顺序鲁棒性](next_research_phase_role_conditioned_action_head.md)
- [下一研究阶段：no-op 塌缩机理与 PPO 优化稳定性](next_research_phase_noop_optimization_stability.md)

其中，任务一至任务十二均已完成。任务十二证明 all-no-op 同时包含 deterministic argmax 概率碎片化和 PPO 种子分叉；交战-目标因子化虽降低异质场景未分配泄漏占比，但仍出现不开火与高成本开火两极分化，30k 正式筛选 19 项门槛仅通过 6 项，因此未运行 100k。下一阶段应研究交战概率校准及 Actor-Critic 优化稳定性，GNN 继续冻结。

后续每个主要研究阶段应新增独立任务文档，不直接覆盖已经完成阶段的验收记录。
