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
- [下一研究阶段：交战概率校准与反事实分层信用分配](next_research_phase_counterfactual_credit_assignment.md)
- [下一研究阶段：非图结构掩码条件动作价值 Critic](next_research_phase_action_conditioned_q_critic.md)
- [下一研究阶段：动作差异数据与组内排序监督](next_research_phase_q_critic_ranking_refinement.md)
- [下一研究阶段：显式交战与目标分层 Q 诊断](next_research_phase_hierarchical_q_diagnostics.md)
- [下一研究阶段：风险与约束感知的交战效用诊断](next_research_phase_risk_aware_engagement_utility.md)
- [下一研究阶段：安全临界状态与类别平衡交战估值](next_research_phase_critical_state_balanced_engagement.md)
- [下一研究阶段：资源约束交战边界校准](next_research_phase_resource_constrained_engagement_calibration.md)
- [下一研究阶段：状态条件资源预算与显式约束价值](next_research_phase_state_conditioned_constrained_value.md)
- [下一研究阶段：跨场景鲁棒预算与可靠成本差监督](next_research_phase_cross_scenario_robust_budget.md)
- [下一研究阶段：多批次临界状态语料与留一批次泛化](next_research_phase_multibatch_leave_one_out.md)

其中，任务一至任务十四的当前门控阶段均已完成。任务十二证明 all-no-op 同时包含 deterministic argmax 概率碎片化和 PPO 种子分叉；任务十三进一步否决了“统一阈值即可修复”，并确认现有 `V(s)` Critic 不能提供动作条件反事实价值。

多批次临界状态语料与留一批次泛化也已完成。三个训练批次共形成144个状态和193个上下文组，数据功效与批次独立性门槛全部通过；但选中目标的留一批次可行数仅为 `1/3`，最终72状态独立批次完整通过数为 `0/3`。三种子 engage recall 达到 `0.829 / 0.943 / 0.886`，no-op recall 却降至 `0.500 / 0.477 / 0.545`，且全部九个种子-场景组合的 no-op recall 低于0.65。问题已从异质场景漏交战转为系统性过度交战。MCH-PPO、30k/100k 和 GNN 继续冻结；下一入口是不新增 rollout 的留一批次安全-停止 Pareto 可行性审计。项目最终目标仍是稳定、可扩展的防空资源分配，不是 GNN 本身。

后续每个主要研究阶段应新增独立任务文档，不直接覆盖已经完成阶段的验收记录。
