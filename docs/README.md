# Docs Index

本文档是 `docs/` 目录索引。文档按用途分为项目管理、环境设计、算法与文献、实验结果、任务指导和汇报材料。

## Project

项目级文档，记录工程结构、环境配置和总体进度。

- [academic_project_progress.md](project/academic_project_progress.md)：学术项目总进度、阶段总结、下一步计划。
- [project_structure.md](project/project_structure.md)：代码和文档目录结构规划。
- [rl_environment_setup.md](project/rl_environment_setup.md)：`rein-learning` 环境与依赖说明。
- [research_innovation_roadmap.md](project/research_innovation_roadmap.md)：最终研究目标、两级创新假设、创新边界、GNN 定位和论文产出条件。

## Environments

环境建模、架构图和场景图。

- [air_defense_env_v0_design.md](environments/air_defense/air_defense_env_v0_design.md)：AirDefense v0 环境设计。
- [air_defense_rl_environment_model_design.md](environments/air_defense/air_defense_rl_environment_model_design.md)：基于文献的 AirDefense 环境模型设计。
- [air_defense_v1_architecture_diagram.md](environments/air_defense/air_defense_v1_architecture_diagram.md)：AirDefense v1.0 架构图。
- [air_defense_v1_scenario_diagram.md](environments/air_defense/air_defense_v1_scenario_diagram.md)：AirDefense v1.0 场景图说明。
- [air_defense_v1_scenario_seed0.png](environments/air_defense/air_defense_v1_scenario_seed0.png)：AirDefense v1.0 默认场景示意图。
- [air_defense_v1_scenario_profiles.md](environments/air_defense/air_defense_v1_scenario_profiles.md)：冻结默认基准、难度与单因素压力配置、场景校准结果。

## Algorithms

算法实现说明与相关论文阅读整理。

- [implemented_algorithms.md](algorithms/implemented_algorithms.md)：当前已实现算法说明。
- [hungarian_damage_reduction_baseline.md](algorithms/hungarian_damage_reduction_baseline.md)：AirDefense v1.0 Hungarian 一对一即时优化基线、数学定义与验收结果。
- [conflict_free_joint_action_masking.md](algorithms/conflict_free_joint_action_masking.md)：`Discrete(136)` 无冲突联合动作编码、动态掩码与软件接口。
- [autoregressive_conflict_free_policy.md](algorithms/autoregressive_conflict_free_policy.md)：自回归条件动作分布、联合概率、模型签名和筛选结论。
- [autoregressive_order_ablation.md](algorithms/autoregressive_order_ablation.md)：任务十单元顺序参数化、逐决策指标和泄漏归因规则。
- [role_conditioned_autoregressive_policy.md](algorithms/role_conditioned_autoregressive_policy.md)：任务十一共享资源-目标关系 Actor、置换性质、参数匹配与模型签名。
- [factorized_engagement_policy.md](algorithms/factorized_engagement_policy.md)：任务十二交战-目标因子化概率、分层确定性规则、固定探针和实现结论。
- [masked_counterfactual_hierarchical_ppo.md](algorithms/masked_counterfactual_hierarchical_ppo.md)：任务十三动态合法集反事实分解公式、理论边界和候选冻结判定。
- [masked_action_q_critic.md](algorithms/masked_action_q_critic.md)：任务十四非图掩码条件 Q-Critic、共同随机数标签、排序诊断和当前实现边界。
- [hierarchical_masked_q_critic.md](algorithms/hierarchical_masked_q_critic.md)：显式 Q_engage/Q_target 双头结构、层级监督和当前失败边界。
- [risk_aware_engagement_critic.md](algorithms/risk_aware_engagement_critic.md)：分量化反事实效用、下尾 CVaR、安全-资源 oracle 与稀有正类失败边界。
- [balanced_engagement_sign_critic.md](algorithms/balanced_engagement_sign_critic.md)：安全临界状态采样、类别平衡 BCE/margin 与过度交战失败边界。
- [resource_constrained_engagement_boundary.md](algorithms/resource_constrained_engagement_boundary.md)：冻结交战 logit 的全局/资源对偶边界、约束校准与跨批次失败边界。
- [state_conditioned_engagement_value.md](algorithms/state_conditioned_engagement_value.md)：安全收益/增量成本双价值、状态条件资源乘子与跨场景鲁棒性边界。
- [cross_scenario_robust_engagement_value.md](algorithms/cross_scenario_robust_engagement_value.md)：场景-类别最差块损失、可靠成本监督及批次外失败边界。
- [multibatch_engagement_value_generalization.md](algorithms/multibatch_engagement_value_generalization.md)：多独立批次语料、留一批次验证与过度交战失效边界。
- [两篇HARL论文创新点与研究借鉴报告.md](algorithms/两篇HARL论文创新点与研究借鉴报告.md)：HARL/HATRPO/HAPPO 论文创新点和研究借鉴。

## Literature

系统查新、论文差异矩阵和创新边界。

- [task13_counterfactual_credit_novelty_review.md](literature/task13_counterfactual_credit_novelty_review.md)：反事实信用、分层 PPO、动作掩码与 GNN-WTA 的公式级查新和 MCH-PPO 收窄结论。

## Experiments

实验结果、baseline 结果和对比记录。

- [air_defense_v1_baseline_results.md](experiments/air_defense_v1_baseline_results.md)：AirDefense v1.0 第一组规则 baseline 结果。
- [air_defense_v1_learning_baselines.md](experiments/air_defense_v1_learning_baselines.md)：AirDefense v1.0 PPO / Maskable PPO 训练与统一对比说明。
- [air_defense_v1_formal_benchmark_100k.md](experiments/air_defense_v1_formal_benchmark_100k.md)：AirDefense v1.0 正式 100k × 5 seeds 基准结果与分析。
- [air_defense_v1_diagnostic_metrics.md](experiments/air_defense_v1_diagnostic_metrics.md)：高威胁突防、冲突、过度分配、资源效率和决策耗时的统一定义与聚合规范。
- [air_defense_v1_cross_scenario_benchmark.md](experiments/air_defense_v1_cross_scenario_benchmark.md)：训练/测试场景解耦、泛化矩阵、配对差异和任务五 smoke run。
- [air_defense_v1_task6_screening.md](experiments/air_defense_v1_task6_screening.md)：三种子跨场景筛选实验、算法分水岭、典型失效回合与正式实验场景选择。
- [air_defense_v1_task7_formal_100k.md](experiments/air_defense_v1_task7_formal_100k.md)：核心场景 100k × 5 种子正式结果、配对统计、资源效率与联合动作冲突分析。
- [air_defense_v1_task8_screening.md](experiments/air_defense_v1_task8_screening.md)：无冲突联合动作机制的工程验收、30k × 3 种子筛选和正式实验门槛判定。
- [air_defense_v1_task9_screening.md](experiments/air_defense_v1_task9_screening.md)：自回归无冲突动作机制、资源效率恢复和异质目标优先级门槛判定。
- [air_defense_v1_task10_order_diagnostics.md](experiments/air_defense_v1_task10_order_diagnostics.md)：冻结模型诊断、三顺序 30k × 3 筛选、行为归因和 100k 门槛判定。
- [air_defense_v1_task11_role_conditioned_screening.md](experiments/air_defense_v1_task11_role_conditioned_screening.md)：角色条件关系动作头正式筛选、单元塌缩、顺序鲁棒性和优化瓶颈结论。
- [air_defense_v1_task12_noop_stability.md](experiments/air_defense_v1_task12_noop_stability.md)：no-op argmax 放大、10k 训练分叉、因子化候选和 30k 门槛判定。
- [air_defense_v1_task13_credit_diagnostics.md](experiments/air_defense_v1_task13_credit_diagnostics.md)：冻结模型阈值扫描、联合信用诊断、反事实分支和动作条件 Q-Critic 转向结论。
- [air_defense_v1_task14_q_critic.md](experiments/air_defense_v1_task14_q_critic.md)：动作条件 Q-Critic 正式门控、消融、有效比较不足和 MCH-PPO/GNN 暂缓结论。
- [air_defense_v1_task14_ranking_refinement.md](experiments/air_defense_v1_task14_ranking_refinement.md)：独立高功效测试、组内动作差异监督、排序改善和 engage/no-op 剩余瓶颈。
- [air_defense_v1_task14_hierarchical_q.md](experiments/air_defense_v1_task14_hierarchical_q.md)：108 状态显式分层 Q 正式对照、目标层收益与交战层负结果。
- [air_defense_v1_task14_engagement_utility.md](experiments/air_defense_v1_task14_engagement_utility.md)：风险/约束交战效用正式对照、类别功效与估值失败分析。
- [air_defense_v1_task14_balanced_engagement.md](experiments/air_defense_v1_task14_balanced_engagement.md)：144状态定向采样、平衡分类与 time-pressure 过度交战诊断。
- [air_defense_v1_task14_engagement_calibration.md](experiments/air_defense_v1_task14_engagement_calibration.md)：72状态独立测试、资源对偶校准与标量停止边界负结果。
- [air_defense_v1_task14_state_conditioned_value.md](experiments/air_defense_v1_task14_state_conditioned_value.md)：三折交叉拟合、72状态双价值正式实验与逐场景剩余边界。
- [air_defense_v1_task14_cross_scenario_robust_value.md](experiments/air_defense_v1_task14_cross_scenario_robust_value.md)：鲁棒损失消融、新72状态批次与异质场景漏交战诊断。
- [air_defense_v1_task14_multibatch_leave_one_out.md](experiments/air_defense_v1_task14_multibatch_leave_one_out.md)：三训练批次留一验证、最终独立批次和安全-停止边界诊断。

## Presentations

导师汇报、阶段答辩和学术汇报材料。

- [advisor_progress_report_2026-07-16.md](presentations/advisor_progress_report_2026-07-16.md)：2026-07-16 导师阶段汇报内容、讲稿和答疑准备。
- [advisor_progress_report_2026-07-16.pptx](presentations/advisor_progress_report_2026-07-16.pptx)：2026-07-16 导师阶段汇报演示文稿，共 13 页，含讲者备注。
- [advisor_progress_report_2026-07-22.md](presentations/advisor_progress_2026-07-22/advisor_progress_report_2026-07-22.md)：以 MCH-PPO 创新证据链为核心的近期成果、实验边界与后续工作报告。
- [advisor_progress_report_2026-07-22.pptx](presentations/advisor_progress_2026-07-22/advisor_progress_report_2026-07-22.pptx)：2026-07-22 导师汇报演示文稿，共15页，含实验图表与讲者备注。

## Task Guides

后续研究阶段的任务拆分、执行顺序、交付物和验收标准。

- [task_guides/README.md](task_guides/README.md)：任务指导目录说明与索引。
- [next_research_phase_difficulty_generalization.md](task_guides/next_research_phase_difficulty_generalization.md)：场景难度、Hungarian 基线、泛化诊断和图结构算法进入条件。
- [next_research_phase_conflict_free_joint_action.md](task_guides/next_research_phase_conflict_free_joint_action.md)：任务八无冲突联合动作机制、实施步骤、筛选协议和正式验收门槛。
- [next_research_phase_autoregressive_joint_action.md](task_guides/next_research_phase_autoregressive_joint_action.md)：任务九顺序式/自回归无冲突动作生成、概率契约、筛选协议和验收标准。
- [next_research_phase_order_bias_diagnostics.md](task_guides/next_research_phase_order_bias_diagnostics.md)：任务十逐决策诊断、固定单元顺序消融和 GNN 进入条件。
- [next_research_phase_role_conditioned_action_head.md](task_guides/next_research_phase_role_conditioned_action_head.md)：任务十一共享资源-目标关系动作头、单元塌缩抑制、顺序鲁棒性和筛选门槛。
- [next_research_phase_noop_optimization_stability.md](task_guides/next_research_phase_noop_optimization_stability.md)：任务十二固定策略探针、no-op 训练分叉诊断、交战-目标因子化和确认门槛。
- [next_research_phase_counterfactual_credit_assignment.md](task_guides/next_research_phase_counterfactual_credit_assignment.md)：任务十三系统查新、概率校准、信用分配诊断、反事实估值原型和 GNN 进入条件。
- [next_research_phase_action_conditioned_q_critic.md](task_guides/next_research_phase_action_conditioned_q_critic.md)：任务十四动作条件 Q 语义、反事实数据协议、离线估值指标和 MCH-PPO 恢复门槛。
- [next_research_phase_q_critic_ranking_refinement.md](task_guides/next_research_phase_q_critic_ranking_refinement.md)：任务十四修订的独立测试功效、组内动作差异监督和 MCH-PPO 二次门控。
- [next_research_phase_hierarchical_q_diagnostics.md](task_guides/next_research_phase_hierarchical_q_diagnostics.md)：显式 Q_engage/Q_target 语义、独立大样本协议和 MCH-PPO 最终离线门控。
- [next_research_phase_resource_constrained_engagement_calibration.md](task_guides/next_research_phase_resource_constrained_engagement_calibration.md)：冻结 logit 的资源约束边界、独立测试协议和 MCH-PPO 恢复门槛。
- [next_research_phase_state_conditioned_constrained_value.md](task_guides/next_research_phase_state_conditioned_constrained_value.md)：显式安全/成本价值、状态条件预算、交叉拟合和最终独立门控。
- [next_research_phase_cross_scenario_robust_budget.md](task_guides/next_research_phase_cross_scenario_robust_budget.md)：场景-类别鲁棒预算、可靠成本差监督与 MCH-PPO 前最终门控。
- [next_research_phase_multibatch_leave_one_out.md](task_guides/next_research_phase_multibatch_leave_one_out.md)：多批次临界状态语料、留一批次泛化和最终独立门控。

## Suggested Reading Order

如果是重新接手项目，建议按以下顺序阅读：

1. [academic_project_progress.md](project/academic_project_progress.md)
2. [project_structure.md](project/project_structure.md)
3. [air_defense_rl_environment_model_design.md](environments/air_defense/air_defense_rl_environment_model_design.md)
4. [air_defense_v1_architecture_diagram.md](environments/air_defense/air_defense_v1_architecture_diagram.md)
5. [air_defense_v1_baseline_results.md](experiments/air_defense_v1_baseline_results.md)
