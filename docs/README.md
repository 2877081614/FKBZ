# Docs Index

本文档是 `docs/` 目录索引。文档按用途分为项目管理、环境设计、算法与文献、实验结果、任务指导和汇报材料。

## Project

项目级文档，记录工程结构、环境配置和总体进度。

- [academic_project_progress.md](project/academic_project_progress.md)：学术项目总进度、阶段总结、下一步计划。
- [project_structure.md](project/project_structure.md)：代码和文档目录结构规划。
- [rl_environment_setup.md](project/rl_environment_setup.md)：`rein-learning` 环境与依赖说明。
- [research_innovation_roadmap.md](project/research_innovation_roadmap.md)：最终研究目标、两级创新假设、创新边界、GNN 定位和论文产出条件。
- [first_innovation_claim_evidence_matrix.md](project/first_innovation_claim_evidence_matrix.md)：R2独立确认后的第一创新主张、支持/否决证据和论文表述边界。

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
- [oob_safety_stop_pareto_calibration.md](algorithms/oob_safety_stop_pareto_calibration.md)：完整 OOB 阈值集合、分组鲁棒约束、Pareto 前沿与种子尺度诊断。
- [cross_batch_uncertainty_calibration.md](algorithms/cross_batch_uncertainty_calibration.md)：批次-场景-类别平衡 Platt、预测标准误、LCB及其失效边界。
- [boundary_probed_counterfactual_engagement_ppo.md](algorithms/boundary_probed_counterfactual_engagement_ppo.md)：joint PPO安全主干、边界成对探测、严格退化语义与v0机制结论。
- [identifiable_resource_credit_candidate_matrix.md](algorithms/identifiable_resource_credit_candidate_matrix.md)：N1 四分量资源信用语义、候选 A/B/C 比较、软件契约和 N1-E4 否决边界。
- [future_coverability_responsibility_certificate.md](algorithms/future_coverability_responsibility_certificate.md)：N2 未来可覆盖性责任证书、精确匹配公式、软件接口、静态门控和使用边界。
- [两篇HARL论文创新点与研究借鉴报告.md](algorithms/两篇HARL论文创新点与研究借鉴报告.md)：HARL/HATRPO/HAPPO 论文创新点和研究借鉴。

## Literature

系统查新、论文差异矩阵和创新边界。

- [task13_counterfactual_credit_novelty_review.md](literature/task13_counterfactual_credit_novelty_review.md)：反事实信用、分层 PPO、动作掩码与 GNN-WTA 的公式级查新和 MCH-PPO 收窄结论。
- [n1_identifiable_resource_credit_search_protocol.md](literature/n1_identifiable_resource_credit_search_protocol.md)：N1 检索边界、查询族、纳入标准、去重规则和核心一手来源。
- [n1_identifiable_resource_credit_novelty_review.md](literature/n1_identifiable_resource_credit_novelty_review.md)：N1 最近工作五层差异矩阵、伪创新压力测试和 N1-P2 否决结论。
- [n2_future_coverability_novelty_review.md](literature/n2_future_coverability_novelty_review.md)：N2 相对约束 RL、reachability、shield、自回归分配和动态 WTA 的创新距离审查。
- [lr_01_counterfactual_effect_decomposition.md](literature/algorithm_innovation_reading/lr_01_counterfactual_effect_decomposition.md)：ICML 2025 反事实效应分解公式卡、R2/N1 路径对照、创新覆盖压力测试与 LR-05 移交边界。
- [lr_03_gradient_shaping_multi_constraint_safe_rl.md](literature/algorithm_innovation_reading/lr_03_gradient_shaping_multi_constraint_safe_rl.md)：L4DC 2024 GradS 多约束公式、梯度冲突矩阵、AirDefense 约束语义与 cost-critic 前置门控。
- [lr_04_paspo_constrained_allocation.md](literature/algorithm_innovation_reading/lr_04_paspo_constrained_allocation.md)：NeurIPS 2024 PASPO 可行分配公式、初始化去偏、Task 8–11 差异矩阵与离散强基线判决。
- [lr_05_capo_sequential_counterfactual_credit.md](literature/algorithm_innovation_reading/lr_05_capo_sequential_counterfactual_credit.md)：arXiv v2 COSAC/SeqAU 公式、版本变化、Gram 覆盖审计及 MCH/BPCE 创新边界。
- [lr_06_offline_to_online_critic_reconstruction.md](literature/algorithm_innovation_reading/lr_06_offline_to_online_critic_reconstruction.md)：NeurIPS 2024 OCR-CFT 两类错配、Critic 重构、策略对齐、CFT 及 AirDefense 在线接入 no-go 条件。

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
- [air_defense_v1_mch_ppo_mechanism_stress_test.md](experiments/air_defense_v1_mch_ppo_mechanism_stress_test.md)：MCH-PPO 最小在线训练实现、冻结三种子压力实验及门控失败结论。
- [air_defense_v1_rg_mch_ppo_stress_test.md](experiments/air_defense_v1_rg_mch_ppo_stress_test.md)：可靠度门控 MCH-PPO、GAE 锚定的在线正向证据与剩余种子塌缩边界。
- [air_defense_v1_sa_rg_mch_ppo_stress_test.md](experiments/air_defense_v1_sa_rg_mch_ppo_stress_test.md)：支持感知、累计KL锚点的失败实验及独立层级clipping安全退化缺陷。
- [air_defense_v1_bpce_ppo_stress_test.md](experiments/air_defense_v1_bpce_ppo_stress_test.md)：BPCE-PPO v0软件验收、等预算随机探测、10k三种子双场景结果与覆盖瓶颈。
- [air_defense_v1_bpce_label_semantics_audit.md](experiments/air_defense_v1_bpce_label_semantics_audit.md)：72上下文三标签语义对照、deterministic continuation失败与双向覆盖结论。
- [air_defense_v1_bpce_short_horizon_label_audit.md](experiments/air_defense_v1_bpce_short_horizon_label_audit.md)：TTI事件窗三态标签、异质资源STOP有效性与跨场景失败结论。
- [air_defense_v1_action_substitution_opportunity_cost_audit.md](experiments/air_defense_v1_action_substitution_opportunity_cost_audit.md)：N/E/E-R嵌套反事实、未来射击替代、单发弹药机会价值及其跨场景可辨识性边界。
- [air_defense_v1_action_substitution_confirmation.md](experiments/air_defense_v1_action_substitution_confirmation.md)：新策略种子、新状态和三场景下的动作替代独立确认、成本账本修正与资源类型边界。
- [air_defense_v1_n1_offline_semantic_audit.md](experiments/air_defense_v1_n1_offline_semantic_audit.md)：冻结 R2 四分量恒等式、标量含混率、候选门控和 N1-E4 判决。
- [air_defense_v1_n2_static_coverability_audit.md](experiments/air_defense_v1_n2_static_coverability_audit.md)：冻结 R2 的 243 个合法前缀动作、FCRC 非退化性、非冗余性和计算预算审计。
- [air_defense_v1_n3_fcrc_paired_predictive_validation.md](experiments/air_defense_v1_n3_fcrc_paired_predictive_validation.md)：32 个全新上下文的 FCRC 共同随机数成对验证、增量预测失败和 N3-E3 判决。
- [air_defense_v1_task14_ranking_refinement.md](experiments/air_defense_v1_task14_ranking_refinement.md)：独立高功效测试、组内动作差异监督、排序改善和 engage/no-op 剩余瓶颈。
- [air_defense_v1_task14_hierarchical_q.md](experiments/air_defense_v1_task14_hierarchical_q.md)：108 状态显式分层 Q 正式对照、目标层收益与交战层负结果。
- [air_defense_v1_task14_engagement_utility.md](experiments/air_defense_v1_task14_engagement_utility.md)：风险/约束交战效用正式对照、类别功效与估值失败分析。
- [air_defense_v1_task14_balanced_engagement.md](experiments/air_defense_v1_task14_balanced_engagement.md)：144状态定向采样、平衡分类与 time-pressure 过度交战诊断。
- [air_defense_v1_task14_engagement_calibration.md](experiments/air_defense_v1_task14_engagement_calibration.md)：72状态独立测试、资源对偶校准与标量停止边界负结果。
- [air_defense_v1_task14_state_conditioned_value.md](experiments/air_defense_v1_task14_state_conditioned_value.md)：三折交叉拟合、72状态双价值正式实验与逐场景剩余边界。
- [air_defense_v1_task14_cross_scenario_robust_value.md](experiments/air_defense_v1_task14_cross_scenario_robust_value.md)：鲁棒损失消融、新72状态批次与异质场景漏交战诊断。
- [air_defense_v1_task14_multibatch_leave_one_out.md](experiments/air_defense_v1_task14_multibatch_leave_one_out.md)：三训练批次留一验证、最终独立批次和安全-停止边界诊断。
- [air_defense_v1_task14_oob_pareto_audit.md](experiments/air_defense_v1_task14_oob_pareto_audit.md)：零新增 rollout 的 OOB 安全-停止可行性、阈值区间和独立确认判定。
- [air_defense_v1_task14_independent_confirmation.md](experiments/air_defense_v1_task14_independent_confirmation.md)：冻结 OOB 阈值的唯一独立批次、数据隔离与 `0/3` 门控结论。
- [air_defense_v1_task14_cross_batch_calibration.md](experiments/air_defense_v1_task14_cross_batch_calibration.md)：四种统一概率/LCB候选的外层留一批次验证与零新增rollout结论。

## Presentations

导师汇报、阶段答辩和学术汇报材料。

- [advisor_progress_report_2026-07-16.md](presentations/advisor_progress_report_2026-07-16.md)：2026-07-16 导师阶段汇报内容、讲稿和答疑准备。
- [advisor_progress_report_2026-07-16.pptx](presentations/advisor_progress_report_2026-07-16.pptx)：2026-07-16 导师阶段汇报演示文稿，共 13 页，含讲者备注。
- [advisor_progress_report_2026-07-22.md](presentations/advisor_progress_2026-07-22/advisor_progress_report_2026-07-22.md)：以 MCH-PPO 创新证据链为核心的近期成果、实验边界与后续工作报告。
- [advisor_progress_report_2026-07-22.pptx](presentations/advisor_progress_2026-07-22/advisor_progress_report_2026-07-22.pptx)：2026-07-22 导师汇报演示文稿，共15页，含实验图表与讲者备注。

## Task Guides

后续研究阶段的任务拆分、执行顺序、交付物和验收标准。

- [task_guides/README.md](task_guides/README.md)：任务指导目录说明与索引。
- [next_research_phase_identifiable_resource_credit.md](task_guides/next_research_phase_identifiable_resource_credit.md)：N1 可辨识资源信用定义、系统查新、离线证伪、预注册与 N1-E4 出口。
- [next_research_phase_future_coverability_certificate.md](task_guides/next_research_phase_future_coverability_certificate.md)：N2 未来可覆盖性责任问题、任务门槛、静态审计和 N2-E1 出口。
- [next_research_phase_fcrc_paired_predictive_validation.md](task_guides/next_research_phase_fcrc_paired_predictive_validation.md)：N3 样本与统计预注册、共同随机数验证和 N3-E3 否决出口。
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
- [next_research_phase_oob_pareto_feasibility.md](task_guides/next_research_phase_oob_pareto_feasibility.md)：完整阈值审计、跨批次/场景约束和独立确认放行条件。
- [next_research_phase_independent_calibration_confirmation.md](task_guides/next_research_phase_independent_calibration_confirmation.md)：冻结目标、模型与阈值后的一次独立确认及 MCH-PPO 恢复条件。
- [next_research_phase_cross_batch_uncertainty_calibration.md](task_guides/next_research_phase_cross_batch_uncertainty_calibration.md)：统一概率、显式不确定性、外层验证和MCH-PPO就绪门控。
- [next_research_phase_bpce_ppo_v0.md](task_guides/next_research_phase_bpce_ppo_v0.md)：BPCE-PPO v0冻结语义、探测预算、严格退化验收和10k机制证伪。
- [next_research_phase_bpce_label_semantics_and_dose_audit.md](task_guides/next_research_phase_bpce_label_semantics_and_dose_audit.md)：标签A/B/C、辅助剂量和选点覆盖的顺序门控；阶段A失败后停止B/C。
- [next_research_phase_bpce_short_horizon_component_label_audit.md](task_guides/next_research_phase_bpce_short_horizon_component_label_audit.md)：短视窗安全—资源三态标签、上下文强校验和阶段A2停止条件。

## Suggested Reading Order

如果是重新接手项目，建议按以下顺序阅读：

1. [academic_project_progress.md](project/academic_project_progress.md)
2. [project_structure.md](project/project_structure.md)
3. [air_defense_rl_environment_model_design.md](environments/air_defense/air_defense_rl_environment_model_design.md)
4. [air_defense_v1_architecture_diagram.md](environments/air_defense/air_defense_v1_architecture_diagram.md)
5. [air_defense_v1_baseline_results.md](experiments/air_defense_v1_baseline_results.md)
