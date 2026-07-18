# Docs Index

本文档是 `docs/` 目录索引。文档按用途分为项目管理、环境设计、算法与文献、实验结果、任务指导和汇报材料。

## Project

项目级文档，记录工程结构、环境配置和总体进度。

- [academic_project_progress.md](project/academic_project_progress.md)：学术项目总进度、阶段总结、下一步计划。
- [project_structure.md](project/project_structure.md)：代码和文档目录结构规划。
- [rl_environment_setup.md](project/rl_environment_setup.md)：`rein-learning` 环境与依赖说明。

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
- [两篇HARL论文创新点与研究借鉴报告.md](algorithms/两篇HARL论文创新点与研究借鉴报告.md)：HARL/HATRPO/HAPPO 论文创新点和研究借鉴。

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

## Presentations

导师汇报、阶段答辩和学术汇报材料。

- [advisor_progress_report_2026-07-16.md](presentations/advisor_progress_report_2026-07-16.md)：2026-07-16 导师阶段汇报内容、讲稿和答疑准备。
- [advisor_progress_report_2026-07-16.pptx](presentations/advisor_progress_report_2026-07-16.pptx)：2026-07-16 导师阶段汇报演示文稿，共 13 页，含讲者备注。

## Task Guides

后续研究阶段的任务拆分、执行顺序、交付物和验收标准。

- [task_guides/README.md](task_guides/README.md)：任务指导目录说明与索引。
- [next_research_phase_difficulty_generalization.md](task_guides/next_research_phase_difficulty_generalization.md)：场景难度、Hungarian 基线、泛化诊断和图结构算法进入条件。
- [next_research_phase_conflict_free_joint_action.md](task_guides/next_research_phase_conflict_free_joint_action.md)：任务八无冲突联合动作机制、实施步骤、筛选协议和正式验收门槛。
- [next_research_phase_autoregressive_joint_action.md](task_guides/next_research_phase_autoregressive_joint_action.md)：任务九顺序式/自回归无冲突动作生成、概率契约、筛选协议和验收标准。
- [next_research_phase_order_bias_diagnostics.md](task_guides/next_research_phase_order_bias_diagnostics.md)：任务十逐决策诊断、固定单元顺序消融和 GNN 进入条件。
- [next_research_phase_role_conditioned_action_head.md](task_guides/next_research_phase_role_conditioned_action_head.md)：任务十一共享资源-目标关系动作头、单元塌缩抑制、顺序鲁棒性和筛选门槛。
- [next_research_phase_noop_optimization_stability.md](task_guides/next_research_phase_noop_optimization_stability.md)：任务十二固定策略探针、no-op 训练分叉诊断、交战-目标因子化和确认门槛。

## Suggested Reading Order

如果是重新接手项目，建议按以下顺序阅读：

1. [academic_project_progress.md](project/academic_project_progress.md)
2. [project_structure.md](project/project_structure.md)
3. [air_defense_rl_environment_model_design.md](environments/air_defense/air_defense_rl_environment_model_design.md)
4. [air_defense_v1_architecture_diagram.md](environments/air_defense/air_defense_v1_architecture_diagram.md)
5. [air_defense_v1_baseline_results.md](experiments/air_defense_v1_baseline_results.md)
