# Docs Index

本文档是 `docs/` 目录索引。文档按用途分为项目管理、环境设计、算法与文献、实验结果四类。

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

## Algorithms

算法实现说明与相关论文阅读整理。

- [implemented_algorithms.md](algorithms/implemented_algorithms.md)：当前已实现算法说明。
- [两篇HARL论文创新点与研究借鉴报告.md](algorithms/两篇HARL论文创新点与研究借鉴报告.md)：HARL/HATRPO/HAPPO 论文创新点和研究借鉴。

## Experiments

实验结果、baseline 结果和对比记录。

- [air_defense_v1_baseline_results.md](experiments/air_defense_v1_baseline_results.md)：AirDefense v1.0 第一组规则 baseline 结果。
- [air_defense_v1_learning_baselines.md](experiments/air_defense_v1_learning_baselines.md)：AirDefense v1.0 PPO / Maskable PPO 训练与统一对比说明。
- [air_defense_v1_formal_benchmark_100k.md](experiments/air_defense_v1_formal_benchmark_100k.md)：AirDefense v1.0 正式 100k × 5 seeds 基准结果与分析。

## Presentations

导师汇报、阶段答辩和学术汇报材料。

- [advisor_progress_report_2026-07-16.md](presentations/advisor_progress_report_2026-07-16.md)：2026-07-16 导师阶段汇报内容、讲稿和答疑准备。
- [advisor_progress_report_2026-07-16.pptx](presentations/advisor_progress_report_2026-07-16.pptx)：2026-07-16 导师阶段汇报演示文稿，共 13 页，含讲者备注。

## Suggested Reading Order

如果是重新接手项目，建议按以下顺序阅读：

1. [academic_project_progress.md](project/academic_project_progress.md)
2. [project_structure.md](project/project_structure.md)
3. [air_defense_rl_environment_model_design.md](environments/air_defense/air_defense_rl_environment_model_design.md)
4. [air_defense_v1_architecture_diagram.md](environments/air_defense/air_defense_v1_architecture_diagram.md)
5. [air_defense_v1_baseline_results.md](experiments/air_defense_v1_baseline_results.md)
