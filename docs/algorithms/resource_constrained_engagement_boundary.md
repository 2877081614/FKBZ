# 资源约束交战边界校准

更新时间：2026-07-21  
实现状态：离线正式实验完成，门控未通过  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`

## 1. 方法目的

类别平衡 BCE+margin 已能识别必要交战，但零阈值产生过多高成本交战。本方法冻结 Critic 参数，只对 `z_engage-z_noop` 的决策边界加入可解释资源压力，检验停止错误是否可由低维约束校准修复。

## 2. 决策变量

当前单元资源压力定义为：

```text
p_resource = cost_norm * (2 - ammo_fraction)
```

它在高成本、低弹药时增大。为消除不同模型 logit 尺度差异，使用 validation 标准差：

```text
engage iff
z_engage - z_noop
> tau + lambda * std_validation(logit) * p_resource
```

`lambda=0` 为全局阈值；资源对偶候选为 `0.25/0.5/1.0/2.0`。阈值候选由资源修正后 validation logit 的相邻中点构成。

## 3. 约束选择

每个冻结模型种子独立选择 `tau/lambda`，但方法族在三个种子间统一。validation 约束包括总体 `BA>=0.70`、engage recall `>=0.60`、no-op recall `>=0.65`，以及相同的逐场景类别召回下界。先比较可行种子数，再比较平均 BA，平局偏向全局阈值。

正式 test 不参与参数或方法族选择。该协议与任务十三不同：任务十三扫描原 PPO Actor 概率，本方法校准的是 oracle 监督后的交战符号 logit。

## 4. 实现接口

`engagement_boundary_calibration.py` 提供：

- `resource_pressure_from_observations`：按冻结观测布局提取单元成本和弹药；
- `EngagementBoundaryConfig`：记录阈值、对偶权重、logit 尺度和可行性；
- `apply_engagement_boundary`：执行全局或资源感知边界；
- `calibrate_engagement_boundary`：只在校准集上枚举并按约束排序；
- `scenario_classification_metrics`：输出逐场景双类诊断。

## 5. 正式结果与失败边界

validation 上资源对偶平均 BA `0.759`，略高于全局阈值的 `0.750`，但两个方法族均为 `0/3` 可行。独立 test 中，资源对偶三种子 BA 为 `0.593/0.612/0.605`，no-op recall 仅 `0.38/0.32/0.34`，wasteful-engage 达到 `0.62/0.68/0.66`。

test 中 no-op 的平均资源压力 `0.610` 高于 engage 的 `0.467`，说明特征方向合理但不可单独分离两类。成本与弹药只是停止条件的一部分；目标紧迫度、剩余资源预算、未来目标到达和其他单元替代能力共同决定是否应交战。单一线性压力项无法表达该状态条件边界。

因此本实现保留为受控负基线，不接入 PPO。下一候选应使用显式预算状态、约束价值或状态条件拉格朗日乘子，并避免继续对同一 validation 增加标量网格。

## 6. 实现入口

```text
rein_learning/common/engagement_boundary_calibration.py
scripts/run_air_defense_v1_task14_engagement_calibration.py
tests/test_air_defense_v1_task14_engagement_calibration.py
docs/experiments/air_defense_v1_task14_engagement_calibration.md
```
