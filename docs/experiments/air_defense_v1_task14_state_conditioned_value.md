# AirDefense v1 状态条件资源预算与显式双价值实验

更新时间：2026-07-21  
实验状态：正式实验完成  
结论状态：总体性能通过，逐场景鲁棒性未通过

## 1. 实验目的

验证直接预测安全收益与增量成本、并使用状态条件资源乘子，能否解决固定标量边界在独立状态上的过度交战问题。

## 2. 数据协议

| 项目 | 数值 |
| --- | ---: |
| 历史非 test 训练组 | 202 |
| grouped cross-fitting | 3折 |
| 全新正式 test 状态 | 72 |
| test 上下文组 | 97 |
| 每分支 rollout | 32 |
| 可靠 oracle 组 | 87 |
| engage / no-op | 30 / 57 |
| medium / time / heterogeneity | 30 / 29 / 28 |
| 旧观测重叠 | 全部0 |
| 回报重构最大误差 | `1.53e-05` |
| 数据生成时间 | 632.40 s |

数据完整性和全部功效门槛通过。

## 3. 交叉拟合选择

| seed | safety-only BA | global-budget BA | state-budget BA | state-budget 可行 |
| ---: | ---: | ---: | ---: | --- |
| 20 | 0.741 | 0.745 | 0.778 | 否 |
| 21 | 0.745 | 0.745 | 0.756 | 是 |
| 22 | 0.752 | 0.748 | 0.764 | 是 |

state-budget 在2/3种子满足逐场景约束，因此在不查看正式 test 的情况下被选中。seed20 唯一 OOF 失败项为 time-pressure no-op recall `0.634`，距离0.65门槛为0.016。

## 4. 总体正式结果

| seed | 方法 | BA | engage recall | no-op recall | false-noop | wasteful-engage |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 20 | zero margin | 0.756 | 0.933 | 0.579 | 0.067 | 0.421 |
| 20 | risk regression | 0.697 | 0.833 | 0.561 | 0.167 | 0.439 |
| 20 | state budget | **0.834** | **0.967** | **0.702** | **0.033** | **0.298** |
| 21 | zero margin | 0.755 | 0.967 | 0.544 | 0.033 | 0.456 |
| 21 | risk regression | 0.741 | 0.833 | 0.649 | 0.167 | 0.351 |
| 21 | state budget | **0.776** | **0.833** | **0.719** | **0.167** | **0.281** |
| 22 | zero margin | 0.782 | 0.967 | 0.596 | 0.033 | 0.404 |
| 22 | risk regression | 0.639 | 0.700 | 0.579 | 0.300 | 0.421 |
| 22 | state budget | **0.768** | **0.800** | **0.737** | **0.200** | **0.263** |

三种子全部通过总体 BA、双类召回、false-noop、wasteful-engage、安全收益符号和推理成本门槛。相较上一阶段资源对偶的 `0.62-0.68` wasteful-engage，本轮下降到 `0.263-0.298`。

## 5. 逐场景结果

| seed | 场景 | engage recall | no-op recall | 结果 |
| ---: | --- | ---: | ---: | --- |
| 20 | medium | 0.889 | 0.810 | 通过 |
| 20 | time-pressure | 1.000 | 0.563 | no-op失败 |
| 20 | heterogeneity | 1.000 | 0.700 | 通过 |
| 21 | medium | 0.667 | 0.810 | 通过 |
| 21 | time-pressure | 0.846 | 0.625 | no-op失败 |
| 21 | heterogeneity | 1.000 | 0.700 | 通过 |
| 22 | medium | 0.556 | 0.905 | engage失败 |
| 22 | time-pressure | 0.846 | 0.750 | 通过 |
| 22 | heterogeneity | 1.000 | 0.550 | no-op失败 |

失败不再是统一的 all-noop 或全局过度交战，而是不同训练种子在不同场景上的局部边界漂移。

## 6. 双价值诊断

| seed | safety MAE | safety corr | safety sign | cost MAE | cost corr | budget mean±std |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 4.535 | 0.530 | 0.753 | 0.168 | -0.044 | 0.955±0.194 |
| 21 | 4.467 | 0.523 | 0.728 | 0.163 | 0.034 | 1.268±0.657 |
| 22 | 4.890 | 0.494 | 0.716 | 0.152 | 0.128 | 2.039±1.430 |

安全收益达到中等相关和超过0.70的符号准确率；成本差虽然 MAE 较小，但方差更小、相关性很弱。state-budget 的提升主要来自状态条件边界与安全监督，尚不能宣称形成了可靠 cost-value 估计。

## 7. 门控结论

| 门控 | 结果 |
| --- | --- |
| 数据与功效 | 全部通过 |
| 总体 BA/双类召回 | 3/3通过 |
| 对风险回归双非劣 | 3/3通过 |
| safety sign >=0.70 | 3/3通过 |
| 逐场景 engage/no-op recall | 0/3完整通过 |
| 至少2/3种子整体通过 | 未通过：0/3 |
| 恢复 MCH-PPO | 否 |
| 进入 GNN | 否 |

## 8. 下一步

当前距离 MCH-PPO 只剩一次有针对性的跨场景鲁棒修订：

- 冻结双价值网络与数据语义；
- 用逐场景最坏召回或分布鲁棒目标选择 epoch/模型；
- 对低方差 cost-delta 使用可靠性加权、排序或异方差监督；
- 保持新的独立 test，不对本轮 test 调参；
- 至少2/3种子完整通过后立即冻结 MCH-PPO 接口。

## 9. 结果入口

```text
results/air_defense_v1/task14_state_conditioned_value/test_dataset.npz
results/air_defense_v1/task14_state_conditioned_value/crossfit_curves.csv
results/air_defense_v1/task14_state_conditioned_value/crossfit_predictions.csv
results/air_defense_v1/task14_state_conditioned_value/model_metrics.csv
results/air_defense_v1/task14_state_conditioned_value/test_group_diagnostics.csv
results/air_defense_v1/task14_state_conditioned_value/gate_summary.json
results/air_defense_v1/task14_state_conditioned_value/experiment_config.json
```
