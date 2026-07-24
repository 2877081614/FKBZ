# AirDefense v1.0 冻结 OOB 校准独立确认

更新时间：2026-07-22
实验状态：完成，主门控未通过

## 1. 实验目的

验证在完全冻结模型、目标函数和 OOB 种子级阈值后，安全-停止决策边界能否泛化到
唯一一批全新临界状态。该实验是恢复 MCH-PPO 前的最终独立门控，不承担调参功能。

## 2. 冻结设置

```text
objective: scenario_robust_reliable_cost
model seeds: 20, 21, 22
frozen thresholds: 0.105205, 0.028757, 0.354024
confirmation eval_seed: 887000
source seeds: 8, 10
scenarios: medium, time_pressure, heterogeneity_pressure
states: 72
rollouts per branch: 32
```

确认批次生成前已将上述配置与 `score > threshold` 规则写入任务文档。模型没有重训，
阈值没有根据确认标签重新扫描。

## 3. 数据审计

| 项目 | 结果 |
|---|---:|
| 状态数 | 72 |
| 上下文组数 | 87 |
| 可靠组数 | 81 |
| engage / no-op | 35 / 46 |
| medium engage / no-op | 10 / 19 |
| time-pressure engage / no-op | 14 / 12 |
| heterogeneity engage / no-op | 11 / 15 |
| 历史数据集重叠 | 0 / 19 |
| 回报重构最大误差 | 7.63e-06 |
| 数据与功效门控 | 全部通过 |

因此本次失败不能归因于样本类别缺失、历史数据泄漏或回报计算错误。

## 4. 总体结果

| seed | frozen threshold | BA | engage | no-op | 最差场景 engage | 最差场景 no-op | safety sign | 通过 |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 20 | 0.1052 | 0.625 | 0.771 | 0.478 | 0.636 | 0.333 | 0.740 | 否 |
| 21 | 0.0288 | 0.646 | 0.857 | 0.435 | 0.727 | 0.333 | 0.740 | 否 |
| 22 | 0.3540 | 0.625 | 0.686 | 0.565 | 0.364 | 0.467 | 0.753 | 否 |

三个种子的 safety sign 均超过0.70，但 BA、no-op recall 和逐场景门槛没有形成完整
通过。冻结阈值为 `0/3`，零阈值也为 `0/3`。

## 5. 场景结果

| seed | 场景 | engage recall | no-op recall |
|---:|---|---:|---:|
| 20 | medium | 0.900 | 0.632 |
| 20 | time_pressure | 0.786 | 0.417 |
| 20 | heterogeneity_pressure | 0.636 | 0.333 |
| 21 | medium | 1.000 | 0.579 |
| 21 | time_pressure | 0.857 | 0.333 |
| 21 | heterogeneity_pressure | 0.727 | 0.333 |
| 22 | medium | 0.900 | 0.684 |
| 22 | time_pressure | 0.786 | 0.500 |
| 22 | heterogeneity_pressure | 0.364 | 0.467 |

seed20/21 的主要问题是跨场景过度交战。seed22 使用更高阈值后仍未恢复 no-op，且
异质场景必要交战识别明显下降。这不是单纯把阈值整体向上移动就能稳定解决的现象。

## 6. 价值诊断

三个种子的 safety sign accuracy 为 `0.740 / 0.740 / 0.753`，说明安全收益方向仍有
信息；safety correlation 仅为 `0.374 / 0.384 / 0.349`。cost correlation 虽为正，
但只有 `0.235 / 0.196 / 0.195`。当前 score 在新批次上的排序和尺度不足以稳定表达
“继续交战的安全收益是否值得资源消耗”。

## 7. 结论与路线

独立确认主门控未通过，不能恢复 MCH-PPO。上一阶段 `3/3` 的 OOB 可行性证明模型
不是完全不可分，但本阶段 `0/3` 证明种子级固定阈值没有跨到新批次。

本确认批次不得用于回调阈值，也不追加新的确认批次。下一步应修改机制，优先研究：

- 跨批次可比较的 score 归一化或概率校准；
- 预测不确定性与拒绝/停止约束；
- 安全收益和资源成本的显式约束决策，而不是单一标量差值。

GNN仍不直接进入，因为当前失败没有证明关系表示容量是主要瓶颈。

## 8. 复现说明

正式结果已经冻结。仅在验证代码时可复用现有数据集：

```powershell
conda run -n rein-learning python scripts\run_air_defense_v1_task14_independent_confirmation.py --reuse-test-dataset
```

不得删除结果后使用其他 `eval_seed` 重复确认。
