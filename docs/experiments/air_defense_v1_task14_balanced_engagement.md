# AirDefense v1 安全临界采样与类别平衡交战实验

更新时间：2026-07-21  
实验状态：正式实验完成，数据功效通过，模型完整门控未通过  
结论状态：必要交战识别显著改善，但 time-pressure 过度交战未受控

## 1. 实验目的

检验安全临界度定向采样能否解决稀有 `oracle_engage` 功效，并比较风险效用回归、类别平衡 BCE 和类别平衡 BCE+margin 在全新独立 test 上的双尾错误。

## 2. 数据与隔离

| 项目 | 数值 |
| --- | ---: |
| 全新 targeted 状态 | 144 |
| targeted 上下文组 | 196 |
| 历史非 test 训练组 | 87 |
| 合并分析组 | 283 |
| train/validation/test 组 | 163 / 39 / 81 |
| 每分支 rollout | 32 |
| 数据生成时间 | 899.87 s |
| 四组旧测试观测重叠 | 0 / 0 / 0 / 0 |
| state split 泄漏 | 0 |
| 回报重构最大误差 | `7.63e-06` |

历史 engagement utility 的原 test 63组完全排除。validation 和 test 均只来自本轮新状态。

## 3. 定向采样功效

新 test 81组中74组有可靠 oracle：

| 场景 | 可靠组 | engage | no-op |
| --- | ---: | ---: | ---: |
| medium | 25 | 9 | 16 |
| time_pressure | 22 | 11 | 11 |
| heterogeneity_pressure | 27 | 8 | 19 |
| 合计 | 74 | 28 | 46 |

engage 比例为 `37.8%`，相比上一阶段随机 test 的 `5.3%` 提高 `32.6` 个百分点。总数、两类数量、逐场景数量、跨场景 engage 和富集率门槛全部通过。

## 4. Validation 方法选择

| 方法 | seed20 | seed21 | seed22 | 平均 |
| --- | ---: | ---: | ---: | ---: |
| balanced BCE | 0.683 | 0.683 | 0.719 | 0.695 |
| BCE + margin | 0.689 | 0.755 | 0.719 | 0.721 |

正式候选冻结为 `balanced_bce_margin`。

## 5. 总体测试结果

| seed | 方法 | BA | engage recall | no-op recall | false-noop | wasteful-engage |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 20 | risk regression | 0.612 | 0.571 | 0.652 | 0.429 | 0.348 |
| 20 | BCE + margin | 0.758 | 0.929 | 0.587 | 0.071 | 0.413 |
| 21 | risk regression | 0.583 | 0.536 | 0.630 | 0.464 | 0.370 |
| 21 | BCE + margin | 0.711 | 0.857 | 0.565 | 0.143 | 0.435 |
| 22 | risk regression | 0.583 | 0.536 | 0.630 | 0.464 | 0.370 |
| 22 | BCE + margin | 0.708 | 0.786 | 0.630 | 0.214 | 0.370 |

候选三种子均通过总体 BA、相对提升和 false-noop 非劣；seed20/21 的 wasteful-engage 恶化，seed22持平。

## 6. 典型失效场景

`time_pressure` 是剩余瓶颈：

| seed | engage recall | no-op recall | wasteful-engage |
| ---: | ---: | ---: | ---: |
| 20 | 1.000 | 0.455 | 0.545 |
| 21 | 1.000 | 0.182 | 0.818 |
| 22 | 0.909 | 0.273 | 0.727 |

候选几乎识别了所有必要交战，却把大量应停止的决策也判为 engage。这说明类别平衡目标解决了 all-noop 方向，但尚未形成资源约束下的停止边界。

## 7. 门控结论

| 门控 | 结果 |
| --- | --- |
| 数据完整性 | 通过 |
| 正类与逐场景功效 | 全部通过 |
| 总体 BA 与相对提升 | 三种子通过 |
| false-noop 非劣 | 三种子通过 |
| wasteful-engage 非劣 | 仅 seed22通过 |
| 每场景 no-op recall >=0.65 | 三种子均未通过 |
| 至少2/3种子整体通过 | 未通过：0/3 |
| 恢复 MCH-PPO | 否 |
| 进入 GNN | 否 |

## 8. 科学解释与下一步

本轮第一次同时获得了足够的必要交战样本和跨种子 `BA>0.70` 的交战分类器，因此数据功效和基本表示能力不再是主瓶颈。失败已经收窄为：

> 类别平衡零阈值过度重视稀有 engage，缺少显式 wasteful-engage/资源约束校准，尤其在 time-pressure 场景。

下一阶段应冻结本轮模型和 test，在新的 validation 语料上比较：

- 满足 no-op recall 下界的约束阈值；
- false-noop 与 wasteful-engage 的对偶代价；
- 全局阈值与状态依赖预算变量；
- 校准前后跨场景可靠性。

只有校准方法在全新 test 上至少2/3种子同时通过两类错误与逐场景门槛，才允许最小 MCH-PPO 30k。

## 9. 结果入口

```text
results/air_defense_v1/task14_balanced_engagement/targeted_dataset.npz
results/air_defense_v1/task14_balanced_engagement/analysis_dataset.npz
results/air_defense_v1/task14_balanced_engagement/gate_summary.json
results/air_defense_v1/task14_balanced_engagement/model_metrics.csv
results/air_defense_v1/task14_balanced_engagement/training_curves.csv
results/air_defense_v1/task14_balanced_engagement/test_group_diagnostics.csv
```
