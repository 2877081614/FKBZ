# 下一研究阶段：资源约束交战边界校准

更新时间：2026-07-21  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`  
阶段状态：已完成；数据与功效通过，边界校准门控未通过  
阶段主题：在不重新训练 Critic 的条件下，校准“必须交战”与“应当停止”的资源约束边界

## 1. 阶段定位

安全临界采样和类别平衡 BCE+margin 已经恢复必要交战识别，但在 `time_pressure` 中出现明显过度交战。上一轮三个训练种子的 engage recall 为 `0.786-0.929`，而该场景 no-op recall 仅为 `0.182-0.455`。当前瓶颈已由“看不见 engage”收窄为“缺少资源约束下的停止边界”。

本阶段只校准冻结的 `balanced_bce_margin` 输出，不修改环境、奖励、oracle、Critic 参数、临界状态采样协议或 conditional-target 层。MCH-PPO、30k/100k 训练和 GNN 继续冻结。

## 2. 与任务十三阈值扫描的区别

任务十三扫描的是原 PPO Actor 的交战概率。不同训练种子的概率尺度和校准误差差异很大，统一阈值无法修复全部场景。

本阶段处理的是由可靠 oracle、类别平衡监督和 margin 训练得到的交战符号：

```text
logit = z_engage - z_noop
```

阈值按冻结模型种子在历史 validation 上独立校准，并显式引入当前单元资源压力。因此，本阶段检验的是“已恢复的交战信号能否形成资源受限决策边界”，不是再次对原 Actor 做经验阈值搜索。

## 3. 冻结内容

- `balanced_bce_margin` seeds `20/21/22` 检查点；
- 同种子的 `risk_regression` 对照检查点；
- 安全-资源 oracle：`harm=30*damage+20*high_threat_leak`；
- 90% 成对置信规则、每分支32次共同随机数 rollout；
- factorized source policies seeds `8/10`；
- `medium`、`time_pressure`、`heterogeneity_pressure` 三个场景；
- 安全临界度定向采样规则；
- AirDefense v1.0 环境、奖励、观测和动作空间；
- 任务十四前序全部正式 test 及其门槛。

禁止根据本轮独立 test 修改压力公式、候选网格、约束或通过门槛。

## 4. 资源约束边界

从冻结 v1 观测中读取当前决策单元：

```text
ammo_fraction = unit_features[3]
unit_cost_norm = unit_features[10]
resource_pressure = unit_cost_norm * (2 - ammo_fraction)
```

比较三种决策边界：

```text
zero_margin:
    engage iff logit > 0

global_threshold:
    engage iff logit > tau

resource_dual:
    engage iff logit > tau
                        + lambda * std_validation(logit) * resource_pressure
```

`std_validation(logit)` 只用于消除不同模型种子的 logit 尺度差异。`lambda` 候选冻结为 `0.25 / 0.5 / 1.0 / 2.0`；全局边界等价于 `lambda=0`。

## 5. 校准协议

校准数据只取上一阶段 `analysis_dataset.npz` 的 `validation` split。对每个模型种子独立搜索 `tau`，候选由资源修正后 validation logit 的相邻中点构成。

validation 可行约束为：

- overall balanced accuracy 不低于 `0.70`；
- overall engage recall 不低于 `0.60`；
- overall no-op recall 不低于 `0.65`；
- 含 engage 的每个场景 engage recall 不低于 `0.60`；
- 含 no-op 的每个场景 no-op recall 不低于 `0.65`。

先比较可行种子数，再比较平均 validation balanced accuracy；平局时选择更简单的全局边界。每个种子允许有不同 `tau`，但方法族必须统一。正式 test 不参与任何选择。

## 6. 独立测试协议

正式新增：

```text
2 source policies * 3 scenarios * 12 states = 72 states
```

每个状态仍按可用单元形成 no-op/engage 上下文组，每分支使用32次共同随机数 rollout。新 test 与所有前序正式 test 的观测交集必须为0。

数据与功效门槛：

- 恰好72个全新状态，每分支32次 rollout；
- 可可靠判别组不少于40；
- engage 与 no-op 各不少于10；
- 每场景可可靠组不少于8；
- 至少两个场景包含 engage；
- 总回报分量重构误差不超过 `1e-4`；
- 与所有旧正式 test 观测重叠为0。

## 7. 模型门槛

校准候选对每个种子必须同时满足：

- balanced accuracy 不低于 `0.70`；
- false-noop 不高于同种子 `risk_regression`；
- wasteful-engage 不高于同种子 `risk_regression`；
- 相对零阈值 margin 的 wasteful-engage 至少降低 `0.10`；
- 相对零阈值 margin 的 false-noop 增幅不超过 `0.10`；
- 含 no-op 的每个场景 no-op recall 不低于 `0.65`；
- 含 engage 的每个场景 engage recall 不低于 `0.60`。

三个冻结模型种子至少两个整体通过，才允许恢复最小 MCH-PPO 实现。无论通过或失败，本阶段都不直接进入 GNN。

## 8. 交付物

```text
docs/task_guides/next_research_phase_resource_constrained_engagement_calibration.md
rein_learning/common/engagement_boundary_calibration.py
scripts/run_air_defense_v1_task14_engagement_calibration.py
tests/test_air_defense_v1_task14_engagement_calibration.py
docs/algorithms/resource_constrained_engagement_boundary.md
docs/experiments/air_defense_v1_task14_engagement_calibration.md
results/air_defense_v1/task14_engagement_calibration/
```

## 9. 阶段完成定义

本阶段在代码、测试、72状态正式实验和结果文档全部完成后结束。若至少2/3种子通过，下一阶段进入最小 MCH-PPO 的30k独立种子筛选；若资源感知边界明显优于全局阈值但仍未通过，则审查显式预算/拉格朗日约束；若两者均无改善，则说明停止错误并非单一资源阈值问题，应回到 oracle、状态条件或分布式估值机理，而不能用 GNN 掩盖该问题。

## 10. 执行结果

正式实验生成72个全新状态、84个上下文组和每分支32次 rollout。81个可靠组包含31个 engage 与50个 no-op；三个场景分别有24、29、28个可靠组，旧观测重叠均为0，总回报重构误差为 `7.63e-06`。数据与功效门槛全部通过。

validation 上，全局阈值三种子 BA 为 `0.741 / 0.755 / 0.755`，资源对偶边界为 `0.755 / 0.769 / 0.755`。资源对偶平均略高，因此按预注册规则被选中；但两种方法族的三种子均不存在满足全部逐场景召回约束的可行解。

独立 test 上，资源对偶校准三种子 BA 为 `0.593 / 0.612 / 0.605`，no-op recall 为 `0.38 / 0.32 / 0.34`，wasteful-engage 为 `0.62 / 0.68 / 0.66`。相对零阈值既未达到 `0.10` 的浪费性交战改善，也未优于风险回归基线；整体通过数为 `0/3`。

资源压力具有弱方向性：test 中 no-op 平均压力为 `0.610`，engage 为 `0.467`，但两类范围高度重叠。为保持 engage recall，validation 选择的负阈值抵消了部分资源惩罚，无法形成稳定停止边界。因此本阶段不恢复 MCH-PPO，也不进入 GNN。下一阶段应研究状态条件预算或显式受限优化，并采用独立/交叉拟合校准，而不是继续扩大标量阈值网格。
