# 下一研究阶段：动作差异数据与组内排序监督

更新时间：2026-07-19  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`  
阶段编号：任务十四·修订  
阶段状态：已完成；附加对照通过，原门控未通过  
阶段主题：Q-Critic 标签功效、组内动作差异学习与 MCH-PPO 二次门控

## 1. 阶段定位

任务十四证明非图 MLP 能把动作条件 Q 的 MAE 相对 `V(s)` 降低 `36.4%-40.1%`，但高置信动作对只有 8 个，排序准确率仅 `0.250-0.375`。完整输入消融也没有稳定优于简化输入。当前不能区分：

1. 测试标签功效不足，导致排序门槛不可判定；
2. 纯绝对 Q 回归主要学习状态共同价值，没有学习组内动作差值；
3. 非图关系 MLP 在可靠监督下仍缺少表达能力。

本阶段先控制前两项。它不是任务十五，不训练 PPO Actor，也不实现 GNN。

## 2. 核心研究问题

> 在不改变环境、冻结策略和 Q-Critic 表示结构的条件下，提高独立测试标签功效并加入组内动作差异监督，是否能使非图 Q-Critic 稳定恢复候选动作排序和 engage/no-op advantage 符号？

只有修订模型在全新测试状态上通过原任务十四门槛，才允许进入任务十五 MCH-PPO。

## 3. 冻结边界

继续冻结：

- AirDefense v1.0 环境、奖励、终止条件和三个核心场景；
- 任务十二 factorized 策略 seeds 8/10、顺序 `012` 和模型参数；
- `Q^pi(s,h_i,a_i)` 的自回归反事实语义；
- 当前 `MaskedActionQCritic` 完整输入和 MLP `256/128` 结构；
- 任务十四测试指标、置信规则和通过阈值；
- 训练种子 14/15/16；
- 不使用 GNN、GAT、Transformer；
- 不更新 PPO Actor，不运行 30k/100k。

允许改变：

- 训练损失；
- 独立测试状态数和每候选 rollout 数；
- 数据留档模式与配对功效诊断。

## 4. 数据隔离协议

### 4.1 训练与验证

复用任务十四数据，但只允许使用：

- 原 `split=train` 作为训练集；
- 原 `split=validation` 作为早停与模型选择集；
- 原 `split=test` 永久排除，不得重新标记为训练数据。

### 4.2 全新正式测试集

使用新的环境随机种子采集：

| 项目 | 冻结值 |
| --- | ---: |
| 来源策略 | seed 8、10 |
| 场景 | medium、time_pressure、heterogeneity_pressure |
| 每个来源种子 × 场景状态数 | 6 |
| 总测试状态数 | 36 |
| 每个候选配对 rollout | 32 |
| 折扣因子 | 0.98 |
| 测试集用途 | 仅最终一次门控 |

测试状态采用随机 reservoir 抽样，不按回报差异筛选。所有候选继续使用共同环境和策略随机数。

任务十四功效审计显示：原 18 状态测试集从 8 提升到 32 rollout 时，预计三个场景分别产生约 20、23、21 个高置信动作对。将随机测试状态扩大到 36 个后，预计每场景超过 30 对。该估计只用于样本量设计，不参与模型选择。

## 5. 对照方法

同一模型、训练数据、优化器和训练种子比较：

### 5.1 `absolute_mse`

任务十四原始基线：

```text
L_abs = mean((Q_hat - Q_mc)^2)
```

### 5.2 `difference_aware`

预注册主候选：

```text
L = L_abs + L_center + 0.5 * L_pair
```

其中：

```text
L_center = mean(((Q_hat - group_mean(Q_hat))
                 - (Q_mc - group_mean(Q_mc)))^2)

L_pair = weighted SmoothL1(
    (Q_hat_a - Q_hat_b) - (Q_mc_a - Q_mc_b)
)
```

`group = state_id + unit_index`。配对权重由 8-rollout 训练标签的配对信噪比确定并截断到 `[0.25, 4.0]`：

```text
w_ab = clip(|delta_ab| / (1.96 * SE_ab + eps), 0.25, 4.0)
```

验证早停只使用：

```text
validation_score = normalized_MAE + centered_MAE
```

不得根据正式测试排序结果选择 epoch、损失权重或模型结构。

## 6. 工程任务

### 6.1 功效审计

- 实现测试集在不同 rollout 数下的高置信比较数投影；
- 输出总体及逐场景结果；
- 检查投影只用于规模设计。

### 6.2 组内监督工具

- 构造不跨 `state_id + unit_index` 的配对索引；
- 计算组内中心化标签和预测；
- 计算配对标准误、可靠性权重与 SmoothL1 损失；
- 对单候选组、零标准误和极低信噪比稳定处理。

### 6.3 统一训练与正式测试脚本

- 加载旧 train/validation，排除旧 test；
- 生成并缓存全新 36 状态测试集；
- 训练 `absolute_mse` 与 `difference_aware`，各 3 个种子；
- 保存模型、曲线、逐候选预测、指标、门控和配置；
- 支持 `--reuse-test-dataset`，但不得把测试集并入训练。

## 7. 验收标准

### 7.1 工程验收

- 旧 test 行进入训练或验证的数量必须为 0；
- 新测试 `state_id` 与旧数据交集必须为 0；
- 新测试集必须正好包含 36 个状态和 32 rollout；
- 配对索引不得跨组；
- 测试集不得参与早停；
- 单元测试和项目全量测试通过；
- 正式配置、模型和原始逐 rollout 回报完整留档。

### 7.2 科学门槛

沿用任务十四主门槛：

| 类别 | 门槛 |
| --- | --- |
| Q 数值 | MAE 相对 `V(s)` 改善 >= 10% |
| 总体排序 | accuracy >= 0.70，有效对 >= 30 |
| engage/no-op | 符号准确率 >= 0.70，有效组 >= 30 |
| 目标排序 | accuracy >= 0.65，有效对 >= 30 |
| top-1 | accuracy >= 0.50，有效状态 >= 30 |
| 场景 | 三场景排序均 >= 0.60，且各有效对 >= 30 |
| 稳定性 | `difference_aware` 至少 2/3 seeds 整体通过 |
| 效率 | Q 推理快于 32-rollout Monte Carlo |

附加对照要求：`difference_aware` 的总体排序准确率相对同种子 `absolute_mse` 平均提高至少 0.10，且不能使 Q MAE 相对恶化超过 10%。附加要求用于证明组内监督有实际贡献，不替代原门槛。

## 8. 决策规则

```text
新测试集是否达到各场景有效比较 >= 30？
├─ 否：数据功效设计仍不足，停止算法结论
└─ 是
   └─ difference_aware 是否通过原门槛且优于 absolute_mse？
      ├─ 是：冻结动作 Q 接口，允许进入任务十五 MCH-PPO
      └─ 否
         ├─ 两种模型排序都弱：非图动作差异学习未成立
         ├─ 仅 MAE 好：状态价值主导问题仍存在
         └─ 数据可靠且结构稳定失败：才允许论证更强关系 Critic
```

普通 MLP 失败不会自动触发 GNN；还需要确认失败不是训练目标、测试功效或优化不稳定造成。

## 9. 交付物

```text
docs/task_guides/next_research_phase_q_critic_ranking_refinement.md
rein_learning/common/q_critic_training.py
scripts/analyze_air_defense_v1_task14_power.py
scripts/run_air_defense_v1_task14_ranking_refinement.py
tests/test_air_defense_v1_task14_ranking_refinement.py
docs/experiments/air_defense_v1_task14_ranking_refinement.md
results/air_defense_v1/task14_q_critic_ranking_refinement/
```

## 10. 阶段完成定义

满足以下任一条件即可完成本阶段：

- 修订模型通过全部原门槛和附加对照，允许进入任务十五；
- 测试功效达标但排序监督仍失败，形成可靠负结果；
- 测试功效未达标，明确给出继续增加样本的量化需求；
- 工程或数据隔离检查失败时停止实验并修复，不输出算法结论。

## 11. 执行结果

正式实验完成 36 个全新测试状态、192 个候选动作和每候选 32 次共同随机数 rollout。任务十四旧 test 的 116 行全部排除，新旧状态 ID 交集为 0。

`difference_aware` 相对 `absolute_mse` 的总体排序平均提高 `0.167`，平均 MAE 比值为 `0.993`，两个附加对照均通过。差异感知模型的三种子总体排序为 `0.659 / 0.727 / 0.705`，目标排序为 `0.696 / 0.826 / 0.783`。

但总体高置信动作对虽达到 44，目标对、top-1、engage/no-op 和三个场景的有效数量仍不足 30；engage/no-op 符号准确率仅 `0.545`。完整门控通过种子数仍为 `0/3`。

因此不恢复 MCH-PPO、不进入 GNN。下一入口收窄为离线非图的 engage/target 显式分层 Q 诊断，详见[任务十四修订实验报告](../experiments/air_defense_v1_task14_ranking_refinement.md)。
