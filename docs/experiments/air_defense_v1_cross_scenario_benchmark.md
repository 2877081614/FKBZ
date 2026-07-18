# AirDefense v1.0 跨场景统一实验与泛化矩阵

更新日期：2026-07-17  
实验数据模式：`schema_version = 3`

## 1. 目的

任务五将原单场景 benchmark 升级为：

```text
训练场景 × 训练种子 × 测试场景 × 方法
```

学习方法在每个“训练场景×训练种子”上独立训练一次，训练完成后复用同一模型评估所有测试场景。规则方法不进行训练，但复制到每个训练场景行中，以便与学习方法构成完整、可配对的泛化矩阵。

第一阶段只允许观测维度和联合动作维度完全一致的场景进入同一次实验。

## 2. 命令行入口

单训练场景、三测试场景：

```powershell
conda run -n rein-learning python scripts\compare_air_defense_v1_methods.py `
  --train-scenario medium `
  --eval-scenarios easy medium hard `
  --methods greedy_damage hungarian_damage maskable_ppo
```

完整三乘三难度矩阵：

```powershell
conda run -n rein-learning python scripts\compare_air_defense_v1_methods.py `
  --train-scenario easy medium hard `
  --eval-scenarios easy medium hard `
  --methods greedy_damage hungarian_damage maskable_ppo `
  --seeds 0 1 2
```

`--train-scenario` 为兼容任务协议保留单数名称，但接受一个或多个场景。`--methods` 未提供时仍运行原有全部规则和学习方法；原默认单场景命令继续使用 `medium → medium`。

`--rules-only` 仍可运行全部规则基线，但不能与 `--methods` 同时使用。

## 3. 场景解析与维度校验

场景名称来自 `rein_learning/envs/air_defense_v1/scenarios.py`。别名会转换为正式名称，重复的正式场景会被拒绝。

每个场景在训练开始前生成空间签名：

```text
space_signature = (observation_space.shape, action_space.nvec)
```

所有训练和测试场景必须与第一个训练场景签名一致。若目标数、资源数、区域数或特征维度造成不兼容，实验直接报出参考场景、冲突场景和两组签名，不进入训练。

旧 `env_config=` Python API 仍受支持，但只表示单个名为 `custom` 的训练/测试场景，不能与命名的多场景参数同时使用。

## 4. 配对评估种子

设测试场景数为 `E`，每个场景评估回合数为 `N`：

```text
evaluation_seed
= eval_seed
+ run_index * E * N
+ eval_scenario_index * N
```

因此，同一 `run_index` 和 `eval_scenario` 下：

- 所有规则和学习方法使用相同场景种子块；
- 不同训练场景得到的模型使用相同测试样本；
- 不同测试场景的种子区间互不重叠。

学习曲线只在模型自己的训练场景上评估，最终泛化评估才覆盖全部测试场景。

## 5. 方法选择与训练留档

可选方法：

```text
random_joint, nearest_joint, highest_threat, time_to_impact,
greedy_damage, hungarian_damage, ppo, maskable_ppo
```

模型保存路径包含训练场景：

```text
models/<train_scenario>/<method>_seed<seed>.zip
```

TensorBoard 日志名称同样包含训练场景，避免不同训练分布写入同一运行目录。`runs.csv` 同时记录请求训练步数和 SB3 实际 rollout 步数。

## 6. 结果产物

| 产物 | 内容 |
| --- | --- |
| `experiment_config.json` | 场景完整参数、空间签名、算法参数、种子公式、命令和运行环境 |
| `episodes.csv` | 方法×训练场景×测试场景×种子×回合原始指标 |
| `runs.csv` | 每个配对测试块的聚合指标 |
| `summary.csv` | 方法×训练场景×测试场景的跨种子均值、标准差和 95% CI |
| `paired_differences.csv` | 相同测试块上 `method_a - method_b` 的配对差异及 95% CI |
| `generalization_matrix.csv` | 方法×训练场景×测试场景×指标的长格式泛化矩阵 |
| `learning_curves.*` | 各学习方法在自身训练场景上的学习曲线 |
| `generalization.*` | 平均奖励的训练场景×测试场景热力图 |
| `models/` | 按训练场景隔离的模型 |
| `tensorboard/` | 按训练场景和训练种子隔离的日志 |

配对差异统一定义为：

```text
difference = method_a - method_b
```

置信区间基于匹配的 `run_index` 计算，而不是把两个方法当作独立样本。

## 7. 任务五 Smoke Run

验收实验目录：

```text
results/air_defense_v1/task5_smoke_2x2
```

实验规模：

```text
训练场景：easy, medium
测试场景：easy, hard
方法：greedy_damage, maskable_ppo
训练种子：0, 1
训练预算：16 steps
最终评估：2 episodes / evaluation scenario / seed
```

产物计数：

```text
run_rows:                 16
episode_rows:             32
paired_difference_rows:  68
generalization_rows:     136
curve_rows:               12
models:                    4
TensorBoard runs:          4
```

smoke run 已生成两类 CSV 统计、学习曲线、泛化热力图、4 个模型和 4 组 TensorBoard 日志。16 步训练只用于验证工程链路，不能用于比较算法性能或形成论文结论。

## 8. 兼容性

- 原默认 `medium → medium` benchmark 仍可执行；
- 原有指标名称和含义不变；
- schema 2 的逐回合诊断字段全部保留；
- schema 3 只追加训练场景、测试场景、方法筛选、空间签名和泛化统计；
- 规则方法在多训练场景中的重复行是有意设计，用于与相应学习模型进行同块配对比较。

## 9. 验证

测试覆盖：

- 单场景旧 API 兼容；
- 多训练场景、多测试场景矩阵行数；
- 跨方法和跨训练场景的配对种子；
- 方法白名单和重复场景拒绝；
- 不兼容空间在训练前失败；
- 原始回合可重聚合；
- 配对差异、泛化矩阵、模型、日志和图表落盘。

当前全量测试：`99 passed`。
