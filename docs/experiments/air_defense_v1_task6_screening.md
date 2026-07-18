# AirDefense v1.0 任务六：三种子筛选实验诊断报告

更新时间：2026-07-17  
实验性质：算法与场景筛选，不作为最终论文结果  
结果目录：`results/air_defense_v1/task6_screening_medium_20k_3seeds/`

## 1. 实验目的

本实验回答三个问题：

1. 动作掩码、训练预算和场景压力分别在何处形成算法分水岭；
2. 当前 MLP PPO、Maskable PPO、greedy 和 Hungarian 的典型失效模式是什么；
3. 哪些场景值得进入 5 种子、100k 训练步的正式实验。

本轮结果只用于筛选。3 个训练种子和 20k 训练步不足以支持最终算法优劣结论。

## 2. 实验协议

| 项目 | 配置 |
| --- | --- |
| 训练场景 | `medium` |
| 测试场景 | `easy`、`medium`、`hard`、5 个单因素压力场景 |
| 方法 | `greedy_damage`、`hungarian_damage`、`ppo`、`maskable_ppo` |
| 训练种子 | `0, 1, 2` |
| 请求训练步数 | 20,000 |
| SB3 实际训练步数 | 20,224 |
| 曲线检查点 | 每 5,000 步，10 回合/检查点 |
| 最终评估 | 50 个配对回合/场景/种子 |
| 设备 | CPU |

执行命令：

```powershell
conda run -n rein-learning python scripts\compare_air_defense_v1_methods.py `
  --train-scenario medium `
  --eval-scenarios easy medium hard time_pressure resource_pressure `
    intercept_uncertainty damage_pressure heterogeneity_pressure `
  --methods greedy_damage hungarian_damage ppo maskable_ppo `
  --seeds 0 1 2 `
  --timesteps 20000 `
  --eval-episodes 50 `
  --eval-seed 200 `
  --curve-eval-freq 5000 `
  --curve-eval-episodes 10 `
  --curve-eval-seed 10000 `
  --device cpu `
  --experiment-name task6_screening_medium_20k_3seeds
```

## 3. 完整性审计

| 产物 | 实际数量 | 期望数量 | 状态 |
| --- | ---: | ---: | --- |
| `runs.csv` | 96 | 96 | 完整 |
| `episodes.csv` | 4,800 | 4,800 | 完整 |
| `paired_differences.csv` | 816 | 816 | 完整 |
| `generalization_matrix.csv` | 544 | 544 | 完整 |
| `learning_curves.csv` | 36 | 36 | 完整 |
| 模型 | 6 | 6 | 完整 |
| TensorBoard 运行 | 6 | 6 | 完整 |

96 个运行块来自 `1 个训练场景 × 3 个种子 × 8 个测试场景 × 4 种方法`。规则方法和学习方法在同一 `run_index`、同一测试场景内使用相同的评估种子块。

## 4. 总体结果

### 4.1 平均回合奖励

奖励越大越好，即数值越接近 0 越好。

| 测试场景 | Greedy | Hungarian | PPO | Maskable PPO |
| --- | ---: | ---: | ---: | ---: |
| `easy` | -16.72 | -16.34 | -43.12 | -37.91 |
| `medium` | -37.05 | -36.30 | -92.14 | -74.03 |
| `hard` | -109.16 | -108.67 | -122.71 | -124.93 |
| `time_pressure` | -43.94 | -43.39 | -115.97 | -94.68 |
| `resource_pressure` | -64.09 | -64.45 | -95.29 | -79.86 |
| `intercept_uncertainty` | -51.27 | -50.99 | -91.22 | -79.12 |
| `damage_pressure` | -75.44 | -76.37 | -120.44 | -105.11 |
| `heterogeneity_pressure` | -49.69 | -50.11 | -94.31 | -80.44 |

### 4.2 关键诊断指标

| 场景/方法 | 拦截率 | 高威胁突防率 | 平均损伤 | 平均弹药 | 成功率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `medium` / Hungarian | 0.529 | 0.271 | 0.974 | 15.81 | 0.060 |
| `medium` / Maskable PPO | 0.223 | 0.539 | 1.941 | 8.98 | 0.000 |
| `time_pressure` / Hungarian | 0.600 | 0.381 | 1.339 | 15.61 | 0.133 |
| `time_pressure` / Maskable PPO | 0.247 | 0.680 | 2.456 | 8.25 | 0.000 |
| `hard` / Hungarian | 0.335 | 0.520 | 2.836 | 11.00 | 0.000 |
| `hard` / Maskable PPO | 0.144 | 0.573 | 3.136 | 6.65 | 0.000 |

Maskable PPO 使用较少弹药，但同时具有更低拦截率、更高损伤和更高突防率。因此本轮不能把“少用弹药”解释为资源保留优势，它主要表现为交战不足。

## 5. 算法分水岭

### 5.1 动作约束分水岭：普通 PPO 塌缩为全 no-op

普通 PPO 在未训练检查点平均每回合产生 117.9 次非法动作。训练到 5,000 步后，非法动作降为 0，但三个种子同时变成：

```text
平均弹药 = 0
平均射击 = 0
拦截率 = 0
成功率 = 0
```

最终 8 个场景共 1,200 个 PPO 评估回合中，没有一个回合发生射击。PPO 学到的不是合法分配策略，而是通过始终选择 no-op 规避非法动作惩罚。由此可确认：动作掩码是当前联合离散动作任务的必要约束，不是可有可无的性能技巧。

### 5.2 训练预算分水岭：20k 尚未使 Maskable PPO 稳定

Maskable PPO 的跨种子曲线没有形成持续上升趋势。`seed=0` 在 10k 步时达到奖励 -47.15、弹药 15.0，随后退化到 20,224 步的奖励 -84.83、弹药 1.8。最终 `medium` 结果为：

| 训练种子 | 奖励 | 拦截率 | 平均弹药 | 解释 |
| ---: | ---: | ---: | ---: | --- |
| 0 | -83.10 | 0.092 | 1.98 | 陷入低交战策略 |
| 1 | -69.51 | 0.304 | 13.94 | 保持主动交战 |
| 2 | -69.49 | 0.272 | 11.02 | 保持主动交战，但曲线波动明显 |

已有 100k × 5 种子基准中，Maskable PPO 在 `medium` 的平均奖励为 -35.93。两个实验的评估块不同，不能做严格配对推断，但结果共同说明 20k 更适合暴露早期训练失效，不适合代表 Maskable PPO 的最终能力。

### 5.3 场景分水岭：时间压力最清晰

下表给出 `Hungarian - Maskable PPO` 的配对奖励差；正值表示 Hungarian 更好。

| 测试场景 | 平均差 | 95% CI | 筛选判断 |
| --- | ---: | --- | --- |
| `easy` | +21.57 | [9.65, 33.50] | 有稳定差异 |
| `medium` | +37.73 | [9.16, 66.31] | 有稳定差异 |
| `hard` | +16.27 | [5.20, 27.33] | 有差异，但接近共同失效区 |
| `time_pressure` | +51.30 | [12.35, 90.24] | 最大且可解释的算法分水岭 |
| `resource_pressure` | +15.41 | [-11.31, 42.12] | 种子方差过大，暂不稳定 |
| `intercept_uncertainty` | +28.13 | [0.38, 55.87] | 边界显著，值得复验 |
| `damage_pressure` | +28.74 | [-0.10, 57.58] | 接近显著；Greedy 对比显著 |
| `heterogeneity_pressure` | +30.33 | [6.86, 53.80] | 有稳定差异，适合表示能力研究 |

`time_pressure` 中规则方法仍保持约 0.60 的拦截率和 0.13 的成功率，而 Maskable PPO 的高威胁突防率升至 0.68。这表明场景并非整体不可完成，主要差异来自学习策略在有限决策窗口内的目标选择和交战不足。

### 5.4 即时优化分水岭尚未出现

Greedy 与 Hungarian 在 7 个场景的奖励配对置信区间均跨 0。唯一不跨 0 的场景是 `heterogeneity_pressure`，但差异为 `Greedy - Hungarian = +0.419`，95% CI 为 `[0.128, 0.710]`，效应很小且方向反而偏向 Greedy。

当前 v1.0 的一对一即时收益结构没有产生足够强的组合分配冲突，Hungarian 尚未表现出相对 Greedy 的长时域优势。它仍应作为优化正确性基线保留，但不能宣称其性能优于 Greedy。

## 6. 典型失效回合

以下回合可在 `episodes.csv` 中由 `method + eval_scenario + train_seed + episode_seed` 唯一复现。

| 类型 | 方法/场景 | 标识 | 观测结果 | 诊断 |
| --- | --- | --- | --- | --- |
| no-op 塌缩 | PPO / `medium` | train seed 1, episode seed 695 | 0 射击、0 拦截、4 突防、3 个高威胁突防、损伤 4.081、奖励 -150.93 | 用不行动规避非法动作惩罚 |
| 低交战陷阱 | Maskable PPO / `time_pressure` | train seed 0, episode seed 375 | 1 射击、1 拦截、4 突防、2 个高威胁突防、损伤 3.730、奖励 -137.78 | 动作合法，但资源调用严重不足 |
| 高损伤失效 | Maskable PPO / `damage_pressure` | train seed 0, episode seed 514 | 3 射击、0 拦截、3 突防且均为高威胁、损伤 4.564、奖励 -171.42 | 未形成高价值目标优先级 |
| 场景可行性边界 | Greedy / `hard` | episode seed 309 | 11 射击、1 拦截、4 突防、3 个高威胁突防、损伤 5.073、奖励 -185.77 | 强规则也失败，说明该回合接近资源与时间联合上限 |

`hard` 场景中全部方法成功率均为 0，但规则方法仍能拦截约三分之一目标并显著减少损伤。因此它是“任务成功指标接近不可达”的可行性边界，而不是所有动作都无效的严格不可完成场景。后续应以损伤、拦截率和高威胁突防率为主指标，不能只看成功率。

## 7. 场景筛选结论

### 正式实验保留

- `medium`：训练分布和基准锚点；
- `time_pressure`：最清晰的策略能力分水岭；
- `heterogeneity_pressure`：检验资源-目标关系表示的主要场景；
- `intercept_uncertainty`：检验随机转移鲁棒性的次要场景；
- `damage_pressure`：检验高价值目标优先级和奖励敏感性的次要场景。

### 仅作边界或校验

- `easy`：任务可学习性和基本行为校验；
- `hard`：可行性边界与压力上限，不以成功率作为主要指标；
- `resource_pressure`：当前差异受种子方差和低交战地板效应影响，暂不作为首批主场景。

### 方法选择

- 正式主对比保留 `greedy_damage`、`hungarian_damage`、`maskable_ppo`；
- 普通 `ppo` 只保留为动作掩码消融，不再把它当作有竞争力的主算法；
- 现阶段不能进入正式 GNN 优劣对比。`heterogeneity_pressure` 的差异仍可能来自 20k 训练不足，需要先完成稳定预算下的 MLP 对照。

## 8. 任务七建议协议

1. 在 `medium` 训练 Maskable PPO 100k 步，运行 5 个种子；
2. 正式评估优先覆盖 `medium/time_pressure/heterogeneity_pressure`，再加入 `intercept_uncertainty` 和 `damage_pressure`；
3. 每个种子、每个测试场景使用 100 个配对回合；
4. 预先固定 10k、25k、50k、75k、100k 检查点，并增加曲线评估回合数；
5. 主结果使用预先声明的 100k 最终模型，另报告独立验证种子选择的最佳检查点作为稳定性分析，避免事后挑选；
6. 同时报告奖励、损伤、拦截率、高威胁突防率、资源效率和决策耗时；
7. 只有当 100k 下 MLP Maskable PPO 在异质性场景仍表现出稳定表示瓶颈，才进入 Graph Maskable PPO。

## 9. 结论

任务六已经找到三个层次的分水岭：

```text
动作合法性：普通 PPO -> no-op 策略塌缩
训练稳定性：20k Maskable PPO -> 跨种子低交战陷阱
场景压力：time_pressure / heterogeneity_pressure -> 规则与学习策略稳定分离
```

同时，`hard` 暴露的是场景可行性边界，Greedy 与 Hungarian 尚未形成有实际意义的性能分离。项目可以进入任务七的 5 种子正式实验，但应先完成稳定的 MLP Maskable PPO 对照，暂不直接转入 GNN 算法实现。

