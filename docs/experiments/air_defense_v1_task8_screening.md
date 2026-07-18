# AirDefense v1.0 任务八无冲突联合动作筛选实验

更新时间：2026-07-17  
阶段状态：工程验收与 30k 筛选完成，100k 正式实验暂缓  
结果目录：`results/air_defense_v1/task8_conflict_free_screening_30k_3seeds/`

## 1. 研究问题

任务七发现 Maskable PPO 仍有约 1.6%–2.5% 的联合分配冲突，并在异质性场景中出现更高的高威胁目标突防。本实验在环境、奖励、状态表示、MLP 和 PPO 超参数不变时，将原始 `MultiDiscrete([6,6,6])` 替换为只含一对一分配的 `Discrete(136)`，检验显式无冲突动作空间能否：

1. 严格消除冲突、过度分配和非法动作；
2. 保持或改善奖励和毁伤；
3. 降低异质性场景的高威胁突防；
4. 保留时间压力场景中的资源效率。

## 2. 实验协议

协议在运行前已冻结：

```text
训练场景：medium
测试场景：medium / time_pressure / heterogeneity_pressure
方法：greedy_damage / hungarian_damage /
      maskable_ppo / conflict_free_maskable_ppo
训练种子：0 / 1 / 2
训练预算：30,000 requested steps
最终评估：50 个配对回合/场景/种子
曲线检查点：10k / 20k / 30k
统计：跨种子 Student-t 95% CI
```

运行命令：

```powershell
conda run -n rein-learning python scripts\compare_air_defense_v1_methods.py `
  --train-scenario medium `
  --eval-scenarios medium time_pressure heterogeneity_pressure `
  --methods greedy_damage hungarian_damage maskable_ppo conflict_free_maskable_ppo `
  --seeds 0 1 2 --timesteps 30000 --eval-episodes 50 `
  --eval-seed 200 --curve-eval-freq 10000 `
  --curve-eval-episodes 10 --curve-eval-seed 10000 `
  --device cpu --output-dir results\air_defense_v1 `
  --experiment-name task8_conflict_free_screening_30k_3seeds
```

## 3. 产物审计

实验正常完成并写入 schema 4：

```text
运行汇总：36 行
原始评估回合：1,800 行
配对差异：306 行
泛化统计：204 行
学习曲线：30 行
学习模型：6 个
TensorBoard 日志：6 组
```

任务八 smoke run 也已完成，目录为 `results/air_defense_v1/task8_conflict_free_smoke/`。其配置正确记录原始方法的 `MultiDiscrete([6,6,6])` 和新方法的 `Discrete(136)`，只用于工程链路验收。

## 4. 跨种子结果

下表为三种子均值。括号内为 `conflict_free_maskable_ppo - maskable_ppo`：

| 场景 | 方法 | 奖励 | 总毁伤 | 拦截率 | 高威胁突防率 | 资源成本 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| medium | Maskable PPO | -73.61 | 1.957 | 0.249 | 0.548 | 10.00 |
| medium | 无冲突 Maskable PPO | -65.38 (+8.23) | 1.716 (-0.240) | 0.317 | 0.494 | 13.66 (+3.66) |
| time_pressure | Maskable PPO | -92.38 | 2.415 | 0.269 | 0.690 | 9.92 |
| time_pressure | 无冲突 Maskable PPO | -80.27 (+12.10) | 2.135 (-0.280) | 0.361 | 0.635 | 13.42 (+3.49) |
| heterogeneity_pressure | Maskable PPO | -76.70 | 2.015 | 0.245 | 0.554 | 10.95 |
| heterogeneity_pressure | 无冲突 Maskable PPO | -64.10 (+12.60) | 1.698 (-0.317) | 0.333 | 0.490 | 12.87 (+1.92) |

无冲突方法在三个场景的非法动作率、分配冲突率和过度分配率均严格为 `0`。原始 Maskable PPO 的冲突率在三个场景分别约为 `0.87%`、`0.46%` 和 `1.30%`。

所有主要配对差异的 95% CI 都跨越 0。例如，`time_pressure` 的奖励差为 `+12.10`，95% CI `[-63.18, 87.38]`；资源成本差为 `+3.49`，95% CI `[-8.19, 15.17]`。因此这些结果是筛选趋势，不能表述为显著改进或统计等价。

## 5. 冻结门槛判定

| 预设门槛 | 观测结果 | 判定 |
| --- | ---: | --- |
| 非法动作率、冲突率、过度分配率均为 0 | 全部为 0 | 通过 |
| medium 奖励下降不超过 5 | `+8.23` | 通过 |
| medium 毁伤增加不超过 0.10 | `-0.240` | 通过 |
| time_pressure 奖励下降不超过 5 | `+12.10` | 通过 |
| time_pressure 资源成本增加不超过 0.50 | `+3.49` | **未通过** |
| heterogeneity 高威胁突防平均下降至少 0.02 | `-0.064` | 通过 |
| heterogeneity 高威胁突防至少 2/3 种子同向改善 | 2/3 | 通过 |
| heterogeneity 毁伤增加不超过 0.10 | `-0.317` | 通过 |

按运行前协议，只要任一性能门槛未通过，就不直接扩大到 100k × 5 种子。因此本轮不运行条件性正式实验，也不根据结果回溯修改 `+0.50` 的门槛。

## 6. 典型种子与机理解释

- 种子 0 在三个场景均明显改善奖励和高威胁突防，但资源成本大幅增加；
- 种子 1 效果较弱，异质性场景高威胁突防反而升高；
- 种子 2 在 `medium` 和 `time_pressure` 上奖励退化，但异质性高威胁突防改善。

结果表明 `Discrete(136)` 已解决动作协调的结构性错误，并促使策略更积极参与拦截，所以平均奖励、毁伤和高威胁突防同时改善。但它也改变了联合动作探索与策略分布，使模型使用更多弹药；`time_pressure` 的每发弹药毁伤降低量由 `0.159` 降至 `0.126`，资源效率损失并非只有指标口径造成。

由于三种子方差很大，当前证据还不足以判断联合动作约束是否已解决主要性能瓶颈，更不能据此进入 GNN。

## 7. 阶段结论与下一步

任务八的工程目标已完成，`Discrete(136)` 无冲突方法应保留为结构消融基线；但它没有通过资源效率门槛，任务八的 100k 正式实验暂缓。

下一步保持 AirDefense v1.0、奖励、MLP 和 PPO 主体不变，优先设计“逐资源选择目标并实时屏蔽已分配目标”的顺序式或自回归无冲突动作生成。该方案需要与本轮 `Discrete(136)` 和原始 Maskable PPO 做同协议的小规模筛选，先验证是否同时满足零冲突、毁伤不退化和资源效率保持，再决定是否进入 100k 正式消融。暂不实现 GNN。
