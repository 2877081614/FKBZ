# 多批次交战价值泛化与 Leave-One-Batch-Out

更新时间：2026-07-21  
实现状态：正式实验完成，OOB与最终门控未通过  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`

## 1. 方法目的

普通 state folds 无法发现同一场景内部的临界状态批次漂移。本方法建立三个完全独立的定向采样训练批次，并以整个 batch 为验证折，直接测量 engagement Critic 的批次外泛化。

## 2. 批次协议

每个训练批次包含：

```text
2 source policies * 3 scenarios * 8 states = 48 states
32 paired rollouts per branch
```

三个批次的环境采样种子、状态 ID 和反事实随机流独立。`batch_id` 直接映射到验证 fold；同一批次的任何状态都不会进入对应训练折。鲁棒候选使用 `(batch,scenario,class)` 等权块。

## 3. 数据功效

| batch | groups | reliable | engage | no-op |
| --- | ---: | ---: | ---: | ---: |
| 701000 | 68 | 63 | 24 | 39 |
| 719000 | 63 | 61 | 21 | 40 |
| 737000 | 62 | 59 | 23 | 36 |

全部批次通过双类与逐场景功效，且观测重叠为0。因而 OOB 失败不能归因于某个批次没有 engage 或 no-op。

## 4. OOB 结果

最终选择 `scenario_robust_reliable_cost`：

| seed | BA | engage recall | no-op recall | worst batch | worst scenario | 可行 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20 | 0.815 | 0.838 | 0.791 | 0.650 | 0.621 | 否 |
| 21 | 0.819 | 0.838 | 0.800 | 0.667 | 0.690 | 是 |
| 22 | 0.817 | 0.912 | 0.722 | 0.475 | 0.586 | 否 |

只有seed21满足全部批次与场景约束。高总体 BA 不能保证每个留出批次的双类边界稳定。

## 5. 第四批次结果

新批次上三种子 engage recall 为 `0.829/0.943/0.886`，no-op recall 为 `0.500/0.477/0.545`。多批次训练消除了此前异质场景 engage recall `0.273/0.182` 的安全漏判，却使所有场景出现资源停止不足。

这说明多批次覆盖改变了错误方向，但没有解决安全-资源权衡：分类目标在跨批次不确定性下倾向保护 engage recall，最终形成系统性激进边界。

## 6. 当前边界

现有证据支持：

- 批次多样性确实改善必要交战泛化；
- 网络可以表示安全收益的主要方向；
- 单一零分数边界不能稳定满足批次级双约束；
- 继续增加相同类型批次不能保证恢复 no-op。

下一步应在 OOB 预测上构造安全-资源 Pareto 前沿，先判断是否存在满足所有批次/场景约束的统一边界。若不存在，需要显式双约束或场景条件校准；若存在，再在新批次一次性确认。

## 7. 实现入口

```text
rein_learning/common/multibatch_diagnostics.py
scripts/run_air_defense_v1_task14_multibatch_leave_one_out.py
tests/test_air_defense_v1_task14_multibatch_leave_one_out.py
docs/experiments/air_defense_v1_task14_multibatch_leave_one_out.md
```
