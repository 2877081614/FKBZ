# AirDefense v1 RG-MCH-PPO 机制压力实验

## 总结

- 总门控：`false`
- 候选塌缩场景种子数：2
- `structural_zero`：`true`
- `no_collapsed_candidate_runs`：`false`
- `all_noop_noninferiority`：`true`
- `high_threat_improvement`：`true`
- `reward_damage_safety`：`true`
- `resource_cost`：`false`
- `improves_mch_v0_both_scenarios`：`true`

## 训练诊断

- `mch_engagement_reliability`：0.883579
- `mch_target_reliability`：0.574506
- `mch_engagement_residual_abs`：0.295173
- `mch_target_residual_abs`：0.206970
- `mch_engagement_gate_active_rate`：0.887607
- `mch_target_gate_active_rate`：0.578678
- `training_time_ratio_vs_baseline`：1.249606

## heterogeneity_pressure

- 奖励差 vs baseline：14.489464
- 损伤差 vs baseline：-0.320052
- 高威胁突防差 vs baseline：-0.081019
- 奖励差 vs MCH v0：23.028256
- 损伤差 vs MCH v0：-0.605575

| seed | collapsed | all-noop差 | 突防差 | 奖励差 | 损伤差 | 成本差 |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 8 | true | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 9 | false | 0.000000 | -0.020833 | 4.841082 | -0.172297 | 0.000000 |
| 10 | false | -0.933333 | -0.222222 | 38.627309 | -0.787857 | 3.990000 |

## time_pressure

- 奖励差 vs baseline：-1.793679
- 损伤差 vs baseline：0.123269
- 高威胁突防差 vs baseline：0.052370
- 奖励差 vs MCH v0：8.297579
- 损伤差 vs MCH v0：-0.171123

| seed | collapsed | all-noop差 | 突防差 | 奖励差 | 损伤差 | 成本差 |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 8 | false | 0.000000 | -0.150000 | 37.105007 | -0.702081 | -0.516667 |
| 9 | false | 0.000000 | 0.019231 | 0.129286 | 0.006410 | 0.000000 |
| 10 | true | 0.833333 | 0.287879 | -42.615331 | 1.065478 | -16.783333 |

## 解释边界

该结果是冻结三种子机制筛选。不得选择单个优势种子替代总体门控；只有总门控通过才允许进入更大预算实验。
