# AirDefense v1 RG-MCH-PPO 机制压力实验

## 总结

- 总门控：`false`
- 候选塌缩场景种子数：1
- `structural_zero`：`true`
- `no_collapsed_candidate_runs`：`false`
- `all_noop_noninferiority`：`false`
- `high_threat_improvement`：`false`
- `reward_damage_safety`：`false`
- `resource_cost`：`true`
- `improves_mch_v0_both_scenarios`：`true`

## 训练诊断

- `mch_engagement_reliability`：0.933897
- `mch_target_reliability`：0.638914
- `mch_engagement_residual_abs`：0.346697
- `mch_target_residual_abs`：0.211282
- `mch_engagement_gate_active_rate`：0.928357
- `mch_target_gate_active_rate`：0.646000
- `training_time_ratio_vs_baseline`：0.063824

## time_pressure

- 奖励差 vs baseline：-35.168019
- 损伤差 vs baseline：0.794190
- 高威胁突防差 vs baseline：0.133333
- 奖励差 vs MCH v0：6.472078
- 损伤差 vs MCH v0：-0.246736

| seed | collapsed | all-noop差 | 突防差 | 奖励差 | 损伤差 | 成本差 |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 8 | true | 1.000000 | 0.133333 | -35.168019 | 0.794190 | -17.000000 |

## 解释边界

该结果是冻结三种子机制筛选。不得选择单个优势种子替代总体门控；只有总门控通过才允许进入更大预算实验。
