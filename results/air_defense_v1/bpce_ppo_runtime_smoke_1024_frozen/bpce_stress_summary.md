# AirDefense v1 BPCE-PPO v0 机制压力实验

## 总结

- 总门控：`false`
- 候选塌缩场景种子数：0
- `structural_zero`：`true`
- `no_collapsed_candidate_runs`：`true`
- `all_noop_noninferiority`：`true`
- `high_threat_improvement`：`false`
- `reward_damage_safety`：`true`
- `resource_cost`：`true`
- `boundary_beats_equal_budget_random`：`false`
- `training_time_within_2x`：`true`

## 探测与训练诊断

- `bpce_probe_cumulative_probe_rollouts`：2.000000
- `bpce_probe_cumulative_selected_count`：4.000000
- `bpce_probe_cumulative_accepted_count`：1.000000
- `bpce_probe_cumulative_acceptance_rate`：0.250000
- `bpce_probe_cumulative_positive_count`：0.000000
- `bpce_probe_cumulative_negative_count`：1.000000
- `bpce_probe_cumulative_mean_abs_delta`：5.978121
- `bpce_probe_cumulative_mean_sign_agreement`：1.500000
- `bpce_probe_cumulative_effect_pass_rate`：0.750000
- `bpce_probe_cumulative_agreement_pass_rate`：0.750000
- `bpce_probe_cumulative_mean_informative_repeats`：2.500000
- `bpce_probe_cumulative_mean_opposite_repeats`：1.000000
- `bpce_probe_cumulative_extra_transitions`：2341.000000
- `training_time_ratio_vs_baseline`：1.610552

## time_pressure

- 奖励差 vs baseline：0.000000
- 损伤差 vs baseline：0.000000
- 高威胁突防差 vs baseline：0.000000
- 奖励差 vs random probe：0.000000
- 损伤差 vs random probe：0.000000

| seed | collapsed | all-noop差 | 奖励差 | 损伤差 | vs随机奖励 | vs随机损伤 |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 8 | false | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## 解释边界

该结果使用冻结种子、场景和门控。随机探测与边界探测使用相同分支预算；不得选择单个优势种子替代总体结论。
