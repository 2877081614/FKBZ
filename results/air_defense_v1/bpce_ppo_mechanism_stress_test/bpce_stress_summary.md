# AirDefense v1 BPCE-PPO v0 机制压力实验

## 总结

- 总门控：`false`
- 候选塌缩场景种子数：2
- `structural_zero`：`true`
- `no_collapsed_candidate_runs`：`false`
- `all_noop_noninferiority`：`false`
- `high_threat_improvement`：`true`
- `reward_damage_safety`：`false`
- `resource_cost`：`false`
- `boundary_beats_equal_budget_random`：`false`
- `training_time_within_2x`：`true`

## 探测与训练诊断

- `bpce_probe_cumulative_probe_rollouts`：20.000000
- `bpce_probe_cumulative_selected_count`：40.000000
- `bpce_probe_cumulative_accepted_count`：10.500000
- `bpce_probe_cumulative_acceptance_rate`：0.262500
- `bpce_probe_cumulative_positive_count`：4.166667
- `bpce_probe_cumulative_negative_count`：6.333333
- `bpce_probe_cumulative_mean_abs_delta`：7.286359
- `bpce_probe_cumulative_mean_sign_agreement`：3.133333
- `bpce_probe_cumulative_effect_pass_rate`：0.750000
- `bpce_probe_cumulative_agreement_pass_rate`：0.962500
- `bpce_probe_cumulative_mean_informative_repeats`：5.466667
- `bpce_probe_cumulative_mean_opposite_repeats`：2.333333
- `bpce_probe_cumulative_extra_transitions`：17369.500000
- `bpce_train_cumulative_auxiliary_train_calls`：7.500000
- `bpce_train_cumulative_mean_auxiliary_loss`：0.209816
- `training_time_ratio_vs_baseline`：1.940320

## heterogeneity_pressure

- 奖励差 vs baseline：21.686369
- 损伤差 vs baseline：-0.508722
- 高威胁突防差 vs baseline：-0.129484
- 奖励差 vs random probe：-13.563061
- 损伤差 vs random probe：0.339791

| seed | collapsed | all-noop差 | 奖励差 | 损伤差 | vs随机奖励 | vs随机损伤 |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 8 | false | -1.000000 | 39.014659 | -1.079809 | 6.236755 | -0.159295 |
| 9 | true | 1.000000 | -36.963988 | 1.023037 | -75.335185 | 1.636322 |
| 10 | false | -1.000000 | 63.008436 | -1.469394 | 28.409246 | -0.457654 |

## time_pressure

- 奖励差 vs baseline：-24.953195
- 损伤差 vs baseline：0.587395
- 高威胁突防差 vs baseline：0.153458
- 奖励差 vs random probe：13.619749
- 损伤差 vs random probe：-0.313744

| seed | collapsed | all-noop差 | 奖励差 | 损伤差 | vs随机奖励 | vs随机损伤 |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 8 | false | 0.000000 | 0.995622 | -0.029536 | 3.226748 | -0.075774 |
| 9 | true | 1.000000 | -58.960953 | 1.383563 | 0.000000 | 0.000000 |
| 10 | false | 0.033333 | -16.894253 | 0.408158 | 37.632500 | -0.865456 |

## 解释边界

该结果使用冻结种子、场景和门控。随机探测与边界探测使用相同分支预算；不得选择单个优势种子替代总体结论。
