# AirDefense v1 SA-RG-MCH-PPO 机制压力实验

## 总门控

- 通过：`false`
- 候选塌缩数：0
- RG-MCH 塌缩数：2
- `structural_zero`：`true`
- `no_collapsed_candidate_runs`：`true`
- `all_noop_noninferiority`：`true`
- `reward_damage_safety`：`false`
- `high_threat_improvement`：`false`
- `resource_cost`：`true`
- `improves_mch_v0_both_scenarios`：`true`
- `reduces_rg_mch_collapse_count`：`true`
- `noncatastrophic_vs_rg_mch`：`false`

## 训练诊断

- `mch_engagement_reliability`：0.117971
- `mch_target_reliability`：0.016050
- `mch_engagement_support`：0.126521
- `mch_target_support`：0.021616
- `mch_engagement_residual_abs`：0.054388
- `mch_target_residual_abs`：0.006573
- `mch_anchor_kl`：0.000002
- `mch_anchor_penalty`：0.000000
- `mch_anchor_excess_rate`：0.000000
- `training_time_ratio_vs_baseline`：0.072912

## time_pressure

- 奖励差 vs baseline：-14.571501
- 损伤差 vs baseline：0.211009
- 突防差 vs baseline：0.133333
- 奖励差 vs RG-MCH：-51.676508
- 损伤差 vs RG-MCH：0.913090

| seed | collapsed | all-noop差 | 突防差 | 奖励差 | 损伤差 | 成本差 |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 8 | false | 0.000000 | 0.133333 | -14.571501 | 0.211009 | 0.000000 |
