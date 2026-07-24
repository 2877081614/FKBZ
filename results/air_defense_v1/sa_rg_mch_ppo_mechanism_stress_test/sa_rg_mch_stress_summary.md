# AirDefense v1 SA-RG-MCH-PPO 机制压力实验

## 总门控

- 通过：`false`
- 候选塌缩数：5
- RG-MCH 塌缩数：2
- `structural_zero`：`true`
- `no_collapsed_candidate_runs`：`false`
- `all_noop_noninferiority`：`false`
- `reward_damage_safety`：`false`
- `high_threat_improvement`：`false`
- `resource_cost`：`true`
- `improves_mch_v0_both_scenarios`：`false`
- `reduces_rg_mch_collapse_count`：`false`
- `noncatastrophic_vs_rg_mch`：`false`

## 训练诊断

- `mch_engagement_reliability`：0.113865
- `mch_target_reliability`：0.014005
- `mch_engagement_support`：0.124429
- `mch_target_support`：0.021757
- `mch_engagement_residual_abs`：0.049448
- `mch_target_residual_abs`：0.007615
- `mch_anchor_kl`：0.017134
- `mch_anchor_penalty`：0.000000
- `mch_anchor_excess_rate`：0.000000
- `training_time_ratio_vs_baseline`：1.371148

## heterogeneity_pressure

- 奖励差 vs baseline：-12.321329
- 损伤差 vs baseline：0.341012
- 突防差 vs baseline：0.104167
- 奖励差 vs RG-MCH：-26.810793
- 损伤差 vs RG-MCH：0.661064

| seed | collapsed | all-noop差 | 突防差 | 奖励差 | 损伤差 | 成本差 |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 8 | true | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 9 | true | 1.000000 | 0.312500 | -36.963988 | 1.023037 | -15.400000 |
| 10 | true | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## time_pressure

- 奖励差 vs baseline：-30.929289
- 损伤差 vs baseline：0.745057
- 突防差 vs baseline：0.217016
- 奖励差 vs RG-MCH：-29.135610
- 损伤差 vs RG-MCH：0.621788

| seed | collapsed | all-noop差 | 突防差 | 奖励差 | 损伤差 | 成本差 |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 8 | true | 1.000000 | 0.283333 | -41.640097 | 1.040926 | -17.000000 |
| 9 | false | 0.000000 | 0.019231 | 3.378984 | -0.079370 | 0.000000 |
| 10 | true | 1.000000 | 0.348485 | -54.526753 | 1.273615 | -16.950000 |
