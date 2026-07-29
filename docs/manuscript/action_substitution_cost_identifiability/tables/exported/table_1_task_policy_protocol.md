| Item | Value | Definition |
| --- | --- | --- |
| Environment | AirDefense v1 | 3 units, 5 targets, dynamic legal masks |
| Scenarios | 3 | medium, time_pressure, heterogeneity_pressure |
| Source policy | factorized_engagement_ar_ppo_order_012 | factorized joint PPO; order 0-1-2 |
| Policy seeds | 3 | 17/18/19 |
| Source models | 9 | 9/9 retained without behavior screening |
| Training | 10,000 steps/model | n_steps=256; batch=64; epochs=2 |
| Contexts | 108 | 6 safety + 6 resource per scenario-seed block |
| Resource quota | 3 missile + 3 laser | per scenario-seed resource block |
| Paired repeats | 32 | N/E CRN pairs per context |
| Target action | exact marginalization | conditional on engage |
| Continuation | stochastic | frozen actor and shared uniform tape |
