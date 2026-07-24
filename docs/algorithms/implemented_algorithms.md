# Implemented Algorithms

This project currently includes baseline algorithms for discrete-action reinforcement learning.

## Q-learning

Files:

- `rein_learning/algorithms/tabular/q_learning.py`
- `rein_learning/agents/tabular_q_agent.py`
- `rein_learning/trainers/grid_world_q_learning.py`
- `scripts/train_q_learning_grid_world.py`

Core idea:

```text
Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
```

The implementation uses an epsilon-greedy policy and a Q-table.

Run:

```powershell
conda run -n rein-learning python scripts\train_q_learning_grid_world.py
```

## DQN

Files:

- `rein_learning/models/q_network.py`
- `rein_learning/buffers/replay_buffer.py`
- `rein_learning/agents/dqn_agent.py`
- `rein_learning/trainers/grid_world_dqn.py`
- `scripts/train_dqn_grid_world.py`

Core components:

- online Q-network
- target Q-network
- replay buffer
- epsilon-greedy exploration
- TD target with MSE loss

The current Q-network encodes discrete state ids as one-hot vectors before passing them through an MLP. This keeps the first DQN implementation simple and suitable for `SmallGridWorldEnv`.

Run:

```powershell
conda run -n rein-learning python scripts\train_dqn_grid_world.py
```

## Masked Vector DQN for AirDefenseEnv

Files:

- `rein_learning/models/q_network.py`
- `rein_learning/buffers/replay_buffer.py`
- `rein_learning/agents/dqn_agent.py`
- `rein_learning/trainers/air_defense_dqn.py`
- `scripts/train_air_defense_dqn.py`

Purpose:

```text
Train DQN on vector observations from AirDefenseResourceAssignmentEnv v0.
```

Key differences from the GridWorld DQN:

- uses `VectorQNetwork` instead of one-hot discrete-state encoding
- stores vector observations in `VectorReplayBuffer`
- uses `action_mask` during epsilon-greedy action selection
- masks illegal next actions when computing the DQN target

Run:

```powershell
conda run -n rein-learning python scripts\train_air_defense_dqn.py
```

## REINFORCE

Files:

- `rein_learning/algorithms/policy_gradient/reinforce.py`
- `rein_learning/models/policy_network.py`
- `rein_learning/agents/reinforce_agent.py`
- `rein_learning/trainers/grid_world_reinforce.py`
- `scripts/train_reinforce_grid_world.py`

Core idea:

```text
J(theta) = E[G_t * log pi_theta(a_t | s_t)]
theta <- theta + alpha * gradient_theta J(theta)
```

The implementation uses a stochastic categorical policy over discrete actions,
discounted Monte Carlo returns, optional return normalization, and SGD.
`REINFORCEAgent.update_episode` performs one on-policy policy-gradient update
and returns the scalar loss.

Run:

```powershell
conda run -n rein-learning python scripts\train_reinforce_grid_world.py
```

## AirDefense v1 PPO and Optimization Baselines

The AirDefense v1.0 experiment stack includes PPO, Maskable PPO, five heuristic
joint-action policies, and the `hungarian_damage` one-step global assignment
baseline.

Hungarian baseline details:

- [hungarian_damage_reduction_baseline.md](hungarian_damage_reduction_baseline.md)
- `rein_learning/baselines/air_defense_v1.py`
- `tests/test_air_defense_v1_hungarian.py`

Run all six non-learning baselines:

```powershell
conda run -n rein-learning python scripts\evaluate_air_defense_v1_baselines.py
```

## Conflict-Free Maskable PPO for AirDefense v1

`conflict_free_maskable_ppo` wraps the unchanged AirDefense v1.0 environment
with a deterministic `Discrete(136)` action space containing only one-to-one
unit-target assignments. A dynamic joint mask combines this static conflict
constraint with the base environment's ammunition, cooldown, range, and target
validity masks.

Files:

- `rein_learning/envs/air_defense_v1/wrappers/conflict_free_joint_action.py`
- `rein_learning/trainers/air_defense_v1_ppo.py`
- `rein_learning/experiments/air_defense_v1_benchmark.py`
- `tests/test_air_defense_v1_conflict_free_actions.py`
- [conflict_free_joint_action_masking.md](conflict_free_joint_action_masking.md)

The 30k by three-seed screening eliminated conflicts and overkill, but did not
pass the frozen time-pressure resource-cost gate. It remains a structural
ablation baseline; its conditional 100k experiment was not run.

## Autoregressive Maskable PPO for AirDefense v1

`autoregressive_maskable_ppo` keeps the original `MultiDiscrete([6,6,6])`
environment interface but samples unit actions in fixed order. Targets selected
by earlier units are removed from later conditional masks, and the joint log
probability is the sum of the three conditional log probabilities.

Files:

- `rein_learning/models/autoregressive_action_head.py`
- `rein_learning/algorithms/policy_gradient/autoregressive_ppo.py`
- `tests/test_autoregressive_joint_action.py`
- [autoregressive_conflict_free_policy.md](autoregressive_conflict_free_policy.md)

The task-nine screening retained exactly zero conflicts while reducing
`time_pressure` resource cost below both the original Maskable PPO and the
`Discrete(136)` baseline. Its heterogeneous high-threat leak improvement was
`0.01483`, below the frozen `0.02` gate, so the conditional 100k experiment was
not run.

## Autoregressive Unit-Order Ablation and Decision Diagnostics

Task ten parameterizes the autoregressive unit order and registers three
cyclic variants: `012`, `120`, and `201`. Experiment schema version 6 records
unit-level decisions, conditional legal opportunities, assignment quality,
and mutually exclusive high-threat leak attributions during final evaluation.

Files:

- `rein_learning/common/air_defense_v1_decision_metrics.py`
- `scripts/diagnose_air_defense_v1_task9_models.py`
- `scripts/analyze_air_defense_v1_task10.py`
- [autoregressive_order_ablation.md](autoregressive_order_ablation.md)

The 30k by three-seed screening found that order `201` reduced heterogeneous
high-threat leak by `0.021729`, but raised time-pressure resource cost by
`4.253` relative to order `012`. No fixed-order candidate passed all frozen
gates, so the conditional 100k confirmation was not run.

## Role-Conditioned Autoregressive PPO for AirDefense v1

Task eleven replaces independent positional action logits with shared zone,
target, and unit encoders, a permutation-equivariant unit-target pair scorer,
and a shared no-op scorer. The actor is capacity matched to task ten and the
critic remains unchanged.

Files:

- `rein_learning/models/air_defense_observation_layout.py`
- `rein_learning/models/air_defense_role_conditioned_action_head.py`
- `rein_learning/algorithms/policy_gradient/role_conditioned_autoregressive_ppo.py`
- `scripts/analyze_air_defense_v1_task11.py`
- [role_conditioned_autoregressive_policy.md](role_conditioned_autoregressive_policy.md)

The 30k by three-seed screening retained zero structural conflicts and strong
resource efficiency, but did not remove seed-level all-no-op collapse. The
canonical method had five heterogeneous collapsed unit-runs, 94.8% unassigned
high-threat leaks, and 73.51% higher decision latency. The 100k confirmation
was not run; the next research target is no-op probability and PPO optimization
stability rather than GNN capacity.

## Task 14: Masked Action Q-Critic

Task fourteen adds a non-graph action-conditioned value model for the frozen
autoregressive policy. It estimates `Q(s, h_i, a_i)` from the observation,
selected unit and candidate action, entity-relation features, prefix target
occupancy, and the conditional legal mask. Counterfactual labels fix earlier
unit actions, replace the current action, resample later actions under the
modified prefix, and reuse common environment and policy random seeds.

The implementation also adds grouped state splits, paired-uncertainty ranking,
top-action and engage/no-op sign diagnostics. The formal gate showed a
36.4%-40.1% MAE improvement over `V(s)`, but only 8 high-confidence action
pairs and 0/3 passing training seeds. The model is therefore retained as an
offline diagnostic prototype and is not connected to PPO training.

The Task 14 refinement keeps the same network and adds group-centered and
reliability-weighted pairwise difference losses. On a fresh 36-state,
32-rollout test set, mean ranking accuracy improved by 0.167 over absolute MSE
without MAE degradation. Engagement/no-op sign accuracy remained 0.545 and
hierarchical effective counts stayed below the frozen gates, so PPO integration
remains disabled.

```text
rein_learning/models/masked_action_q_critic.py
rein_learning/common/q_critic_diagnostics.py
scripts/run_air_defense_v1_task14_q_critic.py
```

## Task 14: Hierarchical Masked Q-Critic

The hierarchical diagnostic separates the binary engagement value from the
conditional target value. Its engagement head predicts `[Q_noop, Q_engage]`
from the state, unit, prefix occupancy, unit features, and legal mask. Its
target head reuses the masked action critic but is trained only on legal target
actions. Training combines absolute, group-centered, and reliability-weighted
pairwise losses while keeping the environment and policies frozen.

On a fresh 108-state, 32-rollout formal test, conditional-target ranking
reached 0.830-0.870 and improved by 0.057 on average over the monolithic
baseline. Engagement sign accuracy fell to 0.588-0.706, 0.255 below the
baseline on average, and target MAE worsened by roughly 17%-21%. All three
seeds failed the complete gate. The model therefore establishes target-layer
learnability but is retained only as an offline diagnostic; PPO and GNN
integration remain disabled.

```text
rein_learning/models/hierarchical_masked_q_critic.py
rein_learning/common/hierarchical_q_diagnostics.py
scripts/run_air_defense_v1_task14_hierarchical_q.py
docs/algorithms/hierarchical_masked_q_critic.md
docs/experiments/air_defense_v1_task14_hierarchical_q.md
```

## Task 14: Risk-Aware Engagement Utility Critic

This diagnostic decomposes paired no-op/engage rollouts into operational
return, discounted resource cost, discounted damage, high-threat leaks, and
shots. A validation-frozen utility combines explicit cost and damage weights
with lower-tail CVaR, while an independent safety-resource oracle labels
necessary engagement and wasteful engagement without using the candidate
utility as its own target.

On a fresh 108-state, 32-rollout dataset, the selected utility improved test
balanced accuracy from 0.713 to 0.926, reduced false-noop from 0.333 to zero,
and reduced wasteful-engage from 0.241 to 0.148. Only 3 of 57 reliable test
groups were oracle-engage, however, and three regression critics reached only
0.398/0.435/0.435 balanced accuracy. The utility direction is retained, but
PPO integration remains disabled pending targeted critical-state collection
and class-balanced sign estimation.

```text
rein_learning/models/risk_aware_engagement_critic.py
rein_learning/common/engagement_utility_diagnostics.py
scripts/run_air_defense_v1_task14_engagement_utility.py
scripts/analyze_air_defense_v1_task14_engagement_power.py
docs/algorithms/risk_aware_engagement_critic.md
docs/experiments/air_defense_v1_task14_engagement_utility.md
```

## Task 14: Critical-State Balanced Engagement Sign Critic

This stage replaces uniform state sampling with a pre-decision criticality
score based on damage potential, hit probability, threat, and time to impact.
It also replaces absolute utility regression with class-balanced BCE and an
optional pairwise margin on the engage-minus-noop logit. Historical test rows
remain excluded; only historical train/validation rows augment training.

Targeted test engagement prevalence increased from 5.3% to 37.8%, with 28
reliable engage and 46 reliable no-op groups across all three core scenarios.
Validation selected BCE+margin. Its three test balanced accuracies were
0.758/0.711/0.708 and false-noop fell to 0.071-0.214. Wasteful engagement
worsened for two seeds, and time-pressure no-op recall was only
0.455/0.182/0.273. The estimator therefore remains offline pending explicit
resource-constrained decision-boundary calibration.

```text
rein_learning/common/critical_engagement_sampling.py
rein_learning/common/balanced_engagement_training.py
scripts/run_air_defense_v1_task14_balanced_engagement.py
docs/algorithms/balanced_engagement_sign_critic.md
docs/experiments/air_defense_v1_task14_balanced_engagement.md
```

## Task 14: Resource-Constrained Engagement Boundary

This stage freezes the balanced BCE+margin critics and calibrates their
engage-minus-noop logits with either a global threshold or a resource-aware
dual boundary. Resource pressure combines normalized unit cost and remaining
ammunition; all parameters are selected on the previous validation split and
tested once on 72 fresh critical states.

The independent dataset passed all integrity and power gates with 31 engage
and 50 no-op labels. Neither boundary family had a feasible validation
solution. The selected resource-dual family reached only
0.593/0.612/0.605 balanced accuracy and 0.38/0.32/0.34 no-op recall on test,
for 0/3 passing seeds. Scalar cost-ammo calibration is retained as a negative
baseline; PPO and GNN integration remain disabled pending a state-conditioned
budget or explicit constrained-value mechanism.

```text
rein_learning/common/engagement_boundary_calibration.py
scripts/run_air_defense_v1_task14_engagement_calibration.py
docs/algorithms/resource_constrained_engagement_boundary.md
docs/experiments/air_defense_v1_task14_engagement_calibration.md
```

## Task 14: State-Conditioned Constrained Engagement Value

This stage replaces scalar cost-ammunition calibration with explicit paired
safety-gain and incremental-cost heads plus a non-negative state-conditioned
resource multiplier. Three-fold grouped cross-fitting compares safety-only,
global-budget, and state-budget variants without reusing a fixed test split.

Cross-fitting selected state-budget with 2/3 feasible seeds. On 72 fresh
states, all three seeds passed overall balanced accuracy, class recalls,
error non-inferiority, safety-sign, and inference gates. Balanced accuracy was
0.834/0.776/0.768 and wasteful engagement fell to 0.298/0.281/0.263. Local
scenario recalls still failed for every seed, so PPO integration remains
disabled for one final cross-scenario robustness refinement.

```text
rein_learning/models/state_conditioned_engagement_value.py
rein_learning/common/state_conditioned_value_training.py
scripts/run_air_defense_v1_task14_state_conditioned_value.py
docs/algorithms/state_conditioned_engagement_value.md
docs/experiments/air_defense_v1_task14_state_conditioned_value.md
```

## Task 14: Cross-Scenario Robust Engagement Value

This stage keeps the state-budget architecture fixed and compares standard
training with equal scenario-class blocks, a worst-block penalty, and paired
cost-delta reliability weights. Model selection remains three-fold grouped
cross-fitting and the formal evaluation uses 72 fresh states.

The robust variants did not dominate standard training out of fold. Feasible
seed counts were 2/3 for standard, 2/3 for scenario-robust, and 1/3 for the
reliable-cost variant, so standard remained selected. On the new test batch,
seeds 21/22 fell to 0.273/0.182 heterogeneous engage recall. This reverses the
previous batch's error direction and identifies within-scenario critical-state
batch shift, rather than average scenario weighting, as the remaining blocker.

```text
rein_learning/common/state_conditioned_value_training.py
scripts/run_air_defense_v1_task14_cross_scenario_robust_value.py
docs/algorithms/cross_scenario_robust_engagement_value.md
docs/experiments/air_defense_v1_task14_cross_scenario_robust_value.md
```

## Task 14: Multi-Batch Leave-One-Out Generalization

This stage generates three independent critical-state training batches and
uses each `batch_id` as a held-out fold. The same state-budget architecture is
trained with standard, scenario-robust, and reliable-cost robust objectives.
A fourth unseen batch is reserved for final evaluation.

All batch independence and oracle power gates passed. The selected objective
was feasible in only 1/3 leave-one-batch-out seeds and 0/3 final-test seeds.
Final engage recall recovered to 0.829/0.943/0.886, while no-op recall fell to
0.500/0.477/0.545. This identifies systematic over-engagement, rather than
insufficient data or heterogeneous-scenario under-engagement, as the current
blocker. MCH-PPO and GNN integration remain disabled pending a no-new-rollout
Pareto feasibility audit.

```text
rein_learning/common/multibatch_diagnostics.py
scripts/run_air_defense_v1_task14_multibatch_leave_one_out.py
docs/algorithms/multibatch_engagement_value_generalization.md
docs/experiments/air_defense_v1_task14_multibatch_leave_one_out.md
```

## Task 14: OOB Safety-Stop Pareto Calibration

This stage reuses frozen leave-one-batch-out predictions and enumerates every
distinct binary decision boundary induced by each continuous score. It audits
pooled, per-batch, and per-scenario engage/no-op recall without new rollouts or
final-test access.

At zero threshold, the selected reliable-cost objective was feasible for only
1/3 seeds. Seed-specific robust calibration found 23/20/2 feasible thresholds
for seeds 20/21/22, so the preregistered gate passed at 3/3. Seed 22 retained
zero minimum margin and a narrow feasible interval; a shared raw threshold
passed at most 2/3 seeds. The result identifies usable ranking with seed-scale
drift, not yet stable independent generalization.

```text
rein_learning/common/pareto_feasibility.py
scripts/run_air_defense_v1_task14_oob_pareto_audit.py
docs/algorithms/oob_safety_stop_pareto_calibration.md
docs/experiments/air_defense_v1_task14_oob_pareto_audit.md
```

## Task 14: Frozen OOB Calibration Independent Confirmation

This stage freezes the selected objective, three final checkpoints, and the
seed-specific OOB thresholds before generating one new 72-state batch. It
performs no retraining or threshold search and audits overlap against all
historical task-14 datasets.

Data integrity and power passed, with 81 reliable groups and zero overlap
against 19 historical datasets. Frozen thresholds passed 0/3 seeds. Seeds
20/21 retained engage recall but fell to 0.478/0.435 no-op recall; seed 22
reached 0.565 no-op recall but fell to 0.364 worst-scenario engage recall.
Safety-sign accuracy remained above 0.74, so the remaining failure is unstable
cross-batch score calibration and constrained decision semantics. MCH-PPO
remains frozen.

```text
rein_learning/common/independent_confirmation.py
scripts/run_air_defense_v1_task14_independent_confirmation.py
docs/experiments/air_defense_v1_task14_independent_confirmation.md
```

## Task 14: Cross-Batch Probability and Uncertainty Calibration

This stage fits per-seed weighted logistic calibrators with equal
batch-scenario-class mass and derives conservative decisions from the inverse
Hessian prediction error. Four score/value and confidence-bound candidates are
evaluated with outer leave-one-batch-out validation before any new rollout.

All candidates were feasible for 0/3 seeds. Score-only Platt retained 0.781
mean balanced accuracy, but worst-batch no-op recall was 0.550/0.333/0.475.
Value-context standard errors rose to 1.235-1.797, and confidence bounds traded
over-engagement for complete engage failure in one held batch. The independent
batch gate was therefore not opened. Linear calibration is retained only as a
diagnostic; MCH-PPO remains frozen pending an explicit multi-constraint value
representation.

```text
rein_learning/common/cross_batch_calibration.py
scripts/run_air_defense_v1_task14_cross_batch_calibration.py
docs/algorithms/cross_batch_uncertainty_calibration.md
docs/experiments/air_defense_v1_task14_cross_batch_calibration.md
```

## MCH-PPO Mechanism Stress Prototype

`MaskedCounterfactualHierarchicalPPO` is now implemented as an exploratory
online optimizer. It uses a frozen hierarchical Q-Critic ensemble to construct
masked per-unit engagement and conditional-target advantages, then applies
separate PPO ratios and clipping to both factors. Joint GAE remains the value
function target.

The preregistered 10k, three-seed, two-scenario stress test failed its mechanism
gate. Three of six matched candidate runs collapsed to no-op, and mean
high-threat leak and damage both increased in both core scenarios. The code is
therefore an implemented research prototype, not a validated algorithm. A
single successful `time_pressure/seed9` run must not be selected as proof of
general superiority.

```text
rein_learning/algorithms/policy_gradient/mch_ppo.py
scripts/run_air_defense_v1_mch_ppo_stress_test.py
docs/experiments/air_defense_v1_mch_ppo_mechanism_stress_test.md
```

## Reliability-Gated MCH-PPO

`ReliabilityGatedMCHPPO` keeps normalized on-policy GAE as the actor's global
credit and adds bounded counterfactual residuals weighted by critic-ensemble
directional agreement. It records factor-level reliability, residual magnitude,
and gate activation in the unified benchmark rows.

The frozen 10k, three-seed, two-scenario test improved reward and damage over
MCH-PPO v0 in both scenarios. Heterogeneity pressure also improved reward by
14.49 and damage by 0.320 versus factorized PPO. The overall gate still failed:
two of six matched runs collapsed, and heterogeneity resource cost reached
125.9% of baseline. Mean engagement reliability/gate activation was
0.884/0.888, showing that ensemble agreement is overconfident under shared
distribution shift. RG-MCH is a positive mechanism result but not yet a
validated final algorithm.

```text
rein_learning/algorithms/policy_gradient/mch_ppo.py
scripts/run_air_defense_v1_rg_mch_ppo_stress_test.py
docs/experiments/air_defense_v1_rg_mch_ppo_stress_test.md
```

## Support-Anchored RG-MCH-PPO

`SupportAnchoredRGMCHPPO` adds nearest-neighbor support from the Q-Critic train
split and a cumulative engagement KL constraint relative to the initial actor.
The support context includes observation, unit identity, prefix occupancy, legal
mask, and the selected target for conditional target support.

The frozen 10k stress test failed with five of six matched runs collapsing to
no-op. Mean engagement/target support was 0.124/0.022, reducing counterfactual
residuals to 0.049/0.008. Initial-anchor KL averaged only 0.017, so the 0.10
budget never activated. This exposes a deeper optimizer issue: with the
counterfactual term disabled, separate hierarchical ratios/clips do not recover
the validated joint-ratio factorized PPO objective. SA-RG-MCH is retained as a
falsified ablation and diagnostic implementation, not an active candidate.

```text
rein_learning/common/masked_context_support.py
rein_learning/algorithms/policy_gradient/mch_ppo.py
scripts/run_air_defense_v1_sa_rg_mch_ppo_stress_test.py
docs/experiments/air_defense_v1_sa_rg_mch_ppo_stress_test.md
```

## Boundary-Probed Counterfactual Engagement PPO

`BoundaryProbedCounterfactualEngagementPPO` retains the complete factorized
joint PPO ratio and clipped surrogate. It records on-policy environment
snapshots, selects engagement contexts near the deterministic decision
boundary, and performs paired no-op/engage continuations using target-indexed
common random tapes. Accepted return directions add a small engagement-logit
ranking loss; zero budget or rejected labels call the original joint PPO train
path exactly.

The frozen 10k, three-seed, two-scenario test failed its mechanism gate. Two of
six matched runs collapsed to all-noop. Heterogeneity pressure improved mean
reward by 21.686 and damage by 0.509 versus factorized PPO, but resource cost
reached 192.8% of baseline. Boundary selection beat equal-budget random probing
only in time pressure. Seed9 received only negative accepted labels in both
scenarios and collapsed, identifying bidirectional label coverage and auxiliary
update dosage as the next mechanism bottleneck. BPCE v0 is an implemented,
partially positive research prototype, not a validated final algorithm.

```text
rein_learning/common/boundary_counterfactual_probe.py
rein_learning/algorithms/policy_gradient/bpce_ppo.py
scripts/run_air_defense_v1_bpce_ppo_stress_test.py
docs/experiments/air_defense_v1_bpce_ppo_stress_test.md
```

## Verification

```powershell
conda run -n rein-learning python -m pytest tests
```

Latest verification:

```text
242 passed
BPCE focused regression after final additions: 14 passed
SA-RG-MCH focused regression: 24 passed
Q-learning greedy evaluation: total_reward=3.0, steps=8
DQN greedy evaluation: total_reward=3.0, steps=8, device=cuda
REINFORCE smoke train: episode=001, avg_reward=-33.00, loss=0.0051, device=cuda
AirDefense DQN smoke evaluation: intercept_rate=0.40, leak_rate=0.40, avg_invalid_actions=0.00
AirDefense v1 Hungarian acceptance: avg_reward=-40.48, damage=1.15, invalid=0.00
```
