# Academic Project Progress

Updated: 2026-07-17

## 1. Current Project Position

The project has completed its engineering foundation and is ready to move from "algorithm sandbox construction" toward "research problem definition and domain simulation".

The recommended research direction is:

```text
Dynamic Weapon/Resource-Target Assignment for counter-UAV air-defense grouping,
using deep reinforcement learning and multi-agent reinforcement learning.
```

The current work should be regarded as a research scaffold, not yet the final academic contribution.

## 2. Completed Engineering Foundation

### Environment

The main Python environment is:

```text
conda env: rein-learning
Python: 3.10.20
PyTorch: 2.11.0+cu128
GPU: NVIDIA GeForce RTX 5060 Ti
```

Verified packages include:

```text
Gymnasium
Stable-Baselines3 / SB3-Contrib
Ray / RLlib
PettingZoo / SuperSuit
Tianshou
TorchRL
TensorBoard / W&B
OpenCV / scikit-learn / MuJoCo
```

### Code Structure

The project now uses a reinforcement-learning-oriented package layout:

```text
rein_learning/
  envs/
  agents/
  algorithms/
  models/
  buffers/
  trainers/
  simulators/
  wrappers/
  configs/
  common/
```

This separates task environments, algorithm logic, models, replay buffers, and training flows.

### Toy Environment

Implemented:

```text
SmallGridWorldEnv
```

Purpose:

- test RL algorithm mechanics
- verify environment-agent-trainer workflow
- provide a small discrete MDP for learning and debugging

This is not the final air-defense environment.

### Baseline Algorithms

Implemented and tested:

```text
Q-learning
DQN for discrete-state GridWorld
masked vector-observation DQN for AirDefenseResourceAssignmentEnv v0
REINFORCE
```

Current verification:

```text
pytest: 50 passed
```

Q-learning and the first DQN implementation have been verified on GridWorld. REINFORCE is present as a policy-gradient baseline and has smoke-test coverage. The AirDefense DQN path adds vector observations, action masking, replay buffer support for continuous observations, and a trainer for the air-defense task.

### Air-Defense Environment v0 Initial Implementation

Initial implementation has started for:

```text
AirDefenseResourceAssignmentEnv v0
```

Implemented components:

- Gymnasium environment interface
- air-defense environment config
- defense unit state
- target state
- resource-target discrete action space
- no-op action
- action mask
- target motion
- intercept probability model
- ammo and cooldown updates
- target leak detection
- reward breakdown
- unit tests

Current verification:

```text
pytest: 30 passed
```

### Rule-Based Baselines and Evaluation

Implemented baseline policies for `AirDefenseResourceAssignmentEnv v0`:

- random legal action
- nearest target first
- highest threat first
- greedy expected benefit

Implemented reusable episode and aggregate metrics:

- average reward
- success rate
- intercept rate
- leak rate
- average ammo used
- average shots
- hit rate per shot
- invalid action count

First 30-episode baseline evaluation:

```text
policy              avg_reward  success  intercept  leak   ammo  shots  hit/shot  invalid
random_legal          -19.64     0.10       0.55   0.30  15.47  15.47      0.18     0.00
nearest_target        -25.51     0.10       0.51   0.34  15.90  15.90      0.16     0.00
highest_threat        -34.57     0.07       0.41   0.43  15.43  15.43      0.13     0.00
greedy_expected         1.47     0.30       0.72   0.15  14.83  14.83      0.24     0.00
```

This indicates that the environment has a meaningful decision structure: a stronger rule policy performs clearly better than random/simple heuristics.

### AirDefense DQN Training Path

Implemented the first learning-based trainer for:

```text
AirDefenseResourceAssignmentEnv v0
```

Implemented components:

- vector-observation Q-network
- vector replay buffer
- masked DQN action selection
- masked target-Q calculation
- AirDefense DQN trainer
- AirDefense DQN script entrypoint
- trainer smoke tests

Run:

```powershell
conda run -n rein-learning python scripts\train_air_defense_dqn.py
```

Short smoke evaluation after 5 training episodes:

```text
episodes=2
avg_reward=-33.29
intercept_rate=0.40
leak_rate=0.40
avg_invalid_actions=0.00
```

This confirms the training and evaluation pipeline is executable. The short run is not a performance claim.

### AirDefense RL Environment Model Design

Added a literature-grounded environment design document:

```text
docs/environments/air_defense/air_defense_rl_environment_model_design.md
```

The document synthesizes the local papers in:

```text
research_papers/05_anti_uav_rl_environment_model/
```

It defines the recommended next environment model around:

- protected zones and damage-aware objectives
- hostile UAV targets as spatio-temporal tasks
- heterogeneous defense units and effectors
- centralized single-agent v1.0 and later multi-agent Dec-POMDP versions
- vector, masked, and later graph-based observations
- joint resource-target action spaces
- reward terms for interception, protection, tracking, jamming, resource cost, conflict, overkill, and invalid actions
- a staged roadmap from Gymnasium to PettingZoo/MAPPO-compatible environments

### AirDefenseResourceAssignmentEnv v1.0 Implementation

Implemented the first v1.0 environment:

```text
rein_learning/envs/air_defense_v1/
```

Implemented components:

- multiple protected zones
- hostile targets with `payload`, `target_zone`, and `time_to_impact`
- heterogeneous defense units
- centralized Gymnasium environment
- joint `MultiDiscrete` resource-target action space
- per-unit action masks
- damage-aware reward breakdown
- conflict and overkill penalties
- render and info metrics
- unit tests

Implemented v1.0 baseline policies:

- random legal joint action
- nearest target joint assignment
- highest threat joint assignment
- time-to-impact priority
- greedy expected damage reduction

First 50-episode baseline evaluation:

```text
policy              avg_reward  success  intercept  leak   damage  ammo  shots  hit/shot  invalid
random_joint          -54.15     0.08       0.41   0.41     1.39  15.82  15.82      0.14     0.00
nearest_joint         -49.85     0.10       0.42   0.38     1.35  15.70  15.70      0.13     0.00
highest_threat        -60.47     0.04       0.34   0.45     1.51  15.84  15.84      0.11     0.00
time_to_impact        -58.36     0.04       0.36   0.43     1.47  15.86  15.86      0.11     0.00
greedy_damage         -41.22     0.12       0.48   0.34     1.17  15.52  15.52      0.15     0.00
```

The strongest rule policy is `greedy_damage`, but success remains low. This indicates that the v1.0 environment now exposes a meaningful research pressure point: simple heuristics struggle under joint allocation, finite ammunition, stochastic interception, and damage-aware objectives.

## 3. Literature and Reproduction Progress

The project has collected and organized literature around:

- foundational MARL algorithms: MADDPG, QMIX, MAPPO, HATRPO/HAPPO
- attention and exploration mechanisms: GAT, RND, MAAC, Qatten
- heterogeneous resource coordination
- counter-UAV and air-defense-related task allocation

There is also a reproduction project for:

```text
MADDPG-IA / HELS-UAV-DRTA
```

This reproduction code and paper material are useful as the closest current reference for the later domain environment, especially for:

- dynamic resource-target assignment
- heterogeneous defense resources
- attention-based MARL
- intrinsic reward / exploration
- simulation-based experimental evaluation

## 4. Current Main Gap

The project can now run basic RL algorithms, a first learning-based method on v0, and rule-based baselines on the v1.0 damage-aware air-defense environment. The academic experiment pipeline is becoming concrete, but learning-based comparisons on v1.0 are not yet implemented.

The previous missing core was:

```text
a simplified but publishable air-defense resource-assignment environment.
```

That core now has a v0 implementation, rule-based baselines, a first DQN training path, a literature-grounded v1 environment design, and an executable v1.0 environment with baseline results. The next remaining gap is to train and compare learning-based methods on v1.0:

- PPO / Maskable PPO training on v1.0
- unified comparison script for rules and learning methods
- experiment logging
- scenario parameter sweeps
- analysis of why simple baselines fail

## 5. Recommended Next Research Step

The next priority is not implementing more generic algorithms.

The next priority is:

```text
Train and evaluate PPO / Maskable PPO on AirDefenseResourceAssignmentEnv v1.0.
```

The next comparison loop should include:

- random joint baseline
- greedy damage-reduction baseline
- PPO
- Maskable PPO
- fixed evaluation seeds
- aggregate metrics table
- saved experiment results
- scenario parameter sweeps

## 6. Short-Term Roadmap

### Step 1: Research Problem Definition

Write a concise MDP/POMDP document:

- problem statement
- assumptions
- state space
- action space
- transition dynamics
- reward design
- constraints
- terminal conditions
- evaluation metrics

### Step 2: Implement Environment v0

Add:

```text
rein_learning/envs/air_defense/
rein_learning/simulators/
```

Status: completed as an initial v0.

### Step 3: Baseline Experiments

Run:

- rule-based greedy assignment
- random policy
- DQN
- PPO / SB3

Only after the environment is stable should multi-agent algorithms be added.

### Step 4: Research Innovation Selection

Choose one main innovation axis:

- action masking for constrained allocation
- heterogeneous resource modeling
- attention/GNN state encoding
- reward decomposition
- multi-agent cooperation
- curriculum learning

Avoid combining too many innovations too early.

## 7. Completed Learning Baseline Step

The previously recommended next task after the v1.0 environment and rule baseline run was:

```text
Implement PPO / Maskable PPO trainers for AirDefenseResourceAssignmentEnv v1.0,
then compare them against the v1.0 rule-based baselines.
```

Status: completed as an executable first version.

Implemented output:

```text
rein_learning/trainers/air_defense_v1_ppo.py
scripts/train_air_defense_v1_ppo.py
scripts/compare_air_defense_v1_methods.py
tests/test_air_defense_v1_trainers.py
docs/experiments/air_defense_v1_learning_baselines.md
```

Verification:

```text
conda run -n rein-learning python -m pytest tests
54 passed
```

The unified comparison framework has been upgraded with:

- five complete rule baselines;
- PPO and Maskable PPO;
- multiple training seeds and paired evaluation scenario blocks;
- held-out periodic evaluation curves;
- Student-t confidence intervals across runs;
- raw run rows and aggregated result tables;
- SVG/PDF/PNG learning-curve exports;
- complete environment, training, runtime, and command-line configuration records;
- separate requested and actual SB3 rollout timesteps.

Implemented output:

```text
rein_learning/experiments/air_defense_v1_benchmark.py
tests/test_air_defense_v1_experiments.py
```

Multi-seed smoke comparison:

```text
conda run -n rein-learning python scripts\compare_air_defense_v1_methods.py --timesteps 128 --n-steps 64 --batch-size 32 --eval-episodes 2 --seeds 0 1 --curve-eval-freq 64 --curve-eval-episodes 1 --no-save-models --experiment-name benchmark_smoke_multiseed_v2
```

Smoke artifacts are stored under:

```text
results/air_defense_v1/benchmark_smoke_multiseed_v2/
```

The short smoke run is not a performance claim. It verifies that the complete multi-seed comparison pipeline executes, records reproducibility metadata, and shows the expected action-validity difference: Maskable PPO has zero invalid actions while plain PPO does not.

## 8. Formal v1.0 Learning Benchmark

The first formal controlled benchmark is complete:

```text
Train PPO / Maskable PPO with five seeds on AirDefenseResourceAssignmentEnv v1.0,
compare them with all five rule baselines, inspect learning curves and confidence
intervals, then analyze why each method succeeds or fails.
```

Main result:

```text
Maskable PPO: avg_reward=-35.93, intercept_rate=0.561, damage=1.052
Greedy damage: avg_reward=-38.75, intercept_rate=0.517, damage=1.052
Plain PPO:     avg_reward=-86.52, intercept_rate=0.398, damage=1.534
```

Maskable PPO consistently outperforms plain PPO and eliminates invalid actions. It reaches the strongest rule baseline, but paired confidence intervals do not yet establish a stable advantage over `greedy_damage`.

Formal report:

```text
docs/experiments/air_defense_v1_formal_benchmark_100k.md
```

## 9. Frozen Baseline and Scenario Profiles

The default AirDefense v1.0 formal benchmark configuration is now frozen as the
canonical `medium` scenario. The snapshot is explicit and independent from
future `AirDefenseV1EnvConfig` default changes.

Implemented scenario profiles:

```text
difficulty: easy, medium, hard
pressure:   time_pressure, resource_pressure, intercept_uncertainty,
            damage_pressure, heterogeneity_pressure
```

All profiles keep the same two zones, three defense units, five targets,
observation shape, joint action shape, and action-mask shape.

The first difficulty calibration used `greedy_damage` on 500 paired scenario
seeds per profile:

```text
scenario  avg_reward  avg_damage  intercept  success
easy         -15.91       0.266      0.646    0.132
medium       -40.03       1.055      0.507    0.046
hard        -108.11       2.822      0.334    0.002
```

The paired hard-minus-easy damage difference was `+2.556`, with 95% CI
`[2.492, 2.619]`. The profile ordering therefore passes the initial difficulty
calibration criterion.

Implemented output:

```text
rein_learning/envs/air_defense_v1/scenarios.py
tests/test_air_defense_v1_scenarios.py
docs/environments/air_defense/air_defense_v1_scenario_profiles.md
docs/task_guides/next_research_phase_difficulty_generalization.md
```

## 10. Hungarian Optimization Baseline

The strong one-step optimization baseline is now implemented and registered as
`hungarian_damage`. It uses the same expected damage-reduction score as
`greedy_damage`, excludes illegal and nonpositive assignments, and adds one
independent no-op dummy column per defense unit.

Correctness checks cover known small matrices, one-to-one constraints,
deterministic actions, and equality with brute-force optimal objectives. Rule
and learning policy evaluation now records average decision time per step.

The first 50-episode `medium` evaluation block produced:

```text
method              avg_reward  avg_damage  invalid  decision_ms
greedy_damage          -41.22        1.17      0.00        0.019
hungarian_damage       -40.48        1.15      0.00        0.028
```

This is an implementation acceptance run, not a statistical claim that
Hungarian has better long-horizon performance.

Implemented output:

```text
rein_learning/baselines/air_defense_v1.py
tests/test_air_defense_v1_hungarian.py
docs/algorithms/hungarian_damage_reduction_baseline.md
```

## 11. Diagnostic Evaluation Metrics

The evaluation pipeline now records raw episode diagnostics and aggregates them
with one shared implementation for rule policies, PPO, and Maskable PPO.

Added diagnostics:

```text
high_threat_leak_rate, avg_zone_weighted_damage,
assignment_conflict_rate, overkill_rate,
damage_reduction_per_ammo, avg_resource_cost,
avg_decision_time_ms
```

The experiment bundle schema is now version 2 and adds `episodes.csv`. Raw
numerators and denominators can reproduce every run-level diagnostic. Existing
metric names and formulas remain unchanged, and legacy episode rows can still
reconstruct the old metrics.

Implemented output:

```text
rein_learning/common/air_defense_v1_metrics.py
tests/test_air_defense_v1_diagnostics.py
docs/experiments/air_defense_v1_diagnostic_metrics.md
```

## 12. Cross-Scenario Benchmark and Generalization Matrix

The unified benchmark now supports explicit method selection and one or more
training and evaluation scenarios. Learning models train once per training
scenario and seed, then reuse the model across paired evaluation scenario
blocks. Rule methods share the same blocks and are repeated across training
scenario rows for direct comparison.

Schema version 3 adds:

```text
train_scenario, eval_scenario, scenario space signatures,
paired_differences.csv, generalization_matrix.csv,
generalization figures, scenario-scoped model and TensorBoard paths
```

The task-five acceptance run completed a two-training-scenario by
two-evaluation-scenario matrix with `greedy_damage`, `maskable_ppo`, and two
seeds. It generated 16 run rows, 32 raw episodes, four models, four TensorBoard
runs, paired statistics, learning curves, and generalization figures.

Implemented output:

```text
rein_learning/experiments/air_defense_v1_benchmark.py
scripts/compare_air_defense_v1_methods.py
tests/test_air_defense_v1_experiments.py
docs/experiments/air_defense_v1_cross_scenario_benchmark.md
results/air_defense_v1/task5_smoke_2x2/
```

Current verification:

```text
pytest: 99 passed
```

## 13. Three-Seed Screening and Failure Diagnosis

Task six is complete. The screening experiment trained PPO and Maskable PPO on
`medium` for 20,000 requested steps with seeds `0/1/2`, then evaluated them
with Greedy and Hungarian on eight paired scenarios. The bundle contains 96
run rows, 4,800 raw evaluation episodes, six models, six TensorBoard runs,
paired confidence intervals, learning curves, and generalization figures.

The screening identified three main watersheds:

```text
action validity: plain PPO collapses to an all-no-op policy by 5k steps
training stability: 20k Maskable PPO has a seed-dependent low-engagement trap
scenario pressure: time and heterogeneity pressure separate rules from learning
```

`hard` is a feasibility boundary where success is near-unreachable for every
method, while Greedy and Hungarian remain practically indistinguishable. The
full diagnosis and exact failure episodes are recorded in:

```text
docs/experiments/air_defense_v1_task6_screening.md
results/air_defense_v1/task6_screening_medium_20k_3seeds/
```

## 14. Core-Scenario Formal Benchmark

Task seven is complete. Five Maskable PPO models were trained on `medium` for
100,000 requested steps and evaluated with Greedy and Hungarian on `medium`,
`time_pressure`, and `heterogeneity_pressure`. The completed bundle contains
45 run rows, 4,500 paired episodes, five models, five TensorBoard runs, and
153 rows each of summary, paired-difference, and generalization statistics.

Main findings:

```text
reward and damage: no significant Maskable PPO versus rule difference
medium: Maskable PPO has a significantly higher intercept rate
time pressure: Maskable PPO uses significantly less ammunition and cost
heterogeneity: Maskable PPO has a significantly higher high-threat leak rate
all scenarios: Maskable PPO retains 1.6%-2.5% joint assignment conflicts
```

The 100k budget resolves the persistent low-engagement behavior seen in the
20k screening experiment. The remaining limitation is no longer basic action
validity or insufficient engagement; it is conflict-free joint coordination
and high-value target prioritization under heterogeneous resource relations.

Formal report and result bundle:

```text
docs/experiments/air_defense_v1_task7_formal_100k.md
results/air_defense_v1/task7_formal_medium_100k_5seeds/
```

## 15. Conflict-Free Joint-Action Screening

Task eight implemented a deterministic `Discrete(136)` codec and dynamic joint
mask over all one-to-one assignments. The wrapper leaves AirDefense v1.0
dynamics, reward, observations, MLP, and PPO hyperparameters unchanged. The
trainer and unified benchmark now support `conflict_free_maskable_ppo`, and
experiment schema version 4 records method-specific action-space signatures.

The smoke run passed, followed by a 30k by three-seed screening on `medium`,
`time_pressure`, and `heterogeneity_pressure`. The new method reduced invalid
actions, assignment conflicts, and overkill to exactly zero. Its mean reward,
damage, and high-threat leak trends improved across the three scenarios, but
`time_pressure` resource cost increased by `3.49`, exceeding the frozen `+0.50`
screening limit. All major paired confidence intervals crossed zero.

The conditional 100k by five-seed experiment was therefore not run. The codec
remains a valid structural ablation baseline, but it does not yet preserve the
resource-efficiency signal established in task seven.

```text
docs/algorithms/conflict_free_joint_action_masking.md
docs/experiments/air_defense_v1_task8_screening.md
results/air_defense_v1/task8_conflict_free_screening_30k_3seeds/
```

## 16. Autoregressive Joint-Action Screening

Task nine implemented a fixed-order autoregressive masked policy while keeping
the AirDefense v1.0 environment, reward, shared MLP encoder, critic, and PPO
hyperparameters fixed. The joint action remains one environment decision; its
log probability is reconstructed as the sum of conditional unit log
probabilities. Experiment schema version 5 records the action-generator
contract separately from the Gym action-space signature.

The smoke run passed, followed by a 30k by three-seed screening with the
original Maskable PPO, the `Discrete(136)` conflict-free baseline, Greedy, and
Hungarian. The completed bundle contains 45 run rows, 2,250 raw episodes, nine
models, nine TensorBoard logs, and paired cross-scenario statistics.

Main findings:

```text
structural validity: invalid actions, conflicts, and overkill are exactly zero
time pressure: resource cost is 1.47 below original Maskable PPO
joint enumeration: autoregression saves 4.96 resource-cost units
medium performance: reward and damage remain within screening limits
heterogeneity: high-threat leak falls by 0.01483, below the 0.02 gate
```

All gates except the heterogeneous high-threat leak magnitude passed. The
conditional 100k experiment was therefore not run. The result separates the
resource-efficiency problem from the remaining heterogeneous target-priority
problem more clearly.

```text
docs/algorithms/autoregressive_conflict_free_policy.md
docs/experiments/air_defense_v1_task9_screening.md
results/air_defense_v1/task9_autoregressive_screening_30k_3seeds/
```

## 17. Immediate Next Task

Add unit-level action, no-op, resource-type, target-threat, and assignment-order
diagnostics. Then run a small fixed-order sensitivity ablation to determine
whether seed-dependent high-threat leaks are caused by decoder order or by the
MLP's inability to represent heterogeneous resource-target relations. Do not
start the 100k formal experiment or introduce a GNN before this diagnosis.

The diagnostic fields, three-order ablation, screening gates, independent-seed
confirmation protocol, and GNN entry conditions are frozen in:

```text
docs/task_guides/next_research_phase_order_bias_diagnostics.md
```

## 18. Task-Ten Order-Bias Diagnosis

Task ten is complete. The autoregressive generator now accepts an explicit
unit permutation while preserving environment action indices. Schema version
6 adds final-evaluation decision traces, pooled opportunity metrics, and six
mutually exclusive high-threat leak attributions.

Three frozen task-nine models were replayed for 900 episodes. The earlier seed-2
failure was traced to low engagement: unit 1 never assigned a target, the laser
assigned in only 0.28% of decisions, and 80% of its heterogeneous high-threat
leaks were attributable to legal-but-unassigned opportunities.

The three-order 30k by three-seed screening completed 27 evaluation blocks,
1,350 episodes, 169,887 decision rows, and nine saved models. Order `201`
reduced heterogeneous high-threat leak by 0.021729 with 2/3 seeds improving,
but increased time-pressure resource cost by 4.253 relative to order `012`.
No candidate passed all frozen gates, so the independent 100k confirmation was
not run.

The evidence supports a real order/role interaction, but not a cost-preserving
fixed-order replacement. The next research step is a lightweight
role-conditioned or permutation-equivariant action head while keeping the
environment, reward, and GNN work frozen.

```text
docs/algorithms/autoregressive_order_ablation.md
docs/experiments/air_defense_v1_task10_order_diagnostics.md
results/air_defense_v1/task10_frozen_model_diagnostics/
results/air_defense_v1/task10_order_screening_30k_3seeds/
```

## 19. Task-Eleven Role-Conditioned Action Head

Task eleven is complete. A model-side observation layout and a shared
unit-target relation actor replaced the independent positional logits while
the environment, reward, critic, and PPO hyperparameters remained frozen. The
new actor has 34,946 parameters versus 37,138 in task ten (-5.90%), and the
critic is unchanged. Unit and target permutation tests, non-default-order model
loading, schema-7 parameter records, and all regression tests passed.

The 30k by three-seed screening completed nine models, 1,350 episodes, and
164,868 decision rows. The canonical order `012` improved reward, damage, and
resource cost relative to task ten, but high-threat leak worsened by 0.00335,
only one of three seeds improved, five heterogeneous unit-runs collapsed, and
decision latency increased 73.51%. All cross-order robustness gates failed.

The relation scorer achieved high matching efficiency when it acted, but 94.8%
of canonical heterogeneous high-threat leaks were legal-but-unassigned. The
next bottleneck is therefore no-op probability and PPO optimization stability,
not pair matching capacity. The 100k confirmation and GNN work remain frozen.

```text
docs/algorithms/role_conditioned_autoregressive_policy.md
docs/experiments/air_defense_v1_task11_role_conditioned_screening.md
results/air_defense_v1/task11_role_conditioned_screening_30k_3seeds/
```

## 20. Task Twelve: No-Op Optimization Stability

Task twelve is complete. A frozen corpus of 768 policy-probe states now records
engagement probability, no-op margin, entropy, value, PPO losses, KL, clipping,
advantages, and gradient norms. Frozen task-eleven replay showed that seed 1 is
all-noop under deterministic evaluation in all three core scenarios, while its
stochastic engagement rate remains 0.36-0.41. The failure therefore includes
categorical argmax probability fragmentation rather than probability-level zero
engagement alone.

A 10k five-seed diagnostic reproduced an early training bifurcation: seeds 3
and 6 engaged, while seeds 4, 5, and 7 reached 100% all-noop. A factorized
engagement-target actor was then implemented with the environment, reward,
critic, relation scorer, PPO hyperparameters, and order `012` frozen.

The paired 30k three-seed screening completed six models, 18 evaluation blocks,
900 deterministic episodes, 109,041 decision rows, and 1,047 leak attributions.
The candidate preserved zero invalid actions/conflicts and reduced the
heterogeneous unassigned-leak share by 0.1716, but retained six collapsed
scenario-seed combinations. It also degraded reward, damage, resource cost,
calibration, and latency. Only 6 of 19 gates passed, so the conditional 100k
confirmation was not run.

The next bottleneck is engagement calibration and Actor-Critic optimization
stability. GNN work remains deferred because representation capacity is not yet
the controlled limiting variable.

```text
docs/algorithms/factorized_engagement_policy.md
docs/experiments/air_defense_v1_task12_noop_stability.md
results/air_defense_v1/task12_analysis/
```
