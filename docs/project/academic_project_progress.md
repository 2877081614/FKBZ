# Academic Project Progress

Updated: 2026-07-29

## 1. Current Project Position

The project has completed W1 claim-evidence freeze and the N1/N2/N3 mainline
falsification gates. N1 rejected return-decomposition and generic CMDP candidates.
N2 identified a future-coverability responsibility certificate (FCRC) that passed a
development-only static gate. N3 then rejected its predictive proposition: the frozen
paired effect was not significant and FCRC added no leave-block-out predictive value.
No new online algorithm or standalone performance contribution has been established.

The recommended research direction is:

```text
Dynamic Weapon/Resource-Target Assignment for counter-UAV air-defense grouping,
using deep reinforcement learning and multi-agent reinforcement learning.
```

The current work now constitutes a traceable scientific manuscript module. Target-journal
formatting, public release identifiers, and any new online algorithm remain separate tasks.

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

## 21. 当前研究路线与任务十三

更新时间：2026-07-19。

项目最终目标已重新冻结为：构建面向异质资源、动态目标和复杂约束的稳定、可解释、可扩展防空资源分配方法。GNN 不是最终目标，而是后续关系建模、批量反事实估值和跨规模泛化的候选工具。

当前形成两级创新假设：

1. 第一创新假设：动态合法动作约束下的掩码感知反事实分层 PPO，重点解决单元、交战和目标选择共享联合 advantage 所造成的信用混叠；
2. 第二创新假设：面向变规模防空分配的类型化二部图反事实 Critic，重点解决批量反事实估值效率和跨规模泛化，而不是普通 `PPO + GNN`。

这两个方向目前均属于待验证假设，不得提前宣称创新已经成立。COMA、H-PPO、HAPPO/HATRPO、CAPO 和现有 GNN-WTA 工作构成直接相关边界，任务十三必须先完成系统查新和方法差异矩阵。

任务十三现为当前执行阶段，顺序为：

```text
系统查新
-> 冻结模型阈值与概率校准
-> engage/no-op advantage 与 Critic 误差诊断
-> 掩码感知反事实估值原型
-> 条件性冻结 MCH-PPO
-> 30k 配对筛选
```

环境、奖励、核心场景、任务十二探针、关系 scorer 语义和顺序 `012` 继续冻结。GNN、变规模环境和 100k 正式实验暂不实施。

进入图结构阶段必须同时满足：第一创新候选通过独立种子稳定性门槛；交战不再是主要失败来源；剩余瓶颈集中于关系匹配、反事实估值效率或规模泛化；已建立变规模测试协议。

```text
docs/project/research_innovation_roadmap.md
docs/task_guides/next_research_phase_counterfactual_credit_assignment.md
```

## 22. 任务十三执行结论

更新时间：2026-07-19。

任务十三完成第一轮公式级查新。动作掩码、分层 PPO、反事实 baseline 和顺序 advantage 分解均有直接先例，因此宽泛的 MCH-PPO 组合不能作为充分创新。保留命题已收窄为“动态可行集上的掩码条件反事实估计与层级独立近端更新”。

6 个任务十二冻结模型完成 `0.10-0.90` 的 17 阈值、三核心场景扫描。没有任何阈值能同时通过 18 个模型种子与场景组合；不同因子化种子的最佳阈值分别集中在 `0.10-0.15`、`0.25-0.30` 和 `0.60`。统一阈值修复被否决。

代表性 factorized seeds 8/10 完成逐单元信用与 16 次共同随机数反事实诊断。联合 `return-V(s)` 对 engage/no-op 的均值差远小于组内方差；13 个非零局部反事实样本中有 5 个与联合 value-advantage 方向相反，但 26 个目标分支均未达到单分支显著门槛。现有 PPO Critic 只输出 `V(s)`，结构上不能提供动作反事实 `Q(s,h_i,a_i)`。

因此任务十三冻结了反事实分解公式、动态合法集重归一化规则、阈值接口和诊断数据模式，但没有冻结完整 MCH-PPO，也没有触发 30k、100k 或 GNN。下一阶段先验证非图结构动作条件 Q-Critic 的偏差、排序和 advantage 符号准确率；只有通过后才恢复第一创新候选训练。

```text
docs/literature/task13_counterfactual_credit_novelty_review.md
docs/algorithms/masked_counterfactual_hierarchical_ppo.md
docs/experiments/air_defense_v1_task13_credit_diagnostics.md
results/air_defense_v1/task13_calibration/
results/air_defense_v1/task13_credit_diagnostics/
```

## 23. 任务十四执行结论

更新时间：2026-07-19。

任务十四已完成非图结构掩码条件动作价值 Critic、固定前序动作与后序重采样接口、共同随机数反事实数据生成、按状态分组的数据划分、排序与符号诊断、四结构消融以及正式门控实验。

正式数据来自任务十二 factorized seeds 8/10 和三个核心场景，共 90 个独立状态、571 个合法候选动作样本，每个候选使用 8 次共同随机数 rollout。训练/验证/测试为 338/117/116 行，同一 `state_id` 不跨 split。完整模型包含 83,457 个参数，训练种子为 14/15/16。

三个完整模型的 Q MAE 为 `10.626-11.295`，相对冻结 `V(s)` 的 `17.747` 改善 `36.4%-40.1%`，且推理耗时远低于 Monte Carlo 标签生成。但总体高置信排序只有 8 对，准确率为 `0.250-0.375`；目标排序只有 5 对，top-1 只有 1 个有效状态，engage/no-op 符号只有 2 个有效组。完整模型在消融中也没有形成稳定优势，说明纯回归目标主要学到了状态共同价值，尚未可靠学习同一状态内的动作差值。

任务十四整体通过种子数为 `0/3`。因此不恢复 MCH-PPO，不运行条件性 30k/100k，也不进入 GNN。下一阶段先增加明确动作差异状态的覆盖，采用配对方差缩减和组内 ranking/centered-advantage 监督，并保持现有独立测试协议冻结。只有非图模型在可靠数据上仍稳定失败，才形成更强关系估值或图反事实 Critic 的进入证据。

```text
docs/task_guides/next_research_phase_action_conditioned_q_critic.md
docs/algorithms/masked_action_q_critic.md
docs/experiments/air_defense_v1_task14_q_critic.md
rein_learning/models/masked_action_q_critic.py
rein_learning/common/q_critic_diagnostics.py
scripts/run_air_defense_v1_task14_q_critic.py
results/air_defense_v1/task14_q_critic/
```

## 24. 任务十四排序监督修订结论

更新时间：2026-07-19。

任务十四修订完成了配对功效审计、组内中心化与可靠性加权动作差值损失、旧测试集隔离、全新测试集生成和三种子正式对照。任务十四旧 test 的 116 行全部排除；新测试包含 36 个状态、192 个候选动作和每候选 32 次共同随机数 rollout，新旧状态 ID 无交集。

在网络结构完全相同的条件下，`difference_aware` 相对 `absolute_mse` 的总体排序平均提高 `0.167`，三种子排序达到 `0.659 / 0.727 / 0.705`，目标排序达到 `0.696 / 0.826 / 0.783`。平均 MAE 比值为 `0.993`，说明组内监督改善排序时没有牺牲 Q 数值精度。该结果首次在当前项目中给出“纯绝对 Q 回归受到状态共同价值主导”的受控证据。

原门控仍未通过。目标排序有效对只有 23，top-1 只有 10，engage/no-op 只有 11；三个场景有效动作对分别为 9、28、7。engage/no-op 符号准确率固定在 `0.545`，说明剩余瓶颈开始集中于是否交战的层级信用。功效投影还表明，即使同一批状态增加到 256 rollout，异质场景预计仍只有 22 对、top-1 只有 23，因此继续重复轨迹不如增加独立状态。

任务十四修订整体通过数为 `0/3`，不恢复 MCH-PPO，不运行 30k/100k，也不进入 GNN。下一阶段应离线拆分 `Q_engage` 和 conditional `Q_target`，分别验证交战符号与目标排序；只有两个层级同时通过后，才允许实现最小 MCH-PPO。

```text
docs/task_guides/next_research_phase_q_critic_ranking_refinement.md
docs/experiments/air_defense_v1_task14_ranking_refinement.md
rein_learning/common/q_critic_training.py
scripts/analyze_air_defense_v1_task14_power.py
scripts/run_air_defense_v1_task14_ranking_refinement.py
results/air_defense_v1/task14_q_critic_ranking_refinement/
```

## 25. 任务十四显式交战与目标分层 Q 诊断结论

更新时间：2026-07-20。

本阶段实现了非图结构 `HierarchicalMaskedQCritic`，将动作价值拆分为 `Q_engage(s,h_i,e_i)` 和 conditional `Q_target(s,h_i,target | engage)`。训练数据继续使用任务十四原始 train/validation，正式测试使用 factorized policy seeds 8/10、三个核心场景和 108 个全新状态；共生成 684 个候选动作，每候选使用 32 次共同随机数 rollout。原始 test 116 行、排序修订 test 36 个状态和本轮测试状态均严格隔离。

目标层形成了明确正结果。三个训练种子的 target ranking 为 `0.870 / 0.850 / 0.830`，相对同一测试集上的 monolithic `difference_aware` baseline 平均提高 `0.057`；target top-1 为 `0.875 / 0.833 / 0.750`。三个场景的目标排序均达到门槛，说明给定 engage 后的资源-目标关系估值已经可以由当前非图模型稳定学习。

交战层没有形成改进。engage/no-op 符号准确率为 `0.706 / 0.588 / 0.588`，相对 monolithic baseline 平均下降 `0.255`；target Q MAE 也恶化约 `17%-21%`。三个种子均只通过 4/8 项门槛，正式通过数为 `0/3`。32-rollout 下 engagement 有效组为 17，功效投影显示 64 rollout 可满足总体数量，但异质场景 top-1 仍不足；128 rollout 才能基本满足逐场景数量。由于候选在已有高置信组上已明确退化，本阶段不追加昂贵 rollout。

因此，显式分层本身不能解决交战信用。MCH-PPO、30k/100k 和 GNN 继续冻结。下一阶段转向离线交战效用审计：比较均值回报、资源代价、关键目标泄漏、毁伤以及 CVaR/分位数尾部风险标签，验证风险敏感或显式约束目标能否恢复 engage/no-op 的稳定符号；已通过的 conditional-target 层保持冻结。

```text
docs/task_guides/next_research_phase_hierarchical_q_diagnostics.md
docs/algorithms/hierarchical_masked_q_critic.md
docs/experiments/air_defense_v1_task14_hierarchical_q.md
rein_learning/models/hierarchical_masked_q_critic.py
rein_learning/common/hierarchical_q_diagnostics.py
scripts/run_air_defense_v1_task14_hierarchical_q.py
results/air_defense_v1/task14_hierarchical_q/
```

## 26. 风险与约束感知交战效用诊断结论

更新时间：2026-07-20。

本阶段新增分量化反事实数据协议和 `RiskAwareEngagementCritic`。对同一状态、前缀和单元，使用共同随机数成对执行 no-op 与按冻结条件目标策略采样的 engage，分别记录 operational return、折扣资源成本、折扣毁伤、高威胁泄漏、总回报和射击数。正式实验包含108个全新状态、150个上下文组和每分支32次 rollout；train/validation/test 为58/29/63组。三轮旧测试观测重叠均为0，state split 无泄漏，总回报分量重构最大误差为 `7.63e-06`。

validation 从108组候选中冻结 `cost=2.0, damage=30.0, high=0.0, CVaR beta=0.5, alpha=0.25`。在独立 test 上，该效用相对原始均值回报把 balanced accuracy 从 `0.713` 提高到 `0.926`，false-noop 从 `0.333` 降至 `0`，wasteful-engage 从 `0.241` 降至 `0.148`。这说明显式资源约束与低回报尾部惩罚具有正确方向，但不能因样本不足提前宣称方法成立。

test 的57个可靠 oracle 组只有3个 engage、54个 no-op；全数据也只有12个 engage、120个 no-op。三种子 risk-constraint 回归 Critic balanced accuracy 为 `0.398 / 0.435 / 0.435`，通过数 `0/3`。功效点估计需要152个有效 test 组、约261个总状态；95% Wilson 下界对应约760个总状态。均匀扩样成本过高，且不能解决绝对回归被多数类主导的问题。

因此 MCH-PPO、30k/100k 和 GNN 继续冻结。下一阶段冻结本轮效用与 test，定向采集安全临界状态，并比较类别平衡 BCE、成对 margin/ranking 和分位数估值。只有新的独立 test 功效充分且至少2/3训练种子同时降低 false-noop 与 wasteful-engage，才允许进入最小 MCH-PPO。

```text
docs/task_guides/next_research_phase_risk_aware_engagement_utility.md
docs/algorithms/risk_aware_engagement_critic.md
docs/experiments/air_defense_v1_task14_engagement_utility.md
rein_learning/models/risk_aware_engagement_critic.py
rein_learning/common/engagement_utility_diagnostics.py
scripts/run_air_defense_v1_task14_engagement_utility.py
scripts/analyze_air_defense_v1_task14_engagement_power.py
results/air_defense_v1/task14_engagement_utility/
```

## 27. 安全临界采样与类别平衡交战估值结论

更新时间：2026-07-21。

本阶段实现了不读取未来结果的安全临界度采样，以及类别总权重各0.5的 balanced BCE 和 BCE+margin。正式新增144个状态、196个 targeted 上下文组，每分支32次 rollout；历史非 test 的87组只用于扩充训练，所有历史 test 均排除。合并 train/validation/test 为163/39/81组，四组旧测试观测重叠均为0，总回报重构误差为 `7.63e-06`。

定向采样解决了上一阶段的主要功效问题。新 test 的74个可靠组包括28个 engage 和46个 no-op，engage 比例由 `5.3%` 提升至 `37.8%`；medium、time-pressure、heterogeneity 分别包含9、11、8个 engage，全部功效门槛通过。

validation 平均 balanced accuracy 为 BCE `0.695`、BCE+margin `0.721`，因此冻结 margin 候选。正式 test 三种子 BA 为 `0.758 / 0.711 / 0.708`，相对风险回归提高 `0.146 / 0.128 / 0.125`；false-noop 从 `0.429-0.464` 降至 `0.071-0.214`。这说明必要交战样本和类别平衡符号学习均有效。

候选同时出现新的受控失败：seed20/21 的 wasteful-engage 由 `0.348/0.370` 恶化至 `0.413/0.435`，time-pressure no-op recall 只有 `0.455 / 0.182 / 0.273`。三种子均未通过逐场景停止边界，整体 `0/3`。因此 MCH-PPO、30k/100k 和 GNN 继续冻结。下一阶段转向 oracle 监督后 engagement logit 的资源约束阈值或对偶校准，而不是增加网络容量。

```text
docs/task_guides/next_research_phase_critical_state_balanced_engagement.md
docs/algorithms/balanced_engagement_sign_critic.md
docs/experiments/air_defense_v1_task14_balanced_engagement.md
rein_learning/common/critical_engagement_sampling.py
rein_learning/common/balanced_engagement_training.py
scripts/run_air_defense_v1_task14_balanced_engagement.py
results/air_defense_v1/task14_balanced_engagement/
```

## 28. 资源约束交战边界校准结论

更新时间：2026-07-21。

本阶段冻结上一轮 BCE+margin Critic、风险 oracle、环境和临界采样协议，只在历史 validation 上比较全局阈值与成本-弹药资源对偶边界。正式独立测试新增72个状态、84个上下文组和每分支32次 rollout；81个可靠组包含31个 engage 与50个 no-op，三个场景均达到功效门槛。六组旧数据观测重叠均为0，总回报重构误差为 `7.63e-06`。

validation 上全局阈值平均 BA 为 `0.750`，资源对偶为 `0.759`，但两种方法均没有任何种子满足逐场景双类召回约束。按冻结选择规则进入独立 test 的资源对偶边界，三种子 BA 为 `0.593 / 0.612 / 0.605`，no-op recall 为 `0.38 / 0.32 / 0.34`，wasteful-engage 为 `0.62 / 0.68 / 0.66`，整体通过数 `0/3`。

no-op 样本平均资源压力 `0.610` 高于 engage 的 `0.467`，说明成本-弹药压力方向合理但两类高度重叠。固定标量 `lambda` 无法同时表达当前目标紧迫度、未来风险、剩余任务预算和替代单元能力；validation 为保持 engage recall 选择的负阈值还抵消了部分资源惩罚。

因此不恢复 MCH-PPO，不运行30k/100k，也不进入 GNN。下一阶段停止扩大标量阈值网格，转向状态条件资源预算、显式 cost-value/约束价值和独立或交叉拟合校准。只有新的机制在独立 test 上至少2/3种子同时通过安全召回、资源停止与逐场景门槛，才允许恢复最小 MCH-PPO。

```text
docs/task_guides/next_research_phase_resource_constrained_engagement_calibration.md
docs/algorithms/resource_constrained_engagement_boundary.md
docs/experiments/air_defense_v1_task14_engagement_calibration.md
rein_learning/common/engagement_boundary_calibration.py
scripts/run_air_defense_v1_task14_engagement_calibration.py
results/air_defense_v1/task14_engagement_calibration/
```

## 29. 状态条件资源预算与显式双价值结论

更新时间：2026-07-21。

本阶段实现 `StateConditionedEngagementValue`，直接预测 no-op/engage 成对分支的安全收益和增量资源成本，并使用非负状态条件资源乘子构成约束分数。模型选择采用202个历史非 test 上下文的三折 grouped cross-fitting，不再重复使用单一 validation；safety-only、global-budget 和 state-budget 使用相同 folds 与优化预算。

交叉拟合选择 state-budget，其 OOF BA 为 `0.778 / 0.756 / 0.764`，后两个种子满足全部逐场景约束。正式独立测试新增72个状态、97个上下文组和每分支32次 rollout；87个可靠组包含30个 engage 与57个 no-op。旧观测重叠全部为0，总回报重构误差为 `1.53e-05`，数据与功效门槛全部通过。

正式 test 三种子 BA 为 `0.834 / 0.776 / 0.768`，engage recall 为 `0.967 / 0.833 / 0.800`，no-op recall 为 `0.702 / 0.719 / 0.737`，wasteful-engage 降至 `0.298 / 0.281 / 0.263`。三种子均通过总体双类召回、风险回归双非劣、安全收益符号和推理成本门槛，证明状态条件资源价格相对固定阈值取得实质改善。

完整门槛仍为 `0/3`。seed20/21 的 time-pressure no-op recall 为 `0.563/0.625`；seed22 的 medium engage recall 为 `0.556`，heterogeneity no-op recall 为 `0.550`。安全收益相关系数约 `0.49-0.53`，成本相关系数仅 `-0.04-0.13`，剩余瓶颈已收窄为跨场景鲁棒预算和低方差 cost-delta 辨识。

因此暂不恢复 MCH-PPO，也不进入 GNN。进入 MCH-PPO 前预计只剩一次针对性修订：冻结双价值结构，采用逐场景最坏召回/分布鲁棒选择与可靠性加权成本监督，并在新的独立 test 上要求至少2/3种子完整通过。

```text
docs/task_guides/next_research_phase_state_conditioned_constrained_value.md
docs/algorithms/state_conditioned_engagement_value.md
docs/experiments/air_defense_v1_task14_state_conditioned_value.md
rein_learning/models/state_conditioned_engagement_value.py
rein_learning/common/state_conditioned_value_training.py
scripts/run_air_defense_v1_task14_state_conditioned_value.py
results/air_defense_v1/task14_state_conditioned_value/
```

## 30. 跨场景鲁棒预算与可靠成本监督结论

更新时间：2026-07-21。

本阶段冻结 state-budget 网络结构，比较 standard、场景-类别等权最差块损失和可靠成本差加权。三种目标使用相同三折 grouped cross-fitting、模型种子和优化预算。正式测试新增72个状态、88个上下文组和每分支32次 rollout；81个可靠组包含38个 engage 与43个 no-op。旧观测重叠全部为0，总回报重构误差为 `7.63e-06`。

OOF 可行种子数分别为 `2/3、2/3、1/3`。鲁棒目标只改善 seed20，却使 seed21/22 的最差召回或成本相关性退化；standard 的平均最差召回更高，因此按冻结规则继续选择 standard。可靠成本权重没有形成稳定 cost correlation 改善。

在新 test 上，冻结 state-budget 三种子 BA 为 `0.758 / 0.662 / 0.634`。seed20 只在 heterogeneity no-op recall `0.611` 上失败；seed21/22 的 heterogeneity engage recall 降至 `0.273 / 0.182`，安全收益符号准确率也降至 `0.675 / 0.675`。这与上一批异质场景 engage recall 全为1.0形成方向翻转。

因此上一阶段“只剩一次场景鲁棒修订”的估计过于乐观。普通按状态交叉拟合共享同一批次生成机制，不能检验新的临界状态子分布。当前不恢复 MCH-PPO，也不进入 GNN。下一阶段应生成多个独立训练/校准批次，以 batch_id 做 leave-one-batch-out，并保留新的最终 test；不得使用本轮 test 回灌训练。

```text
docs/task_guides/next_research_phase_cross_scenario_robust_budget.md
docs/algorithms/cross_scenario_robust_engagement_value.md
docs/experiments/air_defense_v1_task14_cross_scenario_robust_value.md
scripts/run_air_defense_v1_task14_cross_scenario_robust_value.py
results/air_defense_v1/task14_cross_scenario_robust_value/
```

## 31. 多批次临界状态语料与留一批次泛化结论

更新时间：2026-07-21。

本阶段生成三个独立训练批次，每批包含48个状态、三个核心场景和每分支32次共同随机数 rollout；合并后共有144个状态、193个上下文组和183个可靠组，其中68个 engage、115个 no-op。批次间、与历史训练/测试数据以及最终独立批次的状态重叠均为0，三个批次分别满足最低可靠组与双类别功效门槛。

留一批次实验比较 standard、scenario-robust 和 scenario-robust-reliable-cost 三种目标。冻结规则选择 reliable-cost 目标，但其三种子只有 seed21 同时满足最差批次、最差场景、双类别召回和安全收益符号门槛，可行数仅为 `1/3`。这说明普通混合批次平均指标会掩盖批次外停止边界漂移。

最终测试使用第四个全新72状态批次，共87个上下文组和79个可靠组，其中35个 engage、44个 no-op。三种子 balanced accuracy 为 `0.664 / 0.710 / 0.716`；engage recall 为 `0.829 / 0.943 / 0.886`，no-op recall 为 `0.500 / 0.477 / 0.545`。全部九个种子-场景组合的 no-op recall 均低于0.65，完整通过数为 `0/3`。相较上一批异质场景漏交战，本轮误差方向转为系统性过度交战。

因此数据功效和批次覆盖已不再是首要问题，MCH-PPO、30k/100k 和 GNN 继续冻结。下一阶段不立即生成新 rollout，而是复用三训练批次与现有 OOB 预测，开展安全召回、停止召回、最差批次和最差场景之间的 Pareto 可行性审计。若不存在跨批次共同可行边界，则必须修改价值语义或显式约束结构。

```text
docs/task_guides/next_research_phase_multibatch_leave_one_out.md
docs/algorithms/multibatch_engagement_value_generalization.md
docs/experiments/air_defense_v1_task14_multibatch_leave_one_out.md
rein_learning/common/multibatch_diagnostics.py
scripts/run_air_defense_v1_task14_multibatch_leave_one_out.py
results/air_defense_v1/task14_multibatch_leave_one_out/
```

## 32. OOB 安全-停止 Pareto 可行性审计结论

更新时间：2026-07-22。

本阶段复用三训练批次的1737行 OOB 预测，对每个连续 score 枚举所有能产生不同二分类结果的阈值。主目标继续冻结为 `scenario_robust_reliable_cost`，没有新增 rollout，也没有读取最终 test 标签。门控同时约束总体、最差批次、最差场景的 engage/no-op recall、总体 BA 和 safety sign accuracy。

零阈值仅 seed21 完整通过，可行数为 `1/3`。种子级鲁棒阈值扫描后，seed20/21/22 分别存在23、20、2个可行阈值，选定阈值为 `0.1052 / 0.0288 / 0.3540`，BA 为 `0.798 / 0.832 / 0.757`，主门控达到 `3/3`。因此当前连续 score 并未失去安全-停止排序能力，上一阶段的主要失败来自默认零阈值和随机种子间 score 尺度漂移。

该结果仍然脆弱。seed22 的可行区间仅为 `0.3540-0.3629`，选定点最小约束余量为0；共享原始阈值最多只能使2/3种子通过，不能证明模型间 score 已具有统一尺度。本阶段只证明历史 OOB 上存在可行边界，不证明新批次泛化。

按照预注册规则，下一阶段只运行一次全新独立确认批次，并冻结目标函数、种子级鲁棒阈值选择协议和全部门槛。确认至少2/3种子完整通过后，才允许恢复最小 MCH-PPO；30k/100k 和 GNN 继续冻结。

```text
docs/task_guides/next_research_phase_oob_pareto_feasibility.md
docs/algorithms/oob_safety_stop_pareto_calibration.md
docs/experiments/air_defense_v1_task14_oob_pareto_audit.md
rein_learning/common/pareto_feasibility.py
scripts/run_air_defense_v1_task14_oob_pareto_audit.py
results/air_defense_v1/task14_oob_pareto_audit/
```

## 33. 冻结 OOB 校准协议的独立确认结论

更新时间：2026-07-22。

本阶段在生成数据前冻结 reliable-cost 目标、三个最终模型和 OOB 种子级阈值 `0.1052 / 0.0288 / 0.3540`，然后只生成一次 `eval_seed=887000` 的独立确认批次。模型没有重训，阈值没有重新扫描。

确认批次包含72个状态、87个上下文组和每分支32次 rollout。81个可靠组包含35个 engage 与46个 no-op，三个场景均有双类别；与19个历史数据集重叠全部为0，总回报重构最大误差为 `7.63e-06`。数据完整性和功效门控全部通过。

冻结阈值下三种子 BA 为 `0.625 / 0.646 / 0.625`，engage recall 为 `0.771 / 0.857 / 0.686`，no-op recall 为 `0.478 / 0.435 / 0.565`，完整通过数为 `0/3`。seed20/21 的最差场景 no-op recall 均为 `0.333`；seed22 的异质场景 engage recall 仅为 `0.364`。零阈值同样为 `0/3`。

三个种子的 safety sign accuracy 仍达到 `0.740 / 0.740 / 0.753`，说明安全价值方向没有完全失效；但固定 OOB 边界不能泛化到新批次。当前结论从“阈值可能可校准”收紧为“历史批次内可校准，但跨批次 score 尺度与约束语义不稳定”。

因此不恢复 MCH-PPO，不运行30k/100k，也不进入 GNN。确认标签不得回灌阈值，不追加第二确认批次。下一阶段应修改机制，研究跨批次尺度对齐、预测不确定性和显式安全-资源约束，而不是继续扩大阈值网格或随机批次数。

```text
docs/task_guides/next_research_phase_independent_calibration_confirmation.md
docs/experiments/air_defense_v1_task14_independent_confirmation.md
rein_learning/common/independent_confirmation.py
scripts/run_air_defense_v1_task14_independent_confirmation.py
results/air_defense_v1/task14_independent_confirmation/
```

## 34. 跨批次统一概率校准与不确定性约束结论

更新时间：2026-07-22。

本阶段只使用三训练批次的 OOB 预测，预注册 score Platt、value-context Platt 以及 `z=0.5/1.0` 的保守置信下界四个候选。每个模型种子独立拟合 L2 逻辑回归，样本按批次-场景-类别等权，采用外层 leave-one-batch-out；失败的809000和887000独立标签均未用于拟合或选择。

四个候选完整通过数均为 `0/3`。最佳 score Platt 平均 BA 为0.781、Brier为0.159，但 seed20/21/22 的最差批次 no-op recall 仅为 `0.550 / 0.333 / 0.475`，最差场景 no-op recall 为 `0.517 / 0.552 / 0.586`。混合总体指标继续掩盖局部过度交战。

value-context 的平均预测标准误为 `1.235 / 1.797 / 1.368`。LCB虽然改善部分 no-op，却使 seed21 的最差批次 engage recall 降为0；不同批次的错误方向不能由统一保守惩罚解决。因此当前失败不只是原始 score 的温度或截距漂移，而是安全与资源信息压缩成单一标量后的约束语义不稳定。

由于 OOB 未达到2/3，脚本没有生成预留的941000独立批次，没有新增 rollout。MCH-PPO就绪状态为 false，30k/100k和GNN继续冻结。下一阶段应实现显式安全收益下界、资源成本上界与停止可行域，不再扩展线性校准、LCB系数、标量阈值或随机批次。

```text
docs/task_guides/next_research_phase_cross_batch_uncertainty_calibration.md
docs/algorithms/cross_batch_uncertainty_calibration.md
docs/experiments/air_defense_v1_task14_cross_batch_calibration.md
rein_learning/common/cross_batch_calibration.py
scripts/run_air_defense_v1_task14_cross_batch_calibration.py
results/air_defense_v1/task14_cross_batch_calibration/
```

## 35. MCH-PPO 最小在线机制压力实验结论

更新时间：2026-07-22。

根据“停止继续卡在外围前置任务、直接进入 MCH-PPO”的研究决策，本阶段实现了可训练的 `MaskedCounterfactualHierarchicalPPO`。候选复用因子化 engagement-target 策略，使用冻结的 hierarchical Q-Critic seeds 14/15/16 集成构造逐单元反事实 advantage，并对 engagement 与 conditional target 的 PPO ratio 独立裁剪；联合 GAE 只训练状态价值头。

实验在读取结果前冻结 `time_pressure/heterogeneity_pressure`、训练种子8/9/10、每模型10k steps和每场景30回合评估。共训练12个模型、执行24个场景评估块。候选保持 invalid action、assignment conflict 和 overkill 为0，但出现3/6个同场景绝对 no-op 塌缩。time_pressure 中高威胁突防率均值增加0.1028、损伤增加0.2944、奖励下降10.09；heterogeneity_pressure 中对应增加0.0903、0.2855和下降8.54。总机制门控失败。

time_pressure/seed9 是唯一同时改善奖励、突防和损伤的配对，但其他种子不复现；该结果只证明机制存在条件性潜力，不能通过选择该种子宣称 MCH-PPO 优势。当前状态从“未实现”推进到“训练原型已实现但机制未成立”，因此不进入30k/100k。下一版必须修改信用接入机制，而不是继续挑种子或单纯扩大预算。

MCH 平均训练时间为113.86秒/模型，对照为73.84秒/模型，当前实现约慢54.2%。后续若恢复扩大实验，需要先缓存 rollout 级反事实优势并向量化候选 Critic 评估。

```text
docs/task_guides/next_research_phase_mch_ppo_mechanism_stress_test.md
docs/algorithms/masked_counterfactual_hierarchical_ppo.md
docs/experiments/air_defense_v1_mch_ppo_mechanism_stress_test.md
rein_learning/algorithms/policy_gradient/mch_ppo.py
scripts/run_air_defense_v1_mch_ppo_stress_test.py
tests/test_mch_ppo.py
results/air_defense_v1/mch_ppo_mechanism_stress_test/
```

## 36. RG-MCH-PPO 核心信用机制验证结论

更新时间：2026-07-22。

本阶段不再追加外围前置诊断，直接实现 Reliability-Gated MCH-PPO。算法保留标准化 on-policy GAE 作为 engagement/target 两层主信用，只将冻结 Critic 集成一致的反事实 advantage 作为系数0.5、绝对幅度不超过0.5的残差。零可靠度时自动退化为层级 GAE，Critic 全程冻结。

实验冻结 time_pressure/heterogeneity_pressure、种子8/9/10、10k steps和每场景30回合评估，只新增训练6个候选模型；baseline 和 MCH v0 复用完全相同协议的上一阶段结果。候选在两个场景的平均奖励和损伤均优于 MCH v0，其中 time_pressure 奖励提高8.30、损伤下降0.171，heterogeneity_pressure 奖励提高23.03、损伤下降0.606。异质场景相对 factorized PPO 也实现奖励提高14.49、损伤下降0.320和高威胁突防率下降0.081。

总机制门控仍为 false。RG-MCH 有2/6个同场景运行绝对塌缩；异质场景成本比为1.259，超过1.10门槛。最后训练更新的 engagement reliability 和门控激活率分别高达0.884与0.888，target 为0.575与0.579。这说明“Critic 集成方向一致”在分布外状态上可能共同错误，当前可靠度过于乐观。

项目因此取得第一项算法核心机制的部分正结果：GAE 锚定已被在线实验证明有效，反事实残差在异质场景有明确收益；但种子稳定性和可靠度语义尚未解决，不能进入30k/100k或论文主结果。下一算法工作应引入状态分布支持/行为锚定可靠度与 engagement 累计漂移约束，而不是继续调整单一融合系数或挑选优势种子。

```text
docs/task_guides/next_research_phase_reliability_gated_mch_ppo.md
docs/experiments/air_defense_v1_rg_mch_ppo_stress_test.md
rein_learning/algorithms/policy_gradient/mch_ppo.py
scripts/run_air_defense_v1_rg_mch_ppo_stress_test.py
tests/test_rg_mch_ppo.py
results/air_defense_v1/rg_mch_ppo_mechanism_stress_test/
```

## 37. SA-RG-MCH-PPO 支持感知与累计漂移实验结论

更新时间：2026-07-23。

本阶段实现了 Critic train split 上的状态-单元-前缀-合法掩码最近邻支持度，并将其与 ensemble agreement 相乘；同时冻结初始 factorized actor，对超出0.10预算的累计 engagement Bernoulli KL施加平方 hinge 惩罚。支持数据严格限制为原 Q-Critic dataset 的338条 train 行。

正式10k三种子双场景实验中，SA-RG-MCH 有5/6个同场景运行绝对 no-op 塌缩。time_pressure 相对factorized PPO的平均奖励下降30.93、损伤增加0.745、高威胁突防增加0.217；heterogeneity_pressure对应下降12.32、增加0.341和增加0.104。所有核心任务门控均失败。

训练诊断提供了新的关键证据。engagement/target context support只有0.124/0.022，组合可靠度0.114/0.014，反事实残差0.049/0.008，表明在线actor访问的上下文大部分位于原始Critic支持域之外。初始anchor KL仅0.017，未产生任何惩罚，说明deterministic 0.5阈值跨越可以在很小的分布KL下造成all-noop。

反事实残差关闭后，算法仍采用joint GAE但对engagement和target分别计算ratio与clip；它并不严格退化为factorized PPO的joint ratio与joint clip。5/6塌缩由此把核心瓶颈进一步定位到独立层级近端更新本身。下一算法版本必须以标准joint PPO surrogate为安全主干，只把支持感知反事实信用作为辅助项，并直接诊断或约束deterministic engagement margin。不得继续调整最近邻尺度或KL预算，也不进入GNN。

```text
docs/task_guides/next_research_phase_support_anchored_rg_mch_ppo.md
docs/experiments/air_defense_v1_sa_rg_mch_ppo_stress_test.md
rein_learning/common/masked_context_support.py
rein_learning/algorithms/policy_gradient/mch_ppo.py
scripts/run_air_defense_v1_sa_rg_mch_ppo_stress_test.py
tests/test_sa_rg_mch_ppo.py
results/air_defense_v1/sa_rg_mch_ppo_mechanism_stress_test/
```

## 38. BPCE-PPO v0 核心算法阶段启动

更新时间：2026-07-23。

任务状态：已完成；软件验收通过，10k机制门控失败。

SA-RG-MCH 的5/6塌缩已经否决“joint GAE + engagement/target 独立
ratio/clipping”作为安全 fallback。项目不再修补离线 Critic 支持距离、KL
预算或标量校准器，正式转入 **Boundary-Probed Counterfactual Engagement
Auxiliary PPO（BPCE-PPO）**。

BPCE-PPO 保留 factorized PPO 的完整 joint ratio、joint clipping、GAE、
value loss 和动态掩码语义，只在当前 rollout 的 engagement 决策边界附近
生成稀疏成对反事实标签。标签只形成 engagement logit 排序辅助损失，不再
替换联合信用，也不为 engagement/target 建立独立 PPO 主干。

v0 冻结以下关键语义：

- rollout 中当前单元之前的动作前缀保持不变；
- engage 分支使用冻结旧策略在动态合法目标上的 masked argmax；
- 当前单元之后的同一步后缀及后续时刻由冻结旧策略确定性补全；
- 命中随机性使用按“环境时间步×目标”索引的共同随机带，而不是仅复用 seed；
- 候选按绝对 engagement margin 排序，并受冻结边界半径约束；
- 可靠性门控使用成对差值方向一致性和最小任务效应，不把小样本 SNR 当作正式置信区间；
- `probe_budget=0`、`lambda_cf=0` 或全部标签被拒绝时，训练路径必须数值退化为现有 factorized joint PPO。

当前先完成环境快照/恢复、共同随机带、探测器、BPCE rollout buffer、joint
PPO 辅助训练和严格退化测试；随后运行256步 smoke，再按冻结协议执行
10k、三种子、双核心场景机制证伪。GNN、target 反事实辅助、30k/100k
扩大实验继续冻结。

```text
docs/task_guides/next_research_phase_bpce_ppo_v0.md
docs/algorithms/boundary_probed_counterfactual_engagement_ppo.md
rein_learning/common/boundary_counterfactual_probe.py
rein_learning/algorithms/policy_gradient/bpce_ppo.py
scripts/run_air_defense_v1_bpce_ppo_stress_test.py
tests/test_bpce_ppo.py
```

正式实验最终使用 `K=2、B=8、interval=2`。每个候选模型平均执行20轮
探测、选择40个边界上下文、接受10.5个标签，训练时间为factorized PPO的
1.940倍。项目完整回归达到242项通过，BPCE定向测试为14项通过。

10k三种子双场景中，BPCE v0有2/6个同场景运行绝对all-noop，均为seed9。
time-pressure相对baseline的平均奖励下降24.953、损伤增加0.587、高威胁
突防增加0.153。heterogeneity-pressure的奖励提高21.686、损伤下降0.509、
高威胁突防下降0.129，但资源成本达到baseline的1.928倍。

边界探测在time-pressure相对等预算随机探测奖励提高13.620、损伤下降
0.314；在heterogeneity-pressure反而奖励下降13.563、损伤增加0.340。
seed9在两个场景分别只接受1/5个边界标签，且全部为负；异质场景seeds
8/10分别获得8/7个正标签并改善安全，但seed8交战率升至0.959。当前瓶颈
由“错误fallback”进一步收窄为“正负证据覆盖和辅助更新剂量随种子分叉”。

总机制门控失败，不进入30k/100k、target辅助或GNN。下一修订保持joint
PPO与paired rollout冻结，只允许增加双向证据覆盖门控、正负类别平衡和
辅助梯度预算。

```text
docs/experiments/air_defense_v1_bpce_ppo_stress_test.md
results/air_defense_v1/bpce_ppo_mechanism_stress_test/
```

## 39. BPCE 标签语义审计结论

更新时间：2026-07-23。

任务状态：阶段 A 已完成但未通过；阶段 B/C 未启动。

BPCE-PPO v0 已证明 joint PPO fallback 和成对反事实探测的工程可行性，但
2/6 all-noop、seed9 单边负标签、异质场景1.928倍资源成本以及边界选点
跨场景反转，说明不能直接把 coverage-balanced loss 作为下一算法版本。
当前先审计反事实标签是否真正表示 engagement 条件价值。

本阶段复用 `mch_ppo_mechanism_stress_test` 中两个核心场景、seeds 8/9/10
的六个10k factorized PPO 冻结模型。每个“场景×种子”固定选择6个安全
临界和6个资源临界上下文，共72个；每个分支执行32次共同随机数 rollout。
同一上下文同时计算当前 argmax-target/deterministic 标签、目标精确边缘化
deterministic 标签和目标边缘化/stochastic-continuation 标签。

正式审计生成72个上下文、2304条重复记录、169条目标记录和266,198个
额外transition，Actor参数最大差为0，项目完整回归247项通过。A/B总体符号一致率为0.901，可靠
反转为0/24，说明masked-argmax target不是当前主要混叠来源；B/C总体一致
率只有0.778，低于0.80门槛，deterministic continuation不能继续作为默认
标签语义。

标签 C 只有25/72可靠；六个块中`time_pressure/seed9`为0，
`heterogeneity_pressure/seed9`为2。两个场景可靠正/负标签分别为10/0和
14/1，资源槽没有可靠负标签。阶段 A 同时未通过总体功效、块级功效、
B/C一致和双向覆盖门控。

因此辅助剂量阶段 B、选点阶段 C、下一版10k均未启动。当前不得直接实现
类别平衡或coverage-balanced loss；下一机制必须先验证随机后续或短视窗
安全/资源分量能否形成跨种子双向可靠标签。30k/100k、target辅助和GNN
继续冻结。

```text
docs/task_guides/next_research_phase_bpce_label_semantics_and_dose_audit.md
rein_learning/common/bpce_label_semantics.py
scripts/run_air_defense_v1_bpce_label_semantics_audit.py
tests/test_bpce_label_semantics.py
results/air_defense_v1/bpce_label_semantics_audit/
```

## 40. BPCE 短视窗安全—资源双分量审计结论

更新时间：2026-07-23。

任务状态：阶段 A2 已完成但未通过；BPCE在线辅助主线暂停。

阶段 A 已确认target argmax不是主要可靠方向错误来源，但deterministic
continuation、总体标签功效和可靠STOP覆盖失败。当前新增阶段 A2，保持
原72个上下文、合法目标精确边缘化、随机策略延续和32次共同随机数重复，
只改变标签读出。

主标签使用由快照目标time-to-impact决定的事件窗，分别估计zone damage、
high-threat leak和resource cost的engage-noop差值，并输出
`ENGAGE / STOP / AMBIGUOUS`。STOP必须同时排除最小安全收益并确认正资源
代价，不能由“回报不显著”直接推断。

正式运行重建72/72个原上下文，目标概率最大误差`4.98e-13`；完成2304条
重复和127,700个额外transition，Actor参数差为0，完整回归255项通过。

短视窗得到15 ENGAGE、16 STOP和41 AMBIGUOUS，可操作标签31/72，完整
回合对照为27/72。异质场景形成10/14双向标签，但time-pressure只有5/2；
四个场景-种子块低于6个可操作标签，所有块均未同时达到2个ENGAGE和2个
STOP。

time-pressure资源槽18个上下文全部AMBIGUOUS，成本差均值平均为-0.034，
成本下界为正为0/18；强制当前交战主要替代后续射击。异质资源槽成本差
均值为+1.036，得到10个STOP。短视窗标签具有资源异质性条件下的局部
可辨识性，但不是跨场景机制。

阶段 A2 门控失败。阶段B/C、coverage-balanced loss和修订版10k均不启动，
BPCE在线辅助主线暂停。保留joint PPO fallback、成对反事实基础设施和
局部标签可辨识性失败机制，后续不得通过增加重复、视窗或训练预算绕过。

```text
docs/task_guides/next_research_phase_bpce_short_horizon_component_label_audit.md
rein_learning/common/bpce_short_horizon_labels.py
scripts/run_air_defense_v1_bpce_short_horizon_label_audit.py
tests/test_bpce_short_horizon_labels.py
results/air_defense_v1/bpce_short_horizon_label_audit/
```

## 41. 动作替代与弹药机会成本可辨识性审计结论

更新时间：2026-07-23。

任务状态：已完成；完整性和P-R1通过，P-R2/P-R3未通过。

BPCE阶段A2未能在`time_pressure/resource`中识别稳定的累计资源成本差，
但观察到当前强制交战可能替代冻结策略后续本会执行的射击。为区分“累计
成本差”与“失去的未来行动选择权”，项目启动独立的只读机制审计，不将其
命名为BPCE-A3，也不恢复任何在线辅助训练。

审计沿用原72个上下文和32次共同随机数重复。除当前no-op分支`N`和正常
交战分支`E`外，新增嵌套反事实`E-R`：保持当前动作、命中、即时成本、
冷却和目标状态完全不变，只在下一次策略观察前为被测单元恢复1枚弹药。
核心估计量为未来射击替代`Sub_shot/Sub_cost`、弹药实际复用
`Reuse_probe`、合法动作集合增量`OptionEdge/OptionThreat`以及分离的
安全收益`AmmoGain_D/AmmoGain_L`。

本任务先检验P-R1动作替代解释，再检验P-R2跨场景机会价值和P-R3资源槽
临界性。只有完整性、干预唯一性和三项机制门控全部通过，才允许另立独立
的机会成本oracle可预测性审计；即使通过，也不得直接接入PPO或恢复BPCE。

```text
docs/task_guides/next_research_phase_action_substitution_resource_opportunity_cost_audit.md
rein_learning/common/action_substitution_opportunity_cost.py
scripts/run_air_defense_v1_action_substitution_opportunity_cost_audit.py
tests/test_action_substitution_opportunity_cost.py
results/air_defense_v1/action_substitution_opportunity_cost_audit/
```

正式运行重建72/72个上下文，目标概率最大误差`4.98e-13`；完成2304条
上下文—重复记录、5408条目标—重复干预和219,142个额外transition。
E/E-R当前步身份与单发干预唯一性均为100%，Actor参数差为0，完整回归
259项通过，最大成本分解误差为`4.00e-15`。

P-R1获得强支持：`time_pressure/resource`的18个上下文全部满足
`mean(Sub_shot)>0`且95%下界大于0；11个`cost(E)-cost(N)<=0`上下文全部
由正`Sub_cost`或未来成本构成解释。平均而言，该槽当前交战替代0.990次
未来射击和1.995单位未来资源成本。

P-R2/P-R3未通过。可靠资源机会价值在time和heterogeneity场景分别只有
5/18和2/18；异质场景seed8/seed10均为0，所有可靠资源上下文均为
`missile`。虽然恢复弹药普遍增加未来射击和合法动作边，但其最终安全收益
没有形成跨场景、跨种子和跨资源类型的稳定置信下界。

路线按预注册收敛为：保留“累计成本差受未来动作替代混叠”的机制结论，
停止“通用弹药机会成本在线辅助”路线。不训练机会成本oracle，不恢复
BPCE/MCH-PPO，不用增加重复或选择优势种子绕过失败门控。

## 42. 动作替代测量失真独立确认启动

更新时间：2026-07-23。

任务状态：已完成；P-C1/P-C2通过，P-C3未通过。

R1已确认旧种子8/9/10的`time_pressure/resource`累计成本差受到未来射击
替代混叠，但该结论尚未经过来源策略和状态均独立的确认。R2使用全新
factorized PPO策略种子17/18/19，在`medium`、`time_pressure`和
`heterogeneity_pressure`分别训练9个10k来源模型，并采集与旧正式数据
observation hash零重叠的108个上下文。

resource槽预注册为每块3个missile和3个laser；只运行N/E分支，不运行E-R，
不训练Actor/Critic或机会成本网络。主要门控依次为成本账本完整性P-C1、
`time_pressure/resource`跨新种子复现P-C2以及missile/laser边界P-C3。
无论结果如何，本任务结束后停止扩展机制实验，转入论文贡献冻结或重新
定义第一算法问题。

```text
docs/task_guides/next_research_phase_action_substitution_independent_confirmation.md
rein_learning/common/action_substitution_confirmation.py
scripts/run_air_defense_v1_action_substitution_confirmation.py
tests/test_action_substitution_confirmation.py
results/air_defense_v1/action_substitution_confirmation/
```

正式任务训练并保留9/9个新10k factorized PPO来源模型，选择108个与旧
正式数据hash零重叠的新上下文；resource槽在每块严格保持3个missile和
3个laser。完成3456条重复、7776条目标成本账本和157,485个额外
transition，Actor参数差为0，完整回归264项通过。

首轮审计发现原P-C1公式漏记无冲突自回归后缀在当前步发生的其他单元动作
替代。287/7776条账本受影响，future-only残差最大为2.0；包含当前其他
单元差的扩展恒等式误差为`8.88e-16`。按预注册只修复账本，将总替代成本
拆成同一步其他单元替代、未来被测单元替代和未来其他单元替代，并保存
首轮无效结果后按原配置完整重跑一次。

修正后P-C1通过。P-C2也通过：全新`time_pressure/resource`中13/18个
上下文的`Sub_shot`均值和95%下界为正，三个新种子块下界全部大于0，
7个非正累计成本差上下文全部具有正替代成本。动作替代导致累计成本测量
失真的机制获得独立确认。

P-C3未通过。time场景laser的`rho_sub=1.175`、符号掩盖上下文5个；
missile的`rho_sub=0.571`、掩盖上下文仅2个，未达到3个门槛。当前贡献
冻结为资源类型与场景条件的测量/可辨识性结论，不是跨资源通用规律，也
不是PPO性能改进算法。项目停止追加机制实验，进入claim–evidence冻结和
论文准备。

## 43. W1 主张冻结、双语整稿与对抗性审稿完成

更新时间：2026-07-28。

任务状态：W1-01 至 W1-10 全部完成；T10 通过，阶段出口为 L2/M2。

W1 将 R1/R2 正式证据、三分量成本公式、文献定位、图表、Methods、Results、
Discussion、Limitations、Introduction、Related Work、Abstract 和 Conclusion
整合为中英文终稿。两稿保留 66 个对应 Paragraph ID，关键数字、公式方向、
P-C1/P-C2/P-C3 和所有负边界一致。

W1-10 以三份不同侧重点的模拟审稿完成技术、原创性和可读性压力测试。冻结数据
只读复核确认 9 个模型无行为筛选、108 个 context、3,456 个 repeat、7,776 条
账本、Actor 参数差为 0、P-C1/P-C2 通过且 P-C3 失败。所有 R2/R3/RX 问题已经
关闭；跨环境、跨算法、替代顺序、在线性能和 GNN 仅作为 R4 后续问题登记。

终稿新增 context 选择无结果窥视、小样本正态近似区间边界和真实 Data/Code
Availability。当前没有公共仓库 DOI、accession 或许可证，因此不声称已达到
外部投稿格式定稿。正式出口为：

```text
第一创新测量与可辨识性模块
        W1/T10通过
                    ↓
L2/M2冻结章节、图表、补充材料和证据追溯
                    ↓
移交较大方法论文整合任务
        目标期刊适配和公共发布另行完成
                    ↓
新在线算法/GNN问题必须重新定义并预注册
```

## 44. N1 可辨识资源信用离线门控完成

更新时间：2026-07-28。

任务状态：已完成；N1-P1/P3/P4/P5 通过，N1-P2 失败；阶段出口为
**N1-E4**，在线训练未授权。

N1 比较了三个候选：分量保持的约束信用、全局 CMDP 约束和受控延续差异
回报。系统查新显示，动作效应经后续智能体动作与状态路径的分解已有直接
工作；差异回报和未来条件反事实信用也已覆盖受控延续的主要思想；累计成本
约束则是 CPO 和安全 MARL 的标准问题。

离线审计复用冻结 R2 的 108 个 context 和 7,776 条目标账本，不增加
rollout。四分量恒等式最大误差为 `8.88e-16`，说明标签语义和实现接口
精确；但 4,731 条账本出现正直接成本被非正回合成本差掩盖，含混率为
60.84%。这支持“回合成本差不是唯一局部信用”的诊断，却不能决定应该
优化全局成本、局部责任还是二者的双层关系。

候选 A 因此只保留为方法/诊断组件，候选 B 冻结为后续必需强基线，候选 C
因创新距离与策略分布风险被否决。已生成机器可读 no-go 预注册；种子、状态、
预算和数值门槛保持为空，任何 10k/30k/100k N1 在线训练均被禁止。

下一主线入口是重新定义规范性算法问题和可证伪机制，而不是继续调节辅助
损失、恢复 BPCE/MCH-PPO 或提前引入 GNN。

## 45. N2 未来可覆盖性责任证书静态门控通过

更新时间：2026-07-29。

任务状态：N2-P1 至 P5 全部达到进入下一证伪阶段的门槛；出口为
**N2-E1**。在线训练仍未授权。

N2 将 N1 未解决的“全局预算—局部责任”关系重新定义为未来任务可行域
外部性。对当前合法动作 \(i\rightarrow j\)，FCRC 从两个比较分支中都
排除当前目标 \(j\)，再比较消耗该资源前后对其余带 TTI 威胁的最大加权
一次覆盖值。该定义不执行后续策略，也不分解实现回报。

实现采用目标子集动态规划，人工轨迹 9 项通过。冻结 R2 的 108 个 context
按原模型、种子和协议重放，`context_id/hash/unit/legal_targets` 全部匹配。
243 个合法前缀动作中 86 个具有正外部性（35.39%），34 个 context 出现
同一单元不同目标的责任跨度。FCRC 与单元成本、目标损伤权重的 Spearman
相关分别为 0.466 和 −0.128；平均计算时间 1.02 ms/context。

查新未发现公式等价工作，但 FCRC 与 WTA 匹配机会价值、reachability 和
safe shield 相邻，因此当前只称“通过静态开发门控的算法候选”。下一任务
必须比较高/低 FCRC 同状态动作在共同随机数 continuation 下的其他威胁
覆盖和条件损伤，并检验其相对成本、威胁、原始匹配分数、二元 shield 和
N1 替代量的增量预测价值。

FCRC 在预测验证通过前不得进入 reward、PPO loss 或 action mask；GNN、
10k/30k/100k 在线训练继续冻结。

## 46. N3 FCRC 冻结成对预测验证完成

更新时间：2026-07-29。

任务状态：N3-P1/P5 通过，N3-P2/P3/P4 失败；阶段出口为
**N3-E3**。在线训练未授权。

N3 使用来源策略种子17/18/19和全新的状态/分支基准种子，在medium、
time-pressure和heterogeneity-pressure九个区组中先按FCRC跨度冻结选择
32个上下文。所有状态与此前135个hash零重叠，各区组3–4个上下文；每个
上下文完成64次no-op/high/low共同随机数比较，共82,219个额外transition，
Actor参数差为0，全量回归283项通过。

high-FCRC相对low-FCRC的其他威胁截获权重损害差均值为0.0184，但单侧
符号翻转`p=0.3511`，未通过主方向门槛。候选级FCRC与截获损害的Spearman
为0.4153，但基线与加入FCRC后的留一区组CV MAE均为0.137041，增量约为0。
三个场景中medium/time为正、heterogeneity为负；泄漏损伤差均值为−0.0500，
也未通过安全一致性门槛。

首次门控实现误加“本批hash必须互异”的非预注册条件。修正只重算P1和出口，
没有重选、重跑或改变统计量；正式判决从错误的E4改为N3-E3。

FCRC只保留为静态可覆盖性解释量，不进入reward、loss、mask、shield或GNN。
项目不追加同分布种子修补该命题。下一主线入口再次回到规范性算法问题重定义，
需要提出与已失败的回报责任、机会成本oracle和FCRC不同的新可证伪机制。

## 47. LR-01 反事实效应分解定向阅读完成

更新时间：2026-07-29。

任务状态：`PASSED`；零实验修改，在线训练未授权。

对 ICML 2025 *Counterfactual Effect Decomposition in Multi-Agent Sequential
Decision Making* 的正文、公式、证明、识别假设和实验附录完成核对。关键公式
边界为 `TCFE = tot-ASE - r-SSE`；普通 SSE 不能与 tot-ASE 直接相加重建
总效应。论文已经一般性覆盖动作效应经后续智能体行为和环境状态传播的解释
问题，因此项目不得声称首次发现一般动作替代或首次完成智能体—状态效应分解。

R2 与该工作的关系判为“部分重合”：N/E 回合成本差是特定结果上的反事实总差，
但四通道成本恒等式按资源事件身份和时间记账，不构造 tot-ASE/r-SSE，也没有
单独识别命中和状态路径。论文正文只干预 `t'>t` 的后续动作，未显式表示
AirDefense 同一步自回归后缀；只有把单元决策展开为 micro-time SCM 后才能
纳入其框架。

最终判决为：该论文是局部责任研究的 `BASELINE`，micro-time 路径图可
`ADAPT`，把 TCFE/ASE/Shapley/ICC 直接接入 PPO 必须 `AVOID`；全局资源约束
与局部解释量之间的规范接口仍为 `OPEN`。结果已移交 LR-05；本节完成时
LR-05 仍等待 LR-04，后续状态见第 48 节。

## 48. LR-04 PASPO 约束分配定向阅读完成

更新时间：2026-07-29。

任务状态：`PASSED`；零实验修改，未下载或运行外部代码，在线训练未授权。

完成 NeurIPS 2024 *Autoregressive Policy Optimization for Constrained
Allocation Tasks* 正式论文、证明附录、实验消融和官方仓库静态实现核对。
PASPO 在固定连续凸多面体上按顺序求每个分量的 LP 可行区间，以缩放 Beta
条件策略直接生成合法分配；其去偏机制从完整多面体近似均匀采样并拟合各位置
Beta 参数，只修改输出层初始 bias，不修改 PPO 梯度、目标或采样顺序。

PASPO 已覆盖“自回归硬约束分配”和“完整可行域初始化去偏”的一般叙事，
项目不得把自回归动作本身称作创新。它不直接适用于 AirDefense 的离散
unit-target matching、状态依赖合法集、每单元 no-op 和未来弹药责任。Task 10
观察到的是训练后异质角色参与与资源成本分叉，Task 11–12 还存在 deterministic
argmax 放大和 PPO 种子级 all-noop，因此不能由 PASPO 的初始化机制直接解释。

最终判决为：PASPO 是 `BASELINE`；完整可行域校准思想可 `ADAPT` 为离散可行
后缀计数均匀初始化；连续 Beta/LP 的直接移植为 `AVOID`；该初始化能否缓解
训练后顺序和 no-op 分叉仍为 `OPEN`。报告给出了不改 PPO 的最小可比基线，
但本任务不授权实现或实验。

LR-05 的 LR-01/LR-04 前置现已齐备，状态从等待 LR-04 改为 `READY`。后续必须
区分 PASPO 的动作生成初始化偏置与 CAPO/COSAC 的顺序信用估计偏差。

## 49. LR-03 GradS 多约束梯度塑形定向阅读完成

更新时间：2026-07-29。

任务状态：`PASSED`；零实验修改，未下载或运行外部代码，在线训练未授权。

完成 L4DC 2024 *Gradient Shaping for Multi-Constraint Safe Reinforcement
Learning* 的多约束 CMDP、MOO 统一框架、梯度关系、GradS 算法、理论上界、
实验与局限核对。GradS 并非构造共同下降投影，而是按 cost-gradient 余弦相似度
删除过于同向和过于反向的约束，再从候选集合随机抽取一个梯度。它只比较
cost–cost 关系，不显式处理 reward–cost 冲突。

Theorem 4 在 Slater 可行性和梯度有界光滑条件下给出含删除数和随机采样项的
梯度范数上界；这些 noise-ball 项不随训练步数自动消失。该结果不是所有约束
逐次满足、逐轨迹安全或 PPO 深度近似收敛保证。论文实验证明 GradS 在作者构造
的连续多成本任务上具有较强均值表现，但部分 Cost-N 仍超过阈值，尺度实验也
主要通过复制相似阈值/边界约束扩维。

项目对照表明，当前状态条件双价值中的成本相关仅为
`-0.044 / 0.034 / 0.128`，后续跨场景、跨批次和统一校准门控仍失败；
BPCE 也没有独立 damage/leak/resource cost critics，且标签存在种子级单边
缺失。因此当前问题首先是约束语义、预算可行性与 cost-gradient 可靠性，而非
已知正确梯度的组合方式。此时接入 GradS 可能只会把估计噪声误判为冲突或冗余。

最终判决为：GradS 是未来显式多约束 PPO 的 `BASELINE`；分层 cosine 与置信度
审计可 `ADAPT` 为只读诊断；当前接入 BPCE/MCH/PPO 为 `AVOID`；damage、leak、
resource 的规范身份、可行预算、可靠 cost critics 与尾部安全接口仍为 `OPEN`。

LR-02 与 LR-03 现已形成完整的规范目标层边界：先建立同结构
centralized constrained factorized PPO，再把
`Vanilla / CRPO / Min-Max / GradS` 作为约束梯度聚合消融；在前置门控完成前
不产生在线算法任务。

## 50. LR-05 COSAC 顺序反事实信用定向阅读完成

更新时间：2026-07-29。

任务状态：`PASSED`；零算法与实验修改，未下载或运行外部代码，在线训练未授权。

核验 arXiv `2604.17693` 的两个版本：2026-04-20 v1 名称为 CAPO，
2026-05-09 当前 v2 已更名为 COSAC，并新增四个 Qwen3-0.6B 代理的 ARC
实验。两者核心均为 Sequential Aristocrat Utility，不是两个独立算法。
v1 的理想 on-policy gradient-MSE theorem 没有保留在 v2，因此当前版本只应
引用 SeqAU 唯一性、advantage bias 和 variance 结果，不能声称实际深度 PPO
梯度已获有限样本保证。

COSAC 直接覆盖固定顺序、前缀条件 baseline、上游抵消、direct/indirect
advantage 和无需环境调用的虚拟策略后缀。其 critic-free 含义是不训练
Bellman Q/V Critic，但每批仍通过 ridge 拟合 action-only additive team-reward
surrogate。它避免一种冻结 Critic，却没有消除 batch coverage、非加性残差和
context/state drift。

理论审计还发现：论文用 `d=KA` 的全 per-agent one-hot 拼接定义
`G_mu=E[psi psi^T]`，同时假设其最小特征值大于零。当 `K>1` 时，各 agent
block 的列和均恒等于1，完整设计至少有 `K-1` 个线性依赖，所以
`lambda_min(G_mu)=0`。ridge 可保证求逆，但不能把正动作边际自动变成完整
特征空间覆盖；论文的 `N*kappa_mu` 方差收缩需要在 reference/effect coding
或 action-contrast 子空间中重述。

项目压力测试判定：COSAC 的同一步 downstream indirect effect 与 MCH 叙事
高度重合，但论文只处理 sequential bandit。AirDefense 的焦点动作还会改变
后缀动态合法支持，并经命中、目标存活、弹药和冷却影响未来环境步骤。动态
mask 本身只是已有自回归策略的工程实例；“动态支持变化 + full MDP +
非加性交互”才是尚未解决的数学问题，目前只够重写研究问题，不足以授权
新算法。

最终决策为：SeqAU/COSAC 是 `BASELINE`；prefix-conditional estimand 和
direct/indirect 误差审计可 `ADAPT`；把 action-only ridge、虚拟后缀或独立
层级 clipping 直接接入当前 PPO 为 `AVOID`；完整 MDP、动态支持、跨时间动作
替代、多约束规范和严格 joint PPO 接口仍为 `OPEN`。MCH-PPO v0 放弃作为
主算法，RG-MCH 仅保留 GAE 锚定的机制证据，BPCE 保持暂停；R2 动作替代
测量贡献继续保留。

LR-06 现已转为 `READY`。下一步审计必须同时比较冻结 Q-Critic、每批 COSAC
ridge surrogate 和 BPCE/C3 真实 replay 标签的 policy、context/state 与
feasible-support 三类漂移，并继续要求零辅助系数严格恢复 factorized joint PPO。

## 51. LR-06 OCR-CFT 离线到在线接入审计完成

更新时间：2026-07-29。

任务状态：`PASSED`；LR-01 至 LR-06 阅读任务包全部完成；零算法与实验修改，
未下载或运行外部代码，在线训练未授权。

完成 NeurIPS 2024 *Optimistic Critic Reconstruction and Constrained
Fine-Tuning for General Offline-to-Online RL* 的正文、证明、实验、实现细节和
三套伪代码核对。论文把 O2O 失败拆为 evaluation mismatch、improvement
mismatch 和后续在线分布漂移：先固定可靠离线 actor，用目标在线算法的评价规则
重构 Critic；再以 actor 高概率动作为锚压低误导性 Q；最后用历史参考策略的
KL/MSE 或辅助 advantage 约束在线微调并逐步放松。

审计确认该方法不是 uncertainty penalty。其保证依赖可靠离线 actor、单策略
覆盖集中性、完整离线转移/轨迹、较小 Bellman 误差和可可靠更新的参考策略；
内部 value-alignment 区间不是相对真实价值的安全界，CFT 的渐近
`lambda*=0` 也不是有限步安全或单调改进保证。O2PPO 没有把离线 Q 直接替换
GAE，而是保留 GAE 并加入参考策略对数概率辅助 advantage。

项目对照把历史叙事进一步收窄：普通 Task 14 Q-Critic 只证明 MAE 改善，未通过
动作排序；后续只有 target 排序或总体 BA 出现局部强信号。MCH/RG-MCH 的问题
不是“已正确离线 Critic 上线后突然失效”，而是把固定分布局部能力和 ensemble
一致性过度外推为在线 policy-improvement certificate。SA-RG 的
engagement/target support 仅为 `0.1244/0.0218`，且小 KL 不能阻止
deterministic argmax 跨越 0.5 边界；BPCE 虽更接近当前分布，仍受双向标签覆盖
和辅助剂量分叉影响。

最终判决为：OCR-CFT 是 `BASELINE` 机制参照；“估值器重构/刷新 → 改进方向
对齐 → 在线支持与行为约束 → 严格 joint-PPO fallback”可 `ADAPT` 为审计框架；
直接乐观重构、连续动作 value alignment、小 KL 放行和历史最优 return 选参考
策略为 `AVOID`；动态 state-prefix-mask 支持、联合 argmax、多约束规范和 exact
fallback 下的可拒绝改进方向证书仍为 `OPEN`。

任务包出口保持指导文件约束：六篇结果进入人工头脑风暴，只形成覆盖矩阵、强
基线、no-go 清单和可证伪问题，不自动创建 N4、Critic 重训或在线算法任务。

## 52. DST-01 动态支持敏感信赖域研究契约冻结完成

更新时间：2026-07-29。
任务状态：`PASSED`；零训练、零策略修改、零正式 DS 结果读取；`DST-02` 已解锁。

依据 DS-TR 拆分执行包完成了研究契约、字段字典、机器可读门控表和证据源清单。
v0 主度量固定为可行后缀集合的 Jaccard 距离，并由旧策略概率形成结构风险，再以
结构风险加权新旧策略总变差。最后决策位置不属于主命题；空后缀或空并集为枚举
完整性错误；completion count 只作描述，禁止 Q、reward、威胁和资源成本加权。

P1 已冻结为基础变量模型与基础变量加 DS 模型的场景—策略种子分组外增量比较，
动作对不得跨折泄漏；P2 固定比较 `K0=KL+clip+entropy`、`K1=K0+普通翻转` 和
`K2=K1+DS 加权翻转`，并将“小 KL”冻结为 `approx_kl <= 0.01`；P3 固定为
heterogeneity 10k、种子 8/9/10、order 012 的 factorized joint PPO 配对筛选，
同时要求安全、资源、target 排序和非冻结门通过。

契约产物位于
`results/air_defense_v1/dynamic_support_trust_region/dst_01_contract/`。下一步只执行
DST-02 的精确后缀枚举器、DS 度量性质和环境掩码交叉验证；仍无训练授权。

## 53. DST-02 精确后缀枚举器与 DS 度量验证完成

更新时间：2026-07-29。
任务状态：`PASSED`；零训练、零策略修改、环境合法性规则未修改；`DST-03` 已解锁。

新增独立于策略网络的动态支持模块，直接复用 AirDefense-v1 正式 `action_mask()`，
并只叠加现有自回归动作头的前缀目标占用语义。模块能够枚举任意事实前缀后的全部
有序合法后缀，计算 Jaccard 动作对距离、合法动作代价矩阵、旧策略结构风险和
策略级 DS 距离；最后决策位置、非法动作和空并集保持 DST-01 的显式边界。

定向测试 14 项全部通过；与环境、决策跟踪及 conflict-free joint codec 组合后的
36 项回归测试全部通过。正式验证覆盖 medium、time pressure、heterogeneity
pressure，5 个环境种子、3 种 unit order、60 个动态状态和 720 个前缀。枚举后缀
与 `Discrete(136)` 联合动作真值的对称差为 0，重复后缀为 0，环境掩码副作用为 0，
720 次重复枚举均确定性一致。

验收产物位于
`results/air_defense_v1/dynamic_support_trust_region/dst_02_metric_validation/`。
下一步执行 DST-03，从冻结证据源恢复状态—前缀—动作对语料并进行完整性审计；
该任务仍不授权训练。

## 54. DST-03 冻结状态—前缀语料重建与完整性审计完成

更新时间：2026-07-29。
任务状态：`PASSED`；零训练、零环境重新采样；尚未执行 P1 正式统计门；
`DST-04` 已解锁。

可恢复性审计确认，原 169,887 行顺序诊断以及其他决策/聚合 CSV 没有保存精确
observation、基础 mask 或 state snapshot，不能直接近似计算 DS，已完整记入排除
台账。优先级 1 的 Task12 probe corpus 则保存了 768 个互异的冻结 observation 和
正式 action mask；结合优先级 2 对应的 Task11 order-012 冻结模型种子 0/1/2，
可以在不重新采样环境的前提下执行确定性诊断重放。

重建语料统一标记为 `replay`，最终包含 19,073 个合法无序动作对、2,432 个
状态—前缀上下文；time pressure 和 heterogeneity pressure 分别有 811 和 789 个
上下文。2,176 个上下文因合法动作不足 2 个而没有动作对，2,304 个最后位置按
DST-01 规则排除。

完整性审计中，环境基础 mask、条件 mask、事实策略 argmax、动作对覆盖、Jaccard
公式、交换对称性、模型/配置唯一性、重复 ID 和追溯错误全部为 0，主语料追溯率
为 100%。Parquet、上下文摘要和排除台账连续两次重建哈希一致。

验收产物位于
`results/air_defense_v1/dynamic_support_trust_region/dst_03_frozen_corpus/`。
下一步执行 DST-04，只读取这份冻结语料完成 DS-0 非退化与增量机制硬门；不得
回到原始聚合 CSV 补标签或根据结果修改 DST-01 门槛。

## 55. DST-04 DS-0 增量机制审计与硬门完成

更新时间：2026-07-29。
任务状态：`PASSED`；零训练、零策略或环境修改；`DST-05` 已解锁。

正式分析只读取 DST-03 冻结语料中的两个核心场景，共 12,511 个合法动作对、
1,600 个上下文和 6 个场景—策略种子组。每个 `context_id` 在模型和指标中
具有相同总权重，动作对不跨折泄漏。M0 使用 DST-01 预注册基础变量，M1 只增加
`ds_jaccard`，按场景×策略种子留一组外推。

DS 非退化门全部通过：pooled IQR 为 `0.333333`，6/6 组 IQR 不低于 0.05，
18/18 个样本量合格的 position×noop_pair_type×legal_count 分层中 DS 极差
不低于 0.10。高威胁合法但未分配变化的 AUROC/平衡准确率增量为
`0.066528/0.055062`，前缀阻断变化为 `0.087856/0.100790`；两者 bootstrap
95% 区间下界均大于 0、两个核心场景方向均非负、6/6 块 log-loss 改善非负，
且 1,000 次冻结分层内置换的 max-T FWER p 均为 `0.000999`。因此 P1
按冻结硬门通过。

防伪检查表明，DS 与合法动作数的 Spearman 相关仅 `-0.013103`；高威胁结果在
engage-engage/noop-engage 与 position 0/1 子集中均保留正 AUROC 增量，普通
downstream argmax flip 在两个通过结果上的 AUROC/BA 最大增量只有 `0.012044`。
因此主结果不能由合法动作数、no-op、单一位置、单一种子或普通确定性翻转单独
解释。与此同时，`engagement_extreme_direction_nonzero` 未通过增量和分块
方向门，前缀阻断增量主要来自 noop-engage；这两项限制已作为创新主张边界保留。

正式产物位于
`results/air_defense_v1/dynamic_support_trust_region/dst_04_ds0_audit/`，
报告为 `docs/experiments/air_defense_v1_ds0_dynamic_support_audit.md`。
当前结论只确认冻结重放语料上的增量解释力，不证明时间先行、因果效应或
DS-TR 算法收益。下一步执行 DST-05，增加更新级只读诊断字段并审计现有轨迹/
检查点的可重放性；DST-05 仍不授权训练。

## 56. DST-05 更新级诊断仪表与可重放性完成

更新时间：2026-07-30。
任务状态：`PASSED`；零 PPO 训练、零环境 step、零策略或环境语义修改；
后续新增 `DST-05.5` 作为 DST-06 前置硬门。

新增通用只读 `DynamicSupportInstrumentationCallback`。它只在显式附加后，于
`model._n_updates` 表示的完成 PPO update 之后比较新旧策略；关闭时不加载 probe、
不建立上下文、不写文件。指标包括冻结 K0 的 `approx_kl/clip_fraction/entropy`、
K1 的 `unweighted_prefix_flip_rate`、K2 的 `ds_weighted_flip_mass`，以及
margin crossing、双向 engage/noop flip、joint argmax flip、精确后缀数变化和
三项 probe 退化率。

Task12 的冻结 probe 在 DS 任务前已经生成。本任务保留 time pressure 与
heterogeneity pressure 的全部 512 状态，不按历史塌缩时刻或结果筛选，并将
order 012 的所有可行前缀确定性展开为 5,881 个唯一 `context_id`；其中
1,752 个位置具有下游决策并进入 DS 主指标，最后位置只记录普通边界变化。
合法动作数覆盖 1—6，高威胁可达/不可达上下文为 2,125/3,756。历史
timestep-0 probe 聚合确认 seeds 9/10 在核心场景中同时覆盖 engagement margin
两侧。

`ds_weighted_flip_mass` 落实为唯一合格上下文上的
`mean[1(a_old != a_new) * r_old(a_new)]`，其中 `r_old` 使用 DST-01 冻结的
Jaccard 结构风险；同时保留完整概率质量的 `ds_policy_distance` 作为描述量。
该实现不读取动作对表，所以动作对行复制不会改变更新级聚合。

不干扰性审计使用同一冻结 factorized 模型的两份逐位相同拷贝，执行一次不保存、
不接触环境的冻结批次合成梯度步。仪表开启/关闭两路的训练 RNG、合成 rollout
actions、loss 和更新后参数全部逐位一致；两次 probe 重放的离散事件逐位一致，
连续量最大误差为 0。相关定向与 DS 回归测试 18 项全部通过。

历史重放审计确认 Task12 seeds 8/9/10 各保存了 16 行训练诊断，但每个种子只有
一个 30,208 步最终模型；日志没有中间权重、上下文级概率或 DS 加权 flip。
因此 `replay_insufficient=true`，没有把其他实验最终模型拼成伪时间序列。
DST-05 本身不产生 P2 证据；下一步先执行 DST-05.5，冻结正式事件评估、
rollout 时间轴，并验证真实 callback 不改变训练轨迹；通过后 DST-06 才能执行
冻结的 `heterogeneity_pressure, 10k × seeds 8/9/10` 短跑。

机器产物位于
`results/air_defense_v1/dynamic_support_trust_region/dst_05_instrumentation/`，
正式报告为
`docs/experiments/air_defense_v1_ds_update_instrumentation_audit.md`。

## 57. DST-05.5 事件时间轴冻结与真实 Callback 集成预检完成

更新时间：2026-07-30。
任务状态：`PASSED`；仅执行两路各 512-step 的集成 smoke；未形成 P2 证据，
未保存模型，未执行正式 10k，未实现 DS-TR；`DST-06` 已解锁。

正式事件与同步 probe 诊断已经分离。塌缩事件只由
`heterogeneity_pressure` 独立评估环境上的 50 回合 CRN 结果产生，episode
seeds 固定为 `73000...73049`；事件条件仍为 all-noop episode rate 不低于
0.98，或 actionable engagement rate 低于 0.01。每个策略种子只选择首次
onset；训练前已塌缩种子不作为可判定事件种子；未来 1—3 更新标签和
onset 前 6 更新窗口均只使用新冻结的 `rollout_update_index`。

真实 SB3 时间轴已核对：一轮 256-step rollout 完成 10 epochs PPO train 后，
`rollout_update_index` 增加 1，而原始 `_n_updates` 增加 10。因此正式 10k
请求将记录 40 次 rollout 更新、`sb3_n_updates=10...400` 和实际 10,240
timesteps，禁止再用 `_n_updates±1` 构造窗口。

两路等价性预检使用相同 seed 8 和 512-step 配置：Route A 不附加仪表且不加载
probe，Route B 附加 DS 仪表和正式事件评估。两轮 actions、rewards、dones、
advantages、returns 全部逐位一致；loss、KL、clip fraction、entropy 绝对误差
均为 0；初始/最终参数、最终 optimizer state 和两轮采样前 RNG 均逐位一致。
Route B 恰好生成 2 行更新与 3 个事件点，时间轴为
`(1,10,256)`、`(2,20,512)`。三次评估均保持训练环境、timestep、scheduler、
参数、梯度、optimizer、策略模式和全局 RNG 不变。

11 项事件逻辑测试覆盖 49/50、0.009/0.01、初始塌缩、首次 onset、前向标签、
并发/事件后/尾部排除、SB3 跳 10、六更新窗口、50 回合完整性和 seed-bank hash，
全部通过。机器产物位于
`results/air_defense_v1/dynamic_support_trust_region/dst_05_5_event_timeline_preflight/`，
正式报告为
`docs/experiments/air_defense_v1_dst05_5_event_timeline_preflight.md`。

本任务只确认 DST-06 数据接口有效。下一步执行 DST-06 冻结的
`heterogeneity_pressure, requested 10k × seeds 8/9/10` 诊断短跑；不得把本次
无事件 smoke 解释为 DS 没有先行性，也不得提前实现 DST-07 的 DS-TR。
