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
