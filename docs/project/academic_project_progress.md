# Academic Project Progress

Updated: 2026-07-10

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

## 9. Immediate Next Task

The next task is environment difficulty stratification and generalization testing:

```text
Define easy, medium, and hard AirDefense v1.0 scenario profiles,
then rerun the same multi-seed protocol to determine where Maskable PPO,
greedy assignment, and later graph-based policies separate.
```

Recommended additions before the first novel model:

- fixed scenario profiles;
- Hungarian/optimization assignment baseline;
- conflict and overkill metrics;
- cross-difficulty and unseen-scenario evaluation;
- resource-target graph encoder as the leading algorithmic direction.
