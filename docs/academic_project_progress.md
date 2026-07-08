# Academic Project Progress

Updated: 2026-07-08

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
DQN
REINFORCE
```

Current verification:

```text
pytest: 18 passed
```

Q-learning and DQN have been verified on GridWorld. REINFORCE is present as a policy-gradient baseline and has smoke-test coverage.

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

The project can now run basic RL algorithms, but the academic research problem is not yet fully formalized.

The previous missing core was:

```text
a simplified but publishable air-defense resource-assignment environment.
```

The first implementation of this environment has now started, and rule-based baselines are available. The next remaining gap is to train and compare learning-based methods:

- DQN/PPO training scripts
- experiment logging
- scenario parameter sweeps

## 5. Recommended Next Research Step

The next priority is not implementing more generic algorithms.

The next priority is:

```text
Design AirDefenseResourceAssignmentEnv v0.
```

Suggested first version:

- multiple defense units
- multiple incoming UAV targets
- discrete time steps
- finite ammunition / resource capacity
- target threat level
- target movement toward protected asset
- action: assign defense resource to target, or no-op
- reward: intercept high-threat targets, protect asset, avoid wasted fire, penalize leakage

This environment should first be simple enough for DQN/PPO/MARL baselines, then gradually extended.

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

Start with deterministic or lightly stochastic simulation. Keep physics simple.

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

## 7. Immediate Next Task

Recommended next task after the initial v0 environment implementation:

```text
Implement the first DQN/PPO training scripts for AirDefenseResourceAssignmentEnv v0.
```

Target output:

```text
rein_learning/trainers/air_defense_dqn.py
rein_learning/trainers/air_defense_ppo.py
scripts/train_air_defense_dqn.py
scripts/train_air_defense_ppo.py
tests/test_air_defense_trainers.py
```

This will turn the environment from a code-level simulation into an experiment-ready research platform.
