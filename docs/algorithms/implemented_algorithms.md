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

## Verification

```powershell
conda run -n rein-learning python -m pytest tests
```

Latest verification:

```text
140 passed
Q-learning greedy evaluation: total_reward=3.0, steps=8
DQN greedy evaluation: total_reward=3.0, steps=8, device=cuda
REINFORCE smoke train: episode=001, avg_reward=-33.00, loss=0.0051, device=cuda
AirDefense DQN smoke evaluation: intercept_rate=0.40, leak_rate=0.40, avg_invalid_actions=0.00
AirDefense v1 Hungarian acceptance: avg_reward=-40.48, damage=1.15, invalid=0.00
```
