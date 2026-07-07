# Implemented Algorithms

This project currently includes three baseline algorithms for discrete-state, discrete-action environments.

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

## Verification

```powershell
conda run -n rein-learning python -m pytest tests
```

Latest verification:

```text
18 passed
Q-learning greedy evaluation: total_reward=3.0, steps=8
DQN greedy evaluation: total_reward=3.0, steps=8, device=cuda
REINFORCE smoke train: episode=001, avg_reward=-33.00, loss=0.0051, device=cuda
```
