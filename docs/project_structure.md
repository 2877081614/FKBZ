# Project Structure Plan

This project is organized around reinforcement-learning boundaries: environments define tasks, agents select and learn actions, models approximate policies or values, algorithms implement update rules, and trainers wire everything together.

## Current Layout

```text
.
├── rein_learning/
│   ├── envs/
│   │   └── discrete/
│   │       └── grid_world.py
│   ├── agents/
│   ├── algorithms/
│   ├── buffers/
│   ├── common/
│   ├── configs/
│   ├── models/
│   ├── simulators/
│   ├── trainers/
│   └── wrappers/
├── rl_envs/
├── scripts/
├── tests/
├── docs/
├── datasets/
├── research_papers/
└── reproduction_projects/
```

`rl_envs/` is kept only as a compatibility import path for the first GridWorld environment. New code should import from `rein_learning`.

## Dependency Direction

```text
trainers
  -> agents
  -> algorithms
  -> models
  -> buffers
  -> envs
  -> simulators
  -> common
```

Recommended rules:

- Environments do not import agents, models, algorithms, or trainers.
- Models do not import environment implementations.
- Algorithms should operate on tensors, arrays, batches, or interfaces, not concrete environments.
- Trainers are allowed to compose everything.
- Domain simulation logic goes in `simulators/`; Gymnasium/PettingZoo interfaces go in `envs/`.

## Where New Code Goes

Use `rein_learning/envs/discrete/` for small toy environments with finite states and finite actions.

Use `rein_learning/envs/air_defense/` later for air-defense task environments. Keep reusable physics, sensor, weapon, and target-motion logic in `rein_learning/simulators/`.

Use `rein_learning/agents/` for objects that expose methods such as `select_action()` and `update()`.

Use `rein_learning/models/` for neural networks or tabular value/policy representations.

Use `rein_learning/algorithms/` for Q-learning, SARSA, DQN, PPO, SAC, MADDPG, HAPPO, or other update rules.

Use `rein_learning/buffers/` for replay buffers and rollout buffers.

Use `rein_learning/trainers/` for executable training loops and evaluation orchestration.

Use `scripts/` for small command-line entry points and checks. Keep scripts thin: they should import a package-level trainer and call `main()`.

Use `tests/` for unit tests and fast integration tests.

## Import Examples

```python
from rein_learning.envs import SmallGridWorldEnv
from rein_learning.envs.discrete import GridWorldConfig
from rein_learning.trainers.grid_world_q_learning import train
```

The old import still works for compatibility:

```python
from rl_envs import SmallGridWorldEnv
```
