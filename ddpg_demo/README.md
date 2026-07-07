# Simple DDPG Demo

This is a lightweight browser-based DDPG demonstration. It does not depend on PyTorch, Gym, or a local server.

Open `index.html` in a browser, then click `Train DDPG`.

## Task

The environment is a tiny 1D continuous-control problem:

- State: `[position, velocity]`
- Action: continuous acceleration in `[-1, 1]`
- Goal: move the point mass to position `0` and stop
- Reward: penalizes position error, velocity, and action energy

## DDPG Components

- Actor network: maps state to continuous action
- Critic network: estimates `Q(state, action)`
- Target Actor / Target Critic: slow-moving target networks
- Replay Buffer: stores transitions
- Exploration noise: added to actor output during training
- Soft update: `target = tau * online + (1 - tau) * target`

The neural networks are small hand-written JavaScript MLPs with manual backpropagation. This keeps the demo portable and easy to inspect.

## Outputs

The page shows:

- reward curve
- smoothed reward curve
- loss curves for Actor and Critic
- rollout trajectory after training
- learned policy action over position
- live animation of the trained controller
