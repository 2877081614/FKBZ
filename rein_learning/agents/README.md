# Agents

Agents own action selection and learning state.

Examples:

- `tabular_q_agent.py`
- `dqn_agent.py`
- `ppo_agent.py`
- `maddpg_agent.py`

Agents may use models, buffers, and algorithm update functions, but environment code should not import agents.
