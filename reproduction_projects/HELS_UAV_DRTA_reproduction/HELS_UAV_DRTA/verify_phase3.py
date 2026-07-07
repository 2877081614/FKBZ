"""
Phase 3 验证: Gym仿真环境 (6 scenario configs + random agent test)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from config.scenario_config import ALL_SCENARIOS, list_scenarios
from env.drta_env import HELS_UAV_DRTA_Env


def test_all_scenario_configs():
    """验证所有6个场景配置可以创建环境"""
    print("\n--- Test 1: All 6 Scenario Configs ---")
    for s in ALL_SCENARIOS:
        env = HELS_UAV_DRTA_Env(s)
        print(f"  {s['name']:>15s}: {s['n_hels']} HELS vs {s['n_uavs']} UAVs, "
              f"env={s['env_type']}, obs_dim={env.single_obs_dim}, "
              f"actions={env.n_actions}")
        env.close()
    print("  [PASS]")


def test_observation_action_spaces():
    """验证观测和动作空间"""
    print("\n--- Test 2: Observation & Action Spaces ---")
    env = HELS_UAV_DRTA_Env(ALL_SCENARIOS[0])  # small_rural
    obs, info = env.reset(seed=42)

    for i in range(env.n_hels):
        key = f'agent_{i}'
        assert key in obs, f"Missing {key} in obs"
        assert obs[key].shape == (env.single_obs_dim,), \
            f"Obs shape mismatch: {obs[key].shape} vs ({env.single_obs_dim},)"
        assert obs[key].dtype == np.float32, f"Obs dtype: {obs[key].dtype}"
        print(f"  Agent {i}: obs shape={obs[key].shape}, range=[{obs[key].min():.2f}, {obs[key].max():.2f}]")

    # Check action space
    for i in range(env.n_hels):
        key = f'agent_{i}'
        assert env.action_space[key].n == env.n_actions
        print(f"  Agent {i}: action space = Discrete({env.action_space[key].n})")

    env.close()
    print("  [PASS]")


def test_random_agent_rollout():
    """验证随机智能体可以完成一个完整episode"""
    print("\n--- Test 3: Random Agent Rollout ---")
    env = HELS_UAV_DRTA_Env(ALL_SCENARIOS[0])  # small_rural (2v10)
    obs, info = env.reset(seed=123)

    total_reward = np.zeros(env.n_hels)
    step = 0
    terminated, truncated = False, False

    while not (terminated or truncated):
        actions = {}
        for i in range(env.n_hels):
            actions[f'agent_{i}'] = env.action_space[f'agent_{i}'].sample()
        obs, rewards, terminated, truncated, info = env.step(actions)
        for i in range(env.n_hels):
            total_reward[i] += rewards[f'agent_{i}']
        step += 1

    print(f"  Steps: {step}, Terminated: {terminated}, Truncated: {truncated}")
    print(f"  Rewards: {total_reward}")
    print(f"  Damage rate: {info['damage_rate']:.2%}")
    print(f"  Kills: {info['total_kills']}, Total = {sum(info['total_kills'])}")
    print(f"  Battery remaining: {[f'{b:.1f}s' for b in info['battery_remaining']]}")
    print(f"  Time elapsed: {info['time']:.1f}s")

    assert step > 0, "Should take at least 1 step"
    env.close()
    print("  [PASS]")


def test_large_scale_rollout():
    """验证大规模场景(5v50)随机智能体rollout"""
    print("\n--- Test 4: Large Scale (5v50) Rollout ---")
    s = [s for s in ALL_SCENARIOS if s['name'] == 'large_rural'][0]
    env = HELS_UAV_DRTA_Env(s)
    obs, info = env.reset(seed=456)

    step = 0
    terminated, truncated = False, False
    while not (terminated or truncated):
        actions = {f'agent_{i}': env.action_space[f'agent_{i}'].sample()
                   for i in range(env.n_hels)}
        obs, rewards, terminated, truncated, info = env.step(actions)
        step += 1
        if step >= 200:
            truncated = True

    print(f"  Steps: {step}, Damage rate: {info['damage_rate']:.2%}")
    print(f"  Kills: {sum(info['total_kills'])}/{env.n_uavs}")
    env.close()
    print("  [PASS]")


def test_deterministic_seed():
    """验证固定种子的确定性"""
    print("\n--- Test 5: Deterministic Seed ---")
    env1 = HELS_UAV_DRTA_Env(ALL_SCENARIOS[0])
    env2 = HELS_UAV_DRTA_Env(ALL_SCENARIOS[0])

    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)

    for i in range(env1.n_hels):
        key = f'agent_{i}'
        diff = np.abs(obs1[key] - obs2[key]).max()
        assert diff < 1e-6, f"Non-deterministic obs for agent {i}: diff={diff}"
    print(f"  Same seed → identical observations (max diff < 1e-6)")

    env1.close()
    env2.close()
    print("  [PASS]")


def test_global_state():
    """验证全局状态获取"""
    print("\n--- Test 6: Global State ---")
    env = HELS_UAV_DRTA_Env(ALL_SCENARIOS[0])
    obs, _ = env.reset(seed=42)
    gs = env.get_global_state()
    expected_dim = env.single_obs_dim * env.n_hels
    assert gs.shape == (expected_dim,), f"Global state dim {gs.shape} != {expected_dim}"
    print(f"  Global state dim: {gs.shape}")
    env.close()
    print("  [PASS]")


if __name__ == '__main__':
    print("=" * 60)
    print("Phase 3: Gym Environment Verification")
    print("=" * 60)
    test_all_scenario_configs()
    test_observation_action_spaces()
    test_random_agent_rollout()
    test_large_scale_rollout()
    test_deterministic_seed()
    test_global_state()
    print("\n" + "=" * 60)
    print("Phase 3: ALL TESTS PASSED")
    print("=" * 60)
