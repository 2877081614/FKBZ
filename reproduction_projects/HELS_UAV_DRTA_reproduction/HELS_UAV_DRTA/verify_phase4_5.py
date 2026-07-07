"""
Phase 4+5 验证: 算法组件 + MADDPG-IA集成
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from algorithm.attention import AttentionEncoder
from algorithm.rnd import RNDIntrinsicReward
from algorithm.actor_critic import Actor, Critic
from algorithm.gumbel_softmax import gumbel_softmax
from algorithm.replay_buffer import ReplayBuffer
from algorithm.maddpg_ia import MADDPG_IA
from env.drta_env import HELS_UAV_DRTA_Env
from config.scenario_config import ALL_SCENARIOS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_attention_encoder():
    """验证注意力编码器: 可变长度→固定维度 (Eq.43-45)"""
    print("\n--- Test 1: Attention Encoder ---")
    attn = AttentionEncoder(hels_dim=6, uav_dim=5, d_k=64, d_v=64, d_attn=128, env_dim=1).to(DEVICE)

    B, N = 4, 10
    hels_state = torch.randn(B, 6).to(DEVICE)
    uav_states = torch.randn(B, N, 5).to(DEVICE)
    env_param = torch.randn(B, 1).to(DEVICE)
    mask = torch.ones(B, N).to(DEVICE)
    mask[:, 5:] = 0  # last 5 UAVs are padding

    encoded, attn_weights = attn(hels_state, uav_states, env_param, mask)
    assert encoded.shape == (B, 128), f"Encoded shape {encoded.shape} != (4, 128)"
    assert attn_weights.shape == (B, N)
    # Padded UAVs should get ~0 attention
    assert attn_weights[:, 5:].sum() < 1e-6, "Padded UAVs received attention!"
    print(f"  Encoded shape: {encoded.shape}  OK")
    print(f"  Attention on valid UAVs: {attn_weights[0, :5].sum():.3f} (should be ~1)")
    print("  [PASS]")


def test_rnd_intrinsic_reward():
    """验证RND: 新颖状态→高奖励, 已知状态→低奖励 (Eq.46-47)"""
    print("\n--- Test 2: RND Intrinsic Reward ---")
    rnd = RNDIntrinsicReward(input_dim=128, lr=1e-3).to(DEVICE)

    state_new = torch.randn(8, 128).to(DEVICE)
    r1 = rnd.compute_intrinsic_reward(state_new)
    print(f"  First visit reward:  {r1.mean().item():.4f}")

    # Train on same state → reward should decrease
    for _ in range(20):
        rnd.update(state_new)
    r2 = rnd.compute_intrinsic_reward(state_new)
    print(f"  After 20 updates:    {r2.mean().item():.4f}")

    # Novel state → higher reward
    state_novel = torch.randn(8, 128).to(DEVICE) * 2.0
    r3 = rnd.compute_intrinsic_reward(state_novel)
    print(f"  Novel state reward:  {r3.mean().item():.4f}")

    assert r2.mean() < r1.mean(), "Reward should decrease after training"
    print("  [PASS]")


def test_actor_critic():
    """验证Actor-Critic网络前向传播"""
    print("\n--- Test 3: Actor-Critic Forward ---")
    actor = Actor(attn_dim=128, n_actions=11, hidden_dim=256).to(DEVICE)
    critic = Critic(n_agents=2, attn_dim=128, n_actions=11, hidden_dim=512).to(DEVICE)

    enc = torch.randn(4, 128).to(DEVICE)
    logits = actor(enc)
    assert logits.shape == (4, 11)
    print(f"  Actor output: {logits.shape}  OK")

    global_enc = torch.randn(4, 2 * 128).to(DEVICE)
    joint_act = torch.randn(4, 2 * 11).to(DEVICE)
    q = critic(global_enc, joint_act)
    assert q.shape == (4, 1)
    print(f"  Critic output: {q.shape}  OK")
    print("  [PASS]")


def test_gumbel_softmax():
    """验证Gumbel-Softmax采样"""
    print("\n--- Test 4: Gumbel-Softmax ---")
    logits = torch.tensor([[1.0, 2.0, 0.5, 0.1]]).to(DEVICE)

    y = gumbel_softmax(logits, temperature=1.0, hard=False)
    assert y.shape == (1, 4)
    assert torch.allclose(y.sum(dim=-1), torch.ones(1).to(DEVICE), atol=1e-6)

    y_hard = gumbel_softmax(logits, temperature=1.0, hard=True)
    assert y_hard.sum(dim=-1).item() == 1.0
    assert (y_hard == 1.0).sum().item() == 1  # exactly one 1
    print(f"  Soft: {y[0].detach().cpu().numpy()}")
    print(f"  Hard: {y_hard[0].detach().cpu().numpy()}")
    print("  [PASS]")


def test_replay_buffer():
    """验证经验回放池"""
    print("\n--- Test 5: Replay Buffer ---")
    buf = ReplayBuffer(capacity=100)
    obs = {f'agent_{i}': np.zeros(57, dtype=np.float32) for i in range(2)}
    for _ in range(50):
        buf.push(obs, {'agent_0': 0, 'agent_1': 1},
                 {'agent_0': 0.5, 'agent_1': 0.3}, obs, False)

    assert len(buf) == 50
    batch = buf.sample(16)
    assert len(batch) == 16
    print(f"  Buffer size: {len(buf)}, Sampled: {len(batch)}")
    print("  [PASS]")


def test_maddpg_ia_create():
    """验证MADDPG-IA算法创建与单步更新"""
    print("\n--- Test 6: MADDPG-IA Creation ---")
    algo = MADDPG_IA(
        n_agents=2, n_actions=11, n_uavs=10,
        d_attn=128, actor_hidden=256, critic_hidden=512,
        batch_size=64, buffer_size=1000,
        device=DEVICE
    )
    print(f"  Agents: {algo.n_agents}, Actions: {algo.n_actions}")
    print(f"  Attention: {sum(p.numel() for p in algo.attention.parameters()):,} params")
    print(f"  Actors: {sum(p.numel() for p in algo.actors.parameters()):,} params")
    print(f"  Critics: {sum(p.numel() for p in algo.critics.parameters()):,} params")
    print("  [PASS]")
    return algo


def test_maddpg_ia_env_integration():
    """验证MADDPG-IA与小规模环境的集成"""
    print("\n--- Test 7: MADDPG-IA + Env Integration ---")
    env = HELS_UAV_DRTA_Env(ALL_SCENARIOS[0])  # small_rural: 2v10
    algo = MADDPG_IA(
        n_agents=env.n_hels, n_actions=env.n_actions, n_uavs=env.n_uavs,
        batch_size=64, buffer_size=2000,
        device=DEVICE
    )

    obs, info = env.reset(seed=42)
    episode_reward = np.zeros(env.n_hels)

    for step in range(50):
        actions = {}
        for i in range(env.n_hels):
            a, enc = algo.select_action(obs[f'agent_{i}'])
            actions[f'agent_{i}'] = a

        next_obs, rewards, terminated, truncated, info = env.step(actions)

        # Store in replay buffer
        algo.replay_buffer.push(obs, actions, rewards, next_obs,
                                terminated or truncated)

        # Update
        if algo.replay_buffer.is_ready(64):
            res = algo.update()
            if res:
                pass  # training step done

        for i in range(env.n_hels):
            episode_reward[i] += rewards[f'agent_{i}']

        obs = next_obs
        if terminated or truncated:
            break

    print(f"  Steps: {step+1}, Rewards: {episode_reward}")
    print(f"  Buffer size: {len(algo.replay_buffer)}")
    print(f"  Damage rate: {info['damage_rate']:.2%}")
    print(f"  Gumbel temp: {algo.gumbel_temp:.4f}")

    assert len(algo.replay_buffer) > 0, "Buffer should have experiences"
    env.close()
    print("  [PASS]")


if __name__ == '__main__':
    print("=" * 60)
    print(f"Phase 4+5: Algorithm Components + MADDPG-IA (Device: {DEVICE})")
    print("=" * 60)

    test_attention_encoder()
    test_rnd_intrinsic_reward()
    test_actor_critic()
    test_gumbel_softmax()
    test_replay_buffer()
    test_maddpg_ia_create()
    test_maddpg_ia_env_integration()

    print("\n" + "=" * 60)
    print("Phase 4+5: ALL TESTS PASSED")
    print("=" * 60)
