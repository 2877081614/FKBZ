"""Quick smoke test of all modules"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("1. Testing physics...", end=" ", flush=True)
t0 = time.time()
from env.physics import HELSDamageModel
m = HELSDamageModel('rural')
r = m.compute_damage_time(30e3, 5000, 500, 0.1)
assert r['t_damage'] > 0
print(f"OK ({time.time()-t0:.1f}s)")

print("2. Testing UAV model...", end=" ", flush=True)
t0 = time.time()
from env.uav_model import UAVSwarmDensityModel
model = UAVSwarmDensityModel()
times = model.generate_arrival_times(50, 300.0)
assert len(times) >= 40
print(f"OK ({time.time()-t0:.1f}s)")

print("3. Testing Gym env...", end=" ", flush=True)
t0 = time.time()
from config.scenario_config import ALL_SCENARIOS
from env.drta_env import HELS_UAV_DRTA_Env
env = HELS_UAV_DRTA_Env(ALL_SCENARIOS[0])
obs, info = env.reset(seed=0)
acts = {f'agent_{i}': 10 for i in range(env.n_hels)}  # all wait
obs, rew, term, trunc, info = env.step(acts)
assert info['time'] > 0
env.close()
print(f"OK ({time.time()-t0:.1f}s)")

print("4. Testing algorithm (CPU)...", end=" ", flush=True)
t0 = time.time()
import torch
from algorithm.maddpg_ia import MADDPG_IA
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
algo = MADDPG_IA(n_agents=2, n_actions=11, n_uavs=10, batch_size=64, buffer_size=500, device=DEV)
print(f"OK ({time.time()-t0:.1f}s, device={DEV})")

print("5. Testing MADDPG-IA update...", end=" ", flush=True)
t0 = time.time()
env = HELS_UAV_DRTA_Env(ALL_SCENARIOS[0])
obs, info = env.reset(seed=1)
for step in range(100):
    acts = {}
    for i in range(env.n_hels):
        a, _ = algo.select_action(obs[f'agent_{i}'])
        acts[f'agent_{i}'] = a
    nobs, rew, term, trunc, info = env.step(acts)
    mixed = {}
    for i in range(env.n_hels):
        mixed[f'agent_{i}'] = algo.compute_mixed_reward(nobs[f'agent_{i}'], rew[f'agent_{i}'])
    algo.replay_buffer.push(obs, acts, mixed, nobs, term or trunc)
    if algo.replay_buffer.is_ready(64):
        res = algo.update()
    obs = nobs
    if term or trunc: break
print(f"OK ({time.time()-t0:.1f}s, {step+1} steps, bufsize={len(algo.replay_buffer)})")
env.close()
print(f"\nALL QUICK TESTS PASSED [{DEV}]")
