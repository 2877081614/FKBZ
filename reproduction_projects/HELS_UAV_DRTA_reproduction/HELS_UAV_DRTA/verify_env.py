"""验证环境安装"""
import torch
import gymnasium as gym
import numpy as np
import matplotlib
import seaborn
import scipy
import pandas
import tensorboard
import tqdm

print("=" * 50)
print("MADDPG-IA 复现环境验证")
print("=" * 50)
print(f"PyTorch:      {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version:  {torch.version.cuda}")
    print(f"GPU:           {torch.cuda.get_device_name(0)}")
    print(f"GPU memory:    {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"Gymnasium:    {gym.__version__}")
print(f"NumPy:        {np.__version__}")
print(f"Matplotlib:   {matplotlib.__version__}")
print(f"Seaborn:      {seaborn.__version__}")
print(f"SciPy:        {scipy.__version__}")
print(f"Pandas:       {pandas.__version__}")
print(f"TensorBoard:  {tensorboard.__version__}")
print(f"tqdm:         {tqdm.__version__}")
print("=" * 50)

# CUDA 功能测试
if torch.cuda.is_available():
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.mm(x, y)
    print(f"CUDA matmul (1000x1000): OK")
    print(f"GPU memory used: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    del x, y, z
    torch.cuda.empty_cache()

# Gymnasium 测试
env = gym.make("CartPole-v1")
obs, info = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
print("Gymnasium CartPole test: OK")

print("=" * 50)
print("所有环境验证通过!")
