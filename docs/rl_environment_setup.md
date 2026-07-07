# Reinforcement Learning Environment

The conda environment is named `rein-learning`. This is the recommended project environment for subsequent reinforcement-learning and simulation development.

## Activate

```powershell
conda activate rein-learning
```

If PowerShell still uses the base Python after activation, run commands through `conda run`:

```powershell
conda run -n rein-learning python scripts\check_rl_dependencies.py
```

## GPU PyTorch

PyTorch was installed from the official CUDA 12.8 wheel index:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verified GPU:

```text
NVIDIA GeForce RTX 5060 Ti
torch: 2.11.0+cu128
cuda_available: True
cuda_runtime: 12.8
```

## Main RL Packages

- `gymnasium`: environment API
- `stable-baselines3`, `sb3-contrib`: common single-agent RL algorithms
- `ray[rllib]`: distributed and multi-agent RL
- `pettingzoo`, `supersuit`: multi-agent environment API and wrappers
- `tianshou`: research-oriented RL algorithms
- `torchrl`: PyTorch RL components
- `tensorboard`, `wandb`: experiment tracking

## Reinstall Missing Packages

Install GPU PyTorch first:

```powershell
conda run -n rein-learning python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then install the RL stack:

```powershell
conda run -n rein-learning python -m pip install -r requirements-rl.txt
```

## Verify

```powershell
conda run -n rein-learning python scripts\verify_rl_env.py
conda run -n rein-learning python scripts\check_rl_dependencies.py
```

Note: `gymnasium[all]` is intentionally avoided because `box2d-py` can fail to build on Windows. The installed package set is enough for custom Gymnasium/PettingZoo environments, GPU training, SB3 algorithms, Ray/RLlib multi-agent training, and follow-on development.
