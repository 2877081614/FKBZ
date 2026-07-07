import importlib

import torch


def check_module(name: str, import_name: str | None = None) -> None:
    module_name = import_name or name
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", "unknown")
    print(f"{name}: {version}")


print(f"torch: {torch.__version__}")
print(f"cuda_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_runtime: {torch.version.cuda}")
    print(f"gpu: {torch.cuda.get_device_name(0)}")

check_module("gymnasium")
check_module("stable_baselines3")
check_module("sb3_contrib")
check_module("ray")
check_module("ray.rllib")
check_module("pettingzoo")
check_module("supersuit")
check_module("tianshou")
check_module("torchrl")

print("RL environment verification passed.")
