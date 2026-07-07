import importlib


MODULES = [
    "gymnasium",
    "torch",
    "stable_baselines3",
    "sb3_contrib",
    "ray",
    "ray.rllib",
    "pettingzoo",
    "supersuit",
    "tianshou",
    "torchrl",
    "wandb",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "cv2",
    "pygame",
    "tensorboard",
    "hydra",
]


def main() -> None:
    for module_name in MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"{module_name}: ok ({version})")
        except Exception as exc:
            print(f"{module_name}: missing ({type(exc).__name__})")


if __name__ == "__main__":
    main()
