from rein_learning.envs import SmallGridWorldEnv as NewSmallGridWorldEnv
from rl_envs import SmallGridWorldEnv as CompatibleSmallGridWorldEnv


def test_new_and_compatible_env_import_paths_point_to_same_class() -> None:
    assert NewSmallGridWorldEnv is CompatibleSmallGridWorldEnv
