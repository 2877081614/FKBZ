"""
6个实验场景配置参数
Based on: Liu et al., Aerospace 2025, 12, 729, Section 4.1-4.2
"""
import numpy as np

PROTECTED_POS = (0.0, 0.0, 0.0)
DETECTION_RANGE = 10000.0  # 10 km
SCENARIO_DURATION = 300.0  # 300 s max per episode
DT = 1.0  # 1s decision step

# ==============================================================================
# 小规模场景: 2 HELS vs 10 UAVs
# ==============================================================================
SMALL_SCALE = {
    'rural': {
        'n_hels': 2,
        'n_uavs': 10,
        'env_type': 'rural',
        'hels_positions': [
            (3000.0, 0.0, 100.0),     # HELS 1: east
            (-3000.0, 0.0, 100.0),    # HELS 2: west
        ],
        'hels_powers': [30e3, 30e3],  # both 30 kW
        'battery_max': 200.0,          # 200 s continuous irradiation
        'max_episodes': 2000,
    },
    'desert': {
        'n_hels': 2,
        'n_uavs': 10,
        'env_type': 'desert',
        'hels_positions': [
            (3000.0, 0.0, 100.0),
            (-3000.0, 0.0, 100.0),
        ],
        'hels_powers': [30e3, 30e3],
        'battery_max': 200.0,
        'max_episodes': 2000,
    },
    'coastal': {
        'n_hels': 2,
        'n_uavs': 10,
        'env_type': 'coastal',
        'hels_positions': [
            (3000.0, 0.0, 100.0),     # inland side
            (-3000.0, 0.0, 100.0),    # sea side
        ],
        'hels_powers': [30e3, 30e3],
        'battery_max': 200.0,
        'max_episodes': 2000,
    },
}

# ==============================================================================
# 大规模场景: 5 HELS vs 50 UAVs
# ==============================================================================
# HELS部署: 环绕保护资产形成防御圈 (论文 Fig.11)
# HELS 5位于中心(保护资产位置),其余4个在3km距离上均匀分布
LARGE_SCALE = {
    'rural': {
        'n_hels': 5,
        'n_uavs': 50,
        'env_type': 'rural',
        'hels_positions': [
            (3000.0, 0.0, 100.0),         # HELS 1: E
            (1500.0, 2598.0, 100.0),      # HELS 2: NE (60°)
            (-1500.0, 2598.0, 100.0),     # HELS 3: NW (120°)
            (-3000.0, 0.0, 100.0),        # HELS 4: W (180°)
            (0.0, 0.0, 100.0),            # HELS 5: center (protected asset)
        ],
        # 论文: HELS 4最高功率(50kW,前置), HELS 5最低(20kW,保护资产)
        'hels_powers': [30e3, 40e3, 30e3, 50e3, 20e3],
        'battery_max': 200.0,
        'max_episodes': 2000,
    },
    'desert': {
        'n_hels': 5,
        'n_uavs': 50,
        'env_type': 'desert',
        'hels_positions': [
            (3000.0, 0.0, 100.0),
            (1500.0, 2598.0, 100.0),
            (-1500.0, 2598.0, 100.0),
            (-3000.0, 0.0, 100.0),
            (0.0, 0.0, 100.0),
        ],
        'hels_powers': [30e3, 40e3, 30e3, 50e3, 20e3],
        'battery_max': 200.0,
        'max_episodes': 2000,
    },
    'coastal': {
        'n_hels': 5,
        'n_uavs': 50,
        'env_type': 'coastal',
        'hels_positions': [
            (3000.0, 0.0, 100.0),
            (1500.0, 2598.0, 100.0),
            (-1500.0, 2598.0, 100.0),
            (-3000.0, 0.0, 100.0),
            (0.0, 0.0, 100.0),
        ],
        'hels_powers': [30e3, 40e3, 30e3, 50e3, 20e3],
        'battery_max': 200.0,
        'max_episodes': 2000,
    },
}

# ==============================================================================
# 生成所有场景列表
# ==============================================================================
ALL_SCENARIOS = []
for scale_name, scale_config in [('small', SMALL_SCALE), ('large', LARGE_SCALE)]:
    for env_type in ['rural', 'desert', 'coastal']:
        scenario = dict(scale_config[env_type])
        scenario['name'] = f'{scale_name}_{env_type}'
        scenario['scale'] = scale_name
        scenario['dt'] = DT
        scenario['scenario_duration'] = SCENARIO_DURATION
        scenario['detection_range'] = DETECTION_RANGE
        scenario['protected_pos'] = PROTECTED_POS
        ALL_SCENARIOS.append(scenario)


def get_scenario(name):
    """根据名称获取场景配置"""
    for s in ALL_SCENARIOS:
        if s['name'] == name:
            return s
    raise ValueError(f"Unknown scenario: {name}")


def list_scenarios():
    """列出所有场景"""
    return [s['name'] for s in ALL_SCENARIOS]
