"""
超参数与物理常量配置
Based on: Liu et al., Aerospace 2025, 12, 729, Table 1
"""
import numpy as np

# =============================================================================
# 激光系统参数 (HELS)
# =============================================================================
LASER_WAVELENGTH = 1.064e-6      # lambda: Nd:YAG 激光波长 [m]
LASER_APERTURE = 0.6             # D: 发射孔径 [m]
LASER_POWER_RANGE = (20e3, 50e3) # P0: 激光功率范围 [W]
BETA0 = 1.0                      # beta_0: 衍射极限光束质量因子
ETA0 = 0.85                      # eta_0: 系统总效率 (含光学损耗)
SIGMA_I_OVER_SIGMA_D = 0.3       # sigma_i/sigma_d: 跟踪抖动均方根比

# FSM (Fast Steering Mirror) 参数
FSM_RESPONSE_TIME = 0.05         # t_FSM: 快速转向镜响应时间 [s] (<50ms)
FSM_MAX_ANGLE_DEG = 15.0         # FSM最大偏转角 [deg]
FSM_MAX_ANGLE_RAD = np.deg2rad(15.0)

# 机械转台
MECH_ANGULAR_SPEED_DEG_S = 60.0  # 机械转台角速度 [deg/s]
MECH_ANGULAR_SPEED_RAD_S = np.deg2rad(60.0)

# 电池
BATTERY_CAPACITY = 200.0         # 电池弹匣容量 [s] 连续照射时间

# =============================================================================
# 大气环境参数 (3种环境: 乡村/沙漠/沿海)
# =============================================================================
ATMOSPHERE_PARAMS = {
    'rural': {
        'Cn2_0': 1e-17,          # C_n^2(0): 近地面折射率结构常数 [m^(-2/3)]
        'visibility': 10e3,      # nu: 能见度 [m] (晴空/sunshine)
        'K': 2.828,              # 气溶胶模型常数 (乡村型)
        'wind_speed': 5.0,       # v_g: 地面风速 [m/s]
        'description': '乡村/晴空/弱湍流'
    },
    'desert': {
        'Cn2_0': 1e-15,          # 中湍流
        'visibility': 5e3,       # 5km (轻霾/light haze)
        'K': 2.496,              # 沙漠型气溶胶
        'wind_speed': 5.0,
        'description': '沙漠/轻霾/中湍流'
    },
    'coastal': {
        'Cn2_0': 1e-17,          # 弱湍流
        'visibility': 10e3,      # 10km (晴空)
        'K': 4.453,              # 海洋型气溶胶
        'wind_speed': 5.0,
        'description': '沿海/晴空/弱湍流'
    }
}

# H-V 湍流模型参数
HV_h0 = 0.1e3                    # h0_HV: 特征高度 [m]
HV_RMS_WIND = 27.0               # RMS风速 [m/s]

# =============================================================================
# 目标材料属性 (2024铝合金, 论文 Fig.6 / Eq.17-18)
# =============================================================================
MATERIAL_PROPS = {
    'Al2024': {
        'name': '2024 Aluminum Alloy',
        'rho': 2780.0,           # rho: 密度 [kg/m^3]
        'c_s': 875.0,            # c_s: 固态比热容 [J/(kg.K)]
        'k': 121.0,              # k: 热导率 [W/(m.K)]
        'T_m': 775.0,            # T_m: 熔点 [K] (~502 degC)
        'T_0': 300.0,            # T_0: 环境温度 [K] (~27 degC)
        'L_m': 397e3,            # L_m: 熔化潜热 [J/kg]
        'R': 0.90,               # R: 表面反射率 (抛光铝 @1.064um)
        'z_d': 0.005,            # z_d: 蒙皮厚度 [m] (5mm)
    },
    'Ti_alloy': {
        'name': 'Titanium Alloy',
        'rho': 4420.0,
        'c_s': 560.0,
        'k': 7.2,
        'T_m': 1877.0,
        'T_0': 300.0,
        'L_m': 305e3,
        'R': 0.75,
        'z_d': 0.005,
    },
    'Steel_HS': {
        'name': 'High-Strength Steel',
        'rho': 7850.0,
        'c_s': 475.0,
        'k': 45.0,
        'T_m': 1773.0,
        'T_0': 300.0,
        'L_m': 247e3,
        'R': 0.80,
        'z_d': 0.005,
    },
    'CFRP': {
        'name': 'Carbon Fiber Reinforced Polymer',
        'rho': 1600.0,
        'c_s': 800.0,
        'k': 1.0,
        'T_m': 800.0,            # 分解温度而非熔点
        'T_0': 300.0,
        'L_m': 500e3,            # 有效分解焓
        'R': 0.95,               # 高反射率
        'z_d': 0.005,
    }
}

# 预计算毁伤阈值 e_th = z_d * rho * [c_s*(T_m-T_0) + L_m]   (Eq.18)
for mat_key in MATERIAL_PROPS:
    m = MATERIAL_PROPS[mat_key]
    m['e_th'] = m['z_d'] * m['rho'] * (m['c_s'] * (m['T_m'] - m['T_0']) + m['L_m'])

# =============================================================================
# 威胁与效益因子权重 (Eq.30)
# =============================================================================
LAMBDA1 = 0.5                    # 威胁权重
LAMBDA2 = 0.5                    # 效益权重 (lambda1 + lambda2 = 1)
LAMBDA11 = 0.4                   # 高度威胁子权重
LAMBDA12 = 0.3                   # 速度威胁子权重
LAMBDA13 = 0.3                   # 距离威胁子权重
LAMBDA21 = 0.5                   # 资源消耗子权重
LAMBDA22 = 0.5                   # HELS应用价值子权重

# 威胁因子参数
HEIGHT_REF = 500.0               # h_ref: 参考高度 [m]
THREAT_V0 = 25.0                 # v0: 预设最具威胁速度 [m/s]
THREAT_SIGMA_V = 5.0             # sigma_v: 速度威胁带宽 [m/s]
SAFE_DISTANCE = 1000.0           # L_safe: 安全距离阈值 [m]
RC_SIGMA = 1.0                   # sigma_rc: 资源消耗sigmoid尺度
RC_ALPHA = 0.01                  # alpha_rc: 资源消耗sigmoid形状

# =============================================================================
# MADDPG-IA 训练超参数 (Table 1)
# =============================================================================
MAX_EPISODES = 2000
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 2048
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
SOFT_UPDATE_TAU = 0.01           # varpi in paper
GAMMA = 0.95                     # discount factor
GUMBEL_TEMP_INIT = 1.0
GUMBEL_TEMP_MIN = 0.1
GUMBEL_ANNEAL = 0.9995

# RND参数
RND_BETA_R0 = 0.1                # 内在奖励初始系数
RND_K_R = 1e-4                   # 课程学习衰减率
RND_HIDDEN = 128
RND_OUTPUT = 64

# 混合奖励
GAMMA_MIX = 0.9                  # 外在/内在奖励混合系数
WAIT_REWARD_BETA_D = 0.01        # 等待奖励衰减系数

# 网络结构
D_ATTN = 128                     # 注意力编码输出维度
D_K = 64                         # Key维度
D_V = 64                         # Value维度
ACTOR_HIDDEN = 256
CRITIC_HIDDEN = 512

# 状态维度
HELS_SELF_DIM = 6                # (x,y,z, battery, duration, axis_direction)
UAV_FEAT_DIM = 5                 # (distance, height, speed, azimuth, elevation)
ENV_PARAM_DIM = 1                # Cn2_0 value

# 动作空间
MAX_UAVS = 50                    # 最大UAV数量

# =============================================================================
# 无人机参数
# =============================================================================
UAV_SPEED_MIN = 20.0             # [m/s]
UAV_SPEED_MAX = 30.0             # [m/s]
UAV_HEIGHT_MIN = 200.0           # [m] (0.2 km)
UAV_HEIGHT_MAX = 800.0           # [m] (0.8 km)
UAV_DETECTION_RANGE = 10000.0    # [m] (10 km)
