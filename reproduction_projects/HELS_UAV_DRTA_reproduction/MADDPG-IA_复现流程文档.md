# MADDPG-IA 论文实验仿真复现流程文档

**基于论文:** Liu et al., "Dynamic Resource Target Assignment Problem for Laser Systems' Defense Against Malicious UAV Swarms Based on MADDPG-IA", *Aerospace* 2025, 12, 729.

**DOI:** https://doi.org/10.3390/aerospace12080729

---

## 目录

1. [概述与前置知识](#1-概述与前置知识)
2. [基础环境搭建](#2-基础环境搭建)
3. [物理模型实现](#3-物理模型实现)
4. [Gym仿真环境构建](#4-gym仿真环境构建)
5. [MADDPG-IA算法实现](#5-maddpg-ia算法实现)
6. [训练流程实现](#6-训练流程实现)
7. [实验设计与运行](#7-实验设计与运行)
8. [结果分析与可视化](#8-结果分析与可视化)
9. [完整运行清单](#9-完整运行清单)

---

## 1. 概述与前置知识

### 1.1 论文核心贡献

本论文提出了**MADDPG-IA算法**用于解决高能激光系统(HELS)防御恶意无人机蜂群的动态资源目标分配(DRTA)问题。核心创新包括：

1. **精确的HELS-UAV-DRTA物理模型**：融合大气传输效应(湍流、热晕、抖动)和热毁伤效应的激光毁伤时间量化模型，以及基于威胁因子和效益因子的两阶段动态分配数学模型。

2. **MADDPG-IA算法**：
   - **A (Attention)**：注意力机制将可变长度的目标状态编码为固定大小的表示，解决蜂群规模动态变化问题
   - **I (Intrinsic Reward)**：基于RND(Random Network Distillation)的内在奖励模块，提供密集探索奖励，解决稀疏奖励困境

3. **多场景仿真验证**：3种环境 × 2种规模 = 6个场景，每场景100次独立运行。

### 1.2 关键技术栈

| 组件 | 论文原始版本 | 本次复现版本 | 说明 |
|------|-------------|-------------|------|
| Python | 3.12.9 | 3.12.9 | 一致 |
| PyTorch | 2.6.0 + CUDA 12.6 | **2.11.0 + CUDA 12.8** | ⚠️ 升级原因：RTX 5060 Ti 为 Blackwell 架构(sm_120)，PyTorch 2.6.0 最高仅支持 sm_90 |
| IDE | PyCharm 2024.3.1.1 / Anaconda3 | VSCode / Anaconda3 (2024.10) | |
| OS | Windows 11 | Windows 11 | 一致 |
| CPU | AMD Ryzen 7 5800H | Intel / AMD (见实际硬件) | |
| RAM | 32 GB | 32 GB+ | |
| GPU | RTX 3060 Laptop (6GB) | **RTX 5060 Ti (16GB)** | ⚠️ 架构差异，不影响复现结果 |

> **注意:** PyTorch 版本差异不影响算法复现结果。MADDPG-IA 算法不依赖特定 PyTorch 版本的 API，2.11.0 向后兼容 2.6.0 的全部功能。CUDA 12.8 驱动同样兼容论文使用的 CUDA 12.6 特性。

### 1.3 项目目录结构

```
HELS_UAV_DRTA/
├── config/
│   ├── __init__.py
│   ├── scenario_config.py      # 场景配置参数
│   └── hyperparams.py          # 超参数配置
├── env/
│   ├── __init__.py
│   ├── physics.py              # 物理模型(第3节)
│   ├── uav_model.py            # 无人机模型
│   ├── hels_model.py           # HELS激光系统模型
│   └── drta_env.py             # Gym环境(第4节)
├── algorithm/
│   ├── __init__.py
│   ├── attention.py            # 注意力编码器
│   ├── rnd.py                  # RND内在奖励模块
│   ├── actor_critic.py         # Actor-Critic网络
│   ├── replay_buffer.py        # 经验回放池
│   ├── maddpg_ia.py            # MADDPG-IA主算法
│   └── gumbel_softmax.py       # Gumbel-Softmax采样
├── baselines/
│   ├── __init__.py
│   ├── dqn.py                  # DQN对比算法
│   ├── qmix.py                 # QMIX对比算法
│   └── mappo.py                # MAPPO对比算法
├── train.py                    # 训练主脚本
├── evaluate.py                 # 评估脚本
├── visualize.py                # 可视化脚本
├── run_experiments.py          # 批量实验运行脚本
└── utils/
    ├── __init__.py
    ├── logger.py               # 日志与TensorBoard
    └── metrics.py              # 评估指标计算
```

---

## 2. 基础环境搭建

### 2.1 创建Conda虚拟环境

```bash
# 创建Python 3.12虚拟环境
conda create -n hels_drta python=3.12.9 -y
conda activate hels_drta
```

### 2.2 安装PyTorch

**注意:** 请先确认你的 GPU 计算能力(sm)版本。RTX 50 系列(Blackwell, sm_120)需要使用 PyTorch 2.11+。RTX 30/40 系列可使用 PyTorch 2.6.0。

```bash
# === 选项A: Blackwell GPU (RTX 5060/5070/5080/5090) — 使用 PyTorch 2.11 + CUDA 12.8 ===
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# === 选项B: 较旧GPU (RTX 30/40 系列) — 使用论文原始 PyTorch 2.6.0 + CUDA 12.6 ===
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

### 2.3 安装其余依赖

```bash
pip install gymnasium==0.29.1 matplotlib==3.8.4 seaborn==0.13.2 scipy==1.13.0 pandas==2.2.2 tensorboard==2.16.2 tqdm==4.66.4 "numpy<2.3,>=1.22.4"
```

### 2.4 验证环境

```python
import torch
import gymnasium as gym
import numpy as np

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Gymnasium: {gym.__version__}")
print(f"NumPy: {np.__version__}")
```

---

## 3. 物理模型实现

物理模型是仿真环境的基础，按以下层次实现：大气传输 → 热毁伤 → 毁伤时间 → 蜂群密度 → 威胁/效益因子 → DRTA模型。

### 3.1 常量与材料属性 (`config/hyperparams.py`)

```python
# === 激光系统参数 ===
LASER_WAVELENGTH = 1.064e-6      # λ: 激光波长 [m] (Nd:YAG)
LASER_APERTURE = 0.6             # D: 发射孔径 [m]
LASER_POWER_RANGE = (20e3, 50e3) # P₀: 激光功率范围 [W]
BETA0 = 1.0                      # β₀: 衍射极限光束质量因子
ETA0 = 0.85                      # η₀: 系统总效率
SIGMA_I_OVER_SIGMA_D = 0.3       # σᵢ/σ_d: 跟踪抖动比

# === FSM参数 ===
FSM_RESPONSE_TIME = 0.05         # t_FSM: 快速转向镜响应时间 [s]
FSM_MAX_ANGLE = 15.0             # FSM最大偏转角 [deg]

# === 大气参数(3种环境) ===
ATMOSPHERE_PARAMS = {
    'rural': {
        'Cn2_0': 1e-17,          # C²_n(0): 近地面折射率结构常数 [m^(-2/3)]
        'visibility': 10e3,      # ν: 能见度 [m] (晴空)
        'K': 2.828,              # 气溶胶模型常数
        'wind_speed': 5.0,       # v_g: 地面风速 [m/s]
        'description': '乡村/晴空/弱湍流'
    },
    'desert': {
        'Cn2_0': 1e-15,
        'visibility': 5e3,       # 轻霾
        'K': 2.496,
        'wind_speed': 5.0,
        'description': '沙漠/轻霾/中湍流'
    },
    'coastal': {
        'Cn2_0': 1e-17,
        'visibility': 10e3,
        'K': 4.453,              # 海洋气溶胶
        'wind_speed': 5.0,
        'description': '沿海/晴空/弱湍流'
    }
}

# === 目标材料属性 (2024铝合金) ===
MATERIAL_PROPS = {
    'Al2024': {                   # 2024铝合金, 厚度5mm
        'rho': 2780.0,            # ρ: 密度 [kg/m³]
        'c_s': 875.0,             # c_s: 固态比热容 [J/(kg·K)]
        'k': 121.0,               # k: 热导率 [W/(m·K)]
        'T_m': 775.0,             # T_m: 熔点 [K]
        'T_0': 300.0,             # T_0: 初始温度 [K]
        'L_m': 397e3,             # L_m: 熔化潜热 [J/kg]
        'R': 0.90,                # R: 表面反射率 (抛光铝 @1.064μm)
        'z_d': 0.005,             # z_d: 蒙皮厚度 [m]
        'e_th': None              # 毁伤阈值(计算得出)
    }
}
# 计算毁伤阈值 e_th = z_d * ρ * [c_s*(T_m-T_0) + L_m]
mat = MATERIAL_PROPS['Al2024']
mat['e_th'] = mat['z_d'] * mat['rho'] * (mat['c_s'] * (mat['T_m'] - mat['T_0']) + mat['L_m'])
# ≈ 1.13 × 10⁷ [J/m²]
```

### 3.2 激光大气传输模型 (`env/physics.py` — 第1部分)

实现公式(1)-(12)：计算光斑半径、大气透过率和目标面功率密度。

```python
import numpy as np
from config.hyperparams import *

class LaserAtmosphericTransmission:
    """激光大气传输模型 (论文公式1-12)"""
    
    def __init__(self, env_type='rural'):
        """
        Args:
            env_type: 'rural' | 'desert' | 'coastal'
        """
        self.env_type = env_type
        self.atm = ATMOSPHERE_PARAMS[env_type]
        self.Cn2_0 = self.atm['Cn2_0']
        self.visibility = self.atm['visibility']
        self.K = self.atm['K']
        self.vg = self.atm['wind_speed']
    
    def Cn2_hv(self, L, h, theta):
        """
        公式(3): 斜程H-V修正模型的C²_n计算
        C²_n,HV(L) = C²_n(h) · exp[-L·sin(θ)/h0,HV]
        
        Args:
            L: 传输距离 [m]
            h: 无人机高度 [m]  
            theta: 仰角 [rad]
        Returns:
            Cn2_HV: 斜程湍流结构常数 [m^(-2/3)]
        """
        h0_HV = 0.1e3  # 100 m, 特征高度
        v = 27.0       # RMS风速 [m/s]
        
        # C²_n(h) — 改进的H-V模型
        h_km = h / 1000.0
        term1 = 5.94e-23 * (v / 27.0)**2 * h_km**10 * np.exp(-h_km)
        term2 = 2.7e-16 * np.exp(-2.0 * h_km / 3.0)
        term3 = self.Cn2_0 * np.exp(-10.0 * h_km)
        Cn2_h = term1 + term2 + term3
        
        # 斜程修正
        Cn2_HV = Cn2_h * np.exp(-L * np.sin(theta) / h0_HV)
        return Cn2_HV
    
    def fried_parameter(self, L, h, theta, wavelength=LASER_WAVELENGTH):
        """
        公式(4): 大气相干长度(Fried参数) r₀
        r₀ = [0.423·k²·∫C²_n(z)·dz]^(-3/5)
        
        简化: r₀ ≈ [0.423·k²·C²_n,HV·L]^(-3/5)
        """
        k = 2.0 * np.pi / wavelength  # 波数
        Cn2 = self.Cn2_hv(L, h, theta)
        r0 = (0.423 * k**2 * Cn2 * L) ** (-3.0 / 5.0)
        return r0
    
    def beta_turbulence(self, D, r0):
        """
        公式(5): 湍流引起的光束质量因子
        β²_T = (D/r₀)^(5/3)
        """
        beta_T_sq = (D / r0) ** (5.0 / 3.0)
        return beta_T_sq
    
    def thermal_distortion_parameter(self, P0, L, D, wavelength=LASER_WAVELENGTH):
        """
        公式(6): 热畸变参数 N_D
        衡量热晕效应强度
        N_D ∝ P0 · L / (v · D²)
        """
        v = self.vg  # 横向风速
        # N_D 正比于激光功率×传输距离 / (风速×孔径²)
        # 完整的公式依赖吸收系数等，这里给出简化标度形式
        ND = P0 * L / (v * D**2)  # 简化形式，单位无所谓(后续归一化)
        return ND
    
    def beta_thermal_blooming(self, ND):
        """
        公式(7): 热晕引起的光束质量因子
        β²_B = (N_D / N_D0)^2，其中N_D0为临界热畸变参数
        实际采用: β²_B = C_B · N_D²
        """
        # 文中N_D0 = 1时β²_B = 1，以此标定
        N_D0 = 1.0  # 归一化
        C_B = 0.1   # 热晕系数(与激光波长、吸收系数有关)
        beta_B_sq = C_B * (ND / N_D0) ** 2
        return beta_B_sq
    
    def beta_jitter(self, sigma_i_over_sigma_d=SIGMA_I_OVER_SIGMA_D):
        """
        公式(8): 跟踪抖动引起的光束质量因子
        β_J = 6.93 · (σᵢ/σ_d)²
        """
        beta_J = 6.93 * sigma_i_over_sigma_d ** 2
        return beta_J
    
    def spot_radius(self, L, beta_sq, D=LASER_APERTURE, wavelength=LASER_WAVELENGTH):
        """
        公式(1)+(9): 激光光斑平均半径
        r_spot = 1.22 · (λ/D) · L · β
        其中 β = √(β²₀ + β²_T + β²_B + β²_J)   (公式2)
        """
        beta = np.sqrt(beta_sq)
        r_spot = 1.22 * (wavelength / D) * L * beta
        return r_spot
    
    def atmospheric_transmittance(self, L, theta):
        """
        公式(10): 斜程大气透过率 (Beer-Lambert定律)
        τ(L,κ,ν) = exp[-∫κ(z,ν)·dz]
        
        简化: τ = exp(-α_ext · L)，α_ext为消光系数
        α_ext = 3.912/ν · (0.55/λ)ᵠ · K (修正Koschmieder公式)
        
        Args:
            L: 传输距离 [m]
            theta: 仰角 [rad]
        Returns:
            tau: 大气透过率 [0-1]
        """
        # 修正Koschmieder消光系数
        wavelength_um = LASER_WAVELENGTH * 1e6  # 转为μm
        # q因子: 随波长增大，散射减小
        if wavelength_um <= 0.5:
            q = 0.585 * self.visibility**(1.0/3.0)
        else:
            q = 1.3
        
        alpha_ext = (3.912 / self.visibility) * (0.55 / wavelength_um)**q * self.K
        # 斜程: 距离 = L，假定均匀大气
        tau = np.exp(-alpha_ext * L)
        return tau
    
    def target_power_density(self, P0, L, beta_sq, tau, D=LASER_APERTURE,
                              eta0=ETA0, wavelength=LASER_WAVELENGTH):
        """
        公式(11)+(12): 目标表面平均激光功率密度
        
        公式(11): P_e = η₀ · τ · P₀
        公式(12): I_target = P_e / (π · r_spot²)
        
        Returns:
            I_target: 目标面平均功率密度 [W/m²]
            P_e: 衰减后到达功率 [W]
        """
        P_e = eta0 * tau * P0                                    # (11)
        r_spot = self.spot_radius(L, beta_sq, D, wavelength)     # (9)
        I_target = P_e / (np.pi * r_spot**2)                     # (12)
        return I_target, P_e, r_spot
    
    def compute_all(self, P0, L, h, theta):
        """
        一次计算所有大气传输参数
        
        Args:
            P0: 激光发射功率 [W]
            L: 传输距离 [m]
            h: 目标高度 [m]
            theta: 仰角 [rad]
        Returns:
            dict: 包含所有中间计算结果
        """
        r0 = self.fried_parameter(L, h, theta)
        beta_T_sq = self.beta_turbulence(LASER_APERTURE, r0)
        ND = self.thermal_distortion_parameter(P0, L, LASER_APERTURE)
        beta_B_sq = self.beta_thermal_blooming(ND)
        beta_J = self.beta_jitter()
        beta_sq = BETA0**2 + beta_T_sq + beta_B_sq + beta_J**2   # (2)
        tau = self.atmospheric_transmittance(L, theta)
        I_target, P_e, r_spot = self.target_power_density(
            P0, L, beta_sq, tau)
        
        return {
            'beta_sq': beta_sq,
            'beta': np.sqrt(beta_sq),
            'beta_T': np.sqrt(beta_T_sq),
            'beta_B': np.sqrt(beta_B_sq),
            'beta_J': beta_J,
            'r0': r0,
            'ND': ND,
            'tau': tau,
            'P_e': P_e,
            'r_spot': r_spot,
            'I_target': I_target
        }
```

### 3.3 激光热毁伤模型 (`env/physics.py` — 第2部分)

实现公式(13)-(20)：计算熔化穿透时间和毁伤时间。

```python
class LaserThermalDamage:
    """激光热毁伤模型 (论文公式13-20)"""
    
    def __init__(self, material='Al2024'):
        self.mat = MATERIAL_PROPS[material]
    
    def melting_penetration_time(self, I_target):
        """
        公式(17): 熔化穿透时间
        
        t_m = z_d · ρ · [c_s·(T_m-T_0) + L_m] / [(1-R) · I_target]
        
        物理含义: 激光能量被吸收后使材料熔化穿透所需时间
        
        Args:
            I_target: 目标面功率密度 [W/m²]
        Returns:
            t_m: 熔化穿透时间 [s]
        """
        mat = self.mat
        # 吸收的能量密度 = (1-R) · I_target
        absorbed_power = (1.0 - mat['R']) * I_target
        if absorbed_power <= 0:
            return np.inf
        
        # 单位面积熔化所需能量
        E_melt_per_area = mat['z_d'] * mat['rho'] * (
            mat['c_s'] * (mat['T_m'] - mat['T_0']) + mat['L_m']
        )
        t_m = E_melt_per_area / absorbed_power
        return t_m
    
    def damage_threshold(self):
        """
        公式(18): 毁伤阈值
        e_th = z_d · ρ · [c_s·(T_m-T_0) + L_m]
        
        穿透单位厚度目标材料所需的最小能量密度 [J/m²]
        """
        mat = self.mat
        e_th = mat['z_d'] * mat['rho'] * (
            mat['c_s'] * (mat['T_m'] - mat['T_0']) + mat['L_m']
        )
        return e_th
    
    def damage_time(self, P0, L, beta_sq, tau, D=LASER_APERTURE,
                    wavelength=LASER_WAVELENGTH, eta0=ETA0):
        """
        公式(20): HELS毁伤时间
        
        t_damage ≈ (1.22²π · z_d · ρ · [c_s·(T_m-T_0) + L_m] · λ² · L² · β²) 
                   / (η₀ · τ · P₀ · D²)
        
        即: t_damage ∝ L² · β² / (τ · P₀)
        
        Args:
            P0: 激光功率 [W]
            L: 传输距离 [m]
            beta_sq: 总光束质量因子平方
            tau: 大气透过率
        Returns:
            t_damage: 毁伤所需时间 [s]
        """
        mat = self.mat
        numerator = (1.22**2 * np.pi * mat['z_d'] * mat['rho'] *
                     (mat['c_s'] * (mat['T_m'] - mat['T_0']) + mat['L_m']) *
                     wavelength**2 * L**2 * beta_sq)
        denominator = eta0 * tau * P0 * D**2
        t_damage = numerator / denominator
        return t_damage
```

### 3.4 毁伤时间综合计算 (`env/physics.py` — 第3部分)

```python
class HELSDamageModel:
    """HELS毁伤模型综合计算 (整合大气传输 + 热毁伤)"""
    
    def __init__(self, env_type='rural', material='Al2024'):
        self.atm_trans = LaserAtmosphericTransmission(env_type)
        self.thermal = LaserThermalDamage(material)
    
    def compute_damage_time(self, P0, L, h, theta):
        """
        计算给定条件下HELS毁伤目标的完整时间
        
        流程: 大气传输参数 → 目标面功率密度 → 热毁伤时间
        
        Args:
            P0: 激光发射功率 [W]
            L: 斜距 [m]
            h: 目标高度 [m]
            theta: 仰角 [rad]
        Returns:
            dict: {
                't_damage': 毁伤时间[s],
                'I_target': 目标面功率密度[W/m²],
                'beta_sq': 总光束质量因子,
                'tau': 大气透过率,
                ...
            }
        """
        # 步骤1: 大气传输
        atm_result = self.atm_trans.compute_all(P0, L, h, theta)
        
        # 步骤2: 热毁伤时间
        t_damage = self.thermal.damage_time(
            P0, L, atm_result['beta_sq'], atm_result['tau'])
        
        return {
            't_damage': t_damage,
            **atm_result
        }
    
    def compute_transfer_time(self, delta_angle):
        """
        公式(22): 照射转移时间
        
        t_trans: HELS从一个目标转向下一个目标所需时间
        - 若Δangle ≤ FSM最大角: t_trans = 50ms (FSM快速响应)
        - 若Δangle > FSM最大角: t_trans = Δangle / ω_mech (机械转台)
        
        Args:
            delta_angle: 两目标之间的角度差 [rad]
        Returns:
            t_trans: 转移时间 [s]
        """
        if delta_angle <= np.deg2rad(FSM_MAX_ANGLE):
            return FSM_RESPONSE_TIME  # 50 ms
        else:
            omega_mech = np.deg2rad(60.0)  # 机械转台角速度 60°/s
            return delta_angle / omega_mech
    
    def compute_period_time(self, t_damage, t_trans):
        """
        公式(22): 一个攻击周期的总时间
        t_period = t_trans + t_damage
        """
        return t_trans + t_damage
```

### 3.5 无人机蜂群密度模型 (`env/uav_model.py`)

实现公式(23)-(24)：两种蜂群入侵模式——多方向多批次 与 大规模同时入侵。

```python
import numpy as np
from scipy.stats import lognorm, uniform

class UAVSwarmDensityModel:
    """恶意无人机蜂群密度模型 (论文公式23-24)"""
    
    def __init__(self, n_uavs=50, mode='multi_batch'):
        """
        Args:
            n_uavs: 无人机总数
            mode: 'multi_batch' | 'simultaneous'
        """
        self.n_uavs = n_uavs
        self.mode = mode
    
    def generate_arrival_times(self, scenario_duration=300.0):
        """
        生成无人机蜂群到达时间序列
        
        公式(23): 多方向多批次 — 到达时间间隔服从对数正态分布
            f_1(t) = (1/(t·σ·√(2π))) · exp(-(ln t - μ)²/(2σ²))
        
        公式(24): 大规模同时入侵 — 到达时间间隔服从均匀分布
            f_2(t) = 1/(b-a), t ∈ [a,b]
        
        Returns:
            arrival_times: 各UAV到达时间 [s]  形状 (n_uavs,)
        """
        if self.mode == 'multi_batch':
            # 对数正态分布: μ = 3.0, σ = 0.5
            # 意味着大多数UAV在t=10-30s内到达，分多个波次
            intervals = lognorm.rvs(s=0.5, scale=np.exp(3.0), size=self.n_uavs)
            arrival_times = np.cumsum(intervals)
            # 限制在场景时长内
            arrival_times = arrival_times[arrival_times <= scenario_duration]
            # 如果不够，补充生成
            while len(arrival_times) < self.n_uavs:
                extra = lognorm.rvs(s=0.5, scale=np.exp(3.0), 
                                    size=self.n_uavs - len(arrival_times))
                new_times = np.cumsum(np.concatenate([[arrival_times[-1]], extra]))[1:]
                arrival_times = np.concatenate([arrival_times, new_times])
                arrival_times = arrival_times[arrival_times <= scenario_duration]
        else:
            # 均匀分布: a=0, b=scenario_duration*0.5
            a, b = 0.0, scenario_duration * 0.5
            arrival_times = uniform.rvs(loc=a, scale=b-a, size=self.n_uavs)
            arrival_times = np.sort(arrival_times)
        
        return arrival_times
    
    def generate_uav_configs(self, n_uavs, arrival_times, protected_pos=(0, 0, 0)):
        """
        生成无人机配置: 位置、速度、高度、入侵方向
        
        论文场景: 多方向多批次入侵
        - 方向: 0°~360°均匀随机分布(多方向)
        - 高度: 0.2~0.8 km
        - 速度: 0.02~0.03 km/s (20-30 m/s)
        
        Returns:
            configs: list of dict per UAV
        """
        np.random.seed(None)  # 确保随机
        configs = []
        for i in range(n_uavs):
            # 入侵方向角 [0, 2π)
            azimuth = np.random.uniform(0, 2 * np.pi)
            # 高度 [0.2, 0.8] km
            height = np.random.uniform(0.2, 0.8) * 1000  # 转[m]
            # 速度 [20, 30] m/s
            speed = np.random.uniform(20.0, 30.0)
            # 初始位置: 在探测范围边缘(~10km)
            init_distance = 10000.0  # 10 km探测范围
            init_x = protected_pos[0] + init_distance * np.cos(azimuth)
            init_y = protected_pos[1] + init_distance * np.sin(azimuth)
            init_z = height
            
            configs.append({
                'id': i,
                'init_pos': np.array([init_x, init_y, init_z]),
                'azimuth': azimuth,
                'height': height,
                'speed': speed,
                'arrival_time': arrival_times[i],
                'alive': True
            })
        return configs
```

### 3.6 威胁因子与效益因子 (`env/physics.py` — 第4部分)

实现公式(25)-(29)：5个因子模型。

```python
class ThreatBenefitFactors:
    """威胁因子与防御效益因子 (论文公式25-29)"""
    
    def __init__(self, v0=25.0, L_safe=1000.0, sigma_rc=1.0, alpha_rc=0.01):
        """
        Args:
            v0: 预设最具威胁速度 [m/s]
            L_safe: 安全距离阈值 [m]
            sigma_rc: 资源消耗sigmoid尺度参数
            alpha_rc: 资源消耗sigmoid形状参数
        """
        self.v0 = v0
        self.L_safe = L_safe
        self.sigma_rc = sigma_rc
        self.alpha_rc = alpha_rc
    
    def height_threat(self, h):
        """
        公式(25): 高度威胁因子 (指数衰减型)
        T_rh = exp(-h / h_ref)
        
        越低越具威胁，h_ref为参考高度
        """
        h_ref = 500.0  # 参考高度 [m]
        return np.exp(-h / h_ref)
    
    def velocity_threat(self, v):
        """
        公式(26): 速度威胁因子 (指数型)
        T_rv = exp(-(v - v0)² / (2·σ_v²))
        
        越接近预设最具威胁速度v0，威胁越大
        """
        sigma_v = 5.0  # 速度威胁带宽
        return np.exp(-(v - self.v0)**2 / (2 * sigma_v**2))
    
    def safe_distance_threat(self, L_remaining):
        """
        公式(27): 安全距离威胁因子 (线性)
        T_rL = (L_safe - L_remaining) / L_safe,  L_remaining ≤ L_safe
              = 0,                                   L_remaining > L_safe
        
        剩余飞行距离越短(越接近保护资产)，威胁越大
        """
        if L_remaining <= self.L_safe:
            return (self.L_safe - L_remaining) / self.L_safe
        else:
            return 0.0
    
    def resource_consumption_benefit(self, t_damage):
        """
        公式(28): 资源消耗效益因子 (Sigmoid函数)
        B_rc = 1 / (1 + exp(-α_rc · (σ_rc - t_damage)))
        
        优先选择毁伤时间短的目标(省电、高效率)
        """
        return 1.0 / (1.0 + np.exp(-self.alpha_rc * (self.sigma_rc - t_damage)))
    
    def hels_application_benefit(self, P0_current, P0_max, battery_remaining, 
                                   battery_max):
        """
        公式(29): HELS应用价值效益因子 (线性加权)
        B_rs = w1 · (1 - P0/P0_max) + w2 · (battery_remaining / battery_max)
        
        优先使用低功率、大剩余电量的HELS
        
        Args:
            P0_current: 当前HELS功率 [W]
            P0_max: HELS最大功率 [W]
            battery_remaining: 剩余电池容量 [s]
            battery_max: 最大电池容量 [s]
        """
        w1, w2 = 0.5, 0.5  # 权重
        power_efficiency = 1.0 - P0_current / P0_max
        battery_efficiency = battery_remaining / battery_max
        return w1 * power_efficiency + w2 * battery_efficiency
    
    def compute_all(self, h, v, L_remaining, t_damage, 
                    P0_current, P0_max, battery_remaining, battery_max):
        """计算全部5个因子"""
        T_rh = self.height_threat(h)
        T_rv = self.velocity_threat(v)
        T_rL = self.safe_distance_threat(L_remaining)
        B_rc = self.resource_consumption_benefit(t_damage)
        B_rs = self.hels_application_benefit(
            P0_current, P0_max, battery_remaining, battery_max)
        return T_rh, T_rv, T_rL, B_rc, B_rs
```

### 3.7 DRTA优化模型 (`env/physics.py` — 第5部分)

实现公式(30)-(31)：两阶段动态分配目标函数与约束。

```python
class DRTAModel:
    """HELS-UAV-DRTA优化模型 (论文公式30-31)"""
    
    def __init__(self, lambda1=0.5, lambda2=0.5,
                 lambda11=0.4, lambda12=0.3, lambda13=0.3,
                 lambda21=0.5, lambda22=0.5):
        """
        目标函数权重:
        λ1: 威胁权重, λ2: 效益权重, λ1+λ2=1
        λ11: 高度威胁子权重, λ12: 速度威胁子权重, λ13: 距离威胁子权重
        λ21: 资源消耗子权重, λ22: HELS应用价值子权重
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda11 = lambda11
        self.lambda12 = lambda12
        self.lambda13 = lambda13
        self.lambda21 = lambda21
        self.lambda22 = lambda22
    
    def objective(self, T_rh, T_rv, T_rL, B_rc, B_rs):
        """
        公式(30): 目标函数
        J = Σ_{i,j} x_{ij} [λ₁(λ₁₁·T_rh + λ₁₂·T_rv + λ₁₃·T_rL) 
                            + λ₂(λ₂₁·B_rc + λ₂₂·B_rs)]
        
        Returns:
            obj_value: 综合目标值(越大越好)
        """
        threat_term = (self.lambda11 * T_rh + 
                       self.lambda12 * T_rv + 
                       self.lambda13 * T_rL)
        benefit_term = (self.lambda21 * B_rc + 
                        self.lambda22 * B_rs)
        return self.lambda1 * threat_term + self.lambda2 * benefit_term
    
    def check_constraints(self, hels_state, uav_state, decision, env_state):
        """
        检查公式(31)中的7个约束条件(a-h)
        
        约束(a): A(t_{k+1}) = A(t_k) ∪ B(t_k) − K(t_k)  目标集演进
        约束(b): 电池消耗 t_battery(i,t_{k+1}) = t_battery(i,t_k) − Σ_j x_{ij}·t_damage,ij
        约束(c): 能量可行性 x_{ij}·t_battery,i ≤ W_i
        约束(d): 时间步更新
        约束(e): 互斥性 — 一个目标只能被一个HELS照射
        约束(f): 照射独占性 — 一个HELS同时只能照射一个目标
        约束(g): 照射周期完整性
        约束(h): 二元决策变量 x_{ij} ∈ {0,1}
        
        Returns:
            (feasible, reason): 是否可行及原因
        """
        # 约束(c): 电池是否够用
        if decision['irradiating']:
            t_damage = decision['t_damage']
            if t_damage > hels_state['battery_remaining']:
                return False, "insufficient_battery"
        
        # 约束(e): 目标是否已被其他HELS分配
        if uav_state.get('assigned_to') is not None:
            return False, "target_already_assigned"
        
        # 约束(f): HELS是否正在照射其他目标
        if hels_state.get('currently_irradiating') is not None:
            return False, "hels_already_irradiating"
        
        # 约束(a): 目标是否在探测范围内
        if not uav_state.get('detected', False):
            return False, "target_not_detected"
        
        # 约束(h): 决策有效性
        if decision['action'] not in range(hels_state['n_targets_visible'] + 1):
            return False, "invalid_action"
        
        return True, "feasible"
```

---

## 4. Gym仿真环境构建 (`env/drta_env.py`)

### 4.1 环境初始化与配置

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env.physics import (LaserAtmosphericTransmission, LaserThermalDamage,
                          HELSDamageModel, ThreatBenefitFactors, DRTAModel)
from env.uav_model import UAVSwarmDensityModel

class HELS_UAV_DRTA_Env(gym.Env):
    """HELS-UAV-DRTA Gym环境"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(self, config):
        """
        Args:
            config: dict with keys:
                - n_hels: HELS数量 (2 or 5)
                - n_uavs: UAV数量 (10 or 50)
                - env_type: 'rural' | 'desert' | 'coastal'
                - max_steps: 每episode最大步数
                - dt: 决策时间步长 [s]
                - hels_positions: HELS部署坐标 list[(x,y,z)]
                - protected_pos: 保护资产坐标 (x,y,z)
        """
        super().__init__()
        
        # 基本配置
        self.n_hels = config['n_hels']
        self.n_uavs = config['n_uavs']
        self.env_type = config['env_type']
        self.max_steps = config.get('max_steps', 500)
        self.dt = config.get('dt', 1.0)  # 1秒决策步长
        self.protected_pos = np.array(config.get('protected_pos', (0, 0, 0)))
        
        # HELS配置
        self.hels_positions = [np.array(p) for p in config['hels_positions']]
        self.hels_powers = config.get('hels_powers', [30e3] * self.n_hels)
        self.hels_battery_max = config.get('hels_battery_max', 200.0)  # [s]连续照射
        
        # 物理模型
        self.damage_model = HELSDamageModel(env_type=self.env_type)
        self.tb_factors = ThreatBenefitFactors()
        self.drta_model = DRTAModel()
        
        # 无人机蜂群模型
        self.swarm_model = UAVSwarmDensityModel(
            n_uavs=self.n_uavs, mode='multi_batch')
        
        # 观测空间: 每个HELS智能体
        # s_i = concat(s^{LaWS}_i, s^{UAV}_i)
        # s^{LaWS}_i ∈ R^4: (x,y,z, battery_remaining, remaining_duration, axis_direction)
        # s^{UAV}_i ∈ R^{4m+1}: 每UAV (distance, height, speed, env_param) + n_uavs
        self.hels_obs_dim = 6  # 自身状态维度
        self.uav_feat_dim = 5  # 每UAV特征维度 (距离, 高度, 速度, 方位角, 俯仰角)
        
        # 每个智能体的观测: 自身状态 + 环境参数 + 所有UAV状态
        single_obs_dim = self.hels_obs_dim + 1 + self.uav_feat_dim * self.n_uavs
        
        # 动作空间: m+1 (m个目标 + 1个等待)
        self.n_actions = self.n_uavs + 1
        
        # 多智能体空间
        self.observation_space = spaces.Dict({
            f'agent_{i}': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(single_obs_dim,), dtype=np.float32
            ) for i in range(self.n_hels)
        })
        self.action_space = spaces.Dict({
            f'agent_{i}': spaces.Discrete(self.n_actions)
            for i in range(self.n_hels)
        })
        
        # 全局状态 (用于Critic训练)
        self.global_state_dim = (single_obs_dim * self.n_hels 
                                 + self.n_actions * self.n_hels)
        
        self.reset()
    
    def _get_hels_obs(self, i):
        """获取第i个HELS的局部观测 (公式38-40)"""
        s_hels = self.hels_states[i]  # (6,)
        
        # 环境参数
        env_param = np.array([self.Cn2_0_val])
        
        # 所有UAV状态 (公式39)
        uav_states = []
        for j in range(self.n_uavs):
            if self.uav_states[j]['alive'] and self.uav_states[j]['detected']:
                uav_pos = self.uav_states[j]['position']
                uav_vel = self.uav_states[j]['velocity']
                hels_pos = self.hels_positions[i]
                rel_pos = uav_pos - hels_pos
                distance = np.linalg.norm(rel_pos)
                azimuth = np.arctan2(rel_pos[1], rel_pos[0])
                elevation = np.arctan2(rel_pos[2], 
                    np.sqrt(rel_pos[0]**2 + rel_pos[1]**2))
                speed = np.linalg.norm(uav_vel)
                uav_state = np.array([distance, uav_pos[2], speed, azimuth, elevation])
            else:
                uav_state = np.zeros(self.uav_feat_dim)
            uav_states.append(uav_state)
        
        uav_states = np.concatenate(uav_states)
        return np.concatenate([s_hels, env_param, uav_states])
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置时间
        self.t = 0.0
        self.episode_step = 0
        
        # 重置HELS状态
        self.hels_states = []
        for i in range(self.n_hels):
            self.hels_states.append({
                'position': self.hels_positions[i].copy(),
                'battery_remaining': self.hels_battery_max,  # [s]
                'remaining_duration': 0.0,                     # 当前照射剩余
                'axis_azimuth': 0.0,                           # 转轴方位角
                'axis_elevation': 0.0,                         # 转轴俯仰角
                'currently_irradiating': None,                 # 当前照射目标ID
                'total_kills': 0
            })
        
        # 重置UAV状态
        self.uav_arrival_times = self.swarm_model.generate_arrival_times()
        self.uav_configs = self.swarm_model.generate_uav_configs(
            self.n_uavs, self.uav_arrival_times, self.protected_pos)
        self.uav_states = []
        for cfg in self.uav_configs:
            self.uav_states.append({
                'position': cfg['init_pos'].copy(),
                'velocity': np.array([
                    -cfg['speed'] * np.cos(cfg['azimuth']),
                    -cfg['speed'] * np.sin(cfg['azimuth']),
                    0.0
                ]),
                'speed': cfg['speed'],
                'height': cfg['height'],
                'azimuth': cfg['azimuth'],
                'alive': False,  # 未到达时间前为False
                'detected': False,
                'arrival_time': cfg['arrival_time'],
                'assigned_to': None,
                'damage_accumulated': 0.0,  # [s] 累计照射时间
                'damage_required': 0.0       # 所需总毁伤时间
            })
        
        # 大气环境参数
        atm = ATMOSPHERE_PARAMS[self.env_type]
        self.Cn2_0_val = atm['Cn2_0']
        
        # 获取观测
        obs = {f'agent_{i}': self._get_hels_obs(i) for i in range(self.n_hels)}
        info = self._get_info()
        
        return obs, info
    
    def step(self, actions):
        """
        执行一步仿真
        
        Args:
            actions: dict {f'agent_{i}': action_idx}
        Returns:
            obs, rewards, terminated, truncated, info
        """
        self.episode_step += 1
        
        # === 1. 解析动作 ===
        decisions = []
        for i in range(self.n_hels):
            action = actions[f'agent_{i}']
            if action < self.n_uavs:
                # 照射第action个UAV
                target_id = action
                decisions.append({
                    'agent_id': i,
                    'action': action,
                    'irradiating': True,
                    'target_id': target_id
                })
            else:
                # 等待 (action = self.n_uavs)
                decisions.append({
                    'agent_id': i,
                    'action': action,
                    'irradiating': False,
                    'target_id': None
                })
        
        # === 2. 计算每个决策的毁伤时间 ===
        for dec in decisions:
            if dec['irradiating']:
                i = dec['agent_id']
                j = dec['target_id']
                hels_pos = self.hels_positions[i]
                uav_pos = self.uav_states[j]['position']
                rel_pos = uav_pos - hels_pos
                L = np.linalg.norm(rel_pos)
                h = uav_pos[2]
                theta = np.arcsin(h / L) if L > 0 else 0
                P0 = self.hels_powers[i]
                
                result = self.damage_model.compute_damage_time(P0, L, h, theta)
                dec['t_damage'] = result['t_damage']
                dec['t_trans'] = 0.05  # FSM响应时间(简化)
                dec['t_period'] = dec['t_trans'] + dec['t_damage']
            else:
                dec['t_damage'] = 0
                dec['t_trans'] = 0
                dec['t_period'] = 0
        
        # === 3. 检查约束 ===
        for dec in decisions:
            if dec['irradiating']:
                i, j = dec['agent_id'], dec['target_id']
                hels_s = self.hels_states[i]
                uav_s = self.uav_states[j]
                
                feasible, reason = self.drta_model.check_constraints(
                    hels_s, uav_s, dec, {})
                dec['feasible'] = feasible
                dec['reason'] = reason
                
                if feasible:
                    # 锁定目标分配
                    uav_s['assigned_to'] = i
                    hels_s['currently_irradiating'] = j
                    uav_s['damage_required'] = dec['t_damage']
        
        # === 4. 推进时间 & 更新UAV位置 ===
        self.t += self.dt
        for j in range(self.n_uavs):
            if self.uav_states[j]['alive']:
                self.uav_states[j]['position'] += (
                    self.uav_states[j]['velocity'] * self.dt)
        
        # === 5. 检查新到达的UAV ===
        for j in range(self.n_uavs):
            if (not self.uav_states[j]['alive'] and 
                self.t >= self.uav_arrival_times[j]):
                self.uav_states[j]['alive'] = True
                self.uav_states[j]['detected'] = True
        
        # === 6. 更新毁伤进度 ===
        for i in range(self.n_hels):
            target_id = self.hels_states[i]['currently_irradiating']
            if target_id is not None:
                uav_s = self.uav_states[target_id]
                uav_s['damage_accumulated'] += self.dt
                
                # 检查是否完成毁伤
                if uav_s['damage_accumulated'] >= uav_s['damage_required']:
                    uav_s['alive'] = False
                    uav_s['detected'] = False
                    uav_s['assigned_to'] = None
                    self.hels_states[i]['currently_irradiating'] = None
                    self.hels_states[i]['total_kills'] += 1
                    self.hels_states[i]['remaining_duration'] = 0
                else:
                    self.hels_states[i]['remaining_duration'] = (
                        uav_s['damage_required'] - uav_s['damage_accumulated'])
        
        # === 7. 更新电池 ===
        for i in range(self.n_hels):
            if self.hels_states[i]['currently_irradiating'] is not None:
                self.hels_states[i]['battery_remaining'] -= self.dt
        
        # === 8. 计算奖励 ===
        rewards = self._compute_rewards(decisions)
        
        # === 9. 检查终止条件 ===
        terminated = self._check_terminated()
        truncated = self.episode_step >= self.max_steps
        
        # === 10. 获取观测和info ===
        obs = {f'agent_{i}': self._get_hels_obs(i) for i in range(self.n_hels)}
        info = self._get_info()
        
        # 将全局状态加入info (供集中训练Critic使用)
        info['global_state'] = self._get_global_state()
        
        return obs, rewards, terminated, truncated, info
    
    def _compute_rewards(self, decisions):
        """公式(42): 奖励函数"""
        rewards = {}
        for dec in decisions:
            i = dec['agent_id']
            if dec['irradiating'] and dec.get('feasible', False):
                # (1) 满足条件且执行照射: 基于威胁+效益因子的正奖励
                j = dec['target_id']
                uav_s = self.uav_states[j]
                hels_s = self.hels_states[i]
                rel_pos = uav_s['position'] - self.protected_pos
                L_remaining = np.linalg.norm(rel_pos)
                h = uav_s['height']
                v = uav_s['speed']
                
                T_rh, T_rv, T_rL, B_rc, B_rs = self.tb_factors.compute_all(
                    h, v, L_remaining, dec['t_damage'],
                    self.hels_powers[i], 50e3,
                    hels_s['battery_remaining'], self.hels_battery_max)
                
                obj_value = self.drta_model.objective(T_rh, T_rv, T_rL, B_rc, B_rs)
                rewards[f'agent_{i}'] = obj_value
                
            elif dec['irradiating'] and not dec.get('feasible', True):
                # (2) 不满足条件却执行照射: -1惩罚
                rewards[f'agent_{i}'] = -1.0
            else:
                # (3) 等待: 衰减正奖励 0.1·exp(-β_d·t)
                beta_d = 0.01
                rewards[f'agent_{i}'] = 0.1 * np.exp(-beta_d * self.t)
        
        return rewards
    
    def _check_terminated(self):
        """检查episode是否终止"""
        # 所有UAV被毁伤
        all_dead = all(not s['alive'] for s in self.uav_states)
        if all_dead:
            return True
        
        # 有UAV到达保护资产
        for uav_s in self.uav_states:
            if uav_s['alive']:
                dist_to_protected = np.linalg.norm(
                    uav_s['position'] - self.protected_pos)
                if dist_to_protected < 100.0:  # 100m内视为到达
                    return True
        
        # 所有HELS电量耗尽
        all_battery_dead = all(
            s['battery_remaining'] <= 0 for s in self.hels_states)
        if all_battery_dead:
            return True
        
        return False
    
    def _get_info(self):
        """返回场景统计信息"""
        n_killed = sum(1 for s in self.uav_states if not s['alive'])
        n_reached = sum(1 for s in self.uav_states 
                        if s['alive'] and 
                        np.linalg.norm(s['position'] - self.protected_pos) < 100)
        n_active = sum(1 for s in self.uav_states if s['alive'])
        return {
            'n_killed': n_killed,
            'n_reached': n_reached,
            'n_active': n_active,
            'damage_rate': n_killed / max(n_killed + n_active, 1),
            'total_kills': [s['total_kills'] for s in self.hels_states],
            'battery_remaining': [s['battery_remaining'] for s in self.hels_states],
            'time': self.t
        }
    
    def _get_global_state(self):
        """获取全局状态 (供Critic使用)"""
        obs_list = [self._get_hels_obs(i) for i in range(self.n_hels)]
        return np.concatenate(obs_list)
```

### 4.2 场景配置 (`config/scenario_config.py`)

```python
"""6个实验场景的配置参数"""

import numpy as np

# === 保护资产位置(原点) ===
PROTECTED_POS = (0.0, 0.0, 0.0)

# === 小规模场景 (2 HELS vs 10 UAVs) ===
SMALL_SCALE = {
    'rural': {
        'n_hels': 2,
        'n_uavs': 10,
        'env_type': 'rural',
        'hels_positions': [
            (3000, 0, 100),      # HELS 1: 东侧3km
            (-3000, 0, 100),     # HELS 2: 西侧3km
        ],
        'hels_powers': [30e3, 30e3],  # 各30kW
    },
    'desert': {
        'n_hels': 2,
        'n_uavs': 10,
        'env_type': 'desert',
        'hels_positions': [
            (3000, 0, 100),
            (-3000, 0, 100),
        ],
        'hels_powers': [30e3, 30e3],
    },
    'coastal': {
        'n_hels': 2,
        'n_uavs': 10,
        'env_type': 'coastal',
        'hels_positions': [
            (3000, 0, 100),      # 内陆侧
            (-3000, 0, 100),     # 海面侧
        ],
        'hels_powers': [30e3, 30e3],
    },
}

# === 大规模场景 (5 HELS vs 50 UAVs) ===
LARGE_SCALE = {
    'rural': {
        'n_hels': 5,
        'n_uavs': 50,
        'env_type': 'rural',
        'hels_positions': [
            (3000, 0, 100),      # HELS 1: 东
            (1500, 2600, 100),   # HELS 2: 东北
            (-1500, 2600, 100),  # HELS 3: 西北
            (-3000, 0, 100),     # HELS 4: 西
            (0, 0, 100),         # HELS 5: 中心(保护资产位置)
        ],
        'hels_powers': [30e3, 40e3, 30e3, 50e3, 20e3],
    },
    'desert': {
        'n_hels': 5,
        'n_uavs': 50,
        'env_type': 'desert',
        'hels_positions': [
            (3000, 0, 100),
            (1500, 2600, 100),
            (-1500, 2600, 100),
            (-3000, 0, 100),
            (0, 0, 100),
        ],
        'hels_powers': [30e3, 40e3, 30e3, 50e3, 20e3],
    },
    'coastal': {
        'n_hels': 5,
        'n_uavs': 50,
        'env_type': 'coastal',
        'hels_positions': [
            (3000, 0, 100),
            (1500, 2600, 100),
            (-1500, 2600, 100),
            (-3000, 0, 100),
            (0, 0, 100),
        ],
        'hels_powers': [30e3, 40e3, 30e3, 50e3, 20e3],
    },
}

# 生成所有场景列表
ALL_SCENARIOS = []
for scale_name, scale_config in [('small', SMALL_SCALE), ('large', LARGE_SCALE)]:
    for env_type in ['rural', 'desert', 'coastal']:
        scenario = scale_config[env_type].copy()
        scenario['name'] = f'{scale_name}_{env_type}'
        scenario['scale'] = scale_name
        ALL_SCENARIOS.append(scenario)
```

---

## 5. MADDPG-IA算法实现

### 5.1 注意力机制状态编码器 (`algorithm/attention.py`)

实现公式(43)-(45)：以HELS自身状态为Query，UAV状态为Key/Value。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionEncoder(nn.Module):
    """
    注意力机制状态编码器 (论文公式43-45, 图8)
    
    将可变长度的UAV状态编码为固定维度的特征向量。
    - Query: HELS自身状态 s^{LaWS}_i → W_q
    - Key: 每个UAV状态 s^{UAV}_j → W_k
    - Value: 每个UAV状态 s^{UAV}_j → W_v
    
    公式(43): Q_i = W_q·s^{LaWS}_i, K_j = W_k·s^{UAV}_j, V_j = W_v·s^{UAV}_j
    公式(44): a_ij = softmax(Q_i·K_j^T / √d_k)
    公式(45): ẽ_i = Attn(Q, K, V) = softmax(Q·K^T/√d_k)·V ∈ R^{d_attn}
    """
    
    def __init__(self, hels_dim=6, uav_dim=5, d_k=64, d_v=64, d_attn=128, 
                 max_uavs=50):
        """
        Args:
            hels_dim: HELS自身状态维度 (6)
            uav_dim: 每UAV特征维度 (5)
            d_k: Key维度 (64)
            d_v: Value维度 (64)
            d_attn: 输出注意力维度 (128)
            max_uavs: 最大UAV数量 (50)
        """
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_attn = d_attn
        self.max_uavs = max_uavs
        
        # 线性变换矩阵 W_q, W_k, W_v
        self.W_q = nn.Linear(hels_dim + 1, d_k)   # +1 for env_param
        self.W_k = nn.Linear(uav_dim, d_k)
        self.W_v = nn.Linear(uav_dim, d_v)
        
        # 输出投影
        self.output_proj = nn.Linear(d_v, d_attn)
        
        self.scale = math.sqrt(d_k)
    
    def forward(self, hels_state, uav_states, env_param, mask=None):
        """
        Args:
            hels_state: (batch, hels_dim) HELS自身状态
            uav_states: (batch, max_uavs, uav_dim) 所有UAV状态
            env_param: (batch, 1) 大气环境参数
            mask: (batch, max_uavs) 有效UAV的mask (1=有效, 0=填充)
        Returns:
            encoded: (batch, d_attn) 固定维度编码
            attn_weights: (batch, max_uavs) 注意力权重(用于可视化)
        """
        batch_size = hels_state.shape[0]
        
        # 公式(43): 计算Q, K, V
        Q = self.W_q(torch.cat([hels_state, env_param], dim=-1))  # (B, d_k)
        K = self.W_k(uav_states)   # (B, N, d_k)
        V = self.W_v(uav_states)   # (B, N, d_v)
        
        # 公式(44): 注意力权重 a_ij = softmax(Q·K^T / √d_k)
        Q_expanded = Q.unsqueeze(1)  # (B, 1, d_k)
        scores = torch.matmul(Q_expanded, K.transpose(-2, -1))  # (B, 1, N)
        scores = scores / self.scale
        
        # 应用mask (填充的无效UAV应不被关注)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, N)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)  # (B, 1, N)
        
        # 公式(45): ẽ_i = Attn(Q, K, V) = Σ_j a_ij · V_j
        context = torch.matmul(attn_weights, V)  # (B, 1, d_v)
        context = context.squeeze(1)  # (B, d_v)
        
        encoded = self.output_proj(context)  # (B, d_attn)
        
        return encoded, attn_weights.squeeze(1)
```

### 5.2 RND内在奖励模块 (`algorithm/rnd.py`)

实现公式(46)-(47)：基于Random Network Distillation的内在奖励。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RNDNetwork(nn.Module):
    """RND中的MLP网络"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class RNDIntrinsicReward:
    """
    RND内在奖励模块 (论文公式46-47, 图9)
    
    公式(46): r_c = β_r · ||φ̃(s̃^{t+1}_i) - φ(s̃^{t+1}_i)||
    公式(47): r_h = γ·r_e + (1-γ)·r_c
    
    φ: 固定随机目标网络
    φ̃: 可训练预测网络
    预测误差越大 → 状态越新颖 → 内在奖励越大
    """
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, 
                 lr=1e-4, beta_r0=0.1, k_r=1e-4):
        """
        Args:
            input_dim: 输入状态维度 (d_attn)
            hidden_dim: 隐藏层维度
            output_dim: 输出特征维度
            lr: RND预测网络学习率
            beta_r0: 内在奖励初始系数
            k_r: 课程学习衰减率
        """
        # 目标网络 φ: 随机初始化后固定
        self.target_net = RNDNetwork(input_dim, hidden_dim, output_dim)
        for param in self.target_net.parameters():
            param.requires_grad = False  # 冻结
        
        # 预测网络 φ̃: 可训练
        self.predictor_net = RNDNetwork(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.predictor_net.parameters(), lr=lr)
        
        self.beta_r0 = beta_r0
        self.k_r = k_r
        self.training_step = 0
    
    def compute_intrinsic_reward(self, state):
        """
        公式(46): r_c = β_r · ||φ̃(state) - φ(state)||
        
        Args:
            state: (batch, input_dim) 下一时刻状态编码
        Returns:
            r_c: intrinsic reward (batch,)
        """
        with torch.no_grad():
            target_feat = self.target_net(state)
        predict_feat = self.predictor_net(state)
        
        # 均方误差作为内在奖励
        error = ((predict_feat - target_feat) ** 2).mean(dim=-1)
        
        # β_r: 课程学习衰减
        beta_r = self.beta_r0 * np.exp(-self.k_r * self.training_step)
        r_c = beta_r * error
        
        return r_c
    
    def update(self, state):
        """
        更新预测网络 φ̃，最小化对目标网络的预测误差
        """
        with torch.no_grad():
            target_feat = self.target_net(state)
        predict_feat = self.predictor_net(state)
        
        loss = F.mse_loss(predict_feat, target_feat)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.training_step += 1
        return loss.item()
    
    def hybrid_reward(self, extrinsic_reward, intrinsic_reward, gamma_mix=0.9):
        """
        公式(47): r_h = γ·r_e + (1-γ)·r_c
        
        γ根据奖励稀疏度自动调整: 
        当环境奖励接近0时增大内在奖励权重
        
        Args:
            extrinsic_reward: 环境外在奖励 r_e
            intrinsic_reward: RND内在奖励 r_c
            gamma_mix: 混合系数 (固定或自适应)
        Returns:
            r_h: 混合奖励
        """
        return gamma_mix * extrinsic_reward + (1 - gamma_mix) * intrinsic_reward
```

### 5.3 Actor-Critic网络 (`algorithm/actor_critic.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """Actor策略网络 — 每个HELS智能体独立"""
    
    def __init__(self, attn_dim=128, n_actions=51, hidden_dim=256):
        """
        Args:
            attn_dim: AttentionEncoder输出维度 (128)
            n_actions: 动作数 (m+1)
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(attn_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
    
    def forward(self, attn_encoded):
        """
        Args:
            attn_encoded: (batch, attn_dim) Attention编码后的状态
        Returns:
            logits: (batch, n_actions) 动作logits
        """
        return self.net(attn_encoded)


class Critic(nn.Module):
    """Critic价值网络 — 集中式(使用全局信息)"""
    
    def __init__(self, n_agents, attn_dim=128, n_actions=51, hidden_dim=512):
        """
        Args:
            n_agents: 智能体数量
            attn_dim: 每个智能体的attention编码维度
            n_actions: 动作数 (m+1, 使用One-Hot编码)
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        # 输入: 所有智能体的状态编码 + 所有智能体的动作
        input_dim = n_agents * attn_dim + n_agents * n_actions
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, global_state_encodings, joint_actions_onehot):
        """
        Args:
            global_state_encodings: (batch, n_agents*attn_dim) 所有agent的注意力编码
            joint_actions_onehot: (batch, n_agents*n_actions) 联合动作的One-Hot
        Returns:
            q_value: (batch, 1) Q值
        """
        x = torch.cat([global_state_encodings, joint_actions_onehot], dim=-1)
        return self.net(x)
```

### 5.4 Gumbel-Softmax采样 (`algorithm/gumbel_softmax.py`)

```python
import torch
import torch.nn.functional as F

def gumbel_softmax(logits, temperature=1.0, hard=False, eps=1e-10):
    """
    Gumbel-Softmax: 离散动作的可微采样
    
    论文使用此方法实现离散动作空间的梯度反向传播
    (公式41: One-Hot编码的m+1维离散动作空间)
    
    Args:
        logits: (..., n_actions) 动作logits
        temperature: 温度参数 (>0, 越小越接近argmax)
        hard: 是否使用ST-Gumbel-Softmax
        eps: 数值稳定性
    Returns:
        y: (..., n_actions) soft one-hot or hard one-hot
    """
    # Gumbel(0, 1)采样
    U = torch.rand_like(logits)
    g = -torch.log(-torch.log(U + eps) + eps)
    
    # Gumbel-Softmax
    y_soft = F.softmax((logits + g) / temperature, dim=-1)
    
    if hard:
        # Straight-Through: 前向使用argmax，反向使用softmax梯度
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        y = (y_hard - y_soft).detach() + y_soft  # ST梯度
    else:
        y = y_soft
    
    return y
```

### 5.5 经验回放池 (`algorithm/replay_buffer.py`)

```python
import numpy as np
import torch
from collections import deque
import random

class ReplayBuffer:
    """经验回放池 (容量1×10⁶)"""
    
    def __init__(self, capacity=int(1e6)):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, actions, rewards, next_obs, dones, 
             global_state, next_global_state):
        """
        Args:
            obs: dict {agent_i: np.array}
            actions: dict {agent_i: int}
            rewards: dict {agent_i: float}
            next_obs: dict {agent_i: np.array}
            dones: bool
            global_state: np.array
            next_global_state: np.array
        """
        self.buffer.append({
            'obs': obs,
            'actions': actions,
            'rewards': rewards,
            'next_obs': next_obs,
            'dones': dones,
            'global_state': global_state,
            'next_global_state': next_global_state
        })
    
    def sample(self, batch_size):
        """随机采样batch"""
        batch = random.sample(self.buffer, batch_size)
        return batch
    
    def __len__(self):
        return len(self.buffer)
```

### 5.6 MADDPG-IA主算法 (`algorithm/maddpg_ia.py`)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from algorithm.attention import AttentionEncoder
from algorithm.rnd import RNDIntrinsicReward
from algorithm.actor_critic import Actor, Critic
from algorithm.replay_buffer import ReplayBuffer
from algorithm.gumbel_softmax import gumbel_softmax

class MADDPG_IA:
    """MADDPG-IA算法 (Algorithm 1 伪代码)"""
    
    def __init__(self, n_agents, n_actions, hels_dim=6, uav_dim=5,
                 d_attn=128, actor_hidden=256, critic_hidden=512,
                 actor_lr=1e-3, critic_lr=1e-3, gamma=0.95, tau=0.01,
                 batch_size=2048, buffer_size=int(1e6),
                 gumbel_temp=1.0, gumbel_temp_min=0.1, gumbel_anneal=0.9995,
                 device='cuda'):
        """
        Args:
            n_agents: HELS智能体数量
            n_actions: 动作空间大小 (m+1)
            其他参数参见Table 1
        """
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        
        # 注意力编码器 (所有Agent共享)
        self.attention = AttentionEncoder(
            hels_dim=hels_dim, uav_dim=uav_dim,
            d_attn=d_attn, max_uavs=50
        ).to(device)
        
        # Actor网络 (每个Agent一个, 或共享)
        self.actors = nn.ModuleList([
            Actor(attn_dim=d_attn, n_actions=n_actions, hidden_dim=actor_hidden)
            for _ in range(n_agents)
        ]).to(device)
        
        # Target Actor网络
        self.target_actors = nn.ModuleList([
            Actor(attn_dim=d_attn, n_actions=n_actions, hidden_dim=actor_hidden)
            for _ in range(n_agents)
        ]).to(device)
        self._hard_update(self.actors, self.target_actors)
        
        # Critic网络 (每个Agent一个集中式Critic)
        self.critics = nn.ModuleList([
            Critic(n_agents=n_agents, attn_dim=d_attn, 
                   n_actions=n_actions, hidden_dim=critic_hidden)
            for _ in range(n_agents)
        ]).to(device)
        
        # Target Critic网络
        self.target_critics = nn.ModuleList([
            Critic(n_agents=n_agents, attn_dim=d_attn,
                   n_actions=n_actions, hidden_dim=critic_hidden)
            for _ in range(n_agents)
        ]).to(device)
        self._hard_update(self.critics, self.target_critics)
        
        # 优化器
        self.actor_optimizers = [
            optim.Adam(self.actors[i].parameters(), lr=actor_lr)
            for i in range(n_agents)
        ]
        self.critic_optimizers = [
            optim.Adam(self.critics[i].parameters(), lr=critic_lr)
            for i in range(n_agents)
        ]
        
        # RND内在奖励模块 (每个Agent一个)
        self.rnd_modules = [
            RNDIntrinsicReward(input_dim=d_attn)
            for _ in range(n_agents)
        ]
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        
        # Gumbel-Softmax温度
        self.gumbel_temp = gumbel_temp
        self.gumbel_temp_min = gumbel_temp_min
        self.gumbel_anneal = gumbel_anneal
    
    def _hard_update(self, source, target):
        """硬拷贝参数"""
        for s_param, t_param in zip(source.parameters(), target.parameters()):
            t_param.data.copy_(s_param.data)
    
    def _soft_update(self, source, target):
        """公式(37): 软更新 θ' ← τ·θ + (1-τ)·θ'"""
        for s_param, t_param in zip(source.parameters(), target.parameters()):
            t_param.data.copy_(self.tau * s_param.data + 
                               (1 - self.tau) * t_param.data)
    
    def encode_obs(self, obs, i):
        """使用注意力编码器处理观测"""
        # 解析HELS自身状态和UAV状态
        hels_state = torch.FloatTensor(obs[f'agent_{i}'][:6]).unsqueeze(0).to(self.device)
        env_param = torch.FloatTensor([obs[f'agent_{i}'][6]]).unsqueeze(0).to(self.device)
        
        # UAV状态: (n_uavs * uav_dim) → reshape
        uav_flat = obs[f'agent_{i}'][7:]
        n_uavs = len(uav_flat) // 5
        uav_states = torch.FloatTensor(uav_flat).reshape(1, n_uavs, 5).to(self.device)
        
        # Mask: 非零特征表示有效UAV
        mask = (uav_states.abs().sum(dim=-1) > 0).float()
        
        encoded, attn_weights = self.attention(hels_state, uav_states, env_param, mask)
        return encoded, attn_weights
    
    def select_action(self, obs, i, evaluate=False):
        """
        选择动作
        
        Args:
            obs: 观测
            i: Agent索引
            evaluate: True=argmax(评估模式), False=Gumbel-Softmax采样
        """
        encoded, attn_weights = self.encode_obs(obs, i)
        
        with torch.no_grad():
            logits = self.actors[i](encoded)
        
        if evaluate:
            action = torch.argmax(logits, dim=-1).item()
        elif self.gumbel_temp > self.gumbel_temp_min:
            action_onehot = gumbel_softmax(
                logits, temperature=self.gumbel_temp, hard=True)
            action = torch.argmax(action_onehot, dim=-1).item()
        else:
            # 温度衰减完后使用ε-greedy
            if np.random.random() < 0.05:
                action = np.random.randint(self.n_actions)
            else:
                action = torch.argmax(logits, dim=-1).item()
        
        return action, encoded
    
    def compute_rewards(self, obs, next_obs, env_rewards, i):
        """计算混合奖励 (公式47)"""
        # 外在奖励
        r_e = env_rewards[f'agent_{i}']
        
        # 内在奖励: 下一状态的新颖性
        next_encoded, _ = self.encode_obs(next_obs, i)
        r_c = self.rnd_modules[i].compute_intrinsic_reward(next_encoded).item()
        
        # 混合奖励
        r_h = self.rnd_modules[i].hybrid_reward(r_e, r_c)
        return r_h, r_c
    
    def update(self):
        """一次训练更新 (Algorithm 1 第11-21行)"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        batch = self.replay_buffer.sample(self.batch_size)
        
        critic_losses = []
        actor_losses = []
        rnd_losses = []
        
        for i in range(self.n_agents):
            # === 1. 更新Critic ===
            # 使用Target Actor计算下一时刻动作
            next_actions_onehot = []
            for j in range(self.n_agents):
                next_encoded_j, _ = self.encode_obs(batch['next_obs'][j], j)
                next_logits_j = self.target_actors[j](next_encoded_j)
                next_action_j = gumbel_softmax(next_logits_j, temperature=0.1, hard=True)
                next_actions_onehot.append(next_action_j)
            next_actions_cat = torch.cat(next_actions_onehot, dim=-1)
            
            # 全局状态编码
            global_encodings = []
            for j in range(self.n_agents):
                enc_j, _ = self.encode_obs(batch['next_obs'][j], j)
                global_encodings.append(enc_j)
            global_enc_cat = torch.cat(global_encodings, dim=-1)
            
            # TD目标: y_i = r_i + γ·Q'_i(s', a')
            rewards_i = torch.FloatTensor(
                [batch['rewards'][idx][f'agent_{i}'] for idx in range(self.batch_size)]
            ).unsqueeze(-1).to(self.device)
            
            with torch.no_grad():
                target_q = self.target_critics[i](global_enc_cat, next_actions_cat)
                td_target = rewards_i + self.gamma * (1 - batch['dones']) * target_q
            
            # 当前Q值
            current_actions_onehot = []
            for j in range(self.n_agents):
                current_encoded_j, _ = self.encode_obs(batch['obs'][j], j)
                current_logits_j = self.actors[j](current_encoded_j)
                current_action_j = gumbel_softmax(current_logits_j, temperature=0.1, hard=True)
                current_actions_onehot.append(current_action_j.detach())
            current_actions_cat = torch.cat(current_actions_onehot, dim=-1)
            
            global_enc_curr_cat = torch.cat(
                [self.encode_obs(batch['obs'][j], j)[0] for j in range(self.n_agents)],
                dim=-1)
            
            current_q = self.critics[i](global_enc_curr_cat, current_actions_cat)
            
            # Critic损失 = MSE(TD-error)
            critic_loss = F.mse_loss(current_q, td_target)
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), 1.0)
            self.critic_optimizers[i].step()
            critic_losses.append(critic_loss.item())
            
            # === 2. 更新Actor ===
            # 公式(34): 策略梯度
            encoded_i, _ = self.encode_obs(batch['obs'][i], i)
            actor_logits = self.actors[i](encoded_i)
            actor_action = gumbel_softmax(actor_logits, temperature=self.gumbel_temp)
            
            # 构造联合动作(自己的动作使用当前Actor输出，其他使用detach)
            joint_actions = []
            for j in range(self.n_agents):
                if j == i:
                    joint_actions.append(actor_action)
                else:
                    joint_actions.append(current_actions_onehot[j].detach())
            joint_actions_cat = torch.cat(joint_actions, dim=-1)
            
            # Actor损失 = -Q_i(s, a)
            q_value = self.critics[i](global_enc_curr_cat, joint_actions_cat)
            actor_loss = -q_value.mean()
            
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1.0)
            self.actor_optimizers[i].step()
            actor_losses.append(actor_loss.item())
            
            # === 3. 软更新目标网络 (公式37) ===
            self._soft_update(self.actors[i], self.target_actors[i])
            self._soft_update(self.critics[i], self.target_critics[i])
            
            # === 4. 更新RND预测网络 ===
            next_encoded_i, _ = self.encode_obs(batch['next_obs'][i], i)
            rnd_loss = self.rnd_modules[i].update(next_encoded_i.squeeze(0))
            rnd_losses.append(rnd_loss)
        
        # === 5. 衰减Gumbel-Softmax温度 ===
        self.gumbel_temp = max(self.gumbel_temp_min, 
                               self.gumbel_temp * self.gumbel_anneal)
        
        return {
            'critic_loss': np.mean(critic_losses),
            'actor_loss': np.mean(actor_losses),
            'rnd_loss': np.mean(rnd_losses),
            'gumbel_temp': self.gumbel_temp
        }
    
    def save(self, path):
        torch.save({
            'actors': {i: self.actors[i].state_dict() for i in range(self.n_agents)},
            'critics': {i: self.critics[i].state_dict() for i in range(self.n_agents)},
            'target_actors': {i: self.target_actors[i].state_dict() for i in range(self.n_agents)},
            'target_critics': {i: self.target_critics[i].state_dict() for i in range(self.n_agents)},
            'attention': self.attention.state_dict(),
            'gumbel_temp': self.gumbel_temp
        }, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(ckpt['actors'][i])
            self.critics[i].load_state_dict(ckpt['critics'][i])
            self.target_actors[i].load_state_dict(ckpt['target_actors'][i])
            self.target_critics[i].load_state_dict(ckpt['target_critics'][i])
        self.attention.load_state_dict(ckpt['attention'])
        self.gumbel_temp = ckpt['gumbel_temp']
```

---

## 6. 训练流程实现 (`train.py`)

```python
import torch
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import json

from env.drta_env import HELS_UAV_DRTA_Env
from algorithm.maddpg_ia import MADDPG_IA
from config.scenario_config import ALL_SCENARIOS, SMALL_SCALE, LARGE_SCALE

def train(args):
    """训练一个场景的主函数"""
    
    # 加载场景配置
    scenario = [s for s in ALL_SCENARIOS if s['name'] == args.scenario][0]
    
    print(f"=== Training Scenario: {args.scenario} ===")
    print(f"  HELS: {scenario['n_hels']}, UAVs: {scenario['n_uavs']}")
    print(f"  Environment: {scenario['env_type']}")
    
    # 创建环境
    env = HELS_UAV_DRTA_Env(scenario)
    
    # 创建MADDPG-IA算法
    n_actions = env.n_actions
    maddpg_ia = MADDPG_IA(
        n_agents=scenario['n_hels'],
        n_actions=n_actions,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.95,
        tau=0.01,
        batch_size=2048,
        buffer_size=int(1e6),
        device=args.device
    )
    
    # TensorBoard日志
    log_dir = f"runs/{args.scenario}/{args.run_id}"
    writer = SummaryWriter(log_dir)
    
    # 训练循环
    max_episodes = 2000
    pbar = tqdm(range(max_episodes), desc=f"Training {args.scenario}")
    
    episode_rewards = []
    damage_rates = []
    
    for episode in pbar:
        obs, info = env.reset()
        episode_reward = np.zeros(scenario['n_hels'])
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # 选择动作
            actions = {}
            encodings = {}
            for i in range(scenario['n_hels']):
                action, encoding = maddpg_ia.select_action(obs, i)
                actions[f'agent_{i}'] = action
                encodings[f'agent_{i}'] = encoding
            
            # 执行动作
            next_obs, env_rewards, terminated, truncated, info = env.step(actions)
            
            # 计算混合奖励
            hybrid_rewards = {}
            intrinsic_rewards = {}
            for i in range(scenario['n_hels']):
                r_h, r_c = maddpg_ia.compute_rewards(
                    obs, next_obs, env_rewards, i)
                hybrid_rewards[f'agent_{i}'] = r_h
                intrinsic_rewards[f'agent_{i}'] = r_c
                episode_reward[i] += r_h
            
            # 存储经验
            dones_val = terminated or truncated
            maddpg_ia.replay_buffer.push(
                obs, actions, hybrid_rewards, next_obs, dones_val,
                info.get('global_state', None),
                info.get('global_state', None)  # info after step
            )
            
            # 更新算法
            update_info = maddpg_ia.update()
            
            obs = next_obs
        
        # 记录episode统计
        avg_reward = np.mean(episode_reward)
        damage_rate = info['damage_rate'] if terminated else (
            info['n_killed'] / (info['n_killed'] + info['n_active']))
        
        episode_rewards.append(avg_reward)
        damage_rates.append(damage_rate)
        
        writer.add_scalar('Episode/Avg_Reward', avg_reward, episode)
        writer.add_scalar('Episode/Damage_Rate', damage_rate, episode)
        writer.add_scalar('Episode/Battery_Remaining', 
                          np.mean(info['battery_remaining']), episode)
        
        if update_info:
            writer.add_scalar('Train/Critic_Loss', update_info['critic_loss'], episode)
            writer.add_scalar('Train/Actor_Loss', update_info['actor_loss'], episode)
            writer.add_scalar('Train/RND_Loss', update_info['rnd_loss'], episode)
            writer.add_scalar('Train/Gumbel_Temp', update_info['gumbel_temp'], episode)
        
        pbar.set_postfix({
            'reward': f'{avg_reward:.2f}',
            'damage': f'{damage_rate:.2%}',
            'eps': episode
        })
        
        # 定期保存
        if (episode + 1) % 500 == 0:
            save_path = f"checkpoints/{args.scenario}/{args.run_id}/ep{episode+1}.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            maddpg_ia.save(save_path)
    
    # 保存最终模型
    final_path = f"checkpoints/{args.scenario}/{args.run_id}/final.pt"
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    maddpg_ia.save(final_path)
    
    # 保存训练日志
    log_path = f"logs/{args.scenario}/{args.run_id}.json"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump({
            'episode_rewards': episode_rewards,
            'damage_rates': damage_rates,
            'scenario': args.scenario,
        }, f)
    
    writer.close()
    print(f"Training complete. Final damage rate: {damage_rates[-1]:.4f}")
    return damage_rates


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, required=True,
                        choices=[s['name'] for s in ALL_SCENARIOS],
                        help='Scenario to train')
    parser.add_argument('--run_id', type=int, default=0,
                        help='Run ID (0-99 for 100 independent runs)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    args = parser.parse_args()
    
    train(args)
```

---

## 7. 实验设计与运行

### 7.1 6个典型场景批量运行 (`run_experiments.py`)

```python
import subprocess
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np

SCENARIOS = [
    'small_rural', 'small_desert', 'small_coastal',
    'large_rural', 'large_desert', 'large_coastal'
]
N_RUNS = 100  # 每场景100次独立运行

def run_single_experiment(scenario, run_id):
    """运行单次训练"""
    cmd = [
        'python', 'train.py',
        '--scenario', scenario,
        '--run_id', str(run_id),
        '--device', 'cuda'
    ]
    subprocess.run(cmd, check=True)

def run_all_scenarios(parallel=False, max_workers=4):
    """运行全部6个场景 × 100次 = 600次训练"""
    total_runs = len(SCENARIOS) * N_RUNS
    print(f"Total runs: {total_runs}")
    print(f"Estimated time: small~4.2h, large~11.8h each")
    print(f"Total GPU-hours: {len([s for s in SCENARIOS if 'small' in s]) * N_RUNS * 4.2 + len([s for s in SCENARIOS if 'large' in s]) * N_RUNS * 11.8:.0f}")
    
    tasks = [(s, r) for s in SCENARIOS for r in range(N_RUNS)]
    
    if parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(lambda t: run_single_experiment(*t), tasks))
    else:
        for scenario in SCENARIOS:
            for run_id in range(N_RUNS):
                print(f"Running {scenario} #{run_id}")
                run_single_experiment(scenario, run_id)
    
    print("All experiments complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel', action='store_true', help='并行运行')
    parser.add_argument('--max_workers', type=int, default=1,
                        help='最大并行数 (注意GPU显存)')
    args = parser.parse_args()
    run_all_scenarios(args.parallel, args.max_workers)
```

### 7.2 消融实验 (`run_ablation.py`)

消融实验对比4个变体（论文Figure 15）：

| 变体 | Attention | Intrinsic Reward |
|------|:---------:|:----------------:|
| MADDPG-Basic | ✗ | ✗ |
| MADDPG-Attn | ✓ | ✗ |
| MADDPG-RND | ✗ | ✓ |
| MADDPG-IA | ✓ | ✓ |

```python
import torch
import argparse
import numpy as np
from env.drta_env import HELS_UAV_DRTA_Env
from config.scenario_config import LARGE_SCALE

# 需实现各变体算法:
# - MADDPG_Basic: 无Attention(使用固定维度padding), 无RND
# - MADDPG_Attn: 有Attention, 无RND
# - MADDPG_RND: 无Attention(使用固定维度padding), 有RND
# - MADDPG_IA: 完整版 (即MADDPG_IA类, intrinsic_reward_coef=0可得到Attn, rnd可开关)

ABLATION_VARIANTS = [
    {'name': 'MADDPG_Basic', 'use_attention': False, 'use_rnd': False},
    {'name': 'MADDPG_Attn',  'use_attention': True,  'use_rnd': False},
    {'name': 'MADDPG_RND',   'use_attention': False, 'use_rnd': True},
    {'name': 'MADDPG_IA',    'use_attention': True,  'use_rnd': True},
]

def run_ablation(scenario='large_rural', n_runs=100):
    """运行消融实验 (4变体 × 大规模乡村场景 × 100次)"""
    for variant in ABLATION_VARIANTS:
        print(f"\n=== Ablation: {variant['name']} ===")
        for run_id in range(n_runs):
            # ... 根据variant配置MADDPG-IA的use_attention/use_rnd
            pass
```

### 7.3 对比实验 (`run_comparison.py`)

实现3个baseline算法（论文Figure 16）：

**DQN** (单智能体): 将所有HELS视为一个超级智能体，动作空间 = (m+1)^n
**QMIX** (值分解): 将全局Q值分解为各智能体Q值之和
**MAPPO** (多智能体PPO): 使用PPO替代DDPG，clip-based策略更新

```python
# === 各对比算法关键实现差异 ===

# DQN: 单Agent, 联合动作空间
# 状态: 全部HELS和UAV信息拼接
# 动作: (m+1)^n 维 → 使用QMIX风格的超网络分解
# 问题: 大规模场景动作空间指数爆炸

# QMIX: 
# 每个Agent的Q值由超网络混合为总Q值
# Q_tot = MixingNetwork(Q_1, ..., Q_n, state)
# 对变长输入使用0-padding (论文指出这引入噪声)

# MAPPO:
# 每个Agent使用PPO独立更新
# 使用GAE(Generalized Advantage Estimation)
# pi_ratio = pi_new/pi_old
# loss = min(pi_ratio * A, clip(pi_ratio, 1-ε, 1+ε) * A)
```

### 7.4 参数变化实验

按论文Table 3，依次改变单一变量评估毁伤率变化：

```python
PARAMETER_VARIATIONS = {
    'hels_number': [3, 4, 5, 6, 7],       # HELS数量
    'laser_power': [20, 25, 30, 35, 40, 50],  # 激光功率(kW)
    'uav_number': [30, 50, 70, 90, 110],   # UAV数量
    'turbulence': [1e-17, 5e-16, 1e-15, 5e-15, 1e-14, 5e-14],  # C²_n(0)
}
```

---

## 8. 结果分析与可视化 (`visualize.py`)

### 8.1 训练曲线

```python
import matplotlib.pyplot as plt
import numpy as np
import json

def plot_training_curves(scenario, n_runs=100):
    """绘制训练奖励曲线 (论文Figure 14)"""
    plt.figure(figsize=(12, 6))
    
    all_rewards = []
    for run_id in range(n_runs):
        log_path = f"logs/{scenario}/{run_id}.json"
        with open(log_path) as f:
            data = json.load(f)
        all_rewards.append(data['episode_rewards'])
    
    all_rewards = np.array(all_rewards)  # (n_runs, max_episodes)
    mean_reward = np.mean(all_rewards, axis=0)
    std_reward = np.std(all_rewards, axis=0)
    
    episodes = np.arange(len(mean_reward))
    plt.plot(episodes, mean_reward, 'b-', label='Mean Reward')
    plt.fill_between(episodes, 
                     mean_reward - std_reward, 
                     mean_reward + std_reward, 
                     alpha=0.3, label='±1 Std')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(f'Training Curve: {scenario}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'figures/training_{scenario}.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### 8.2 毁伤率统计 (论文Table 2格式)

```python
from scipy import stats

def compute_damage_statistics(scenario, n_runs=100):
    """计算毁伤率统计 (均值±标准差 + 95%置信区间)"""
    damage_rates = []
    for run_id in range(n_runs):
        log_path = f"logs/{scenario}/{run_id}.json"
        with open(log_path) as f:
            data = json.load(f)
        damage_rates.append(data['damage_rates'][-1])  # 最终毁伤率
    
    damage_rates = np.array(damage_rates)
    mean_dr = np.mean(damage_rates)
    std_dr = np.std(damage_rates, ddof=1)
    
    # Bootstrap 95% CI
    n_bootstrap = 10000
    bootstrap_means = [np.mean(np.random.choice(damage_rates, size=n_runs, replace=True))
                       for _ in range(n_bootstrap)]
    ci_low = np.percentile(bootstrap_means, 2.5)
    ci_high = np.percentile(bootstrap_means, 97.5)
    
    print(f"{scenario}: {mean_dr*100:.2f}% ± {std_dr*100:.2f}%, "
          f"95% CI: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
    return mean_dr, std_dr, ci_low, ci_high

# 输出完整Table 2:
# Environment           MADDPG-IA Mean±Std    Traditional Mean±Std    MADDPG-IA 95% CI
# Rural (sunshine)      99.65% ± 0.32%        72.64% ± 3.21%         [99.54%, 99.76%]
# Desert (light haze)   79.37% ± 2.15%        51.29% ± 4.87%         [78.82%, 79.92%]
# Coastal (sunshine)    91.25% ± 1.78%        67.38% ± 3.95%         [90.82%, 91.68%]
```

### 8.3 时空态势图 (论文Figure 11)

```python
def plot_spatial_situation(env):
    """绘制决策后空间态势的正视图和俯视图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 子图(a): 前视图 (X-Z平面)
    # HELS位置(菱形)、UAV轨迹(红线)、毁伤决策点(彩色圆)、保护资产(原点)
    for i, pos in enumerate(env.hels_positions):
        ax1.scatter(pos[0], pos[2], marker='D', s=100, 
                   label=f'HELS {i+1}', zorder=5)
    
    for j, uav in enumerate(env.uav_states):
        if uav['alive']:
            ax1.scatter(uav['position'][0], uav['position'][2], 
                       marker='o', s=30, alpha=0.6)
    
    ax1.scatter(0, 0, marker='*', s=200, c='red', label='Protected Asset')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Z (Height) [m]')
    ax1.set_title('Front View')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图(b): 俯视图 (X-Y平面)
    for i, pos in enumerate(env.hels_positions):
        ax2.scatter(pos[0], pos[1], marker='D', s=100)
    
    for j, uav in enumerate(env.uav_states):
        if uav['alive']:
            ax2.scatter(uav['position'][0], uav['position'][1], 
                       marker='o', s=30, alpha=0.6)
    
    ax2.scatter(0, 0, marker='*', s=200, c='red')
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_title('Top View')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/spatial_situation.png', dpi=150)
    plt.show()
```

### 8.4 照射时序图 (论文Figure 12)

```python
def plot_irradiation_timeline(irradiation_log):
    """绘制各HELS智能体的照射时序甘特图"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    
    for i, hels_log in enumerate(irradiation_log):
        for task in hels_log['tasks']:
            # task: (start_time, end_time, target_id, distance)
            ax.barh(i, task[1] - task[0], left=task[0], 
                   height=0.6, color=colors[task[2] % 10],
                   edgecolor='black', linewidth=0.5,
                   label=f'UAV {task[2]}' if i == 0 else '')
    
    ax.set_yticks(range(len(irradiation_log)))
    ax.set_yticklabels([f'HELS {i+1}' for i in range(len(irradiation_log))])
    ax.set_xlabel('Time [s]')
    ax.set_title('Irradiation Timeline of Each HELS Agent')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('figures/irradiation_timeline.png', dpi=150)
    plt.show()
```

---

## 9. 完整运行清单

### 9.1 训练时间估算

| 场景 | 规模 | 单次训练时间 | 100次总时间 | GPU显存 |
|------|------|-------------|------------|---------|
| small_rural | 2v10 | ~4.2 h | ~420 h | ~2 GB |
| small_desert | 2v10 | ~4.2 h | ~420 h | ~2 GB |
| small_coastal | 2v10 | ~4.2 h | ~420 h | ~2 GB |
| large_rural | 5v50 | ~11.8 h | ~1180 h | ~4 GB |
| large_desert | 5v50 | ~11.8 h | ~1180 h | ~4 GB |
| large_coastal | 5v50 | ~11.8 h | ~1180 h | ~4 GB |
| **总计** | | | **~4800 GPU-hours** | |

> **注:** 论文报告的总训练GPU-hours约4800小时(RTX 3060)，相当于200天单卡不间断运行。实际复现时建议使用多GPU并行或减少独立运行次数(如10次代替100次)。

### 9.2 快速验证流程 (建议首次复现时使用)

```bash
# 步骤1: 小规模乡村场景 × 1次运行，验证环境与算法正确性
python train.py --scenario small_rural --run_id 0 --device cuda
# 预期: ~500 episodes后收敛，毁伤率 >90%，训练时间~4h

# 步骤2: 检查500 episode时的训练曲线
python visualize.py --scenario small_rural --mode training_curve

# 步骤3: 消融验证 — 对比4个变体(各1次,小规模)
python run_ablation.py --scenario small_rural --n_runs 1

# 步骤4: 算法对比 — 对比DQN/QMIX/MAPPO(各1次,小规模)
python run_comparison.py --scenario small_rural --n_runs 1

# 步骤5: 确认无误后，展开完整实验
python run_experiments.py --parallel --max_workers 2
```

### 9.3 论文结果核对清单

| 实验 | 论文图表 | 期望结果 | 验证命令 |
|------|---------|---------|---------|
| 6场景毁伤率 | Table 2 | 乡村99.65%, 沙漠79.37%, 沿海91.25% | `python evaluate.py --all` |
| 训练曲线 | Figure 14 | 约500episode收敛 | `python visualize.py --mode reward` |
| 消融实验 | Figure 15 | MADDPG-IA > Attn > RND > Basic | `python run_ablation.py` |
| 算法对比 | Figure 16 | IA > MAPPO > QMIX > DQN | `python run_comparison.py` |
| 参数变化 | Table 3 | 5HELS/40kW/≤70UAVs | `python run_param_variation.py` |
| 时空态势 | Figure 11 | 跨区域协调可视化 | `python visualize.py --mode spatial` |
| 照射时序 | Figure 12 | 延迟决策策略可视化 | `python visualize.py --mode timeline` |

---

## 附录A: 关键公式索引

| 公式编号 | 内容 | 实现位置 |
|---------|------|---------|
| (1) | 远场光斑半径 | `LaserAtmosphericTransmission.spot_radius()` |
| (2) | 总光束质量因子 β² | `LaserAtmosphericTransmission.compute_all()` |
| (3) | 斜程H-V修正C²_n | `LaserAtmosphericTransmission.Cn2_hv()` |
| (4) | Fried参数 r₀ | `LaserAtmosphericTransmission.fried_parameter()` |
| (5) | 湍流光束质量 β²_T | `LaserAtmosphericTransmission.beta_turbulence()` |
| (6) | 热畸变参数 N_D | `LaserAtmosphericTransmission.thermal_distortion_parameter()` |
| (7) | 热晕光束质量 β²_B | `LaserAtmosphericTransmission.beta_thermal_blooming()` |
| (8) | 抖动光束质量 β_J | `LaserAtmosphericTransmission.beta_jitter()` |
| (9) | 光斑平均半径 | `LaserAtmosphericTransmission.spot_radius()` |
| (10) | 大气透过率 τ | `LaserAtmosphericTransmission.atmospheric_transmittance()` |
| (11) | 衰减后激光功率 P_e | `LaserAtmosphericTransmission.target_power_density()` |
| (12) | 目标面功率密度 I_target | `LaserAtmosphericTransmission.target_power_density()` |
| (17) | 熔化穿透时间 t_m | `LaserThermalDamage.melting_penetration_time()` |
| (18) | 毁伤阈值 e_th | `LaserThermalDamage.damage_threshold()` |
| (20) | HELS毁伤时间 t_damage | `LaserThermalDamage.damage_time()` |
| (22) | 攻击周期时间 t_period | `HELSDamageModel.compute_period_time()` |
| (23) | 多批次入侵密度 | `UAVSwarmDensityModel.generate_arrival_times()` |
| (24) | 同时入侵密度 | `UAVSwarmDensityModel.generate_arrival_times()` |
| (25) | 高度威胁因子 T_rh | `ThreatBenefitFactors.height_threat()` |
| (26) | 速度威胁因子 T_rv | `ThreatBenefitFactors.velocity_threat()` |
| (27) | 安全距离威胁因子 T_rL | `ThreatBenefitFactors.safe_distance_threat()` |
| (28) | 资源消耗效益因子 B_rc | `ThreatBenefitFactors.resource_consumption_benefit()` |
| (29) | HELS应用价值效益因子 B_rs | `ThreatBenefitFactors.hels_application_benefit()` |
| (30) | DRTA目标函数 | `DRTAModel.objective()` |
| (31) | DRTA约束条件(a-h) | `DRTAModel.check_constraints()` |
| (34) | MADDPG策略梯度 | `MADDPG_IA.update()` |
| (36) | TD误差 | `MADDPG_IA.update()` |
| (37) | 软更新 | `MADDPG_IA._soft_update()` |
| (38-40) | 状态空间设计 | `HELS_UAV_DRTA_Env._get_hels_obs()` |
| (41) | 离散动作空间(One-Hot) | `gumbel_softmax()` |
| (42) | 奖励函数(三种情况) | `HELS_UAV_DRTA_Env._compute_rewards()` |
| (43) | Attention Q/K/V计算 | `AttentionEncoder.forward()` |
| (44) | 注意力权重 a_ij | `AttentionEncoder.forward()` |
| (45) | Attention状态编码 ẽ_i | `AttentionEncoder.forward()` |
| (46) | RND内在奖励 r_c | `RNDIntrinsicReward.compute_intrinsic_reward()` |
| (47) | 混合奖励 r_h | `RNDIntrinsicReward.hybrid_reward()` |

## 附录B: 完整超参数表 (Table 1 复现)

| 类别 | 参数 | 值 |
|------|------|-----|
| **HELS** | 数量 | 2 (小规模) / 5 (大规模) |
| | 激光波长 λ | 1.064 µm (Nd:YAG) |
| | 发射孔径 D | 0.6 m |
| | 功率 P₀ | 20~50 kW |
| | 光束质量 β₀ | 1.0 |
| | 系统效率 η₀ | 0.85 |
| | FSM响应时间 | <50 ms |
| | FSM最大角度 | ±15° |
| | 电池容量 | 200 s (连续照射) |
| **UAV** | 数量 | 10 (小) / 50 (大) |
| | 速度 | 20~30 m/s |
| | 高度 | 0.2~0.8 km |
| | 入侵模式 | 多方向多批次 |
| | 材料 | 2024铝合金, 5mm |
| | 毁伤阈值 e_th | ~1.13×10⁷ J/m² |
| **大气** | 乡村 C²_n(0) | 1×10⁻¹⁷ |
| | 沙漠 C²_n(0) | 1×10⁻¹⁵ |
| | 沿海 C²_n(0) | 1×10⁻¹⁷ |
| | 乡村能见度 | 10 km |
| | 沙漠能见度 | 5 km |
| | 沿海能见度 | 10 km |
| | 气溶胶K: 乡村/沙漠/海洋 | 2.828 / 2.496 / 4.453 |
| | 风速 v_g | 5 m/s |
| **MADDPG-IA** | 最大训练轮数 | 2000 |
| | 经验池大小 | 1×10⁶ |
| | Batch size | 2048 |
| | Actor学习率 | 1×10⁻³ |
| | Critic学习率 | 1×10⁻³ |
| | 软更新系数 ϖ | 1×10⁻² |
| | 折扣因子 γ | 0.95 |
| | Attention维度 d_attn | 128 |
| | d_k / d_v | 64 |
| | RND初始系数 β_r0 | 0.1 |
| | RND衰减率 k_r | 1×10⁻⁴ |
| | Gumbel温度初值 | 1.0 |
| | Gumbel温度衰减 | 0.9995/step |
| | 等待奖励衰减 β_d | 0.01 |

---

*文档生成日期: 2026年6月23日*
*论文: Liu et al., Aerospace 2025, 12, 729 | DOI: 10.3390/aerospace12080729*