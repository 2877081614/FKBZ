# 非 CV 路线：UAV 运动轨迹与模型数据工作方案

整理日期：2026-07-01

## 1. 研究定位调整

如果课题不接触 CV 模型，数据工作不应围绕图像、视频检测或视觉跟踪展开。后续应将重点放在：

1. UAV 运动轨迹数据：位置、速度、加速度、航向、姿态、控制输入。
2. UAV 运动模型：质点模型、Dubins/Unicycle 模型、固定翼模型、多旋翼动力学模型。
3. 威胁行为模型：直线突防、蛇形机动、集群协同、诱饵分流、饱和攻击。
4. 战场环境模型：防御资源部署、保护目标、传感器覆盖、通信延迟、资源消耗。
5. MARL episode 数据：将轨迹、威胁等级、资源状态和奖励统一成可训练格式。

图像数据集不再作为主线数据源。Anti-UAV、VisDrone、UAVDT 等视频数据只保留一个用途：从标注中提取目标轨迹或估计探测误差；若不做感知误差建模，可以完全不使用。

## 2. 数据工作总路线

推荐采用“少量真实飞行轨迹校准 + 大规模参数化模型生成 + 仿真环境训练”的路线。

```text
真实 UAV 轨迹/动力学数据
        ↓ 参数统计
速度、加速度、转弯半径、爬升率、机动频率、轨迹平滑性
        ↓ 模型校准
点质量 / Dubins / Unicycle / Quadrotor / Fixed-wing 运动模型
        ↓ 场景生成
单机突防、蜂群突防、诱饵分流、饱和攻击、协同机动
        ↓ MARL 环境
防御资源动态编组、目标分配、角色分工、风险约束
        ↓ 训练与评估
MADDPG / MAPPO / QMIX / GNN-MARL / 分层 MARL
```

这个路线的优点是：不依赖 CV，数据可控，实验可复现，且更贴合“动态编组与资源分配”的核心科学问题。

## 3. 推荐保留的数据源

### 3.1 EuRoC MAV Dataset

来源：

- ETH Zurich ASL 数据集主页：https://projects.asl.ethz.ch/datasets/

可用数据：

- MAV 飞行轨迹。
- IMU 数据。
- Vicon 或 Leica ground truth。
- 不同难度的室内飞行序列。

使用方式：

- 不使用图像，只读取 ground truth pose、IMU、时间戳。
- 提取速度、加速度、姿态角速度、轨迹曲率。
- 用于校准多旋翼 UAV 的运动边界。

适合支撑：

- 多旋翼运动模型参数。
- 轨迹平滑性和机动强度统计。
- 仿真目标运动模型校准。

### 3.2 Blackbird Dataset

来源：

- MIT Blackbird Dataset：http://blackbird-dataset.mit.edu/
- 论文：The Blackbird Dataset: A large-scale dataset for UAV perception in aggressive flight。

可用数据：

- 168 次飞行。
- 17 类轨迹。
- 多环境高速/激进飞行。
- motion capture ground truth。
- IMU、电机转速等数据。

使用方式：

- 不使用合成相机图像。
- 只读取 mocap 轨迹、速度、IMU、电机转速。
- 提取高机动无人机速度上限、加速度上限、转弯行为。

适合支撑：

- 高机动突防目标建模。
- 激进机动轨迹生成。
- “普通目标 vs 高机动目标”的实验分层。

### 3.3 Race Against the Machine / Drone Racing Dataset

来源：

- 官方仓库：https://github.com/tii-racing/drone-racing-dataset

可用数据：

- 高速自主/人工 FPV 飞行。
- CSV 中包含时间戳、IMU、油门/通道输入、电池电压、位置、姿态、线速度、角速度等字段。
- 仓库提供下载脚本、数据格式说明和轨迹生成脚本。

使用方式：

- 不使用图像和 gate 标注。
- 只用 CSV 中的状态量和控制输入。
- 提取高速飞行状态分布和控制输入变化。

适合支撑：

- 高速穿越/规避式 UAV 运动模型。
- 控制输入约束。
- 高动态突防场景。

### 3.4 INTERACTION / highD 等轨迹数据

来源：

- INTERACTION Dataset：https://interaction-dataset.com/
- highD Dataset：http://www.highD-dataset.com/

注意：

- 这些不是 UAV 飞行数据，而是车辆/交通参与者轨迹数据。
- 可作为多智能体交互、队形变化、避让、合流、竞争/协作运动的行为模板。

使用方式：

- 不直接当作 UAV 物理轨迹。
- 只抽象使用其群体交互模式，例如合流、分散、避让、跟随、穿插。
- 将速度和加速度重新缩放到 UAV 合理范围。

适合支撑：

- 蜂群协同行为生成。
- 多目标交互场景压力测试。
- 饱和攻击和分流诱骗策略模板。

### 3.5 VMAS / gym-pybullet-drones

来源：

- VMAS：https://github.com/proroklab/VectorizedMultiAgentSimulator
- gym-pybullet-drones：https://github.com/utiasDSL/gym-pybullet-drones

定位：

- 不是数据集，而是生成 MARL episode 的核心仿真环境。

使用方式：

- VMAS：用于二维大规模、多智能体、高吞吐训练。
- gym-pybullet-drones：用于加入多旋翼动力学约束。

适合支撑：

- MARL 训练。
- 批量生成战场态势。
- 资源编组、目标分配、角色分工、风险约束实验。

## 4. 建议剔除或降级的数据源

如果明确不做 CV，则以下数据不应作为主线：

| 数据源                    | 原用途      | 建议处理           |
| ---------------------- | -------- | -------------- |
| VisDrone               | 图像检测/跟踪  | 剔除主线；最多用于非核心参考 |
| UAVDT                  | 航拍检测/跟踪  | 剔除主线；不建议投入精力   |
| HIT-UAV                | 红外检测     | 降级为可选感知误差数据    |
| Drone-vs-Bird          | 视觉误警     | 剔除主线           |
| Anti-UAV / Anti-UAV410 | 反无人机视频跟踪 | 只保留“轨迹提取”用途    |
| MMAUD                  | 多模态检测    | 若不做传感器融合，剔除主线  |

保留原则：

- 有位姿、速度、IMU、控制输入、ground truth trajectory 的数据优先。
- 只有图像和检测框的数据不优先。
- 车辆/人群轨迹可以作为群体行为模板，但不能直接作为 UAV 运动物理模型。

## 5. 统一轨迹数据格式

建议后续所有真实数据、生成数据和仿真 episode 都转换为统一 CSV 或 Parquet 格式。

### 5.1 单目标轨迹格式

```text
episode_id
target_id
t
x
y
z
vx
vy
vz
ax
ay
az
heading
pitch
roll
yaw_rate
speed
threat_level
behavior_type
source
```

### 5.2 多目标态势格式

```text
episode_id
t
target_count
defense_unit_count
protected_asset_x
protected_asset_y
protected_asset_z
global_threat_score
communication_delay
sensor_noise_level
scenario_type
```

### 5.3 防御资源状态格式

```text
episode_id
unit_id
t
resource_type
x
y
z
available
cooldown
energy
range
tracking_target_id
role
```

### 5.4 MARL transition 格式

```text
episode_id
t
agent_id
obs
action
reward
next_obs
done
info
```

其中 `obs` 和 `next_obs` 可以是 JSON、NumPy array 或图结构序列化结果。

## 6. 战场环境生成方案

### 6.1 目标类型

建议至少设计四类 UAV 目标：

1. 普通侦察型：速度低，机动弱，威胁中等。
2. 高速突防型：速度高，突防时间短，威胁高。
3. 高机动规避型：转弯频繁，轨迹不稳定，拦截难度高。
4. 诱饵消耗型：威胁低，但用于吸引防御资源。

### 6.2 运动模型

建议从简单到复杂逐级构建：

1. 常速度模型：用于最初调试。
2. 常加速度模型：用于平滑机动。
3. Dubins / Unicycle 模型：用于固定翼或受限转弯半径目标。
4. 多旋翼简化模型：用于悬停、急停、变向、爬升。
5. 轨迹库回放模型：从真实 UAV 轨迹中抽取片段并重采样。
6. 群体行为模型：编队、分散、合围、诱饵分流、饱和突防。

### 6.3 场景类型

建议设计以下训练与测试场景：

1. 单方向突防。
2. 多方向同时突防。
3. 分批次波次攻击。
4. 高低空混合突防。
5. 诱饵目标牵制。
6. 高价值目标掩护低价值目标。
7. 通信延迟或局部观测缺失。
8. 防御资源部分失效。

### 6.4 参数校准

从真实 UAV 轨迹数据中统计：

- 速度分布。
- 加速度分布。
- 最大爬升/下降率。
- 最大转弯角速度。
- 轨迹曲率分布。
- 机动持续时间。
- 悬停/直线/转弯/加速行为比例。

这些统计量用于约束仿真生成器，避免生成“不像 UAV”的轨迹。

## 7. 后续开展步骤

### 阶段 1：建立非 CV 数据目录与元数据表

建议目录：

```text
datasets/
  raw/
    euroc_mav/
    blackbird/
    drone_racing/
    interaction/
    highd/
  processed/
    uav_trajectories/
    motion_statistics/
    scenario_episodes/
    marl_transitions/
  configs/
    motion_models/
    scenario_templates/
    resource_templates/
```

目标：

- 不下载或处理图像文件。
- 只保留轨迹、IMU、姿态、速度、控制输入、场景参数。

### 阶段 2：轨迹抽取与标准化

每个数据集写一个 converter：

```text
raw dataset -> standardized trajectory table
```

输出：

- 标准轨迹表。
- 运动统计表。
- 可视化轨迹图。
- 数据集说明与引用信息。

### 阶段 3：运动模型参数拟合

对每类 UAV 目标拟合参数：

- 速度范围。
- 最大加速度。
- 最大转弯率。
- 轨迹噪声。
- 机动模式切换概率。

输出：

- `configs/motion_models/recon_uav.yaml`
- `configs/motion_models/fast_intruder.yaml`
- `configs/motion_models/agile_intruder.yaml`
- `configs/motion_models/decoy_uav.yaml`

### 阶段 4：场景生成器开发

实现一个不依赖 CV 的 scenario generator：

输入：

- 目标数量。
- 威胁类型比例。
- 进入方向。
- 防御资源配置。
- 保护目标位置。
- 通信/传感器噪声参数。

输出：

- 标准 episode。
- MARL 环境初始状态。
- 威胁等级标签。
- ground truth 轨迹。

### 阶段 5：MARL 环境接入

建议先用 VMAS 做二维高效训练环境，再用 gym-pybullet-drones 做动力学验证。

训练环境应包括：

- UAV 目标运动更新。
- 防御资源状态更新。
- 目标分配动作。
- 编组角色动作。
- 资源冷却和消耗。
- 突防/拦截/压制奖励。

### 阶段 6：实验基线

至少保留以下基线：

- 静态最近目标分配。
- 威胁优先级贪心分配。
- 匈牙利算法 / WTA 优化基线。
- MADDPG。
- MAPPO。
- QMIX。
- 图注意力 MARL。

## 8. 关键结论

后续数据工作应从“找图像数据集”转为“构建可复现的轨迹-模型-场景生成体系”。真实 UAV 轨迹数据的作用不是直接覆盖所有战场场景，而是校准仿真模型，使生成的无人机运动具有合理的速度、加速度、转弯率和机动模式。真正用于 MARL 训练的大规模数据应由参数化战场环境生成。

推荐主线：

```text
EuRoC MAV / Blackbird / Drone Racing Dataset
        ↓
运动参数统计与轨迹片段库
        ↓
UAV 威胁运动模型
        ↓
VMAS / gym-pybullet-drones 反无人机动态编组环境
        ↓
MARL 训练与评估
```

这条路线与“反无人机动态编组”的研究目标最一致，也能避免课题被 CV 检测任务牵着走。
