# 防空编组强化学习环境模型设计文档

更新时间：2026-07-09

本文档基于 `research_papers/05_anti_uav_rl_environment_model/` 中的反无人机、任务分配、多智能体强化学习和仿真平台文献，设计一个面向后续算法实验的防空编组强化学习环境。本文档不是当前 `AirDefenseResourceAssignmentEnv v0` 的代码说明，而是面向下一阶段环境升级的模型规格。

核心目标：

```text
构建一个可训练、可比较、可扩展的防空编组动态资源分配环境，
用于研究来袭无人机威胁下的探测、跟踪、分配、拦截与压制决策。
```

## 1. 文献依据与建模启发

### 1.1 决策级拦截优先级

`P01_2025_RL_Decision_Level_Interception_Prioritization_Drone_Swarm_Defense.pdf` 直接给出了本项目最接近的环境原型。该文将问题建模为决策级无人机蜂群防御任务，防御方需要协调多个 `effector`，即拦截器、定向能武器等防御效应器，对多个来袭无人机进行拦截优先级决策。

对本项目的启发：

- agent 可以先表示集中式防空指挥决策器，而不是单枚导弹或单个目标。
- 状态空间应包含目标位置、目标类别、目标威胁能力、防御资源状态等。
- 动作空间可以设计为“每个防御资源选择一个目标”，即资源-目标分配。
- 奖励函数不应只奖励命中，而应围绕保护高价值目标、减少区域损伤和提高防御效率设计。

### 1.2 反无人机功能链

`P05_2020_Counter_UAS_State_of_the_Art_Challenges_Future_Trends.pdf` 和 `P06_2022_Aerial_Threats_Radar_Communications_Survey.pdf` 强调，C-UAS 系统不是单一拦截动作，而是包含：

```text
detect -> track -> identify -> mitigate
探测 -> 跟踪 -> 识别 -> 处置
```

相关资源包括：

- 雷达
- 光电/视觉传感器
- 被动射频探测
- 声学探测
- 数据融合系统
- 干扰设备
- 物理捕获或拦截武器

对本项目的启发：

- 环境状态不能只包含目标真实位置，还应逐步加入探测置信度、跟踪误差、目标分类置信度。
- 防御动作不能只包含“发射/拦截”，后续应扩展到“探测、跟踪、干扰、拦截、保持”等动作类型。
- 目标小型化、低空飞行、蜂群化会带来观测不确定性和部分可观测问题。

### 1.3 延迟、不确定性与 Dec-POMDP

`P02_2026_Delay_Aware_Active_Triangulation_Uncertainty_Driven_MARL_Counter_UAS.pdf` 将反无人机定位问题建模为带通信延迟的 Dec-POMDP，并强调 Age-of-Information，简称 AoI，对多智能体协同的重要性。

对本项目的启发：

- 后续多智能体版本应采用 Dec-POMDP 表述。
- 每个防御单元只能获得局部观测和延迟共享信息。
- 观测中应包含信息新鲜度 AoI、传感器误差、轨迹协方差或置信度。
- 奖励函数应尽量基于 agent 实际可获得的感知信息，而不是直接使用完美真实状态。

### 1.4 时空任务分配与动态重规划

`P03_2021_Decentralized_Multi_UAV_Spatio_Temporal_Multi_Task_Allocation_Perimeter_Defense.pdf` 将入侵者视为时空任务，防御者需要在时间和空间约束下分配任务。`P11_2018_Partial_Replanning_Decentralized_Dynamic_Task_Allocation.pdf` 进一步强调动态任务出现时，系统需要快速局部重规划，而不是每次全局重算。

对本项目的启发：

- 来袭无人机不只是一个目标点，而是一个带有预计突防位置和预计到达时间的时空任务。
- 状态空间应包含预计突防时间、预计突防区域、拦截窗口、资源到达时间。
- 动作设计应允许动态分配和重分配。
- 奖励可引入时间折扣：越晚完成拦截或处置，收益越低。

### 1.5 异构资源与图结构

`P04_2025_MAGNNET_GNN_Task_Allocation_Autonomous_Vehicles_DRL.pdf` 和 `P08_2023_Multi_Target_Pursuit_Heterogeneous_UAV_Swarm_DMARL.pdf` 都强调异构多智能体任务分配。`P04` 进一步使用 GNN 表示 agent-task 关系，并采用 CTDE，即 centralized training decentralized execution。

对本项目的启发：

- 防空资源应建模为异构单元，不同资源有不同射程、成本、冷却、命中概率和适用目标。
- 资源-目标关系天然是二部图：防御单元是资源节点，来袭目标是任务节点，边表示可拦截性、距离、成功概率和成本。
- 初期可以使用固定长度向量；后续应扩展到图观测和 GNN 策略网络。
- 多智能体阶段应使用 CTDE：训练时可访问全局状态，执行时每个 agent 使用局部观测。

### 1.6 多智能体算法与环境 API

`P12_2017_MADDPG_Mixed_Cooperative_Competitive_Environments.pdf`、`P13_2021_MAPPO_Surprising_Effectiveness_PPO_Cooperative_Multi_Agent_Games.pdf`、`P14_2020_PettingZoo_Gym_for_Multi_Agent_RL.pdf` 和 `P15_2017_MAgent_Many_Agent_RL_Platform.pdf` 给出多智能体算法和环境接口依据。

对本项目的启发：

- 先实现 Gymnasium 单智能体集中决策环境，便于 DQN/PPO baseline。
- 再实现 PettingZoo ParallelEnv 或 AECEnv，支持 MAPPO、MADDPG、QMIX 等 MARL 算法。
- 多智能体环境中应显式区分 global state、local observation、joint action、shared reward。
- 环境应保留 action mask、render、seed、scenario config，保证实验可复现。

## 2. 环境总体定位

建议将下一阶段环境命名为：

```text
AirDefenseResourceAssignmentEnv v1
```

环境定位：

```text
面向防空编组的动态资源-目标分配强化学习环境。
```

它不是导弹制导环境，也不是无人机低层飞控环境，而是更高层的决策级环境。决策对象包括：

- 哪些目标最危险；
- 哪些目标需要优先处置；
- 哪个防御单元负责哪个目标；
- 是否使用拦截、干扰、跟踪或等待；
- 是否需要保留资源应对后续目标。

## 3. 建议采用的建模层次

### 3.1 第一阶段：集中式单智能体

第一阶段建议保留当前 v0 的基本方向：

```text
agent = 防空编组指挥决策器
```

这个 agent 代表一个集中式火控/资源分配决策节点，统一控制所有防御资源。

优点：

- 便于接入 DQN、PPO、Maskable PPO。
- 便于快速验证状态、动作、奖励是否合理。
- 便于与规则 baseline、Hungarian/CBBA 类任务分配方法比较。

缺点：

- 无法体现通信延迟、局部观测和分布式协同。
- 动作空间会随资源数和目标数增长。

### 3.2 第二阶段：多智能体防御单元

第二阶段建议将每个 agent 定义为：

```text
agent_i = 一个防空作战单元 / 防御节点
```

这里的防空作战单元可以是：

- 一套近程防空火力单元；
- 一套雷达-火控-拦截单元；
- 一套干扰单元；
- 一套机动拦截无人机；
- 一个异构防御平台。

不建议把 agent 定义为“单枚导弹”，因为导弹是一次性消耗资源，不是持续决策主体。不建议把 agent 定义为“来袭目标”，因为本项目的学习主体是防御方。

推荐定义：

```text
agent 是具备持续观测、状态保持和动作选择能力的防御作战单元；
missile / laser shot / jamming pulse 是 agent 可调用的资源或动作效果。
```

### 3.3 第三阶段：分层智能体

当环境和 baseline 稳定后，可以扩展为分层结构：

```text
上层 agent：编组指挥节点，负责威胁排序和任务分配
下层 agent：防御单元，负责局部跟踪、干扰、拦截执行
```

但不建议一开始就做分层，否则会同时引入太多变量，难以判断算法失败是因为环境、奖励还是结构复杂。

## 4. MDP / POMDP / Dec-POMDP 形式化

### 4.1 单智能体 POMDP

第一阶段可建模为：

```text
M = <S, A, O, P, R, gamma>
```

其中：

- `S`：环境真实状态；
- `O`：agent 可见观测；
- `A`：资源分配动作；
- `P`：状态转移，包括目标运动、探测更新、拦截结果、资源冷却；
- `R`：奖励函数；
- `gamma`：折扣因子。

虽然实现上可先把真实状态直接作为观测，但文档和接口中应保留 `state` 与 `observation` 的区别。

### 4.2 多智能体 Dec-POMDP

第二阶段可建模为：

```text
M = <I, S, {A_i}, P, R, {O_i}, Z, gamma>
```

其中：

- `I`：防御 agent 集合；
- `S`：全局真实状态；
- `A_i`：第 i 个防御单元的动作空间；
- `P`：联合动作下的状态转移；
- `R`：团队共享奖励；
- `O_i`：第 i 个防御单元的局部观测；
- `Z`：观测模型，包括传感器噪声、通信延迟、AoI；
- `gamma`：折扣因子。

该形式方便后续接入 MAPPO、MADDPG、QMIX 等算法。

## 5. 场景元素设计

### 5.1 被保护目标 / 关键区域

环境中不应只有一个抽象 asset。建议设计多个关键区域：

```text
protected_zones = {zone_1, zone_2, ..., zone_Z}
```

每个区域包含：

| 字段         | 含义                  |
| ---------- | ------------------- |
| `position` | 区域中心位置              |
| `radius`   | 区域半径                |
| `value`    | 保护价值                |
| `damage`   | 已受损程度               |
| `priority` | 作战优先级               |
| `type`     | 指挥所、雷达站、机场、弹药库等抽象类型 |

奖励函数应围绕这些区域的损伤最小化设计，而不是只统计目标是否漏过。

### 5.2 来袭无人机目标

每个目标建议包含：

| 字段                 | 含义                                              |
| ------------------ | ----------------------------------------------- |
| `position`         | 真实位置                                            |
| `velocity`         | 速度向量                                            |
| `heading`          | 航向                                              |
| `target_zone`      | 预计攻击区域                                          |
| `time_to_impact`   | 预计突防/撞击时间                                       |
| `threat_level`     | 威胁等级                                            |
| `class_id`         | 目标类型                                            |
| `payload`          | 载荷/破坏能力                                         |
| `evasion`          | 规避能力                                            |
| `jam_resistance`   | 抗干扰能力                                           |
| `track_confidence` | 跟踪置信度                                           |
| `covariance`       | 位置估计不确定性                                        |
| `aoi`              | 信息新鲜度                                           |
| `status`           | alive / tracked / jammed / intercepted / leaked |

其中 `time_to_impact` 和 `target_zone` 是连接防空任务分配与时空任务分配文献的关键变量。

### 5.3 防御资源 / 作战单元

建议将防御方资源分为四类：

```text
sensor
interceptor
directed_energy
jammer
```

第一阶段可以只实现 `interceptor` 和 `jammer`，后续再扩展传感器链路。

每个防御单元包含：

| 字段                    | 含义                                        |
| --------------------- | ----------------------------------------- |
| `unit_id`             | 单元编号                                      |
| `unit_type`           | missile / laser / jammer / radar / hybrid |
| `position`            | 单元位置                                      |
| `mobility`            | 是否可机动                                     |
| `range`               | 有效作用距离                                    |
| `sector`              | 作用扇区或视场                                   |
| `ammo`                | 弹药数量                                      |
| `energy`              | 能量余量                                      |
| `cooldown`            | 冷却时间                                      |
| `reload_time`         | 再装填时间                                     |
| `base_success_prob`   | 基础成功概率                                    |
| `cost`                | 使用成本                                      |
| `health`              | 单元健康状态                                    |
| `sensor_quality`      | 传感器质量                                     |
| `communication_delay` | 通信延迟                                      |

### 5.4 资源-目标关系

资源和目标之间建议显式构建边特征：

| 边特征                   | 含义        |
| --------------------- | --------- |
| `distance`            | 资源到目标距离   |
| `bearing`             | 目标相对方位    |
| `in_range`            | 是否在射程内    |
| `in_sector`           | 是否在作用扇区内  |
| `line_of_sight`       | 是否可视/可作用  |
| `time_to_intercept`   | 预计拦截时间    |
| `success_prob`        | 成功概率      |
| `expected_benefit`    | 期望收益      |
| `resource_cost`       | 资源消耗      |
| `assignment_conflict` | 是否存在多资源冲突 |

这为后续 GNN 策略提供自然输入。

## 6. 状态空间设计

### 6.1 全局真实状态

环境内部真实状态 `S_t` 建议包含：

```text
S_t = {
  protected_zones,
  targets,
  defense_units,
  tracks,
  assignments,
  communication_state,
  time_state
}
```

这里的 `tracks` 表示传感器融合后的目标航迹，不一定等于真实目标状态。

### 6.2 单智能体观测

第一阶段集中式 agent 的观测可设计为固定长度向量：

```text
O_t = concat(
  zone_features,
  target_features,
  defense_unit_features,
  relation_features,
  global_features
)
```

为了便于 DQN/PPO 训练，先采用最大数量 padding：

```text
max_targets = N_max
max_defense_units = M_max
max_zones = Z_max
```

不足数量用 0 padding，并提供 mask：

```text
target_mask
unit_mask
action_mask
```

### 6.3 目标特征

每个目标的观测特征建议为：

```text
[
  x_norm,
  y_norm,
  vx_norm,
  vy_norm,
  distance_to_nearest_zone_norm,
  time_to_impact_norm,
  threat_level,
  payload_norm,
  class_one_hot,
  track_confidence,
  position_uncertainty_norm,
  aoi_norm,
  is_alive,
  is_jammed,
  is_assigned
]
```

当前 v0 已有位置、速度、距离、威胁和存活状态。v1 应重点补充：

- `time_to_impact`
- `target_zone`
- `track_confidence`
- `aoi`
- `class_id`

### 6.4 防御单元特征

每个防御单元的观测特征建议为：

```text
[
  x_norm,
  y_norm,
  unit_type_one_hot,
  ammo_ratio,
  energy_ratio,
  cooldown_ratio,
  range_norm,
  sector_center,
  sector_width,
  base_success_prob,
  cost_norm,
  health_ratio,
  sensor_quality,
  communication_delay_norm,
  is_available
]
```

当前 v0 已有位置、类型、弹药、冷却、射程和基础命中率。v1 应重点补充：

- 能量/弹药多资源建模；
- 扇区或视场限制；
- 单元健康状态；
- 通信延迟；
- 传感器质量。

### 6.5 关键区域特征

每个保护区域的观测特征建议为：

```text
[
  x_norm,
  y_norm,
  radius_norm,
  value_norm,
  damage_ratio,
  priority,
  zone_type_one_hot
]
```

如果第一阶段只保留一个被保护目标，也应按区域形式实现，避免后续重构。

### 6.6 关系特征

关系特征可以先作为规则计算，不一定直接拼进观测。后续图网络版本建议显式加入：

```text
edge(unit_i, target_j) = [
  distance_norm,
  bearing_sin,
  bearing_cos,
  in_range,
  in_sector,
  time_to_intercept_norm,
  success_prob,
  expected_benefit,
  cost_norm
]
```

### 6.7 全局特征

全局特征建议为：

```text
[
  current_step_norm,
  remaining_time_norm,
  alive_target_ratio,
  leaked_target_ratio,
  available_unit_ratio,
  total_ammo_ratio,
  total_damage_ratio,
  wave_id_norm
]
```

## 7. Agent 设计

### 7.1 推荐结论

当前项目下一阶段建议采用：

```text
v1.0: agent = 集中式防空编组指挥器
v1.5: agent_i = 防空作战单元
```

不要一开始就把 radar、missile、jammer 分别做成完全独立 agent。更稳妥的做法是：

```text
一个 agent 对应一个持续存在的防御节点；
节点内部可以拥有雷达、拦截弹、激光、干扰机等资源。
```

### 7.2 为什么不是“导弹”作为 agent

导弹通常是一次性消耗品，发射后不再参与后续长期决策。把导弹设为 agent 会导致：

- agent 生命周期过短；
- 难以学习长期资源管理；
- 不适合表达弹药存量、冷却和任务重分配；
- 与任务分配文献中的“持续决策主体”不一致。

### 7.3 为什么不是“目标”作为 agent

如果目标也学习，就变成攻防博弈环境。那是后续高级阶段可以做的 adversarial MARL，但不是当前阶段重点。

当前阶段应先固定目标行为模型，让防御方学习：

```text
识别威胁 -> 分配资源 -> 执行处置 -> 保护区域
```

## 8. 动作空间设计

### 8.1 当前 v0 动作

当前 v0 动作是：

```text
选择一个 defense_unit-target pair，或者 no-op
```

即每一步只执行一个资源分配动作：

```text
action = unit_index * num_targets + target_index
noop = num_units * num_targets
```

优点是简单，适合 DQN 入门。缺点是无法表达多个防御单元同时行动。

### 8.2 v1 推荐动作：联合资源分配

v1 建议采用联合动作：

```text
action = [a_1, a_2, ..., a_M]
```

其中 `a_i` 是第 i 个防御单元的动作：

```text
a_i ∈ {no-op, track target_j, engage target_j, jam target_j}
```

如果暂时不加入 track 和 jam，则可简化为：

```text
a_i ∈ {no-op, engage target_1, ..., engage target_N}
```

在 Gymnasium 中可表示为：

```python
spaces.MultiDiscrete([num_targets + 1] * num_defense_units)
```

这种设计最贴近 P01 中“每个 effector 选择一个目标”的形式。

### 8.3 加入动作类型后的动作

更完整的动作可以拆成两部分：

```text
a_i = (mode_i, target_i)
```

其中：

```text
mode_i ∈ {noop, search, track, engage, jam, hold}
target_i ∈ {target_1, ..., target_N, null}
```

可实现为：

```python
spaces.Dict({
    "mode": spaces.MultiDiscrete([num_modes] * num_units),
    "target": spaces.MultiDiscrete([num_targets + 1] * num_units),
})
```

但为了兼容 DQN，初期仍建议使用离散编码：

```text
encoded_action_i = mode_i * (num_targets + 1) + target_i
```

### 8.4 多智能体动作

在 PettingZoo 版本中，每个 agent 的动作空间为：

```text
A_i = {noop, track target_j, engage target_j, jam target_j}
```

环境在同一时刻收集所有 agent 的动作：

```text
joint_action = {agent_i: action_i}
```

然后统一处理冲突、命中、资源消耗和目标状态更新。

### 8.5 动作掩码

动作掩码必须保留，而且应成为环境核心接口。

非法动作包括：

- 目标不存在或已被拦截；
- 防御单元弹药不足；
- 防御单元仍在冷却；
- 目标超出射程；
- 目标不在作用扇区；
- 跟踪置信度低于发射阈值；
- 干扰设备对该目标类型无效；
- 通信延迟导致目标航迹过旧；
- 同一目标已被过量资源分配。

其中“物理上不可能”的动作应 mask 掉，“战术上不优”的动作可保留但通过奖励惩罚。

## 9. 奖励函数设计

### 9.1 总体原则

奖励函数应服务于一个核心目标：

```text
在有限资源和不确定观测下，最小化关键区域损伤，同时控制资源消耗和分配冲突。
```

不建议只用“命中 +1、未命中 -1”这种简单奖励。那会使 agent 过度关注眼前命中，而忽略高价值区域保护、资源保留和任务时序。

### 9.2 推荐总奖励

建议奖励函数写成：

```text
r_t =
  R_intercept
  + R_track
  + R_jam
  + R_protect
  - C_resource
  - C_time
  - P_leak
  - P_damage
  - P_invalid
  - P_conflict
  - P_overkill
```

### 9.3 拦截奖励

拦截奖励应与威胁价值相关：

```text
R_intercept = w_intercept * threat_j * zone_value_j
```

其中：

- `threat_j`：目标威胁等级；
- `zone_value_j`：该目标预计攻击区域的价值。

这样 agent 会优先拦截高威胁、高价值区域方向的目标。

### 9.4 突防与损伤惩罚

目标突防后，应根据其载荷和区域价值给惩罚：

```text
P_damage = w_damage * payload_j * zone_value_j
```

如果只用固定突防惩罚，不区分目标威胁和区域价值，agent 就无法学会威胁排序。

### 9.5 资源消耗成本

不同资源成本不同：

```text
C_resource =
  c_missile * missile_used
  + c_laser * energy_used
  + c_jammer * jammer_time
```

这可以避免 agent 对所有目标无差别开火。

### 9.6 时间惩罚与时序压力

每一步给小的时间惩罚：

```text
C_time = w_time
```

同时可以引入时间折扣收益：

```text
R_intercept_time = R_intercept * exp(-alpha * time_to_impact_j)
```

或者对临近突防目标提高紧迫度：

```text
urgency_j = 1 / (time_to_impact_j + epsilon)
```

### 9.7 跟踪奖励与不确定性惩罚

受 P02 和 P10 启发，后续加入探测/跟踪动作后，应引入：

```text
R_track = w_track * track_confidence_j
P_uncertainty = w_uncertainty * covariance_trace_j
P_aoi = w_aoi * aoi_j
```

这可以鼓励 agent 不只追求开火，也维护高质量航迹。

### 9.8 干扰奖励

干扰动作不一定直接摧毁目标，因此奖励应设计为效果型：

```text
R_jam = w_jam * threat_reduction_j
```

可观测效果包括：

- 目标速度下降；
- 航向偏离；
- 定位误差增加；
- 到达时间延长；
- 攻击成功概率降低。

同时干扰应消耗能量或占用设备：

```text
C_jam = c_jam * duration
```

### 9.9 冲突与过度分配惩罚

如果多个防御单元同时打同一个低价值目标，可能浪费资源。建议加入：

```text
P_conflict = w_conflict * duplicate_assignment_count
P_overkill = w_overkill * max(0, assigned_power_j - required_power_j)
```

但要注意：对高威胁目标允许多资源协同，不应一刀切惩罚所有重复分配。

### 9.10 非法动作惩罚

非法动作惩罚保留：

```text
P_invalid = w_invalid
```

但更推荐优先使用 action mask。非法动作惩罚主要用于算法不支持 mask 或调试环境时。

### 9.11 终局奖励

终局奖励建议包括：

```text
R_terminal =
  success_bonus
  - total_damage_penalty
  + resource_saving_bonus
```

其中 success 不应仅表示“所有目标都被拦截”，而应表示：

```text
关键区域损伤低于任务阈值。
```

这更符合防空任务的实际目标。

## 10. 状态转移与仿真机制

### 10.1 时间推进

环境采用离散时间：

```text
t = 0, 1, 2, ..., T
```

每一步执行：

1. 生成或更新来袭目标；
2. 更新传感器探测与航迹；
3. agent 选择动作；
4. 环境解析动作合法性；
5. 计算拦截、干扰、跟踪效果；
6. 更新目标状态；
7. 更新防御资源状态；
8. 计算奖励；
9. 判断终止。

### 10.2 目标运动

第一阶段使用简单运动模型：

```text
position_{t+1} = position_t + velocity_t * dt
```

目标可朝指定保护区域运动。后续加入：

- 机动规避；
- 编队协同；
- 低空绕飞；
- 目标再规划；
- 诱饵目标。

### 10.3 传感器模型

探测概率可由距离和目标类型决定：

```text
P_detect = f(distance, sensor_quality, target_rcs, clutter)
```

观测误差：

```text
observed_position = true_position + noise
```

航迹置信度随连续观测提高，随丢失和延迟下降。

### 10.4 拦截模型

拦截成功概率建议为：

```text
P_hit = f(
  distance,
  target_evasion,
  unit_base_success_prob,
  track_confidence,
  aspect_angle,
  resource_type_match
)
```

第一阶段可以简化为：

```text
P_hit = base_prob * range_factor * (1 - evasion)
```

### 10.5 干扰模型

干扰效果建议影响目标的一个或多个变量：

- 降低导航精度；
- 降低通信能力；
- 增加航向扰动；
- 增加目标到达时间；
- 降低攻击成功概率。

第一阶段可简化为：

```text
if jam_success:
    target.velocity *= slowdown_factor
    target.heading += random_disturbance
```

### 10.6 通信延迟与 AoI

多智能体阶段加入：

```text
aoi_i_j = 当前时刻 - agent_i 获得 target_j 信息的时刻
```

局部观测不直接使用最新全局状态，而使用延迟航迹：

```text
o_i(t) = track_j(t - delay_i_j)
```

这对应 P02 的延迟感知 Dec-POMDP 建模。

## 11. 实验接口设计

### 11.1 单智能体 Gymnasium 接口

第一阶段：

```python
class AirDefenseResourceAssignmentEnv(gym.Env):
    observation_space = spaces.Box(...)
    action_space = spaces.MultiDiscrete(...)
```

必须支持：

```python
reset(seed=seed)
step(action)
render()
action_mask()
```

### 11.2 多智能体 PettingZoo 接口

第二阶段：

```python
class AirDefenseParallelEnv(ParallelEnv):
    agents = ["unit_0", "unit_1", ..., "unit_M"]
```

每个 agent 返回：

```text
observation_i
reward_i
termination_i
truncation_i
info_i
```

训练 MAPPO 时，可使用共享团队奖励：

```text
reward_i = team_reward
```

也可以加入局部差分奖励：

```text
reward_i = team_reward + local_contribution_i
```

## 12. 推荐环境版本路线

### v1.0：集中式联合动作环境

目标：

```text
从当前 v0 的单资源-单目标动作升级为多资源联合动作。
```

包含：

- 多个防御单元；
- 多个目标；
- 多个保护区域；
- 联合动作 `MultiDiscrete`;
- action mask；
- 威胁-区域价值奖励；
- DQN/PPO 可训练。

### v1.1：目标类型与区域损伤

加入：

- 目标类型；
- 载荷；
- 目标攻击区域；
- 区域价值；
- 区域损伤累计。

### v1.2：探测与跟踪不确定性

加入：

- 探测概率；
- 观测噪声；
- track confidence；
- covariance；
- AoI；
- perception-consistent reward。

### v1.3：干扰资源

加入：

- jammer 单元；
- 干扰动作；
- 干扰成功概率；
- 干扰效果持续时间；
- 干扰能量消耗。

### v1.4：PettingZoo 多智能体环境

加入：

- 每个防御单元一个 agent；
- 局部观测；
- 联合动作；
- 团队共享奖励；
- MAPPO / MADDPG / QMIX 接口。

### v1.5：图观测与 GNN 策略

加入：

- resource-target graph；
- zone-target graph；
- edge features；
- GNN encoder；
- 异构资源任务分配实验。

## 13. 推荐 baseline

### 非学习 baseline

必须保留：

- random legal；
- nearest target first；
- highest threat first；
- greedy expected benefit；
- zone-weighted priority；
- time-to-impact priority。

建议新增：

- Hungarian assignment；
- CBBA / bundle auction；
- greedy expected damage reduction。

### 学习算法 baseline

单智能体：

- DQN；
- PPO；
- Maskable PPO；
- A2C。

多智能体：

- IPPO；
- MAPPO；
- MADDPG；
- QMIX；
- HAPPO / HATRPO。

## 14. 评价指标

建议形成统一实验表：

| 指标                         | 含义        |
| -------------------------- | --------- |
| `avg_reward`               | 平均回合奖励    |
| `success_rate`             | 任务成功率     |
| `total_damage`             | 总区域损伤     |
| `damage_weighted_by_zone`  | 按区域价值加权损伤 |
| `intercept_rate`           | 拦截率       |
| `leak_rate`                | 突防率       |
| `high_threat_leak_rate`    | 高威胁目标突防率  |
| `ammo_used`                | 弹药消耗      |
| `energy_used`              | 能量消耗      |
| `avg_shots`                | 平均开火次数    |
| `hit_rate_per_shot`        | 单次射击命中率   |
| `jam_success_rate`         | 干扰成功率     |
| `track_loss_rate`          | 航迹丢失率     |
| `avg_aoi`                  | 平均信息年龄    |
| `assignment_conflict_rate` | 分配冲突率     |
| `invalid_action_rate`      | 非法动作率     |
| `decision_time`            | 决策耗时      |

## 15. 与当前 v0 的关系

当前 `AirDefenseResourceAssignmentEnv v0` 已经具备：

- 多防御单元；
- 多来袭目标；
- 离散资源-目标动作；
- no-op；
- action mask；
- 目标运动；
- 拦截概率；
- 弹药和冷却；
- 突防检测；
- 奖励分项；
- 规则 baseline；
- DQN 训练入口。

下一阶段不应推翻 v0，而应在 v0 上演化：

```text
v0: 每步选择一个资源-目标动作
v1: 每步为多个资源同时分配动作
v1.2: 加入观测不确定性
v1.4: 扩展为多智能体
```

## 16. 推荐代码结构

建议新增：

```text
rein_learning/envs/air_defense_v1/
  __init__.py
  config.py
  entities.py
  centralized_env.py
  multi_agent_env.py
  observation_builder.py
  action_encoder.py
  reward.py
  masks.py
  scenario_generator.py

rein_learning/simulators/air_defense/
  target_motion.py
  sensor_model.py
  track_model.py
  effector_model.py
  jammer_model.py
  damage_model.py

rein_learning/trainers/
  air_defense_v1_ppo.py
  air_defense_v1_maskable_ppo.py
  air_defense_v1_mappo.py

scripts/
  train_air_defense_v1_ppo.py
  evaluate_air_defense_v1_baselines.py
  compare_air_defense_v1_methods.py

tests/
  test_air_defense_v1_env.py
  test_air_defense_v1_rewards.py
  test_air_defense_v1_masks.py
```

## 17. 先实现哪些内容

建议下一步不要立刻做全部 v1，而是先实现一个可验证的 v1.0：

```text
AirDefenseResourceAssignmentEnv v1.0
```

最小实现范围：

1. 多保护区域；
2. 目标绑定攻击区域；
3. 目标增加 `time_to_impact` 和 `payload`；
4. 防御单元保持 missile / laser 两类；
5. 动作从单个 pair 改为联合动作；
6. 奖励改为区域损伤最小化；
7. 保留 action mask；
8. 保留规则 baseline；
9. 训练 PPO / Maskable PPO；
10. 与 v0 DQN 和规则 baseline 比较。

## 18. 暂不建议加入的内容

以下内容重要，但不建议马上加入：

- 真实雷达方程；
- 复杂三维弹道；
- 完整电子战链路；
- 攻防双方同时学习；
- 大规模蜂群；
- 多层级指挥控制；
- 高保真毁伤模型。

原因是这些因素会显著增加环境复杂度，使算法结果难以解释。当前阶段应优先保证：

```text
环境机制清楚，状态动作可解释，奖励可调，baseline 可比较。
```

## 19. 阶段性结论

根据当前文献，最适合本项目的强化学习环境不是“单纯拦截仿真”，而是：

```text
面向反无人机防空编组的动态资源-目标分配 Dec-POMDP 环境。
```

第一阶段用集中式 agent 代表防空编组指挥器，统一控制各类防御效应器；第二阶段再把每个防空作战单元拆成独立 agent，接入 PettingZoo 和 MAPPO 等多智能体算法。

状态空间应从“目标位置 + 防御资源状态”扩展为：

```text
目标威胁、区域价值、到达时间、资源状态、任务关系、观测置信度和信息延迟。
```

动作空间应从“单资源打单目标”升级为：

```text
多防御单元联合选择目标和动作模式。
```

奖励函数应围绕：

```text
最小化关键区域损伤、优先处置高威胁目标、控制资源消耗、减少冲突和保持航迹质量。
```

这一路线既能继承当前 v0 的实现基础，又能逐步过渡到真正有科研价值的防空编组多智能体环境。

## 20. 文献-设计映射

| 文献    | 对环境设计的主要作用                           |
| ----- | ------------------------------------ |
| `P01` | 决策级拦截优先级、effector-target 动作、区域损伤奖励   |
| `P02` | Dec-POMDP、AoI、观测延迟、不确定性感知奖励          |
| `P03` | 周界防御、时空任务、部分可观测任务分配                  |
| `P04` | 异构资源、GNN、CTDE、冲突约束                   |
| `P05` | C-UAS 探测-跟踪-识别-处置功能链                 |
| `P06` | 雷达/通信探测、小型低空目标、蜂群挑战                  |
| `P07` | 干扰、自定位、SDR 切换、GPS disruption 抽象      |
| `P08` | 异构 UAV swarm、多目标追踪、探索-跟踪角色分化         |
| `P09` | 部分可观测、多任务 MARL、局部观测和联合奖励             |
| `P10` | 视觉跟踪状态、视场约束、跟踪奖励                     |
| `P11` | 动态任务分配、局部重规划、时间折扣收益                  |
| `P12` | MADDPG、CTDE、多智能体非平稳性                 |
| `P13` | MAPPO、共享奖励 Dec-POMDP、合作多智能体 baseline |
| `P14` | PettingZoo API、AEC/Parallel 多智能体接口   |
| `P15` | 大规模多智能体平台、可配置状态/动作/奖励与渲染             |
