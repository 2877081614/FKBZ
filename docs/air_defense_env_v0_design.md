# AirDefenseResourceAssignmentEnv v0 设计文档

更新日期：2026-07-08

## 1. 文档目的

本文档用于定义项目中第一个面向科研实验的防空资源分配强化学习环境：

```text
AirDefenseResourceAssignmentEnv v0
```

该环境不是最终高保真防空仿真系统，而是一个可实现、可训练、可对比、可逐步扩展的研究原型。它的核心目标是把当前项目从 GridWorld 教学环境推进到真正服务论文研究的防空动态资源分配环境。

本文档将明确：

- 研究问题
- 基本假设
- MDP/POMDP 建模
- 状态空间
- 动作空间
- 奖励函数
- 状态转移与仿真规则
- 约束条件
- baseline 与评价指标
- 代码实现规划
- 后续扩展方向

## 2. 研究问题定义

### 2.1 问题背景

在防空/反无人机作战场景中，防御方通常拥有数量有限、能力不同、位置不同的防御资源，例如导弹、激光、电子干扰设备或火力单元。来袭目标可能具有不同速度、距离、航向、威胁等级和突防能力。

防御系统需要在动态变化的态势中决定：

```text
用哪个防御资源，在什么时刻，拦截哪个来袭目标。
```

这类问题本质上属于动态资源-目标分配问题，也可视为动态武器-目标分配问题：

```text
Dynamic Resource-Target Assignment
Dynamic Weapon-Target Assignment, DWTA
```

### 2.2 本项目中的核心问题

本环境关注一个简化但具有科研价值的问题：

> 在多个来袭目标持续接近保护区域的过程中，防御方如何动态分配有限防御资源，使高威胁目标尽可能被拦截，同时减少资源浪费和目标突防损失？

强化学习智能体需要学习一个策略：

```text
pi(a | s)
```

使长期累计收益最大：

```text
max E[sum_t gamma^t r_t]
```

### 2.3 v0 环境边界

v0 版本只做最小可研究环境，暂不引入复杂高保真物理模型。

v0 包含：

- 离散时间步
- 多个防御单元
- 多个来袭目标
- 有限弹药/资源容量
- 目标朝保护区域运动
- 拦截概率由距离、资源类型、目标属性共同决定
- 动作掩码或非法动作惩罚
- 任务成功/失败与资源消耗评价

v0 暂不包含：

- 三维复杂弹道
- 雷达探测链路细节
- 复杂电子战传播模型
- 真实地理地形
- 指挥通信延迟
- 高保真导弹制导律

这些内容可以放到后续 v1/v2 中逐步扩展。

## 3. 环境角色与对象

### 3.1 防御方

防御方拥有若干防御单元：

```text
defense_unit_i, i = 1, 2, ..., N
```

每个防御单元具有：

- 位置
- 资源类型
- 剩余弹药或可用次数
- 射程
- 冷却时间
- 基础命中能力
- 对不同目标类型的适配能力

v0 可以先设置为：

```text
N = 2 或 3
```

防御资源类型可以先简化为两类：

```text
missile：高命中率，高成本，有限数量
laser：低成本，受距离影响明显，可有冷却时间
```

如果实现复杂度需要进一步降低，v0 初始版本可先只使用同构防御单元，后续再扩展为异构资源。

### 3.2 来袭目标

来袭目标表示为：

```text
target_j, j = 1, 2, ..., M
```

每个目标具有：

- 位置
- 速度
- 航向
- 距离保护区域的剩余距离
- 威胁等级
- 生命状态：存活、已拦截、已突防
- 目标类型，可选

v0 可以先设置为：

```text
M = 3 到 8
```

目标数量可以固定，也可以在后续版本中随机生成。

### 3.3 保护对象

保护对象可以抽象为二维平面中的一个固定点或区域：

```text
protected_asset
```

例如：

```text
protected_asset_position = (0, 0)
```

来袭目标朝保护对象运动。若目标进入保护半径，则视为突防成功，防御方受到惩罚。

## 4. MDP/POMDP 建模

### 4.1 MDP 五元组

v0 环境建模为马尔可夫决策过程：

```text
MDP = (S, A, P, R, gamma)
```

其中：

- `S`：态势状态空间
- `A`：资源分配动作空间
- `P`：目标运动、拦截结果、资源状态变化构成的转移概率
- `R`：综合任务收益函数
- `gamma`：折扣因子

### 4.2 是否为 POMDP

真实防空任务更接近 POMDP，因为智能体无法完全观测目标真实状态，例如目标意图、真实速度、电子干扰状态等。

但 v0 建议先按 MDP 实现：

```text
智能体可以观测所有目标和防御资源的完整状态。
```

原因：

- 降低初始实现难度
- 便于验证算法流程
- 便于构建可解释 baseline
- 后续可通过观测噪声、遮蔽、延迟扩展为 POMDP

## 5. 状态空间设计

### 5.1 总体状态

环境状态由三部分组成：

```text
s_t = [防御资源状态, 目标状态, 全局状态]
```

### 5.2 防御资源状态

每个防御单元的状态向量可以定义为：

```text
d_i = [
    x_i,
    y_i,
    resource_type_i,
    ammo_i,
    cooldown_i,
    max_range_i,
    base_hit_prob_i
]
```

说明：

- `x_i, y_i`：防御单元位置
- `resource_type_i`：资源类型编号
- `ammo_i`：剩余弹药或可用次数
- `cooldown_i`：距离下次可用还需等待的时间步
- `max_range_i`：最大有效射程
- `base_hit_prob_i`：基础命中概率

### 5.3 目标状态

每个目标状态向量可以定义为：

```text
u_j = [
    x_j,
    y_j,
    vx_j,
    vy_j,
    distance_to_asset_j,
    threat_j,
    alive_j
]
```

说明：

- `x_j, y_j`：目标位置
- `vx_j, vy_j`：目标速度
- `distance_to_asset_j`：目标到保护对象的距离
- `threat_j`：威胁等级
- `alive_j`：目标是否仍需处理

### 5.4 全局状态

全局状态可以包括：

```text
g_t = [
    current_step,
    remaining_steps,
    number_alive_targets,
    number_available_defense_units
]
```

### 5.5 v0 推荐观测形式

为了尽快实现并接入 DQN/PPO，v0 推荐使用定长连续向量：

```text
observation_space = Box(low=-inf, high=inf, shape=(obs_dim,))
```

其中：

```text
obs_dim = N * dim(defense_unit) + M * dim(target) + dim(global_state)
```

如果目标数量固定，例如：

```text
N = 3
M = 5
```

则状态维度固定，便于使用 MLP。

### 5.6 状态归一化

建议所有连续量都归一化：

- 位置除以战场范围
- 速度除以最大速度
- 距离除以最大距离
- 弹药除以最大弹药量
- 冷却时间除以最大冷却时间
- 威胁等级归一化到 `[0, 1]`

这能显著提高神经网络训练稳定性。

## 6. 动作空间设计

### 6.1 v0 动作含义

每一步，智能体选择一个资源分配动作：

```text
a_t = (defense_unit_id, target_id)
```

表示：

```text
使用某个防御单元拦截某个目标。
```

同时需要允许智能体选择不发射：

```text
no-op
```

表示当前时间步不分配任何资源。

### 6.2 离散动作编码

如果有 `N` 个防御单元、`M` 个目标，则动作数量为：

```text
N * M + 1
```

其中：

```text
0 到 N*M-1：资源-目标分配动作
N*M：no-op
```

动作解码：

```text
defense_unit_id = action // M
target_id = action % M
```

当：

```text
action == N * M
```

表示 no-op。

### 6.3 动作空间

v0 推荐使用：

```text
action_space = Discrete(N * M + 1)
```

这样可直接接入：

- DQN
- PPO
- A2C
- Stable-Baselines3
- 后续 action mask 版本

### 6.4 非法动作

非法动作包括：

- 防御单元弹药为 0
- 防御单元仍处于冷却中
- 目标已经被拦截
- 目标已经突防
- 目标超出资源射程
- 同一时间步重复分配同一资源，若后续支持联合动作

v0 可采用两种处理方式。

第一阶段建议使用非法动作惩罚：

```text
非法动作 -> reward += invalid_action_penalty，状态不执行拦截
```

第二阶段再引入动作掩码：

```text
action_mask[action] = 0 或 1
```

动作掩码更适合后续做约束强化学习或工程优化，但第一版不必过早复杂化。

## 7. 状态转移与仿真规则

### 7.1 时间推进

环境采用离散时间步：

```text
t = 0, 1, 2, ..., T
```

每次 `step(action)` 执行以下流程：

```text
1. 解码动作
2. 判断动作是否合法
3. 若合法，计算拦截结果
4. 更新防御资源状态
5. 更新目标运动状态
6. 判断目标是否突防
7. 计算奖励
8. 判断 episode 是否结束
9. 返回 observation, reward, terminated, truncated, info
```

### 7.2 目标运动

v0 使用简单二维匀速运动：

```text
pos_j(t+1) = pos_j(t) + velocity_j * dt
```

目标朝保护对象方向运动：

```text
velocity_j = speed_j * unit_vector(asset_position - target_position)
```

后续可加入：

- 随机机动
- 多航路点
- 规避策略
- 编队运动
- 智能攻击策略

### 7.3 拦截概率

v0 使用概率模型，而不是复杂物理杀伤模型。

定义防御单元 `i` 对目标 `j` 的命中概率：

```text
p_hit(i, j) = base_hit_prob_i * range_factor(i, j) * target_factor(j)
```

其中距离因子可定义为：

```text
range_factor(i, j) = max(0, 1 - distance(i, j) / max_range_i)
```

目标因子可简化为：

```text
target_factor(j) = 1 - evasion_j
```

v0 若不建模规避能力，可令：

```text
target_factor(j) = 1
```

最终：

```text
hit ~ Bernoulli(p_hit)
```

如果命中：

```text
target_j.alive = False
target_j.status = intercepted
```

如果未命中：

```text
目标继续运动
```

### 7.4 资源更新

合法发射后：

```text
ammo_i -= 1
cooldown_i = cooldown_after_fire_i
```

每个时间步结束时：

```text
cooldown_i = max(0, cooldown_i - 1)
```

### 7.5 突防判定

当目标到保护对象的距离小于保护半径：

```text
distance_to_asset_j <= asset_radius
```

则目标突防：

```text
target_j.status = leaked
target_j.alive = False
```

突防目标会产生较大负奖励。

## 8. 奖励函数设计

### 8.1 奖励设计目标

奖励函数应同时鼓励：

- 拦截目标
- 优先拦截高威胁目标
- 防止目标突防
- 减少弹药浪费
- 避免非法动作
- 在合适时机发射，而不是盲目发射

### 8.2 v0 奖励函数

建议 v0 奖励定义为：

```text
r_t =
    R_intercept
  + R_leak
  + R_cost
  + R_invalid
  + R_time
  + R_terminal
```

### 8.3 拦截奖励

若目标 `j` 被成功拦截：

```text
R_intercept = w_intercept * threat_j
```

推荐初值：

```text
w_intercept = +10
```

高威胁目标被拦截时获得更高奖励。

### 8.4 突防惩罚

若目标 `j` 突防：

```text
R_leak = -w_leak * threat_j
```

推荐初值：

```text
w_leak = 20
```

突防惩罚应大于拦截奖励，以体现保护对象优先级。

### 8.5 资源消耗惩罚

每次合法发射：

```text
R_cost = -cost_i
```

例如：

```text
missile_cost = 2
laser_cost = 0.5
```

这能避免智能体无脑发射。

### 8.6 非法动作惩罚

若动作非法：

```text
R_invalid = -5
```

后续引入动作掩码后，该项可减弱或取消。

### 8.7 时间惩罚

每个时间步给一个小惩罚：

```text
R_time = -0.1
```

作用是鼓励尽快处理威胁。

### 8.8 终局奖励

episode 结束时：

若所有目标被拦截：

```text
R_terminal = +20
```

若保护对象被高威胁目标突防：

```text
R_terminal = -20
```

终局奖励不宜过大，否则可能掩盖中间决策质量。

### 8.9 奖励分项记录

`info` 中应返回奖励分项：

```python
info = {
    "reward_intercept": ...,
    "reward_leak": ...,
    "reward_cost": ...,
    "reward_invalid": ...,
    "reward_time": ...,
    "reward_terminal": ...,
}
```

这对调试奖励函数非常关键。

## 9. 终止条件

### 9.1 terminated

自然终止条件：

```text
所有目标都已被处理：intercepted 或 leaked
```

或者：

```text
保护对象被突破数量达到失败阈值
```

例如：

```text
leaked_targets >= max_allowed_leaks
```

### 9.2 truncated

时间截断条件：

```text
current_step >= max_steps
```

v0 推荐：

```text
max_steps = 50 或 100
```

## 10. 评价指标

科研实验中不能只看累计奖励，需要多维指标。

### 10.1 任务效果指标

- 拦截率
- 高威胁目标拦截率
- 突防率
- 加权突防损失
- 任务成功率

### 10.2 资源效率指标

- 平均弹药消耗
- 单目标平均消耗
- 无效发射次数
- 非法动作次数
- 资源利用率

### 10.3 学习性能指标

- episode reward 曲线
- 收敛速度
- 策略稳定性
- 不同随机种子下的均值和方差

### 10.4 鲁棒性指标

后续版本可加入：

- 不同目标数量下的性能
- 不同威胁分布下的性能
- 不同资源数量下的性能
- 不同目标速度下的性能
- 不同拦截概率噪声下的性能

## 11. Baseline 设计

### 11.1 非学习 baseline

至少应实现以下规则策略：

#### Random Policy

随机选择合法动作。

用途：

```text
提供最低性能参考。
```

#### Nearest Target First

优先拦截距离保护对象最近的目标。

用途：

```text
模拟紧急防御策略。
```

#### Highest Threat First

优先拦截威胁等级最高的目标。

用途：

```text
模拟威胁驱动分配策略。
```

#### Greedy Expected Benefit

选择期望收益最高的资源-目标组合：

```text
score(i, j) = p_hit(i, j) * threat_j - cost_i
```

用途：

```text
作为强规则 baseline。
```

当前实现状态：

```text
已实现 random legal、nearest target first、highest threat first、greedy expected benefit。
已实现 episode 级和多 episode 聚合评价指标。
```

### 11.2 强化学习 baseline

初期建议：

- DQN
- PPO
- A2C

后续加入：

- MAPPO
- MADDPG
- QMIX
- HAPPO/HATRPO

v0 不建议一开始就实现所有 MARL 算法，应先用单智能体集中式决策验证环境是否合理。

## 12. 与当前代码结构的对应关系

### 12.1 推荐新增目录

```text
rein_learning/envs/air_defense/
  __init__.py
  resource_assignment_env.py
  config.py

rein_learning/simulators/
  geometry.py
  target_motion.py
  intercept_model.py
  resource_model.py

rein_learning/baselines/
  random_policy.py
  greedy_assignment.py

tests/
  test_air_defense_env.py
```

### 12.2 环境类命名

建议环境类命名为：

```python
AirDefenseResourceAssignmentEnv
```

配置类命名为：

```python
AirDefenseEnvConfig
```

### 12.3 Gymnasium 接口

环境应实现：

```python
obs, info = env.reset(seed=None)
obs, reward, terminated, truncated, info = env.step(action)
env.render()
env.close()
```

### 12.4 observation_space 和 action_space

v0 建议：

```python
observation_space = gymnasium.spaces.Box(...)
action_space = gymnasium.spaces.Discrete(num_defense_units * num_targets + 1)
```

### 12.5 info 字段

建议返回：

```python
info = {
    "num_intercepted": ...,
    "num_leaked": ...,
    "num_alive": ...,
    "ammo_remaining": ...,
    "invalid_action": ...,
    "hit": ...,
    "selected_defense_unit": ...,
    "selected_target": ...,
    "reward_breakdown": {...},
}
```

## 13. v0 默认参数建议

### 13.1 场景规模

```text
num_defense_units = 3
num_targets = 5
max_steps = 50
map_size = 100.0
asset_position = (0.0, 0.0)
asset_radius = 5.0
```

### 13.2 防御资源

```text
resource 0:
  type = missile
  position = (-10, 0)
  ammo = 3
  max_range = 80
  base_hit_prob = 0.85
  cost = 2.0
  cooldown = 1

resource 1:
  type = missile
  position = (10, 0)
  ammo = 3
  max_range = 80
  base_hit_prob = 0.85
  cost = 2.0
  cooldown = 1

resource 2:
  type = laser
  position = (0, 10)
  ammo = 10
  max_range = 50
  base_hit_prob = 0.65
  cost = 0.5
  cooldown = 0
```

### 13.3 目标生成

目标从地图边缘随机生成，朝保护对象运动：

```text
distance_to_asset: 60 到 100
speed: 1 到 3
threat: 0.5 到 1.0
```

为便于复现实验，必须支持随机种子：

```python
env.reset(seed=seed)
```

## 14. 实现优先级

### P0：必须完成

- 固定数量防御单元和目标
- Gymnasium 环境接口
- 目标匀速运动
- 离散资源-目标动作
- 拦截概率模型
- 弹药消耗
- 目标突防
- 奖励分项
- 单元测试

### P1：建议完成

- 动作掩码：已完成
- 规则 baseline：已完成
- 场景随机化
- 简单渲染
- 与 DQN/PPO 训练脚本对接

### P2：后续扩展

- 异构资源更复杂建模
- 多智能体 PettingZoo 接口
- 注意力或 GNN 状态编码
- 目标规避策略
- 部分可观测与噪声
- 课程学习
- 多场景泛化测试

## 15. 预期科研创新方向

v0 本身主要是环境与问题建模，不一定构成最终创新点。后续可围绕以下方向发展。

### 15.1 动作掩码与约束强化学习

防空分配存在大量非法动作，例如弹药不足、目标超射程、资源冷却中。动作掩码能显著降低探索空间。

可能创新点：

```text
面向异构防御资源分配的约束动作掩码强化学习。
```

### 15.2 异构资源协同

不同防御资源具有不同成本、射程、命中概率和冷却机制。

可能创新点：

```text
考虑异构防御资源能力差异的动态目标分配方法。
```

### 15.3 注意力或图神经网络编码

防御单元与目标之间天然形成二部图：

```text
defense units <-> targets
```

可以用注意力或 GNN 建模资源-目标关系。

可能创新点：

```text
基于资源-目标关系图编码的防空动态分配策略。
```

### 15.4 多智能体协同

每个防御单元可视为一个智能体，学习协同分配策略。

可能创新点：

```text
面向防空编组的多智能体协同资源分配。
```

## 16. v0 实验计划

### 16.1 实验一：环境合理性验证

目标：

```text
验证环境奖励、转移、终止条件是否符合直觉。
```

对比：

- random policy
- nearest target first
- highest threat first
- greedy expected benefit

输出：

- 拦截率
- 突防率
- 平均资源消耗
- 平均奖励

### 16.2 实验二：DQN/PPO 可学习性验证

目标：

```text
验证强化学习算法是否能超过随机策略和简单规则策略。
```

对比：

- random
- greedy
- DQN
- PPO

输出：

- reward 曲线
- 成功率曲线
- 测试集平均性能

### 16.3 实验三：场景规模敏感性

目标：

```text
测试目标数量、防御资源数量变化时策略性能。
```

设置：

```text
num_targets = 3, 5, 8
num_defense_units = 2, 3, 4
```

输出：

- 规模变化下的性能下降曲线
- 策略泛化能力

## 17. 风险与注意事项

### 17.1 奖励设计风险

如果资源消耗惩罚过大，智能体可能不发射。

如果拦截奖励过大，智能体可能无脑发射。

如果突防惩罚过大，训练可能不稳定。

因此必须记录奖励分项并做参数扫描。

### 17.2 动作空间风险

当目标和资源数量增加时：

```text
action_dim = N * M + 1
```

动作空间会快速变大。

后续需要：

- 动作掩码
- 分层动作
- 图匹配动作
- 多智能体分散决策

### 17.3 环境过拟合风险

如果场景固定，算法可能只记住固定模式。

应逐步加入：

- 随机目标初始位置
- 随机速度
- 随机威胁等级
- 随机资源配置

### 17.4 过早复杂化风险

v0 的目标是完成可训练闭环，而不是一次性模拟真实防空系统。

建议遵循：

```text
先可运行，再可信，再复杂，再创新。
```

## 18. 下一步任务清单

建议按以下顺序实施。

1. 创建 `rein_learning/envs/air_defense/` 目录。
2. 创建 `AirDefenseEnvConfig` 配置类。
3. 实现目标、防御资源的数据结构。
4. 实现几何距离与目标运动函数。
5. 实现拦截概率函数。
6. 实现 `AirDefenseResourceAssignmentEnv.reset()`。
7. 实现 `AirDefenseResourceAssignmentEnv.step()`。
8. 编写环境单元测试。
9. 实现 random 和 greedy baseline。
10. 输出第一组环境合理性实验结果。
11. 编写 DQN/PPO 训练脚本。

当前进度：

```text
第 1 到 10 项已完成。
下一步进入第 11 项：DQN/PPO 训练脚本。
```

## 19. 阶段性结论

`AirDefenseResourceAssignmentEnv v0` 应作为当前项目从强化学习工程练习走向防空科研问题建模的关键桥梁。

它的定位是：

```text
一个简化、可解释、可扩展、可用于算法对比的防空动态资源分配环境。
```

只要 v0 能稳定支持规则策略、DQN、PPO 的对比实验，项目就可以进入真正的论文实验阶段。
