# 无冲突联合动作编码与动态掩码

更新时间：2026-07-17  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`  
方法名：`conflict_free_maskable_ppo`

## 1. 设计目的

原始环境使用 `MultiDiscrete([6, 6, 6])`。三个防御单元分别选择五个目标之一或 `no-op`，共包含 `6^3=216` 个联合动作。逐单元动作掩码能排除弹药耗尽、冷却、超射程和失效目标，但不能阻止多个单元同时选择同一目标。

本方法只改变联合动作的表示与合法性约束，不改变环境状态转移、观测、奖励、目标生成、命中概率、MLP 网络和 PPO 超参数。

## 2. 动作空间

只保留所有非 `no-op` 目标互不重复的联合动作。对于三个防御单元和五个目标：

```text
N = sum(C(3, k) * P(5, k), k=0..3)
  = 1 + 15 + 60 + 60
  = 136
```

包装后的空间为 `Discrete(136)`。`ConflictFreeJointActionCodec` 按固定字典序建立一一映射：

```text
离散索引 <-> 长度为 3 的原始联合动作
```

编码表在构造后设为只读；解码返回副本，避免模型保存、加载或评估过程中发生索引漂移。

## 3. 动态联合掩码

`ConflictFreeJointActionWrapper` 先读取基础环境的逐单元布尔掩码。对候选联合动作 `a=(a_1,a_2,a_3)`：

```text
joint_mask(a) = all(base_mask[i, a_i] for i in range(3))
```

候选集合已静态排除重复目标，运行时掩码再排除资源不可用、目标失效和射程不满足等状态相关动作。全 `no-op` 始终保留，因此不会出现空合法动作集。

包装器不做动作修复。策略采样的索引被唯一解码后原样传给基础环境，从而保持 PPO 中“计算概率的动作”和“环境执行的动作”一致。

## 4. 软件接口

主要实现：

```text
rein_learning/envs/air_defense_v1/wrappers/conflict_free_joint_action.py
rein_learning/trainers/air_defense_v1_ppo.py
rein_learning/experiments/air_defense_v1_benchmark.py
```

主要对象和入口：

```python
ConflictFreeJointActionCodec
ConflictFreeJointActionWrapper
train_conflict_free_maskable_ppo
```

`info` 同时保留：

- `encoded_joint_action`：策略输出的离散索引；
- `decoded_joint_action`：传给基础环境的联合动作；
- `joint_action`：基础环境实际执行并记录的联合动作。

统一实验的配置 schema 升级为版本 4，并为每种方法留档动作空间签名。原始方法记录 `MultiDiscrete([6,6,6])`，新方法记录 `Discrete(136)`；两类模型不能在不匹配的环境上互相加载。

## 5. 运行方式

最小训练调用：

```python
from rein_learning.trainers.air_defense_v1_ppo import (
    train_conflict_free_maskable_ppo,
)

model = train_conflict_free_maskable_ppo()
```

统一对比调用示例：

```powershell
conda run -n rein-learning python scripts\compare_air_defense_v1_methods.py `
  --train-scenario medium `
  --eval-scenarios medium time_pressure heterogeneity_pressure `
  --methods maskable_ppo conflict_free_maskable_ppo `
  --seeds 0 1 2 --timesteps 30000
```

## 6. 已验证契约

- 默认配置恰好生成 136 个动作，编码和解码构成双射；
- 所有非 `no-op` 目标在同一联合动作内互不重复；
- 动态掩码随资源耗尽和目标失效同步变化；
- 包装前后观测空间、奖励和终止语义一致；
- 训练、保存、加载和带掩码评估闭环可执行；
- 非法动作、分配冲突和过度分配均为 0；
- 原始 `MultiDiscrete` 模型与新 `Discrete` 模型存在明确的空间兼容性校验。

筛选结果与后续决策见 [air_defense_v1_task8_screening.md](../experiments/air_defense_v1_task8_screening.md)。
