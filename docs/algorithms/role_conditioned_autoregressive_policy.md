# 角色条件化自回归关系策略

更新时间：2026-07-17  
适用环境：`AirDefenseResourceAssignmentEnv v1.0`  
对应阶段：任务十一

## 1. 设计目标

任务十发现固定顺序会改变哪个防御单元进入低交战状态，但单纯换序无法同时保持资源成本和高威胁保护。任务十一将 Actor 的独立位置输出替换为共享资源-目标关系评分器，检验单元塌缩是否来自缺少角色关系归纳偏置。

环境、观察值、动作语义、奖励、Critic、PPO 超参数和自回归目标掩码均保持不变。

## 2. 观察布局

环境仍输出一维观察。模型侧 `AirDefenseV1ObservationLayout` 根据 `Box` 观察空间和 `MultiDiscrete` 动作空间恢复：

```text
zones:   [batch, 2, 7]
targets: [batch, 5, 15]
units:   [batch, 3, 15]
global:  [batch, 8]
```

布局推断会校验观察总维度、单元数、目标数和各实体特征宽度。结构化张量重新展平后必须与原观察逐元素一致，因此没有改变环境接口或引入隐藏状态。

## 3. 共享关系 Actor

所有区域、目标和防御单元分别使用共享编码器：

```text
z_k = f_zone(zone_k)
t_j = f_target(target_j)
u_i = f_unit(unit_i)
g   = f_global(global_features)
```

区域、存活目标和单元 embedding 分别池化后形成置换不变上下文。每个资源-目标动作使用同一个 pair scorer：

```text
pair(i,j) = [u_i, t_j, context, dx, dy, distance, range_margin]
logit(i,j) = f_pair(pair(i,j))
```

每个单元的 no-op 也使用同一个评分器：

```text
noop(i) = f_noop(u_i, context, mean(legal_target_embeddings_i))
```

模型不使用 unit-index embedding。资源类型、射程、命中概率、成本、弹药和冷却均来自现有 unit features。动作 mask 仍由环境和自回归前缀控制。

## 4. 置换性质

如果交换单元特征和对应 action-mask block，输出 logits 会按相同单元排列交换；如果交换目标特征和对应目标动作列，pair logits 会按相同目标排列交换，no-op 不变。

该性质由独立测试覆盖，但自回归动作生成仍需指定顺序。任务十一注册：

```text
role_conditioned_ar_ppo_order_012  # 预注册主方法
role_conditioned_ar_ppo_order_120  # 顺序诊断
role_conditioned_ar_ppo_order_201  # 顺序诊断
```

环境接收的联合动作始终按原始 unit index 排列。

## 5. 参数量控制

| 模块 | 任务十 | 任务十一 | 差异 |
| --- | ---: | ---: | ---: |
| Actor | 37,138 | 34,946 | -5.90% |
| Critic | 34,945 | 34,945 | 0 |
| 总参数 | 72,083 | 69,891 | -3.04% |

Actor 差异处于冻结的 ±15% 容量范围内，Critic 完全不变。统一实验 schema 7 为每个模型输出 `model_parameter_counts.json`。

## 6. 模型签名

模型保存：

```text
type = role_conditioned_autoregressive_conflict_free
unit_order
conditional_target_mask
observation_layout
entity_embedding_dim = 32
context_dim = 96
relation_hidden_dim = 64
unit_index_embedding = false
shared_noop_head = true
```

布局、顺序或策略类型不一致时不能静默加载。

## 7. 主要实现

```text
rein_learning/models/air_defense_observation_layout.py
rein_learning/models/air_defense_role_conditioned_action_head.py
rein_learning/algorithms/policy_gradient/role_conditioned_autoregressive_ppo.py
rein_learning/trainers/air_defense_v1_ppo.py
rein_learning/experiments/air_defense_v1_benchmark.py
tests/test_role_conditioned_action_head.py
```

## 8. 工程验收

- 观察拆分和重新展平一致；
- 单元和目标置换等变性通过；
- 资源角色特征能够改变关系 logits；
- 三顺序动作逆映射正确；
- 非默认顺序模型可保存、加载和评估；
- 三种方法非法动作、冲突和过度分配均为 0；
- 参数量留档完整；
- `tests/` 全量 `140 passed`。

## 9. 实验结论

关系头在实际选择目标时具有较高匹配效率，但未消除种子级 no-op 塌缩；主方法的异质高威胁泄漏中 94.8% 属于 `unassigned`。三顺序性能跨度和决策耗时也未通过门槛，因此不运行 100k。详见[任务十一实验报告](../experiments/air_defense_v1_task11_role_conditioned_screening.md)。
