# 掩码条件动作价值 Critic

更新时间：2026-07-19  
对应阶段：任务十四  
实现状态：已完成离线原型与正式门控实验

## 1. 方法定位

任务十四实现的 `MaskedActionQCritic` 是一个非图结构关系 MLP，用于估计固定自回归前缀下当前单元候选动作的价值：

```text
Q^pi(s, h_i, a_i)
= E[G_t | s, earlier actions=h_i, current action=a_i,
         later actions~pi(.|s,h_i,a_i), future actions~pi]
```

它不是 PPO 的在线 Critic，也没有更新 Actor。其目的仅是验证动作条件价值能否比现有 `V(s)` 更好地区分 `no-op`、交战动作和不同目标，为后续反事实信用分配提供门控证据。

## 2. 输入结构

完整模型连接以下特征：

```text
全局 observation
+ 当前单元 one-hot
+ 候选动作 one-hot
+ 当前单元实体特征
+ 候选目标实体特征或 no-op 零向量
+ 单元-目标相对位置、距离和 no-op 标记
+ 前序目标占用向量
+ 当前条件合法动作掩码
-> MLP(256, 128, Tanh)
-> scalar Q
```

模型只接受合法候选动作。非法动作在前向计算前直接触发异常，避免把无定义的反事实动作混入监督集。

## 3. 反事实标签

标签由冻结的任务十二因子化策略生成。对同一个 `state + unit`：

1. 固定早于当前单元的动作；
2. 当前单元分别替换为 `no-op` 和每个条件合法目标；
3. 根据修改后的前缀重新采样后续单元动作；
4. 后续环境时间步继续使用同一冻结策略；
5. 不同候选使用相同环境种子和策略采样种子。

候选动作回报采用折扣回报均值。排序不确定性使用同一 rollout 下的配对回报差标准误，而不是把两个高方差分支的独立标准误相加。

## 4. 数据与训练约束

- 来源策略：任务十二 `factorized_engagement_ar_ppo_order_012`，seed 8/10；
- 场景：`medium`、`time_pressure`、`heterogeneity_pressure`；
- 数据按 `state_id` 分组，以 60%/20%/20% 划分训练、验证和测试；
- 同一状态的全部单元和候选动作只属于一个 split；
- 主模型使用训练种子 14/15/16；
- 训练采用 MSE、Adam、早停，不根据测试结果选择模型；
- `V(s)` 和一步奖励只作为冻结基线。

## 5. 诊断指标

- Q 回归：MAE、RMSE、bias；
- 动作判别：候选 pairwise ranking、目标间 ranking、top-1；
- 局部信用：策略加权 engage 相对 no-op 的 advantage 符号；
- 场景稳定性：三个核心场景分别统计排序准确率；
- 效率：Q-Critic 全候选推理与 Monte Carlo 标签生成耗时；
- 消融：`no_prefix`、`no_mask`、`observation_action_only`。

只有真实回报差超过配对 `1.96 SE` 的比较才进入排序或符号分母。有效比较不足 30 时，对应门控按“证据不足”处理。

## 6. 实现接口

```python
from rein_learning.models import MaskedActionQCritic, MaskedActionQCriticConfig

critic = MaskedActionQCritic(layout, MaskedActionQCriticConfig())
q_values = critic(
    observations,
    unit_indices,
    candidate_actions,
    prefix_occupancy,
    legal_action_masks,
)
```

自回归分布新增 `sample_with_fixed_actions`：`-1` 表示按策略采样，非负值表示固定该单元动作。固定动作仍需通过修改后前缀对应的动态合法性检查。

## 7. 当前边界

正式实验表明该 MLP 能显著降低绝对 Q 误差，但不能可靠恢复候选动作排序。它目前只能作为离线诊断原型，不能作为已经验证的反事实 advantage 估计器接入 PPO。任务十四也不构成 GNN 进入依据，因为有效动作差异样本仍少，且纯回归目标可能主要学习状态价值而不是动作间差值。

实现与结果：

```text
rein_learning/models/masked_action_q_critic.py
rein_learning/common/q_critic_diagnostics.py
scripts/run_air_defense_v1_task14_q_critic.py
docs/experiments/air_defense_v1_task14_q_critic.md
results/air_defense_v1/task14_q_critic/
```

## 8. 组内动作差异监督修订

任务十四修订保持模型结构不变，将训练目标扩展为：

```text
L = L_absolute + L_group_centered + 0.5 * L_pairwise
```

正式独立测试中，该目标把总体排序相对纯回归平均提高 `0.167`，且 MAE 比值为 `0.993`。这说明纯绝对回归确实会优先拟合状态共同价值，组内中心化和配对差值可以恢复部分动作排序。

不过 engage/no-op 符号仍为 `0.545`，完整门控为 `0/3`。当前实现继续作为离线原型，不接入 PPO；下一步需要把 engagement 和 conditional target 两类价值显式拆分后分别验证。
