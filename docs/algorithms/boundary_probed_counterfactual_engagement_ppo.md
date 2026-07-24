# BPCE-PPO：边界探测式反事实交战辅助 PPO

更新时间：2026-07-23  
文档性质：候选创新设计、实验预注册与首轮验证记录  
当前状态：v0已实现；10k机制门控失败，保留为部分正证据与修订候选  
英文名称：Boundary-Probed Counterfactual Engagement Auxiliary PPO  
候选简称：BPCE-PPO

## 1. 研究定位

本候选沿用项目已经冻结的研究主线：

> 面向异构防空资源、动态目标和复杂约束，研究稳定、节制、可解释并具备泛化能力的智能资源—目标分配方法。

BPCE-PPO 不改变 AirDefense v1 环境、奖励函数、动作空间、因子化自回归策略或现有评估指标，也不提前引入 GNN、GAT、Transformer 或新的高保真仿真层。它只处理当前已经由多轮实验定位出的核心瓶颈：

```text
同一 PPO 配置在不同随机种子下发生交战决策分叉
├─ all-noop / 低交战 / 高威胁目标突防
└─ 高交战 / 高资源成本 / 浪费性射击
```

现有证据表明，非法动作、联合分配冲突、普通动作因子化和简单关系表示容量已不再是当前首要问题。当前主要问题是：

1. `engage/no-op` 的局部价值被联合 GAE 混合；
2. 离线反事实 Critic 对在线 Actor 新分布支持不足；
3. 平滑 KL 不能直接检测 engagement 概率跨越 deterministic `0.5` 边界；
4. engagement 与 target 的独立 ratio/clipping 不能在反事实信号关闭时严格退化为已验证的 joint PPO。

相关证据入口：

- [项目创新路线](../project/research_innovation_roadmap.md)
- [MCH-PPO 候选边界](masked_counterfactual_hierarchical_ppo.md)
- [MCH-PPO 机制压力实验](../experiments/air_defense_v1_mch_ppo_mechanism_stress_test.md)
- [RG-MCH-PPO 机制压力实验](../experiments/air_defense_v1_rg_mch_ppo_stress_test.md)
- [SA-RG-MCH-PPO 机制压力实验](../experiments/air_defense_v1_sa_rg_mch_ppo_stress_test.md)

## 2. 创新诊断

当前判断：

> BPCE-PPO 是一个与现有问题链匹配、具备明确否证条件的候选创新，但仍需实现、实验和持续查新后才能形成论文贡献。

它不把“PPO + 动作掩码 + 反事实信号 + 辅助损失”的组件组合本身视为创新。候选贡献只来自以下不可拆分的机制：

> 在动态可行集自回归分配中，针对当前策略 deterministic 交战边界附近的状态执行稀疏、成对、on-policy 反事实探测，将经过统计验证的局部交战方向作为标准 joint PPO 的辅助 logit 排序信号，并保证探测信号失效时严格恢复现有 factorized joint PPO。

## 3. Problem–Method–Insight

| 层次 | 候选表述 |
| --- | --- |
| Problem | 动态掩码自回归分配中的联合 GAE 无法稳定区分“是否交战”和后续目标选择的边际贡献；离线 Critic 在在线策略分布外产生错误信用后，又会被独立层级 PPO 更新放大为 all-noop 或过度交战。 |
| Method | 保留完整联合动作上的 PPO surrogate，只在当前策略的 engagement 决策边界附近生成成对、共同随机数反事实分支，并把置信度达标的 `engage-noop` 回报差转换为辅助 logit 排序监督。 |
| Insight | 在受约束结构化分配中，局部反事实信用更适合充当当前分布下、经过验证的方向性辅助信号，而不应替代 on-policy 联合信用或形成独立的近端更新主干。 |

## 4. 一句话与三句话版本

### 4.1 一句话版本

BPCE-PPO 通过在交战决策边界附近主动验证局部反事实方向，在不改变 joint PPO 主干的条件下抑制全不交战与过度交战分叉。

### 4.2 三句话版本

1. **问题**：现有因子化 PPO 在不同种子下会分叉为全不交战或高成本交战，而离线反事实 Critic 与独立层级 clipping 会进一步放大这种不稳定性。
2. **方法**：BPCE-PPO 在 on-policy rollout 的临界 engagement 上下文中执行稀疏成对反事实探测，并用可信的 `engage-noop` 差值辅助训练 engagement logit。
3. **洞见**：反事实信用只有作为 joint PPO 下的局部辅助约束并满足严格安全退化语义时，才可能稳定改善结构化资源分配。

## 5. 冻结动作与策略结构

BPCE-PPO 继续使用现有因子化自回归策略。单元 `i` 在状态 `s` 和动作前缀 `h_i` 下，先决定交战变量 `z_i`，再在动态合法目标集合中选择目标 `y_i`：

```text
pi(a_i | s, h_i)
= pi_e(z_i | s, h_i)
  * pi_t(y_i | z_i=1, s, h_i)
```

动态合法目标集合 `L_i(s,h_i)` 必须排除：

- 环境基础非法动作；
- 前序单元已经占用的目标；
- 当前不可达、冷却、弹药耗尽或其他被掩码的目标；
- 反事实分支中与冻结动作上下文冲突的目标。

`no-op` 始终保留为合法 engagement 选择，不进入目标占用集合。

本候选不修改：

- `FactorizedEngagementActorCriticPolicy`；
- 自回归单元顺序与前缀占用语义；
- 目标条件策略的基本结构；
- 环境 action mask；
- value head 与 GAE 计算方式；
- 当前正式环境与奖励定义。

## 6. Joint PPO 安全主干

完整联合动作的 log-prob 定义为：

```text
log pi(a | s)
= sum_i [
    log pi_e(z_i | s,h_i)
    + 1[z_i=1] log pi_t(y_i | s,h_i)
  ]
```

联合概率比为：

```text
r_joint
= exp(
    log pi_theta(a | s)
    - log pi_old(a | s)
  )
```

Actor 主损失必须保持标准 joint PPO clipped surrogate：

```text
L_joint
= E[
    min(
      r_joint * A_GAE,
      clip(r_joint, 1-epsilon, 1+epsilon) * A_GAE
    )
  ]
```

BPCE-PPO 不再为 engagement 和 target 分别构造 PPO ratio 或 clipped surrogate。局部反事实信号不替换 `A_GAE`。

必须建立以下数值回归测试：

> 当探测预算为零、没有可靠探测样本或辅助系数为零时，BPCE-PPO 的 Actor loss、梯度和一次参数更新必须与现有 factorized joint PPO 在容差内一致。

建议容差：

```text
loss absolute difference <= 1e-7
maximum gradient difference <= 1e-6
maximum parameter-update difference <= 1e-6
```

该测试是算法安全语义的一部分，不只是软件测试。

## 7. 边界上下文选择

### 7.1 基础边界条件

对每个存在合法目标的单元，定义：

```text
p_engage = pi_e(z_i=1 | s,h_i)
margin_i = logit_engage - logit_noop
```

基础候选条件为：

```text
abs(p_engage - 0.5) <= delta_p
```

或等价的 logit 条件：

```text
abs(margin_i) <= delta_m
```

`delta_p` 或 `delta_m` 必须在正式结果产生前冻结，不得按成功种子事后选择。

### 7.2 安全—资源临界条件

为提高探测预算的信息增益，可复用项目已有安全临界采样指标，在以下条件中至少满足一项时提高优先级：

- 存在高威胁目标且 `time_to_impact` 较小；
- 当前单元仍有合法目标但弹药或资源预算紧张；
- 保护区潜在毁伤较高；
- 多个单元竞争同一高价值目标；
- 当前状态同时具有漏交战和浪费性交战风险；
- stochastic 与 deterministic engagement 决策不一致。

边界选择不得读取未来回报标签。它只能使用当前 observation、动作前缀、合法掩码、策略概率和可在决策时获得的状态量。

### 7.3 探测预算

第一版不对所有 rollout 状态生成反事实分支。建议冻结为：

```text
每个 rollout 最多 K 个 engagement 上下文
每个上下文每个分支 B 次共同随机数 rollout
```

候选初始范围：

```text
K: 8 或 16
B: 4 或 8
```

正式筛选只能选择一组配置。其他配置可用于 smoke test，不得根据正式结果切换。

v0实现时的成本 smoke 表明上述初始 `K` 会超过2倍时间门槛。正式性能结果
产生前按停止规则冻结为 `K=2、B=8、每2个PPO rollout探测一次`；1024步
匹配时间比降至1.61-1.86x，正式10k时间比为1.940x。

## 8. 成对 on-policy 反事实探测

对选中的 `(s,h_i,i)`，复制仿真状态并冻结其他已发生决策。构造两个分支：

```text
分支0：当前单元执行 no-op
分支1：当前单元执行 engage
```

分支1的目标由冻结的旧条件目标策略在 `L_i(s,h_i)` 上选择。第一版不同时学习新的目标反事实辅助损失，以避免重新混合 engagement 与 target 两个问题。

两个分支使用共同随机数：

```text
seed(branch0, repeat_b) == seed(branch1, repeat_b)
```

局部交战差值为：

```text
Delta_i^(b)
= R_engage_i^(b) - R_noop_i^(b)
```

其中 `R` 使用当前冻结的任务回报定义，不新增事后奖励塑形。若需要记录分量，可继续输出：

- operational return；
- zone damage；
- high-threat leak；
- resource cost；
- ammunition usage；
- total return。

第一版训练标签使用冻结的 total return 差值，其他分量用于失效诊断，不直接形成额外优化目标。

## 9. 置信门控

对 `B` 个成对差值计算：

```text
mean_delta = mean_b(Delta_i^(b))
se_delta = std_b(Delta_i^(b)) / sqrt(B)
signal_to_noise = abs(mean_delta) / max(se_delta, epsilon)
```

可靠门控：

```text
w_i
= 1[
    signal_to_noise >= tau
    and abs(mean_delta) >= delta_return
  ]
```

其中：

- `tau` 控制统计方向置信度；
- `delta_return` 排除虽然方向稳定但任务意义很小的差值；
- 两个门槛必须由 smoke/历史非正式数据冻结；
- 正式 test 标签不得用于选择门槛。

可以记录连续置信权重：

```text
c_i = min(1, signal_to_noise / tau_cap)
```

但第一版正式候选应优先使用简单二值门控，避免把新的权重函数搜索变成主要实验自由度。

v0 smoke 进一步发现，离散命中回报中大量 paired delta 为0，直接要求
`7/8` 同号会把零差值错误视为反方向。正式实现使用稀疏二值门控：至少
2个非零差值、均值方向票数严格多于反方向、反方向最多1个，且
`abs(mean_delta)>=1.0`。该门控仍是工程可靠性规则，不是统计置信区间。

## 10. 反事实交战辅助损失

定义 engagement logit margin：

```text
m_i
= logit_theta(engage | s,h_i)
  - logit_theta(noop | s,h_i)
```

反事实方向标签：

```text
d_i = sign(mean_delta)
```

辅助排序损失：

```text
L_BPCE
= mean_i[
    w_i * softplus(-d_i * m_i)
  ]
```

其语义为：

- `mean_delta > 0`：推动 engagement logit 高于 no-op；
- `mean_delta < 0`：推动 no-op logit 高于 engagement；
- `w_i = 0`：该上下文不产生辅助梯度。

总训练目标：

```text
L_total
= -L_joint
  + lambda_cf * L_BPCE
  + c_v * L_value
  + c_e * L_entropy
```

`lambda_cf` 必须在正式实验前冻结。第一版不引入单独的 engagement PPO ratio、单独 clip、初始 Actor KL 或最近邻支持乘法。

## 11. 为什么该机制针对当前瓶颈

### 11.1 避免离线支持错配

探测上下文来自当前 rollout 和旧策略，局部标签在当前策略访问分布上产生，不再依赖338条历史 Critic train 行是否覆盖在线状态。

### 11.2 避免信用替代

联合 GAE 仍是唯一 PPO 主信用。反事实差值只提供 engagement logit 的局部排序方向，不承担完整策略梯度估计。

### 11.3 避免错误 fallback

辅助信号关闭时，算法直接回到原 joint PPO loss，不存在“联合 GAE + 分层独立 clipping”这一中间优化器。

### 11.4 直接观察 deterministic 边界

探测对象由 `p_engage≈0.5` 或 logit margin 接近零定义，直接对应 deterministic 行为发生翻转的位置，而不是使用可能很小的全分布 KL 间接推断。

### 11.5 控制研究范围

第一版只修正 engagement 层。现有实验已经表明 conditional-target 排序相对更容易学习，因此不应在同一候选中同时修改目标 Critic、图表示或动作顺序。

## 12. 与现有基础设施的复用关系

| 现有基础 | BPCE-PPO 用法 |
| --- | --- |
| `FactorizedEngagementActorCriticPolicy` | 保持策略结构和 joint log-prob |
| `FactorizedEngagementMaskablePPO` | 作为安全主干和数值等价基线 |
| 自回归动态掩码 | 构造合法 engagement/target 上下文 |
| 共同随机数反事实 rollout | 生成成对 `engage-noop` 标签 |
| 安全临界状态采样 | 提高有限探测预算的信息增益 |
| 分量化回报与风险标签 | 诊断安全收益和资源成本来源 |
| benchmark 与 paired seeds | 复用正式对照和统计协议 |
| all-noop、leak、damage、cost 指标 | 判断是否只提高射击率 |
| MCH/RG-MCH/SA-RG-MCH 结果 | 作为失败机制对照，不重新挑种子 |

预计新增模块应限制在：

```text
rein_learning/common/
  boundary_counterfactual_probe.py

rein_learning/algorithms/policy_gradient/
  bpce_ppo.py

scripts/
  run_air_defense_v1_bpce_ppo_stress_test.py

tests/
  test_bpce_ppo.py
```

以上仅为接口建议，不在本设计阶段实施。

## 13. 最小实验协议

### 13.1 第一阶段：软件与语义验收

必须验证：

- 动态合法集合与动作前缀重建正确；
- 两个反事实分支除当前 engagement 决策外保持一致；
- 共同随机数配对可复现；
- no-op 与 engage 分支回报分量可重构；
- 不读取未来标签选择探测状态；
- 非可靠样本不产生辅助梯度；
- `probe_budget=0` 时严格等价于 factorized joint PPO；
- 模型保存、加载和统一评估正常；
- 现有完整回归测试通过。

### 13.2 第二阶段：10k 机制压力实验

冻结协议建议：

| 项目 | 配置 |
| --- | --- |
| 安全主干 | `factorized_engagement_ar_ppo_order_012` |
| 失败参考 | MCH-PPO、RG-MCH-PPO、SA-RG-MCH-PPO |
| 候选 | BPCE-PPO |
| 场景 | `time_pressure`、`heterogeneity_pressure` |
| 种子 | `8、9、10` |
| 训练预算 | `10k steps/model` |
| 评估 | 每场景30回合，完整交叉评估 |
| 目标层 | 保持现有 conditional-target 策略，不增加目标辅助损失 |
| 环境与奖励 | 完全冻结 |

### 13.3 机制门控

进入扩大实验需同时满足：

- 六个同场景候选均无绝对 all-noop 塌缩；
- 两个场景均至少 `2/3` 种子的 all-noop 不劣于 factorized PPO；
- 两个场景的平均奖励差均不低于 `-10`；
- 两个场景的平均损伤差均不高于 `+0.20`；
- 至少一个场景的高威胁突防率均值改善；
- 资源成本不超过 factorized PPO 的 `110%`；
- 边界探测优于等预算随机探测；
- 不允许用单个优势种子替代完整结论；
- 训练时间不高于 factorized PPO 的 `2.0x`，否则需先优化探测预算。

通过后才进入：

```text
30k × 5 seeds 消融
        ↓ 条件通过
100k × 5 seeds 正式主实验
        ↓
第二结构化分配任务或变规模验证
```

GNN 在本候选通过之前继续冻结。

## 14. 必要消融

| 消融 | 目的 |
| --- | --- |
| factorized joint PPO | 确认完整候选相对安全主干的增益 |
| `probe_budget=0` | 验证严格退化语义 |
| 随机探测替代边界探测 | 验证收益来自决策边界信息，而非额外仿真数据 |
| 独立随机数替代共同随机数 | 验证成对方差缩减的贡献 |
| 无置信门控 | 检查噪声反事实标签是否重新诱发分叉 |
| 仅正类 engage 标签 | 检查结果是否只是提高射击率 |
| 仅负类 no-op 标签 | 检查结果是否再次诱发 all-noop |
| 不同探测预算 | 报告性能—计算成本关系，不用于事后选择主结果 |

## 15. 可证伪命题

| 命题 | 支持证据 | 否决证据 | 所需实验 |
| --- | --- | --- | --- |
| P1：边界探测能够减少 engagement 种子分叉 | 两个核心场景、三种子均无绝对塌缩，且优于等预算随机探测 | 塌缩数不低于 factorized PPO，或边界探测与随机探测无稳定差异 | 10k三种子双场景压力实验 |
| P2：on-policy 成对探测比冻结离线 Critic 更适合作为 engagement 方向信号 | 临界上下文符号准确率、跨种子一致性和最终安全指标优于 RG/SA-RG | 边界标签仍跨批次反转，或只有挑选种子受益 | 相同种子和场景配对对照 |
| P3：辅助反事实信用不会破坏 joint PPO 安全主干 | 无可靠探测时 loss、梯度和参数更新与基线数值等价 | gate为零时仍发生非等价更新 | 确定性单步回归测试 |
| P4：收益不是简单增加交战率 | 高威胁突防和毁伤改善，同时资源成本不超过110% | 只降低 no-op，但浪费性交战、成本或毁伤明显恶化 | 安全—资源联合指标与正负标签消融 |

任一核心命题被稳定否决时，应收窄或停止该候选，不得通过增加随机种子、改变正式门槛或选择单一成功运行维持创新主张。

## 16. 查新边界

以下方法构成直接相关边界：

| 方法 | 已有内容 | BPCE-PPO 必须证明的差异 |
| --- | --- | --- |
| PPO | joint clipped surrogate | BPCE-PPO 不修改基本 PPO，而是在其上增加边界反事实辅助协议 |
| Invalid Action Masking | 状态相关合法动作掩码 | 动作掩码是基础设施，不是贡献 |
| COMA | 集中式 Critic 与单智能体动作边缘化反事实 baseline | BPCE-PPO 探测集中式自回归策略内部的 engagement 决策，并使用当前仿真分布的成对标签 |
| Action-dependent baseline | 因子化策略的动作相关方差降低 | BPCE-PPO 不把反事实量作为无偏 baseline 主张，而作为经过验证的局部排序辅助 |
| H-PPO | 层级/参数化动作策略优化 | 双动作头和分层动作不是贡献 |
| HAPPO/HATRPO | 多智能体顺序 advantage 与近端更新 | BPCE-PPO 保留单一 joint surrogate，不执行资源单元或动作因子的独立近端更新 |
| CAPO | 顺序团队反事实信用与闭式/拟合分解 | BPCE-PPO 依赖动态合法集和稀疏仿真探测，且研究对象是策略内部 allocate-or-abstain 边界 |
| Phasic Policy Gradient | 主策略阶段与辅助训练阶段 | 使用辅助损失或分阶段训练不是贡献；贡献必须来自边界探测、动态可行集和严格退化机制 |

主要检索入口：

1. PPO：<https://arxiv.org/abs/1707.06347>
2. Invalid Action Masking：<https://arxiv.org/abs/2006.14171>
3. COMA：<https://doi.org/10.1609/aaai.v32i1.11794>
4. Action-dependent baseline critique：<https://proceedings.mlr.press/v80/tucker18a.html>
5. H-PPO：<https://arxiv.org/abs/1903.01344>
6. HAPPO/HATRPO：<https://arxiv.org/abs/2109.11251>
7. CAPO：<https://arxiv.org/abs/2604.17693>
8. Phasic Policy Gradient：<https://proceedings.mlr.press/v139/cobbe21a.html>

当前不得使用“首次”“全新”或“填补空白”等表述。投稿前至少还需检索：

- on-policy counterfactual probing；
- active querying in reinforcement learning；
- simulator-based policy improvement；
- allocate-or-abstain / act-or-skip policy；
- auxiliary ranking loss for policy optimization；
- dynamic feasible-set autoregressive matching。

## 17. 预期论文贡献表述

以下表述仅在相应证据成立后使用。

1. 识别动态掩码自回归分配中 deterministic engagement 边界的种子分叉问题，并通过受控实验区分联合信用混叠、离线支持错配和独立层级 clipping 三类失效来源。
2. 提出 BPCE-PPO：保留 joint PPO 主干，在当前策略交战边界附近执行稀疏成对反事实探测，并以置信门控的 logit 排序辅助稳定 allocate-or-abstain 决策。
3. 建立严格退化、边界探测、共同随机数和正负交战标签消融，验证收益来自可信局部方向，而不是简单提高射击率或增加训练预算。
4. 在多种子、多场景和第二结构化分配任务上报告稳定性、安全性、资源效率、计算成本与失败边界。

## 18. 每项贡献必须证明什么

| 候选贡献 | 必要证据 | 当前最弱环节 |
| --- | --- | --- |
| engagement 边界是可重复的核心瓶颈 | 多种子 probability/margin、deterministic flip 与任务结果的关联 | 尚未建立正式的跨更新边界轨迹统计 |
| on-policy 边界反事实方向可靠 | 成对差值的符号功效、跨种子稳定性和独立批次复现 | 小探测预算下标签方差可能过高 |
| auxiliary 信号不破坏 joint PPO | gate=0 数值等价、完整候选无新增塌缩 | 非零辅助损失仍可能与主梯度冲突 |
| 方法具有论文级外推价值 | 第二结构化 allocate-or-abstain 任务或变规模验证 | 当前证据只来自 AirDefense v1 |

## 19. 审稿人压力点

| 风险 | 可能的审稿问题 | 修复方式 |
| --- | --- | --- |
| 看起来只是额外仿真标签 | 为什么不是简单用更多数据换性能？ | 使用等预算随机探测对照，并证明边界选择显著提高单位探测的信息增益 |
| 与 COMA/CAPO 重叠 | 反事实信用分配已有大量研究，区别在哪里？ | 明确比较并行智能体边缘化、顺序团队分解与策略内部动态可行 engagement 边界 |
| 与 PPG/辅助任务重叠 | 辅助损失和策略保持并不新 | 不把辅助优化形式作为创新，突出 on-policy边界探测、成对干预标签和严格退化协议 |
| 只提高射击率 | 是否通过牺牲资源效率降低突防？ | 同时报告 false-noop、wasteful-engage、成本、弹药、毁伤和高威胁突防 |
| 计算成本过高 | 每个状态生成反事实分支是否失去实用性？ | 使用稀疏预算、边界优先级和性能—开销曲线，冻结最大开销门槛 |
| 仿真器依赖 | 没有可复制仿真状态时是否还能使用？ | 将主张限定于可查询仿真训练场景，并把迁移到真实系统留作未来工作 |
| 仅防空场景有效 | 是否只是领域特定奖励工程？ | 增加第二个结构化 allocate-or-abstain 分配任务，保持算法与门槛不变 |

## 20. 失败后的决策规则

### 20.1 若边界标签功效不足

- 先检查共同随机数和分支冻结是否正确；
- 报告标签方差与所需预算投影；
- 只允许一次预注册的探测预算扩大；
- 若预算不可接受，则停止候选，不转为无限增加 rollout。

### 20.2 若标签可靠但 Actor 仍塌缩

- 检查非零辅助梯度是否与 joint PPO 主梯度冲突；
- 检查单次更新的 engagement margin crossing；
- 不恢复独立 engagement/target clipping；
- 不通过调大 `lambda_cf` 强行提高交战率。

### 20.3 若安全改善但资源成本恶化

- 结论收窄为“恢复必要交战”，不能宣称稳定资源分配；
- 检查负类 no-op 探测覆盖和标签不平衡；
- 只有正负边界同时可学习时才继续。

### 20.4 若 AirDefense v1 通过但第二任务失败

- 将贡献限定为防空动态资源分配方法；
- 不宣称一般结构化分配算法；
- 分析失败是否来自仿真可复制性、奖励语义或动作结构差异。

## 21. 创新演化日志

| 版本 | 当前洞见 | 新证据 | 修订原因 | 下一否证实验 |
| --- | --- | --- | --- | --- |
| v0：MCH-PPO | 用分层反事实 advantage 替代联合 Actor 信用 | 10k双场景中3/6塌缩 | 冻结离线 Critic 信号被 Actor 放大 | 保留 GAE，反事实只作残差 |
| v1：RG-MCH-PPO | GAE 主信用 + 可靠度反事实残差 | 两场景优于MCH v0，但2/6塌缩 | Critic 集成可能在分布外共同犯错 | 加入状态支持和累计漂移约束 |
| v2：SA-RG-MCH-PPO | 支持感知可靠度 + 初始 Actor KL | 5/6塌缩；支持低、KL未激活 | 独立层级 clipping 不能安全退化为 joint PPO | 恢复 joint PPO 主干 |
| v3：BPCE-PPO v0 | joint PPO主干 + on-policy边界反事实辅助 | 2/6塌缩；异质场景安全改善但成本1.93x；边界只在一个场景优于随机 | 正负标签覆盖和辅助更新剂量随种子分叉 | 双向覆盖门控、类别平衡与辅助梯度预算 |
| v4：标签语义审计 | 三种标签定义只读对照 | A/B一致0.901，B/C一致0.778；C可靠25/72；可靠负标签0/1 | deterministic continuation与全回报停止证据未通过门控 | 随机后续或短视窗分量标签；暂停剂量与选点修订 |
| v5：短视窗双分量审计 | TTI事件窗 + ENGAGE/STOP/AMBIGUOUS | 短窗15/16/41；异质场景10/14，time场景5/2；time资源槽STOP为0 | 当前动作替代后续射击，局部成本增量不跨场景可辨识 | 暂停BPCE在线辅助；保留失败机制与基础设施 |

## 22. 当前结论

BPCE-PPO 是当前项目中优先级高于 GNN、更多离线阈值校准和更大训练预算的候选创新。它与现有路线保持一致：

```text
不改变环境和奖励
        ↓
保留 factorized joint PPO 安全主干
        ↓
只修正已定位的 engagement 边界
        ↓
使用当前分布的稀疏成对反事实证据
        ↓
先做10k机制否证
        ↓ 条件通过
再做30k/100k与第二任务验证
```

v0实现、10k实验和标签语义审计后，准确表述应为：

> BPCE-PPO v0 已证明 joint PPO 安全退化和 on-policy 成对探测可行，并在异质场景获得条件性安全收益；但其正负证据覆盖与资源成本仍不稳定，不是已经成立或已经证明全新的算法。

进一步的72上下文审计没有发现可靠的target-argmax符号反转，但
target-marginal deterministic与stochastic continuation的总体符号一致率
只有0.778，且随机后续标签只有25/72可靠。两个场景的可靠负标签分别为0
和1。因此当前不得把类别平衡、辅助剂量或边界选点写成已验证修订，更不能
把BPCE写成已成立的算法创新。

短视窗双分量审计进一步得到31/72个可操作标签，仍低于48/72。异质场景
能够形成10个ENGAGE和14个STOP，但time-pressure只有5/2，其18个资源槽
全部AMBIGUOUS。当前局部STOP监督具有资源异质性条件依赖，不能支撑跨场景
在线辅助。因此BPCE主线在阶段A2停止，不进入剂量、选点或修订版10k。

完整结果见
[BPCE-PPO v0机制压力实验](../experiments/air_defense_v1_bpce_ppo_stress_test.md)
和
[BPCE标签语义审计](../experiments/air_defense_v1_bpce_label_semantics_audit.md)。

短视窗结果见
[BPCE短视窗标签审计](../experiments/air_defense_v1_bpce_short_horizon_label_audit.md)。
