# 下一研究阶段：BPCE 标签语义、辅助剂量与选点覆盖审计

更新时间：2026-07-23  
任务状态：阶段 A 已完成但未通过；阶段 B/C 按门控未启动  
任务性质：机制诊断与下一候选冻结，不训练30k/100k，不进入GNN

## 1. 阶段结论与研究目标

BPCE-PPO v0 已完成软件验收和10k机制压力实验。以下性质已经通过，可以冻结：

1. 完整 joint PPO surrogate 可作为安全主干；
2. `probe_budget=0` 或全部标签被拒绝时，BPCE 与 factorized PPO 的一次参数更新最大差不超过 `1e-6`；
3. AirDefense v1 状态快照、共同随机数双分支和稀疏 on-policy 探测可执行；
4. 正式配置能够在训练时间不超过基线2倍的条件下生成非零辅助监督。

但是，BPCE-PPO v0 的核心机制门控失败：

- `2/6` 个同场景运行发生绝对 all-noop；
- `time_pressure` 的奖励、损伤和高威胁突防均明显退化；
- `heterogeneity_pressure` 虽有安全收益，但资源成本达到基线的 `1.93x`；
- 边界选点未稳定优于等预算随机选点；
- 标签正负覆盖随随机种子分叉。

因此，下一阶段不直接实现完整的 coverage-balanced BPCE-PPO，也不同时修改选点、标签、损失和梯度约束。本阶段只回答三个按顺序排列的问题：

```text
Q1：当前反事实差值是否真正表示 engagement 价值？
        ↓ 通过
Q2：可靠标签是否以与证据量匹配的剂量进入 joint PPO？
        ↓ 通过
Q3：边界选点是否比等预算随机选点提供更平衡、更有效的信息？
        ↓ 通过
才允许冻结下一版 BPCE 并重新运行10k机制实验
```

对应材料：

- [BPCE-PPO 算法设计](../algorithms/boundary_probed_counterfactual_engagement_ppo.md)
- [BPCE-PPO v0 机制压力实验](../experiments/air_defense_v1_bpce_ppo_stress_test.md)
- [项目创新路线](../project/research_innovation_roadmap.md)

## 2. 当前主要风险

### 2.1 engagement 标签与 target 选择混叠

BPCE-PPO v0 的 engage 分支使用冻结旧策略的 masked-argmax 目标，并使用 deterministic continuation。当前标签实际估计：

```text
在当前 argmax 目标和确定性后续策略下，
强制 engage 相对强制 no-op 的回报差
```

它不必然等价于：

```text
在当前动态合法目标集合下，
engage 相对 no-op 的条件期望价值差
```

当 argmax 目标选择错误或后续 deterministic 行为发生分叉时，负标签可能表示“当前目标或后续策略较差”，而不是“当前不应交战”。

### 2.2 单个标签可能获得完整辅助剂量

当前辅助损失只在有效标签上取均值：

```text
L_aux
= sum_i w_i * softplus(-d_i * margin_i)
  / max(1, sum_i w_i)
```

因此，一个有效负标签和多个有效标签可能产生相近的平均辅助损失尺度。seed9 在两个场景中只有 `0/1` 和 `0/5` 个正/负标签，却均发生 absolute all-noop，说明辅助更新剂量不能只由固定 `lambda_cf=0.05` 控制。

### 2.3 margin top-K 不等于任务信息最优

当前边界模式只选择 `|engagement margin|` 最小的两个上下文。该规则不能保证同时覆盖：

- 高威胁、必须交战的安全临界状态；
- 资源紧张、应当停止的资源临界状态；
- 不同单元、不同决策时刻和不同目标压力。

随机探测在 `heterogeneity_pressure/seed9` 中获得正标签并避免塌缩，而 margin top-K 没有获得正标签。这表明“更接近0.5”尚未被证明等价于“更有训练价值”。

## 3. 冻结项

本阶段不得修改：

- AirDefense v1 环境、场景参数和奖励函数；
- 动作掩码、单元顺序和无冲突自回归动作语义；
- `FactorizedEngagementActorCriticPolicy`；
- joint PPO surrogate、GAE、value loss 和目标条件策略结构；
- BPCE v0 正式实验结果与既有门控；
- `time_pressure`、`heterogeneity_pressure` 两个核心场景；
- 当前共同随机数环境随机带语义；
- 30k/100k、target辅助和GNN的冻结状态。

本阶段允许修改或新增的内容仅限：

- 只读探测诊断接口；
- engagement 标签语义对照；
- 每个有效辅助 minibatch 的梯度诊断；
- 不训练 Actor 的选点覆盖比较；
- 下一候选的预注册配置。

## 4. 阶段 A：engagement 标签语义审计

### 4.1 核心问题

本阶段首先检验：

> 当前 `argmax target + deterministic continuation` 得到的符号，是否可以稳定代表合法目标集合下的 engagement 条件价值方向。

在该命题通过前，不实施类别平衡损失。对不可靠标签做类别平衡，只会更加稳定地放大错误方向。

### 4.2 数据生成策略

优先使用既有 factorized PPO 冻结模型生成状态，不使用 BPCE 辅助更新后的策略作为唯一数据源，以避免标签与候选训练结果循环依赖。

建议数据块：

| 维度 | 配置 |
| --- | --- |
| 场景 | `time_pressure`、`heterogeneity_pressure` |
| 策略种子 | 复用 `8、9、10` 的冻结 factorized PPO |
| 每个“种子×场景”上下文 | 12个 |
| 安全临界槽 | 6个 |
| 资源临界槽 | 6个 |
| 总上下文 | 72个 |
| 每个分支重复数 | 32次共同随机数 rollout |
| Actor更新 | 禁止 |

若冻结 factorized PPO 模型不可用或无法重建对应策略，只允许重新训练完全相同的 factorized PPO；不得在该运行中加入BPCE辅助损失。

### 4.3 三种标签定义

对同一 `(s,h_i,i)` 和同一组环境随机带，计算三种差值。

#### 标签 A：当前 BPCE v0 定义

```text
Delta_argmax_det
= R(
    force engage,
    masked-argmax target,
    deterministic continuation
  )
  - R(
      force no-op,
      deterministic continuation
    )
```

该标签用于复现当前机制，不作为默认真值。

#### 标签 B：目标边缘化、确定性后续

```text
Delta_target_marginal_det
= E_y~pi_target_old[
    R(
      force engage on y,
      deterministic continuation
    )
  ]
  - R(
      force no-op,
      deterministic continuation
    )
```

目标只在动态合法集合 `L_i(s,h_i)` 上重新归一化。实现可以选择：

- 对全部合法目标精确边缘化；或
- 按冻结 `pi_target_old` 分层采样，并记录采样误差。

若合法目标数允许，优先精确边缘化，避免再次把目标采样噪声混入 engagement 标签。

#### 标签 C：目标与后续策略边缘化

```text
Delta_target_marginal_stochastic
= E_{
    y~pi_target_old,
    a_>i~pi_old
  }[
    R(force engage on y)
  ]
  - E_{a_>i~pi_old}[
      R(force no-op)
    ]
```

环境随机数在 engage/no-op 分支间配对。策略采样使用冻结旧策略和预生成 uniform tape，以保证可复现。

标签 C 更接近旧策略下的条件期望，但计算成本最高。它只用于语义审计，不直接进入第一版在线训练。

### 4.4 必须记录的诊断

每个上下文至少记录：

- state/scenario/seed/unit/prefix 标识；
- `engage_probability` 与 logit margin；
- 动态合法目标数；
- masked-argmax 目标；
- 各合法目标的条件概率；
- 三种 `mean_delta`、标准误和符号；
- argmax目标相对最佳合法目标的反事实 regret；
- deterministic 与 stochastic continuation 的符号是否一致；
- 安全临界或资源临界槽位；
- zone damage、high-threat leak、resource cost 与 total return 分量差。

### 4.5 标签语义门控

建议冻结以下门槛：

| 门控 | 通过条件 |
| --- | --- |
| 总体标签功效 | 至少48/72个上下文形成可靠非零标签 |
| 块级功效 | 每个“种子×场景”至少6个可靠标签 |
| A/B符号一致 | 总体不低于0.80，最差场景不低于0.70 |
| B/C符号一致 | 总体不低于0.80，最差场景不低于0.70 |
| argmax目标混淆 | 因目标选择导致的可靠符号反转不超过20% |
| 双向覆盖 | 每个场景至少存在6个可靠正标签与6个可靠负标签 |
| 分量一致性 | total-return正标签不能主要由明显增加毁伤或高威胁突防产生 |

门槛必须在运行审计前冻结。若认为上述数值需要调整，只能根据计算功效预估调整，不得根据正式标签结果修改。

### 4.6 阶段 A 决策

#### A/B/C 均通过

说明当前 argmax-det 标签可作为低成本近似。进入阶段 B，但仍需解决证据剂量和覆盖问题。

#### A/B 不通过，B/C通过

说明主要问题是 target argmax 混叠。下一候选必须使用目标边缘化 engagement 标签；不得只做类别平衡。

#### B/C不通过

说明 deterministic continuation 不能代表冻结旧策略下的 engagement 价值。下一候选应使用随机后续或缩短反事实估值范围；不得继续沿用当前标签。

#### 三者均不稳定

暂停 BPCE 在线辅助主线。保留 joint PPO fallback 与实验性失效结论，重新审视环境回报的局部可辨识性，不扩大反事实 rollout 或转入GNN。

## 5. 阶段 B：辅助更新剂量审计

阶段 B 只在阶段 A 至少得到一种通过门控的标签定义后执行。

### 5.1 当前剂量问题

固定 `lambda_cf` 不能表示以下差异：

- 当前 rollout 只有1个还是10个可靠标签；
- 标签是否同时包含正负方向；
- 辅助梯度是否与 joint PPO 主梯度同量级；
- 同一标签是否在多个 PPO epoch 中被重复放大。

### 5.2 梯度诊断修订

当前梯度诊断不得只记录第一个 minibatch。应对每个含有效标签的 minibatch 记录：

```text
norm(g_joint)
norm(lambda_cf * g_aux)
cos(g_joint, g_aux)
active_positive_count
active_negative_count
active_label_count
epoch_index
minibatch_index
engagement_margin_before
engagement_margin_after
deterministic_flip_count
```

汇总时报告：

- 辅助/主梯度范数比的均值、P90和最大值；
- 正标签与负标签的累计梯度剂量；
- 每个标签被使用的epoch次数；
- 辅助更新前后跨越0.5边界的上下文数量；
- all-noop种子与非塌缩种子的剂量差异。

### 5.3 候选剂量规则

不训练 Actor 的离线梯度回放中比较以下规则。

#### D0：当前固定系数

```text
lambda_eff = 0.05
```

作为失败参考。

#### D1：证据量缩放

```text
coverage_scale
= min(
    1,
    active_label_count / N_ref
  )

lambda_eff
= 0.05 * coverage_scale
```

建议先冻结 `N_ref=4`。一个标签最多获得当前剂量的25%，四个及以上标签才达到完整剂量。

#### D2：双类平衡与证据量缩放

当正负类均存在时：

```text
L_balanced
= 0.5 * mean(L_positive)
  + 0.5 * mean(L_negative)
```

并使用：

```text
coverage_scale
= min(
    1,
    2 * min(N_positive, N_negative) / N_ref
  )
```

任一类别为空时：

```text
coverage_scale = 0
L_aux = 0
```

不允许使用重复复制少数类标签来伪造覆盖。

#### D3：D2加相对梯度剂量上限

```text
g_aux_scaled
= g_aux
  * min(
      1,
      alpha * norm(g_joint)
      / max(epsilon, norm(lambda_eff * g_aux))
    )
```

`alpha` 不通过正式性能搜索选择。建议先在离线梯度回放中选择能满足：

```text
P90 auxiliary/joint gradient norm ratio <= 0.10
```

的最小简单值。

### 5.4 阶段 B 门控

冻结下一候选剂量规则需满足：

- 单标签剂量显著低于四标签剂量；
- 正负类累计梯度剂量比例位于 `[0.5, 2.0]`；
- P90辅助/主梯度范数比不超过 `0.10`；
- 任一类别缺失时严格退化为 joint PPO；
- `probe_budget=0` 时继续满足 loss、梯度和参数更新数值等价；
- 不需要按场景、种子设置不同系数；
- 不读取最终10k性能结果选择剂量规则。

优先选择满足门控的最简单规则。D1通过则不引入D2/D3；只有D1不能控制单边标签和梯度峰值时才升级。

## 6. 阶段 C：选点覆盖审计

阶段 C 不训练 Actor，只比较相同反事实分支预算下的选点信息质量。

### 6.1 候选选点方法

#### S0：等预算随机选点

保持当前随机对照。

#### S1：纯 margin top-K

保持 BPCE v0 选点，作为失败参考。

#### S2：安全/资源双槽位边界选点

在不读取未来回报的前提下，将每轮 `K=2` 固定为：

```text
slot 1：安全临界
  高威胁、低time-to-impact或高潜在毁伤
  候选内部按abs(margin)排序

slot 2：资源临界
  弹药紧张、资源成本高或存在替代单元
  候选内部按abs(margin)排序
```

若某槽位没有候选，其预算可以转给另一槽位，但必须记录缺失率。

### 6.2 预算重分配探索

当前 `K=2、B=8` 每轮最多生成32个分支 rollout。允许在 smoke 中比较：

```text
K=2, B=8
K=4, B=4
```

两者保持相同名义分支数量。`K=4、B=4` 只有在标签语义审计表明4次重复具有足够符号功效时才可使用。

可以增加顺序早停：

- 已达到可靠方向且剩余重复无法改变门控结论时停止；
- 即使全部剩余重复同向也无法通过时停止；
- 节省的预算不得在同一正式运行中动态搜索更多上下文，除非该规则提前冻结。

### 6.3 选点门控

S2相对S0/S1应满足：

- 每个场景均获得正负双向标签；
- 最差“种子×场景”块不再出现全正或全负接受集；
- 单位额外transition的可靠标签数不低于随机选点；
- 可靠标签接受率不低于S1；
- 安全槽的正标签率高于资源槽；
- 资源槽的负标签率高于安全槽；
- 训练时间投影不超过factorized PPO的2倍。

若S2不能稳定优于随机选点，则不得继续把“边界选点优越性”作为创新主张。此时可以保留随机探测作为辅助数据获取方法，但BPCE创新需进一步收窄。

## 7. 下一候选的冻结条件

只有阶段 A、B、C 均通过，才允许冻结下一算法候选。暂不提前命名为 coverage-balanced BPCE-PPO，避免在机制未确定前固定论文叙事。

下一候选最多包含三项已分别验证的改动：

1. 一种通过阶段 A 的 engagement 标签定义；
2. 一种通过阶段 B 的证据剂量规则；
3. 一种通过阶段 C 的选点规则。

不得同时加入：

- target辅助损失；
- 新Critic或GNN；
- 新奖励项；
- 新动作顺序；
- 经验最优 engagement 阈值；
- 多套场景专用超参数。

## 8. 下一次10k机制实验建议

### 8.1 实验对象

| 方法 | 作用 |
| --- | --- |
| factorized joint PPO | 安全主干 |
| BPCE-PPO v0 | 失败机制参考 |
| 等预算随机探测 + 新剂量 | 区分选点价值与辅助剂量价值 |
| 下一BPCE候选 | 完整候选 |

### 8.2 种子策略

`8、9、10` 已被多轮读取并用于机制修订，下一候选若继续只使用这些种子，存在设计过拟合风险。

建议：

- 预注册三个此前未用于BPCE设计的新训练种子；
- 使用相同新种子重新训练factorized PPO和随机探测对照；
- `8、9、10` 只作为历史复现附录，不参与新候选选择；
- 新种子编号在运行前写入实验配置，不根据smoke表现更换。

### 8.3 冻结协议

| 项目 | 配置 |
| --- | --- |
| 场景 | `time_pressure`、`heterogeneity_pressure` |
| 训练预算 | `10k steps/model` |
| PPO epochs | 2 |
| 评估 | 每场景30回合，完整交叉评估 |
| 环境与奖励 | 冻结 |
| joint PPO主干 | 冻结 |
| 探测总预算 | 不超过BPCE v0 |
| 正式候选数量 | 1套，不并行搜索多个系数 |

### 8.4 机制门控

下一候选进入30k前必须同时满足：

- 六个同场景候选均无绝对 all-noop；
- 两个场景均至少 `2/3` 新种子的 all-noop 不劣于配对baseline；
- 两个场景奖励差均不低于 `-10`；
- 两个场景损伤差均不高于 `+0.20`；
- 至少一个场景高威胁突防均值改善；
- 两个场景资源成本均不超过baseline的 `110%`；
- 每个“种子×场景”块均存在正负双向可靠标签；
- 辅助/主梯度范数比P90不超过阶段 B 冻结上限；
- 完整候选优于等预算随机探测；
- 训练时间不超过baseline的 `2.0x`；
- 不能选择单个优势种子形成结论。

## 9. 可证伪命题

| 命题 | 支持证据 | 否决证据 |
| --- | --- | --- |
| P1：BPCE v0 的种子分叉部分来自 engagement 标签语义混叠 | argmax-det 与目标边缘化标签存在稳定符号差异，且差异集中于塌缩种子 | 三种标签符号高度一致，seed9仍稳定只有负标签 |
| P2：稀疏标签被固定均值损失过量放大 | 单标签辅助/主梯度比与多标签相近，证据量缩放显著降低剂量峰值 | 当前剂量已随标签数自然缩小，且塌缩与剂量无关 |
| P3：任务分层选点比纯margin更能覆盖安全与资源两侧 | S2在相同预算下提高双向覆盖和单位transition可靠标签数 | S2仍不优于随机或margin，标签方向继续随种子分叉 |
| P4：语义、剂量和覆盖修订能够稳定改善engagement | 新10k候选无塌缩且同时通过安全与资源门控 | 仍出现all-noop/过度交战，或收益只来自提高资源消耗 |

## 10. 停止与转向条件

满足以下任一条件时，应暂停BPCE在线辅助主线：

1. 三种标签定义在高功效上下文中仍频繁符号冲突；
2. 目标边缘化与随机后续仍不能形成跨种子双向标签；
3. S2选点在相同预算下不优于随机选点；
4. 辅助剂量降到安全范围后不再产生可测行为影响；
5. 新10k实验仍有任一绝对all-noop；
6. 安全收益继续依赖超过baseline `110%` 的资源成本；
7. 计算成本需要超过baseline `2x` 才能获得可靠标签。

暂停后的保留成果：

- joint PPO严格fallback；
- on-policy成对反事实探测基础设施；
- engagement边界和标签覆盖的失效证据；
- 边界选点与随机选点的受控比较；
- 对“局部反事实辅助并非自动稳定”的经验结论。

暂停后优先重新审视：

- 环境回报对局部 engagement 的可辨识性；
- 是否需要短视窗安全/资源分量而非全回报标签；
- 是否将工作转为失效机理与评估协议论文贡献。

不得直接通过GNN、更多随机批次或更大训练预算绕过上述失败。

## 11. 预期产物

阶段 A：

```text
docs/experiments/air_defense_v1_bpce_label_semantics_audit.md
results/air_defense_v1/bpce_label_semantics_audit/
```

阶段 B：

```text
docs/experiments/air_defense_v1_bpce_auxiliary_dose_audit.md
results/air_defense_v1/bpce_auxiliary_dose_audit/
```

阶段 C：

```text
docs/experiments/air_defense_v1_bpce_probe_selection_audit.md
results/air_defense_v1/bpce_probe_selection_audit/
```

若三个阶段均通过：

```text
docs/task_guides/next_research_phase_bpce_revised_10k_screening.md
docs/experiments/air_defense_v1_bpce_revised_10k_screening.md
results/air_defense_v1/bpce_revised_10k_screening/
```

本阶段不预先要求创建所有产物；未通过的阶段之后不得继续创建后续实验结果。

## 12. 创新表述边界

阶段 A–C 完成前，项目只能表述：

> BPCE-PPO v0 证明了 joint PPO 安全退化和稀疏 on-policy 成对探测的工程可行性，但其 engagement 标签语义、正负覆盖和辅助更新剂量尚未稳定。

阶段 A–C 通过但新10k尚未完成时，可以表述：

> 项目已经冻结一种语义验证、剂量受控且覆盖平衡的反事实 engagement 辅助接口，正在验证其是否能稳定减少种子分叉。

只有新10k和后续扩大实验通过后，才可以把该接口写成算法贡献。不得提前使用“解决”“稳定消除”或“首次”等结论。

## 13. 建议执行顺序

```text
1. 增加只读标签语义诊断接口
2. 冻结72个上下文、三种标签定义和阶段A门槛
3. 运行阶段A，不训练Actor
4. 根据A的唯一结论冻结一种标签
5. 在固定数据上进行阶段B梯度回放
6. 冻结最简单的安全剂量规则
7. 在相同分支预算下执行阶段C选点审计
8. 三阶段均通过后再编写下一10k预注册
```

当前最优先任务不是实现 coverage-balanced loss，而是完成阶段 A 的 engagement 标签语义审计。

## 14. 阶段 A 执行冻结记录

冻结时间：2026-07-23。正式结果生成前完成。

本轮仅执行阶段 A，不训练或更新 Actor。使用
`mch_ppo_mechanism_stress_test` 中已保存的六个 factorized PPO 10k
冻结模型：

- 场景：`time_pressure`、`heterogeneity_pressure`；
- 策略种子：`8、9、10`；
- 每个“场景×种子”块：6个安全临界上下文和6个资源临界上下文；
- 每个分支：32次配对环境随机带；
- 标签 B/C 的首个 engage 目标：对全部动态合法目标按冻结条件概率精确边缘化；
- 标签 C 的后续动作：使用冻结旧策略和预生成 uniform tape 采样；
- 可靠标签：`|mean_delta| >= 1.0`，且95%均值置信区间不跨越0；
- 分量一致性：可靠正标签中，zone damage 增量不超过0.05且
  high-threat leak 增量不超过0.10的比例至少为0.80。

上下文选择不读取未来回报。安全槽按当前合法目标的
`threat × payload × protected-zone value / time-to-impact` 排序；资源槽按
弹药稀缺度、单位相对成本和替代单元覆盖率排序；同分时优先选择绝对
engagement margin 更小的上下文。正式审计后不得根据结果修改上述规则或
阶段 A 门槛。

## 15. 阶段 A 正式结论

正式审计已完成72个上下文、2304条重复记录和266,198个额外transition，
Actor最大参数差为0。

- A/B总体符号一致率为0.901，最差场景为0.829，可靠反转为0/24；
- B/C总体符号一致率为0.778，低于0.80门槛；
- 标签 C 只有25/72可靠，最差块为0/12；
- `time_pressure` 的可靠正/负标签为10/0；
- `heterogeneity_pressure` 的可靠正/负标签为14/1；
- 阶段 A 总门控失败。

因此，当前证据不支持把 target argmax 视为主要混叠来源；主要问题是
deterministic continuation 在低功效上下文中的方向不稳定，以及全回报
标签缺少资源停止方向。按照本任务的冻结顺序，不执行阶段 B/C，不实现
coverage-balanced loss，不运行下一版10k。

完整结果见：
[AirDefense v1 BPCE 标签语义审计](../experiments/air_defense_v1_bpce_label_semantics_audit.md)。
