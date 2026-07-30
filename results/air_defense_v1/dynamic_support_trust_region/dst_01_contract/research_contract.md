# DST-01 研究契约：动态支持敏感信赖域

契约版本：`1.0.0`  
冻结日期：`2026-07-29`  
任务状态：`PASSED`  
训练授权：`0`  
适用任务：`DST-02`—`DST-09`  
权威机器可读门控：[gate_registry.json](gate_registry.json)  
字段规范：[field_dictionary.csv](field_dictionary.csv)  
证据快照：[source_manifest.json](source_manifest.json)

## 1. 研究问题与边界

本任务包只检验以下因果链：

> 自回归前缀动作改变后续合法联合动作支持域；这种动态支持扰动若能解释并先行于
> 联合行为塌缩，则用 DS-TR 限制该扰动，可能比普通 KL 更有效。

三个可证伪命题固定为：

- `P1`：DS 对预注册失败结果有超出基础变量的分组外增量解释力；
- `P2`：DS 加权前缀 churn 相对 KL 和未加权 churn 有可重复的事件先行增量；
- `P3`：DS-TR 在固定 joint PPO 主目标下改善稳定性，且收益不是策略冻结、安全
  退化或资源失控的伪象。

本契约不把下列内容算作贡献：

- 自回归动作生成本身；
- 动态掩码或合法动作过滤本身；
- 用 Q、reward、威胁或资源成本重新加权后缀；
- 用旧策略 argmax 锚定、最优传输或支持向量距离替换 v0；
- 在看到 DS 结果后改换主距离、结果变量、种子或阈值。

阴性、停止和不可判定均是有效阶段出口。DST-01 不包含训练、策略更新或 DS 结果读取。

## 2. 数学对象

### 2.1 状态—前缀与可行联合动作

固定环境决策状态 \(s\)、单元顺序 \(u_{1:n}\) 和第 \(k\) 个决策前的事实前缀
\(h_k=(a_1,\ldots,a_{k-1})\)，定义：

\[
x_k=(s,h_k).
\]

\(\mathcal F(s)\) 是在该环境状态、固定单元顺序和项目正式动态动作掩码下可完成的
合法联合动作集合。它只描述同一环境决策步内的联合分配完成，不推进环境时间。

对当前位置合法动作 \(a\)，定义有序后缀集合：

\[
\mathcal F_{>k}(x_k,a)=
\{a_{k+1:n}:(h_k,a,a_{k+1:n})\in\mathcal F(s)\}.
\]

后缀元素是剩余单元按冻结顺序排列的动作元组；每个位置的合法 no-op 作为普通离散
动作保留。集合去重后再计算基数。

### 2.2 动作对动态支持距离

\[
c_{\mathrm{DS}}(x_k;a,b)=
1-\frac{
|\mathcal F_{>k}(x_k,a)\cap\mathcal F_{>k}(x_k,b)|
}{
|\mathcal F_{>k}(x_k,a)\cup\mathcal F_{>k}(x_k,b)|
}.
\]

冻结解释：

- `0`：两个当前动作保留完全相同的可行后缀集合；
- `1`：两个非空后缀集合不相交；
- completion-count ratio 只作描述量，不能替代 \(c_{\mathrm{DS}}\)；
- 主分析不按后缀的 Q、reward、威胁、资源成本或出现概率加权。

### 2.3 旧策略结构风险

\[
r_{\mathrm{old}}(x_k,a)=
\sum_{b\in A(x_k)}
\bar\pi_{\mathrm{old}}(b\mid x_k)
c_{\mathrm{DS}}(x_k;a,b),
\]

其中 \(A(x_k)\) 是当前位置既合法且至少存在一个可行完成的动作集合，
\(\bar\pi_{\mathrm{old}}\) 是在该精确动作集合上重新归一化后的旧策略概率。禁止用
argmax、均匀概率或经验动作频率代替旧策略概率。

### 2.4 DS-TR v0 策略距离

\[
D_{\mathrm{DS}}(\pi_\theta,\pi_{\mathrm{old}})
=
\mathbb E_{x_k\sim\widehat d_{\mathrm{old}}}
\left[
\frac{1}{2}
\sum_{a\in A(x_k)}
\left|
\bar\pi_\theta(a\mid x_k)-\bar\pi_{\mathrm{old}}(a\mid x_k)
\right|
r_{\mathrm{old}}(x_k,a)
\right].
\]

\(\widehat d_{\mathrm{old}}\) 是冻结 rollout 中合格、去重后的旧策略状态—前缀经验分布。
每个唯一 `context_id` 权重相同；动作对展开不能增加上下文权重。新旧策略概率必须在
同一个冻结状态、前缀和合法动作集合上计算。

该对象是以旧策略为参照的定向伪距离，不宣称是数学度量。冻结的数值性质为：

- \(0\le c_{\mathrm{DS}},r_{\mathrm{old}},D_{\mathrm{DS}}\le1\)；
- 新旧策略相同时 \(D_{\mathrm{DS}}=0\)；
- 不要求对称性、三角不等式或严格的同一性；
- 在后缀等价动作之间移动概率质量可能得到 \(D_{\mathrm{DS}}=0\)，这是定义性质，
  不是错误。

## 3. 位置、合法性与空集规则

### 3.1 主分析位置

- 只使用存在下游决策的前缀位置，即 `unit_position < n - 1`；
- 最后一个位置固定记为 `not_applicable`，不得用数值 `0` 混入统计；
- 当前动作必须通过正式环境掩码，并且至少存在一个可行联合动作完成；
- `action_a == action_b` 只用于单元测试，不进入正式动作对语料；
- 正式动作对按动作 ID 规范为无序对，避免 `(a,b)` 与 `(b,a)` 重复计数。

### 3.2 空集与异常

- 合格当前动作的后缀集合理论上必须非空；
- 任一合格动作产生空后缀时，记录 `EMPTY_SUFFIX` 并判定枚举器或掩码交叉检查失败；
- 两集合并集为空时，记录 `EMPTY_UNION`，不得把 Jaccard 距离设为 `0` 或 `1`；
- 当前非法动作记录 `ILLEGAL_CURRENT_ACTION` 并排除；
- 无法恢复精确状态、前缀、顺序、配置或模型哈希时不得近似计算 DS。

任何 `EMPTY_SUFFIX`、`EMPTY_UNION`、掩码不一致或最后位置混入主分析都是硬失败。

## 4. 冻结证据源

DS-0 语料源优先级固定为：

1. `results/air_defense_v1/task12_probe_corpus/`
2. `results/air_defense_v1/task12_task11_frozen_replay/`
3. `results/air_defense_v1/task10_frozen_model_diagnostics/`
4. 对应正式模型与 `experiment_config.json`

若历史诊断不能恢复精确状态和前缀，只允许在原配置、原模型、固定环境种子下进行
确定性诊断重放，并标记 `source_kind=replay`；不得把重放称为原始历史记录。

核心场景固定为 `time_pressure` 与 `heterogeneity_pressure`；`medium` 仅作描述和
稳健性检查。核心策略种子至少三个，正式比较组为 `scenario × policy_seed`。

所有输入文件的相对路径、字节数、UTC 修改时间和 SHA-256 已写入
`source_manifest.json`。若后续输入哈希不一致，必须生成新 manifest 和契约版本，
不得覆盖本版本。

## 5. DS-0 字段与结果语义

完整字段、类型、缺失规则和允许任务见 `field_dictionary.csv`。语料基本单位固定为：

```text
一个冻结状态 × 一个早期单元位置 × 一个事实前缀 × 一个合法无序动作对
```

基础变量固定为：

```text
scenario
policy_seed
unit_position
is_noop
legal_action_count
candidate_target_threat
prefix_engagement_count
```

动作对模型使用对应的对称编码：`noop_pair_type`、候选威胁最小值/最大值/绝对差，
并保留威胁缺失指示。不得加入 Q、reward、回合损伤或资源责任作为基础解释量。

冻结结果为：

- `downstream_argmax_changed`：两个分支下后续旧策略确定性 argmax 元组是否不同；
  只作机械正对照，不能单独使 P1 通过；
- `high_threat_legal_but_unassigned_changed`：高威胁目标在至少一个后续位置合法但最终
  未分配的状态是否因动作分支改变；
- `prefix_denied_changed`：动作分支是否改变“后续至少一个单元原可选目标被前缀占用”
  的状态；
- `engagement_extreme_direction`：从规范 `a→b` 分支看，联合交战数向 all-noop 极端
  记 `-1`，无极端边界改变记 `0`，向全可交战极端记 `+1`。

P1 的共同主要失败结果是后三项的二元化版本：
`high_threat_legal_but_unassigned_changed`、`prefix_denied_changed` 和
`engagement_extreme_direction != 0`。

## 6. P1：增量机制门

### 6.1 模型与分组

- 外层评估：六个核心 `scenario × policy_seed` 组的 leave-one-group-out；
- 同一 `context_id` 的所有动作对必须处于同一折；
- `M0`：冻结基础变量的 L2 logistic regression，`C=1`，
  `class_weight=balanced`；
- `M1`：与 `M0` 完全相同，仅增加 `ds_jaccard`；
- 数值标准化和类别 one-hot 只在训练折拟合；无超参数搜索；
- 概率阈值固定为 `0.5`。

主指标为 pooled out-of-fold AUROC 与 balanced accuracy；log loss 为方向一致性指标。
若某一测试组单类，组内只计算 log loss，不补样本或删组。

### 6.2 非退化门

进入增量比较前必须同时满足：

- 核心语料 pooled `ds_jaccard` 的 IQR `>= 0.05`；
- 六个场景—种子组中至少四组 IQR `>= 0.05`；
- 在样本数 `>=20` 的 `unit_position × noop_pair_type × legal_action_count` 分层中，
  至少 25% 的层内 DS 极差 `>=0.10`。

否则 P1=`STOPPED_CONSTANT_OR_DEGENERATE`。

### 6.3 P1 通过条件

至少一个共同主要失败结果必须同时满足：

1. `M1-M0` 的 pooled OOF AUROC 或 balanced accuracy `>=0.02`；
2. 以 `context_id` 为簇、`scenario × policy_seed` 为块的 10,000 次 bootstrap，
   对应增量的双侧 95% CI 下界 `>0`；
3. 两个核心场景的 OOF log-loss 改善 `M0-M1 >=0`；
4. 六个场景—种子块中至少五个的 OOF log-loss 改善 `>=0`；
5. 1,000 次、随机种子 `20260729` 的分层置换中，
   `ds_jaccard` 仅在 `scenario × policy_seed × unit_position × noop_pair_type`
   内置换；对三个共同主要结果使用 max-T 校正，FWER `p<=0.05`，且置换增量中位数
   `<=0.005`。

机械正对照失败会触发测量审计；机械正对照通过但三个共同主要失败结果均未通过，
P1 仍为 `STOPPED_NO_INCREMENTAL_MECHANISM`。

## 7. P2：更新级先行性门

### 7.1 运行与事件

优先重放现有 checkpoint；不足时只允许冻结 factorized engagement-target joint PPO
在 `heterogeneity_pressure` 上运行 `10k × seeds 8,9,10`。不得改学习率、entropy、
clip、网络、奖励或顺序 `012`。

事件口径复用现有正式定义：

- 塌缩：`all_noop_episode_rate >= 0.98` 或
  `actionable_engagement_rate < 0.01`；
- 高交战极端、安全和资源事件复用 P3 的冻结阈值；
- 时刻 \(t\) 的预警标签表示未来 `1—3` 个完成 PPO 更新内首次出现事件；
- 并发、事件后和距序列尾不足三个更新的行不作先行性主分析。

历史 PPO `approx_kl` 的预冻结 95 分位约为 `0.009492`，因此“小 KL”阈值固定为
`approx_kl <= 0.01`。

### 7.2 固定模型

```text
K0 = approx_kl + clip_fraction + entropy
K1 = K0 + unweighted_prefix_flip_rate
K2 = K1 + ds_weighted_flip_mass
```

模型、标准化、class weight、外层按 policy seed 留一和指标口径与 P1 相同。
`ds_weighted_flip_mass` 按唯一上下文平均，不按动作对行数加权。

### 7.3 P2 通过条件

P2 必须同时满足：

1. 至少两个种子含可判定事件；零或一个事件种子均为 `INCONCLUSIVE`，不追加种子；
2. `K2-K1` pooled OOF AUROC 或 balanced accuracy 增量 `>=0.02`；
3. 10,000 次 seed-block bootstrap 的对应 95% CI 下界 `>0`；
4. 所有事件种子的 OOF log-loss 改善 `K1-K2 >=0`，且至少两个严格 `>0`；
5. 至少两个事件种子中，事件前 `t-3:t-1` 的 DS 加权 flip 中位数高于本种子
   `t-6:t-4` 基线窗口；不足六个前置更新的事件不用于此项；
6. 至少一个真实事件前窗口同时满足 `approx_kl <=0.01`，且 DS 加权 flip 高于
   外层训练种子非事件更新的第 95 分位；
7. `K1-K0` 不能同时达到第 2—4 项而使 `K2-K1 <0.02`；否则未加权 flip 已足够解释。

事件只在发生后改变、单一种子独占效果或方向矛盾均使 P2 停止。相关性不得写成因果。

## 8. DS-TR v0 实现约束

DST-07 仅允许在完整 factorized joint PPO surrogate 外增加拒绝式接受规则：

1. 先执行原 joint PPO 候选更新；
2. 在同一冻结 minibatch 状态—前缀上计算 \(D_{\mathrm{DS}}\)；
3. 若超过冻结半径，按确定性 backtracking 缩放整个 actor 更新；
4. 若回溯耗尽，完整恢复旧 actor 参数和 optimizer state。

初始半径固定为 `δ_DS=0.05`，最多回溯 `8` 次，每次系数乘 `0.5`；数值容差
`1e-8`。实现必须保留 `lambda_ds` 作为严格开关：正式候选固定为 `1`；
`lambda_ds=0` 时不计算、不回溯 DS 分支，loss、梯度、更新参数、动作分布、
joint log-prob、评估动作和随机数消费顺序均与原 baseline 在冻结容差内一致。
仅在 DST-07 合成/回退测试失败时允许返回 DST-01 建立 `1.1.0` 版本；
不得在查看 DST-08 正式结果后修订。

严格禁止：

- 旧 argmax 锚定；
- Q、reward、威胁或资源加权；
- 最优传输或支持向量距离；
- 层级 surrogate 替换 joint PPO；
- 辅助 actor loss、奖励塑形或动作 mask 修改。

## 9. P3：最小异质场景筛选门

冻结比较：

```text
scenario: heterogeneity_pressure
budget: 10k
policy seeds: 8, 9, 10
unit order: 012
baseline: 原 factorized engagement-target joint PPO
candidate: baseline + DS-TR v0
```

评估使用相同环境种子、回合数和 frozen probe/counterfactual target set。不得因结果
补种子或调半径。P3 只有以下全部通过才成立：

1. exact fallback 位级相等，非法动作、分配冲突和过杀均为 `0`；
2. 三个候选运行均不满足塌缩定义；
3. `all_noop_episode_rate` 的候选—基线差在至少 `2/3` 种子 `<=0`，且三种子均值
   `<=0`；
4. `high_threat_leak_rate` 差在至少 `2/3` 种子 `<=0`，三种子均值 `<=0`；
5. 至少一个冻结安全指标严格改善：
   - 高威胁泄漏率均值差 `<=-0.02`，或
   - 平均总损伤均值差 `<=-0.20`；
6. 既有安全非劣门：平均奖励差 `>=-10.0` 且平均总损伤差 `<=0.20`；
7. 既有资源非劣门：候选平均资源成本 / 基线平均资源成本 `<=1.10`；
8. 在冻结反事实 target pair set 上，每个种子至少 30 个有效目标对，候选相对基线
   的 target pairwise ranking accuracy 差在至少 `2/3` 种子 `>=-0.02`，且均值
   `>=-0.02`；
9. 非冻结门：
   - 每个种子的 `actionable_engagement_rate >=0.01`；
   - 每个种子在冻结 probe 上的 joint argmax change rate `>=0.01`；
   - 至少 `2/3` 种子的有益翻转率高于有害翻转率；
10. 任一单一种子移除后，安全改善、all-noop 非劣和非冻结三类总体方向不反转。

高威胁改善和总损伤改善是预注册“至少一项”并集，不得事后增加第三个安全指标。
target 排序只用于能力保持，不进入 DS 权重或 actor loss。

## 10. DST-09 增量控制与阶段出口

仅当 P3 通过，按下列固定顺序逐个增加控制：

1. 通用 KL/churn 约束，匹配 DS-TR 的实测 accepted update norm；
2. 可行后缀计数均匀初始化；
3. DS-TR + 初始化组合。

每个控制沿用 P3 全部门控和相同种子，不允许批量搜索。只有 DS-TR 相对 matched
KL/churn 仍通过全部 P3 门，且主要安全改善至少保留 `75%`，才能主张 DS 特异增量。
若初始化单独达到同等结果，DS-TR 只能降级为诊断或组合组件。

阶段出口：

- `PASS_EXTENSION`：P1、P2、P3 及 matched-control 增量均通过；
- `PASS_DIAGNOSTIC_ONLY`：P1 或 P2 成立，但 P3/控制失败；
- `STOPPED`：任一硬门失败且无需更多数据判定；
- `INCONCLUSIVE`：仅限预注册的事件不足或精确状态不可恢复；
- 不允许以长训练、更多种子或新模块规避前述出口。

## 11. 版本控制与偏离规则

任何偏离必须在运行受影响分析前新建版本记录，至少写明：

```text
old_version
new_version
changed_clause
reason
evidence_seen_before_change
affected_tasks
```

允许的修订来源只有：公式在合成性质测试中失败、环境合法性语义被证明与契约不符、
或输入文件无法恢复精确状态。正式 DS/PPO 结果不构成改阈值或改主指标的理由。

当前版本历史：

| 版本 | 日期 | 说明 |
|---|---|---|
| 1.0.0 | 2026-07-29 | 首次冻结；尚未生成 DS-0、DS-1 或 DS-2 正式结果 |
