# DST-01：研究契约、公式与证据源冻结

任务状态：`PASSED`  
训练授权：无  
前置任务：无

## 1. 目标

在读取任何 DS 结果前，冻结问题、公式、字段、统计比较和停止条件，防止后续
根据结果改写创新定义。

## 2. 输入

- [六篇阅读综合与算法创新决策](../../project/air_defense_v1_literature_synthesis_and_algorithm_innovation_decision_2026-07-29.md)
- [算法实验状态总复盘](../../project/air_defense_v1_algorithm_experiment_status_review_2026-07-29.md)
- [Factorized engagement 算法说明](../../algorithms/factorized_engagement_policy.md)
- [自回归无冲突策略说明](../../algorithms/autoregressive_conflict_free_policy.md)

## 3. 必须冻结的数学对象

对状态—前缀 \(x_k=(s,h_k)\) 和候选动作 \(a\)，定义：

\[
\mathcal F_{>k}(x_k,a)=
\{a_{k+1:n}:(h_k,a,a_{k+1:n})\in\mathcal F(s)\}.
\]

动作对的主 DS 度量固定为：

\[
c_{\mathrm{DS}}(x_k;a,b)=
1-\frac{
|\mathcal F_{>k}(x_k,a)\cap\mathcal F_{>k}(x_k,b)|
}{
|\mathcal F_{>k}(x_k,a)\cup\mathcal F_{>k}(x_k,b)|
}.
\]

对旧策略定义每个动作相对旧策略概率质量的结构风险：

\[
r_{\mathrm{old}}(x_k,a)=
\sum_b \pi_{\mathrm{old}}(b\mid x_k)
c_{\mathrm{DS}}(x_k;a,b).
\]

DS-TR v0 的策略级距离冻结为加权总变差：

\[
D_{\mathrm{DS}}(\pi_\theta,\pi_{\mathrm{old}})
=
\mathbb E_{x_k}
\left[
\frac{1}{2}
\sum_a
\left|
\pi_\theta(a\mid x_k)-\pi_{\mathrm{old}}(a\mid x_k)
\right|
r_{\mathrm{old}}(x_k,a)
\right].
\]

该形式在新旧策略相同时为 0，并把概率质量变化按动作相对旧策略支持结构的差异
加权。它只是 v0 的可证伪工程形式，不提前主张为一般意义上的最优策略度量。

规则：

- 主分析只使用存在非空下游决策的前缀位置；
- 最后位置不纳入主 DS 命题；
- 不按 Q、reward、威胁或资源成本给后缀加权；
- completion-count ratio 可作为描述量，不能替代主度量；
- 结果产生后不得改用另一个集合距离作为主指标。
- DST-07 不得改用旧 argmax 锚定、Q 加权、最优传输或支持向量距离；
- 如果该策略距离在合成测试中不满足基本数值性质，返回 DST-01 修订并留下版本
  记录，不能在查看正式结果后静默换公式。

## 4. 冻结命题

| 命题 | 支持条件 | 否决条件 |
|---|---|---|
| P1：DS 有增量解释力 | 加入 DS 后，对预注册失败指标的分组外预测优于基础变量，且方向跨场景稳定 | DS 近似常数、增量消失或方向不稳 |
| P2：DS 加权 churn 先于崩塌 | 在更新级数据中，相对 KL 和普通 flip 有稳定提前量/增量 | 只在崩塌后变化，或与普通 flip 等价 |
| P3：DS-TR 改善而非冻结 | 稳定性和冻结安全—资源门同时通过，且策略仍有有益变化 | 只减少 churn、冻结旧策略或需要大规模调参 |

## 5. 预注册比较

基础变量至少包括：

```text
scenario
policy_seed
unit_position
is_noop
legal_action_count
candidate_target_threat
prefix_engagement_count
```

主结果：

```text
downstream_argmax_changed
high_threat_legal_but_unassigned
prefix_denied
engagement_extreme_direction
```

主比较为“基础变量模型”与“基础变量 + DS 模型”的分组外差值。分组必须以
场景×策略种子为单位，不能把同一轨迹或同一状态的动作对拆到训练、测试两侧。

## 6. 交付物

在以下位置建立：

```text
results/air_defense_v1/dynamic_support_trust_region/dst_01_contract/
  research_contract.md
  field_dictionary.csv
  gate_registry.json
  source_manifest.json
```

`source_manifest.json` 必须记录所有输入文件的路径、大小、修改时间和哈希。

## 7. 验收

- 公式、位置范围和空集规则明确；
- 主/次指标和分组方式明确；
- 所有门控在查看 DS 结果前写入；
- 明确最后位置不属于 v0 的核心主张；
- 明确阴性结果也是阶段出口；
- 没有训练或策略修改。

验收后进入 DST-02。

## 8. 执行记录

执行日期：`2026-07-29`  
契约版本：`1.0.0`  
执行结果：`PASSED`  
训练与策略修改：`0`

已在以下目录完成四项冻结交付物：

```text
results/air_defense_v1/dynamic_support_trust_region/dst_01_contract/
  research_contract.md
  field_dictionary.csv
  gate_registry.json
  source_manifest.json
```

本轮在任何正式 DS-0、DS-1 或 DS-2 结果产生前，补充冻结了：

- 合格位置、合法动作、最后位置和空并集的精确处理；
- 状态—前缀、动作对、后缀集合、分支结果和更新级指标的字段语义；
- P1 的分组外增量、非退化、bootstrap 和分层置换门；
- P2 的事件窗口、`K0/K1/K2`、小 KL 阈值和事件不足出口；
- P3 的安全、资源、target 排序与反策略冻结门；
- DS-TR v0 的半径、回溯、exact fallback 和禁止扩展项。

验收检查已通过：四个产物存在，CSV/JSON 可解析，源文件哈希已冻结，没有训练、
策略代码或实验结果修改。下一任务为 DST-02。
