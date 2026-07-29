# AirDefense v1 N3：FCRC 冻结成对预测验证

更新时间：2026-07-29  
任务状态：已完成  
阶段出口：N3-E3  
在线训练授权：否

## 1. 研究问题

N2 的静态审计证明 FCRC 可计算且非退化，但没有证明它对应真实策略延续
下的其他威胁损害。N3 在观察结果前冻结检验：

> 同一状态、同一单元中，高 FCRC 合法目标是否比低 FCRC 合法目标造成
> 更大的其他威胁截获权重损失，并在静态基线之外提供增量预测价值？

主结局是排除当前目标后的其他威胁截获权重损失；泄漏损伤增量为次结局。

## 2. 冻结协议

- 场景：`medium`、`time_pressure`、`heterogeneity_pressure`；
- 来源策略种子：17、18、19；
- 状态基准种子：1,483,000；
- 分支基准种子：1,493,000；
- 每区组先采集 12 个上下文，仅按 FCRC 跨度选最多 4 个；
- high/low 目标按最大/最小 FCRC 和目标索引确定；
- 每个上下文 64 次共同随机数重复；
- 每次包含 no-op、high-FCRC、low-FCRC 三分支；
- Actor/Critic 不更新，reward、loss、mask 不修改。

正式选择在任何结果分支执行前写入 `context_selection.csv`，并以 SHA-256
`0295c86922355e42587502b89de7b4eb8a9552be09456d99c4ed551c5890615d`
冻结。

## 3. 样本完整性

| 项目 | 结果 |
| --- | ---: |
| 选择上下文 | 32 |
| 场景—种子区组 | 9/9 |
| 每区组上下文 | 3–4 |
| 已排除历史状态哈希 | 135 |
| 与历史状态重叠 | 0 |
| 每上下文重复 | 64 |
| 新增 transition | 82,219 |
| Actor 更新 | 0 |
| Actor 参数变化 | 0 |
| 全量软件回归 | 283 passed |

同一个全新环境状态可以为不同防御单元形成不同的 unit-level context，因此
32 个上下文对应 26 个不同 observation hash。这不违反“与历史状态零重叠”
的预注册要求。

## 4. 主结果

| 指标 | 观察值 | 预注册门槛 | 结果 |
| --- | ---: | ---: | --- |
| 平均 high-low 截获损害差 | 0.018402 | >0 | 方向为正 |
| 单侧符号翻转 p | 0.351065 | <0.05 | 失败 |
| FCRC—候选截获损害 Spearman | 0.415306 | >0.25 | 通过 |
| 基线留一区组 CV MAE | 0.137041 | — | — |
| 加 FCRC 后 CV MAE | 0.137041 | — | — |
| CV MAE 相对降幅 | 约 0% | ≥5% | 失败 |
| 正方向场景数 | 2/3 | ≥2/3 | 通过 |
| 平均 high-low 泄漏损伤差 | −0.050011 | ≥0 | 失败 |

分场景平均 high-low 截获损害差：

- medium：0.009930；
- time pressure：0.123713；
- heterogeneity pressure：−0.076743。

## 5. 判决

- N3-P1 独立性与完整性：通过；
- N3-P2 因果方向：失败；
- N3-P3 增量预测价值：失败；
- N3-P4 跨场景与安全一致性：失败；
- N3-P5 执行完整性：通过。

按预注册出口优先级，正式出口为：

```text
N3-E3_reject_predictive_proposition
```

FCRC 的候选级秩相关说明它仍可作为描述资源占用结构的静态解释量，但它
没有通过预注册的成对因果方向和增量预测门槛。项目因此停止 FCRC 算法化
路线，不把它加入 reward、loss、action mask、shield 或 GNN。

## 6. 门控修正记录

首次汇总代码额外要求本批 observation hash 彼此唯一，导致 P1 被误报为
失败。预注册只要求与旧批次零重叠，并允许同一状态的不同单元形成不同
上下文。修正删除了这一未预注册条件，仅将 P1 从 false 改为 true，并按
既定优先级把出口从 E4 改为 E3。

修正没有：

- 重选上下文；
- 重跑任何反事实分支；
- 改变统计量、特征或门槛；
- 删除场景、种子或观测；
- 追加样本。

## 7. 复现入口

```powershell
conda run -n rein-learning python -m pytest -q `
  tests/test_fcrc_predictive_validation.py `
  tests/test_future_coverability.py

conda run -n rein-learning python `
  scripts/run_air_defense_v1_fcrc_paired_predictive_validation.py `
  --device cpu `
  --software-tests-passed
```

正式结果已存在时，脚本会拒绝无预注册重跑。

## 8. 结果文件

```text
configs/air_defense_v1/n3_fcrc_paired_predictive_preregistration.json
configs/air_defense_v1/n3_stage_gate.json
results/air_defense_v1/fcrc_paired_predictive_validation/
├── context_selection.csv
├── selection_freeze.json
├── repeat_paired_outcomes.csv
├── candidate_effects.csv
├── context_effects.csv
├── actor_integrity.json
├── gate_summary.json
└── gate_correction.json
```
