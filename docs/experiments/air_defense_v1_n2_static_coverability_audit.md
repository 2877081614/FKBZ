# AirDefense v1 N2 静态未来可覆盖性审计

更新时间：2026-07-29。  
数据性质：冻结 R2 开发状态重放。  
新增反事实 rollout：0。  
Actor 更新：0。  
结论：**N2-E1，允许建立 paired predictive validation；不允许在线训练。**

## 1. 协议

使用 R2 的：

- 场景：`medium`、`time_pressure`、`heterogeneity_pressure`；
- 策略种子：17、18、19；
- 每场景—种子 12 个 context；
- 总计 108 个冻结 context。

脚本按原种子与模型重放快照，并逐项检查：

```text
context_id
observation_hash
unit_index
legal_targets
```

108/108 全部匹配。审计只计算每个被测单元在原自回归前缀下的合法目标，
没有重新选择高 FCRC 状态。

## 2. 执行

```powershell
conda run -n rein-learning python `
  scripts/analyze_air_defense_v1_n2_static_coverability.py `
  --software-tests-passed `
  --literature-gate-passed
```

输出：

```text
results/air_defense_v1/n2_static_coverability_audit/
  experiment_config.json
  context_identity_check.csv
  action_certificates.csv
  context_summary.csv
  gate_summary.json
```

## 3. 总体结果

| 指标 | 结果 |
| --- | ---: |
| context | 108 |
| 合法动作行 | 243 |
| 正 FCRC 动作 | 86 |
| 正外部性比例 | 35.39% |
| 有目标间责任跨度的 context | 34 |
| FCRC—单元成本 Spearman | 0.466 |
| FCRC—目标权重 Spearman | −0.128 |
| FCRC—N1 替代量 Spearman | 0.479 |
| 平均计算时间 | 1.02 ms/context |
| 最大计算时间 | 5.47 ms/context |

## 4. 分组结果

| 场景 | 资源 | 合法动作 | 正外部性比例 | 平均外部性 |
| --- | --- | ---: | ---: | ---: |
| medium | laser | 34 | 5.88% | 0.0063 |
| medium | missile | 47 | 44.68% | 0.0622 |
| time pressure | laser | 36 | 19.44% | 0.0218 |
| time pressure | missile | 35 | 40.00% | 0.0297 |
| heterogeneity pressure | laser | 33 | 18.18% | 0.0191 |
| heterogeneity pressure | missile | 58 | 62.07% | 0.1521 |

责任差异在异质压力场景最常见：14/36 context；time 为 11/36，medium
为 9/36。该顺序与“不可替代资源关系增加外部性”的机制一致，但不能替代
正式预测检验。

## 5. 命题判定

| 命题 | 判定 | 证据 |
| --- | --- | --- |
| N2-P1 形式—实现一致 | 通过 | 9 项人工轨迹测试 |
| N2-P2 信号非退化 | 通过 | 34 个跨度 context；35.39% 正动作 |
| N2-P3 不是成本/威胁换名 | 通过 | 两个绝对相关均远低于 0.90 |
| N2-P4 静态计算可用 | 通过 | 平均 1.02 ms，最大 5.47 ms |
| N2-P5 创新距离 | 有条件通过 | 未发现公式等价工作；与 WTA/shield 邻近 |

## 6. 解释边界

该审计证明：

- FCRC 在当前环境中不是常数；
- 它能区分同一单元的不同合法目标；
- 它不是直接成本或目标威胁的简单换名；
- 计算预算足以支持后续离线成对验证。

该审计没有证明：

- 高 FCRC 一定导致更高未来损伤；
- 把 FCRC 加入 PPO 会改善性能；
- 当前线性运动和一次覆盖证书能处理观测不确定性或未来波次；
- FCRC 优于一般 reachability/shield；
- FCRC 已经构成论文算法创新。

## 7. 阶段出口

五项门控均达到进入下一证伪阶段的条件，因此出口为：

```text
N2-E1_enter_frozen_paired_predictive_validation
```

下一任务只允许建立一次冻结 paired continuation，检验 FCRC 对其他威胁
覆盖下降和条件损伤的增量预测价值。在线训练继续保持未授权。

