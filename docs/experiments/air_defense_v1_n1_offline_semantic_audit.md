# AirDefense v1 N1 离线语义审计

更新时间：2026-07-28。  
实验性质：开发性、零新增 rollout、零在线训练。  
阶段结论：**N1-E4；在线训练未授权。**

## 1. 目的

本审计不检验 PPO 性能，而检验四个前置问题：

1. R2 三分量替代账本能否重建回合成本差；
2. 回合标量作为局部成本标签的含混程度有多大；
3. 候选接口能否保持严格零系数退化；
4. 在看任何新训练结果前，创新距离是否足以授权上线。

输入只来自冻结的 R2 独立确认结果：
`results/air_defense_v1/action_substitution_confirmation/`。旧数据被明确标记为
开发数据，不承担后续算法的正式独立确认。

## 2. 执行与可复现性

分析脚本：
`scripts/analyze_air_defense_v1_n1_offline_semantic_audit.py`

执行命令：

```powershell
conda run -n rein-learning python `
  scripts/analyze_air_defense_v1_n1_offline_semantic_audit.py `
  --software-tests-passed
```

软件测试：

```powershell
conda run -n rein-learning python -m pytest `
  tests/test_identifiable_resource_credit.py `
  tests/test_action_substitution_confirmation.py `
  -q -p no:cacheprovider
```

结果：`12 passed in 27.51s`。

## 3. 主要结果

| 指标 | 结果 |
| --- | ---: |
| context 数 | 108 |
| 目标账本行数 | 7,776 |
| 最大扩展恒等式误差 | `8.881784197001252e-16` |
| 最大直接成本误差 | `0.0` |
| 含混账本行数 | 4,731 |
| 含混账本比例 | 60.84% |
| 总替代量可靠为正的 context 比例 | 86.11% |
| 回合成本差非正的 context 比例 | 37.04% |
| 平均符号掩盖率 | 67.45% |
| 多数账本被掩盖的 context 比例 | 72.22% |
| 平均替代比例 | 69.27% |

这里“含混”指直接成本为正，而回合成本差小于等于零。它不是数据错误，而是
局部标签无法从标量符号中唯一读出的操作性定义。

## 4. 分组边界

| 分组 | 总替代可靠为正 | 回合差非正 | 平均掩盖率 |
| --- | ---: | ---: | ---: |
| laser | 85.71% | 50.00% | 68.92% |
| missile | 86.54% | 23.08% | 65.87% |
| heterogeneity pressure | 97.22% | 44.44% | 80.99% |
| medium | 86.11% | 30.56% | 63.37% |
| time pressure | 75.00% | 36.11% | 57.99% |

同一步其他单元替代只在 15.74% 的 context 中非零，但这仍足以说明
future-only 恒等式不完整。未来被测单元替代可靠为正的比例为 79.63%，
是总替代的主要来源。

## 5. N1 命题判定

| 命题 | 判定 | 依据 |
| --- | --- | --- |
| N1-P1 标签语义可计算 | 通过 | 7,776 行恒等式误差近机器精度 |
| N1-P2 有可辩护的算法差异 | **失败** | A 邻近已有分解且目标未定；B 已覆盖；C 有偏差风险 |
| N1-P3 严格 fallback 契约 | 通过 | 人工轨迹和零系数梯度测试 |
| N1-P4 开发数据边界 | 通过 | 只读 R2，未伪装成独立验证 |
| N1-P5 在线停止规则冻结 | 通过 | 机器可读 no-go 预注册 |

全部命题未通过，因此不得进入 N1-E1。候选 A 虽有精确语义和软件接口，
但只达到“方法组件”级别；由于三条路线均不能同时保持目标一致性和创新
距离，最终出口按任务文档判为 **N1-E4**，而不是 N1-E2。

## 6. 输出

```text
results/air_defense_v1/n1_offline_semantic_audit/
  experiment_config.json
  label_dictionary.json
  seed_usage_audit.json
  support_summary.csv
  candidate_comparison.csv
  gate_summary.json
```

`gate_summary.json` 是本阶段的权威机器可读判决。任何后续脚本都必须检查
`online_training_authorized=false`，不得用命令行开关绕过。

