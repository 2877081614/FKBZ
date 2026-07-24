# 下一研究阶段：跨批次统一概率校准与不确定性约束

更新时间：2026-07-22
执行状态：已完成，OOB主门控未通过，独立批次未生成
所属阶段：任务十四离线 Critic 最终修订
前置任务：冻结 OOB 校准协议的独立确认

## 1. 阶段目标

上一阶段证明原始连续 score 虽在历史 OOB 中存在可行阈值，但种子级固定阈值在唯一
独立批次中为 `0/3`。三个模型的 safety sign accuracy 仍为 `0.740-0.753`，因此问题
不是安全价值完全无信息，而是 score 尺度、批次漂移和安全-资源折中没有统一语义。

本阶段一次完成 MCH-PPO 前剩余的离线前置工作：

1. 把不同种子、批次的价值输出映射到统一 engage 概率；
2. 使用预测方差形成显式保守置信边界；
3. 通过外层 leave-one-batch-out 冻结唯一候选；
4. 仅在 OOB 门控通过后生成一次全新独立批次；
5. 独立门控通过后直接形成 MCH-PPO 接口与就绪判定。

失败的 `eval_seed=887000` 确认批次只保留为历史审计，禁止用于拟合、候选选择、
标准化、正则化或阈值调整。

## 2. 冻结输入

- 训练语料：三个独立训练批次的 OOB 预测，共193个上下文组；
- 可靠标签：68个 engage、115个 no-op；
- 基础目标：`scenario_robust_reliable_cost`；
- 基础模型种子：20、21、22；
- 批次：701000、719000、737000；
- 场景：`medium`、`time_pressure`、`heterogeneity_pressure`；
- 基础价值模型和 balanced margin Critic 检查点全部冻结。

## 3. 统一概率校准器

每个基础模型种子独立拟合同构校准器，保留三种子门控。输入候选为：

```text
score_only:
    raw score

value_context:
    raw score
    safety prediction
    positive cost prediction
    budget multiplier
    scenario one-hot
```

所有连续特征只使用拟合批次计算 median/IQR 鲁棒标准化。校准器为带截距的 L2
逻辑回归，固定 `L2=1e-2`。样本权重按 `batch × scenario × oracle class` 块等权，
避免大批次或多数 no-op 类支配损失。

输出 logit 映射为统一 engage 概率，基础边界固定为0.5，不再扫描标量阈值。

## 4. 显式不确定性约束

由加权逻辑回归 Hessian 逆矩阵估计预测 logit 标准误 `se(x)`：

```text
conservative_logit = calibrated_logit - z * se(x)
engage iff conservative_logit > 0
```

固定候选仅有：

| 候选 | 特征 | z |
|---|---|---:|
| score_platt | score_only | 0.0 |
| value_platt | value_context | 0.0 |
| value_lcb_050 | value_context | 0.5 |
| value_lcb_100 | value_context | 1.0 |

候选表在外层验证前冻结，不根据独立批次扩充。

## 5. 外层留一批次验证

对每个候选和基础模型种子执行三折 outer leave-one-batch-out：每折只在两个批次上
拟合标准化与校准参数，在第三批次上预测。合并三个留出批次后计算：

- BA、engage recall、no-op recall；
- 最差批次 engage/no-op recall；
- 最差场景 engage/no-op recall；
- 概率 Brier score 和 log loss；
- 平均预测标准误、LCB 改判率和参数条件数。

完整约束保持：BA `>=0.70`，总体及最差批次/场景 engage `>=0.60`、no-op
`>=0.65`。候选按“可行种子数、最小约束余量、平均 BA、低复杂度”冻结；至少
`2/3` 种子可行才允许独立测试。

## 6. 唯一独立批次

仅在 OOB 门控通过后执行：

```text
eval_seed: 941000
states: 2 source seeds × 3 scenarios × 12 = 72
episodes_per_stratum: 30
rollouts_per_branch: 32
gamma: 0.98
```

候选结构和 `z` 只由 OOB 验证选择。冻结候选后，加载已经保存的全语料基础模型，
在三个训练批次上重新产生与部署尺度一致的价值特征，并使用全部训练标签拟合一次最终
校准器。这属于预注册的最终模型重拟合，不改变基础模型权重。随后在新批次上产生同构
价值特征。不得使用新标签重新拟合标准化、校准器、`z` 或决策边界。

数据功效、历史重叠和回报重构门槛与上一独立确认相同。最终要求至少2/3基础模型
种子同时满足全部总体、逐场景双类召回和 safety sign `>=0.70`。

## 7. MCH-PPO 就绪验收

若 OOB 与独立门控均通过：

- 冻结 `calibrated_probability`、`conservative_logit`、`prediction_se` 三个接口；
- 冻结校准器参数、特征顺序和鲁棒标准化统计量；
- 生成 `mch_ppo_readiness.json`，标记全部离线前置任务完成；
- 下一阶段直接进入 MCH-PPO 公式/接口冻结和 `30k × 3 seeds` 筛选；
- 不再增加新的离线校准前置任务。

若 OOB 未达到2/3：不生成新 rollout，停止当前线性统一校准路线。若独立测试未达到
2/3：新批次不回灌，MCH-PPO继续冻结，下一步必须升级显式约束表示，而不是增加候选。

## 8. 交付物

```text
docs/task_guides/next_research_phase_cross_batch_uncertainty_calibration.md
rein_learning/common/cross_batch_calibration.py
scripts/run_air_defense_v1_task14_cross_batch_calibration.py
tests/test_air_defense_v1_task14_cross_batch_calibration.py
docs/algorithms/cross_batch_uncertainty_calibration.md
docs/experiments/air_defense_v1_task14_cross_batch_calibration.md
results/air_defense_v1/task14_cross_batch_calibration/
```

## 9. 明确禁止

- 不读取 `eval_seed=809000` 或 `887000` 标签参与拟合和选择；
- 不在新独立批次上扫描阈值或新增候选；
- 不重新训练基础价值模型和 PPO Actor；
- 不修改环境、奖励、oracle 或临界状态生成协议；
- 不把线性校准包装成 MCH-PPO 创新点；
- 不实现 GNN、GAT 或 Transformer。

## 10. 执行结果

本阶段于2026-07-22完成。四个预注册候选均完成3批次外层 leave-one-batch-out，
输入 OOB 文件和基础配置哈希已留档；失败确认批次标签未被读取用于拟合。

候选排名如下：

| 候选 | 可行种子 | 平均 BA | 平均 Brier | 最小约束余量 |
|---|---:|---:|---:|---:|
| score_platt | 0/3 | 0.781 | 0.159 | -0.317 |
| value_lcb_100 | 0/3 | 0.697 | 0.263 | -0.600 |
| value_lcb_050 | 0/3 | 0.668 | 0.263 | -0.600 |
| value_platt | 0/3 | 0.672 | 0.263 | -0.650 |

`score_platt` 保持了较好的总体 BA，但 seed20/21/22 的最差批次 no-op recall 仅为
`0.550 / 0.333 / 0.475`，最差场景 no-op recall 为 `0.517 / 0.552 / 0.586`，全部
低于冻结门槛。线性 value-context 的平均预测标准误增加到 `1.235-1.797`；LCB虽然
能减少部分过度交战，却使 seed21 的最差批次 engage recall 降到0，仍不存在共同可行
边界。

因此 OOB放行条件为 `0/3 < 2/3`，正式脚本按预注册规则没有生成 `eval_seed=941000`
独立批次，也没有新增 rollout。`mch_ppo_readiness.json` 明确标记前置任务未完成，
MCH-PPO和GNN继续冻结。

机制结论是：跨批次失败不仅是单调尺度或概率校准问题。下一阶段必须把安全收益下界、
资源成本上界和停止约束保留为独立决策量，形成显式约束可行域；不再增加 Platt、LCB、
标量阈值或随机批次候选。
