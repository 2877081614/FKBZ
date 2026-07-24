# 跨批次统一概率校准与不确定性约束

更新时间：2026-07-22
算法状态：已实现，外层门控未通过

## 1. 研究动机

原始 state-budget score 在历史 OOB 内可以通过种子级阈值校准，但固定阈值无法泛化
到独立批次。本方法检验一个更具体的机制假设：失败是否主要由单调尺度漂移和线性价值
上下文偏差造成，以及预测置信下界能否同时控制漏交战与过度交战。

## 2. 特征与概率语义

每个基础价值种子独立校准。`score_only` 只输入原始 score；`value_context` 输入 score、
安全收益预测、正成本预测、资源乘子和场景 one-hot。连续特征使用拟合批次的 median 与
IQR 标准化。

校准器为 L2 逻辑回归：

```text
p(engage | x) = sigmoid(beta_0 + beta^T standardize(x))
```

样本按 `batch × scenario × oracle class` 分块，每个已观测块具有相同总权重。这样多数
no-op 类或较大批次不能主导拟合。

## 3. 预测不确定性

使用最终加权 Hessian 的广义逆近似参数协方差。对设计向量 `x`：

```text
se(x) = sqrt(x^T Cov(beta) x)
LCB(x) = calibrated_logit(x) - z * se(x)
engage iff LCB(x) > 0
```

预注册 `z=0/0.5/1.0`。LCB只允许把不确定 engage 改为 no-op，不会增加交战数量。

## 4. 数据隔离

候选结构使用三训练批次 OOB 预测进行外层留一批次验证。每折的标准化、逻辑回归参数
和协方差仅由两个拟合批次产生。历史独立批次只参与项目结论，不进入代码拟合路径。

只有至少2/3种子在合并 OOB 上通过总体、最差批次和最差场景双类召回，才会训练最终
校准器并生成新独立批次。

## 5. 实现接口

`rein_learning/common/cross_batch_calibration.py` 提供：

- `assemble_calibration_features`：冻结特征顺序；
- `equal_block_weights`：批次-场景-类别等权；
- `fit_cross_batch_calibrator`：鲁棒标准化与加权 IRLS；
- `FittedCrossBatchCalibrator.predict`：概率、标准误和 LCB；
- `calibrated_operating_point`：完整鲁棒门控与概率指标。

## 6. 实验结论

四个候选均为 `0/3` 可行。score-only Platt 的平均 BA 为0.781，但最差批次 no-op
recall 最低至0.333；value-context 的标准误明显增大，LCB在修复 no-op 的同时造成
seed21 最差批次 engage recall 为0。

因此当前失败不是单一温度、截距或线性上下文校准能够解决的。概率校准接口可以保留为
诊断工具，但不能作为 MCH-PPO 的可信 Critic 接口，也不构成项目创新点。
