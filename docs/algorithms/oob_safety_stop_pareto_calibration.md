# OOB 安全-停止 Pareto 校准

更新时间：2026-07-22  
适用阶段：AirDefense v1.0 任务十四离线门控修订

## 1. 目的

本方法用于判断一个已经训练完成的交战价值模型是否只是决策阈值失准，还是其连续
score 本身无法同时辨识必要交战和合理停止。它不训练新模型，也不改变环境、奖励、
oracle 或 Actor。

## 2. 输入与隔离

输入为 leave-one-batch-out 产生的 OOB 预测。每行至少包含：

- 连续交战 score；
- 可靠二分类 oracle 标签；
- `batch_id` 与场景名；
- 目标函数和模型种子。

阈值选择只使用 OOB 数据。最终独立 test 标签禁止进入候选生成、排序或门控。

## 3. 完整阈值集合

令唯一分数排序为 `z_1 < ... < z_m`。候选集合由相邻分数中点和两端外侧边界组成，
零阈值作为独立基准点额外计算。预测规则为：

```text
engage, score > threshold
no-op,  score <= threshold
```

因此候选集合覆盖该 score 能产生的全部不同二分类结果，不依赖任意等距网格。

## 4. 鲁棒可行性

每个阈值同时计算 pooled、逐批次和逐场景召回。冻结约束为：

```text
BA >= 0.70
pooled engage/no-op recall >= 0.60/0.65
每批次 engage/no-op recall >= 0.60/0.65
每场景 engage/no-op recall >= 0.60/0.65
safety sign accuracy >= 0.70
```

定义 engage、no-op、BA 和 safety sign 四类约束余量，并使用其中最小值表示阈值的
鲁棒余量。诊断阈值按“最大化最小余量、再最大化 BA、最后优先接近零”确定。

同时以“最差 engage recall、最差 no-op recall”为双目标提取非支配前沿，用于判断
二者是否存在不可调和冲突。可行性结论由完整约束集决定，不以 Pareto 图形替代门控。

## 5. 三种校准强度

1. `zero threshold`：检查原始默认边界。
2. `seed-specific robust threshold`：每个种子在自己的 OOB score 尺度上选择鲁棒阈值，
   是本阶段主判据。
3. `shared raw threshold`：所有种子共享一个原始 score 阈值，只用于检查尺度一致性。

种子级阈值可用于识别“排序有效但尺度漂移”；共享阈值失败则说明不能把不同模型
种子的未校准 score 直接视为同一物理量。

## 6. 实现接口

核心实现位于 `rein_learning/common/pareto_feasibility.py`：

- `complete_threshold_candidates`：构造完整候选集合；
- `threshold_operating_point`：计算总体和分组指标；
- `pareto_frontier_mask`：标识双目标非支配点；
- `audit_pareto_thresholds`：输出全候选、选定点、零阈值和可行区间。

正式脚本为 `scripts/run_air_defense_v1_task14_oob_pareto_audit.py`。输出包含输入哈希、
逐阈值 CSV、逐种子摘要、共享阈值诊断、配置和 JSON 门控结论。

## 7. 方法边界

本方法证明的是历史 OOB 分布上的决策边界存在性，不证明新批次泛化。尤其当可行区间
很窄或最小余量接近0时，必须进行一次完全独立确认。确认通过前，不得把校准结果解释
为 MCH-PPO 已经有效，也不得据此进入 GNN 阶段。
