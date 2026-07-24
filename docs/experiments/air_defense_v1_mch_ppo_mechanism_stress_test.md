# AirDefense v1 MCH-PPO 机制压力实验

更新时间：2026-07-22  
实验状态：已完成  
结论：最小 MCH-PPO 未通过机制门控，不进入 30k/100k 正式实验

## 1. 实验目的

本实验按用户要求停止继续堆叠外围前置任务，直接实现可训练的 MCH-PPO 探索版。实验只检验：在相同因子化策略结构下，冻结层级 Q-Critic 提供的掩码反事实信用与两层独立 PPO clipping，能否稳定改善 no-op 塌缩、高威胁突防和资源效率。

本实验不用于证明 MCH-PPO 的普遍优越性。

## 2. 实现方法

`MaskedCounterfactualHierarchicalPPO` 已实现以下机制：

- 沿用 factorized engagement-target 自回归策略；
- 每轮 PPO 更新前冻结 rollout 旧策略，精确重建两层旧 log-prob；
- 使用任务十四 `hierarchical_seed14/15/16` 三个冻结 Critic 的反归一化集成预测；
- 根据实际动作前缀重建动态合法目标集合与目标占用；
- 构造逐单元 engagement advantage 和 conditional target advantage；
- 对 engagement ratio 和 target ratio 独立裁剪；
- no-op 不产生 target actor loss，非法目标不进入 baseline；
- 联合 GAE 仅训练状态价值头，联合 KL 用于更新监控；
- Critic 参数冻结，不参与 PPO 反向传播。

对照方法与候选使用相同策略头、网络规模、动作顺序、训练预算和 PPO 超参数，唯一主要差异是 actor 信用与更新目标。

## 3. 冻结协议

| 项目 | 配置 |
| --- | --- |
| 对照 | `factorized_engagement_ar_ppo_order_012` |
| 候选 | `mch_ppo_order_012` |
| 训练场景 | `time_pressure`、`heterogeneity_pressure` |
| 评估场景 | 两个场景完整交叉评估 |
| 训练种子 | `8、9、10` |
| 训练预算 | 每模型 10k steps |
| PPO epochs | 2 |
| 最终评估 | 每场景 30 episodes |
| 模型数 | 12 |
| 场景评估块 | 24 |

种子和场景在读取 MCH 结果前已经冻结，没有根据结果更换种子。

## 4. 同场景配对结果

以下差值均为 `MCH-PPO - factorized PPO`。负的高威胁突防率、损伤和 all-noop 差值更好，正的奖励差值更好。

### 4.1 time_pressure

| seed | all-noop 差 | 高威胁突防率差 | 资源成本差 | 奖励差 | 损伤差 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | +1.0000 | +0.2833 | -17.0000 | -41.6401 | +1.0409 |
| 9 | 0.0000 | -0.0962 | -0.4167 | +26.7294 | -0.4588 |
| 10 | 0.0000 | +0.1212 | +0.0500 | -15.3631 | +0.3010 |
| 均值 | +0.3333 | +0.1028 | -5.7889 | -10.0913 | +0.2944 |

seed9 同时改善奖励、高威胁突防和损伤，说明该机制在特定训练轨迹下存在有效可能；但 seed8 直接变为 `all-noop=1.0`，seed10 也退化，因此不能把 seed9 单独作为优势证明。

### 4.2 heterogeneity_pressure

| seed | all-noop 差 | 高威胁突防率差 | 资源成本差 | 奖励差 | 损伤差 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 9 | +0.3667 | +0.2708 | -12.7533 | -25.6164 | +0.8566 |
| 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 均值 | +0.1222 | +0.0903 | -4.2511 | -8.5388 | +0.2855 |

seed8 和 seed10 的两种方法均为全 no-op，所谓“相对非劣”只是双方同时失败；seed9 中 MCH 又把原本高交战策略推向低交战，并显著增加突防和损伤。异质场景没有出现机制收益。

## 5. 门控结论

| 门控 | 结果 |
| --- | :---: |
| 结构违规为零 | 通过 |
| 每场景至少 2/3 种子 all-noop 相对非劣 | 通过 |
| 候选绝对无塌缩 | 失败，3/6 场景种子塌缩 |
| 至少一个场景高威胁突防率均值下降 | 失败 |
| 资源成本不超过对照 110% | 通过 |
| 奖励/损伤无灾难性退化 | 失败 |
| 总门控 | **失败** |

MCH 的低资源成本主要来自不交战，不能解释为资源效率提升。两个场景的高威胁突防率平均分别增加 `0.1028` 和 `0.0903`，损伤平均分别增加 `0.2944` 和 `0.2855`。

## 6. 计算代价

六个同场景训练组合中：

- factorized PPO 平均训练时间：`73.84 s/model`；
- MCH-PPO 平均训练时间：`113.86 s/model`；
- MCH 训练时间约为对照的 `1.54x`，增加约 `54.2%`。

当前实现会在每个 PPO epoch 重复计算冻结 Critic 的候选动作价值。若后续修订版重新进入扩大实验，应先实现 rollout 级反事实优势缓存和候选向量化。

## 7. 科学结论

第一版 MCH-PPO 已经从概念和离线诊断进入真实在线策略优化，但当前证据不支持其稳定优势。失败模式说明：冻结离线 Critic 的相对信用即使在单批次上可辨，也不能保证在 PPO 访问的新状态分布中保持正确；把该信用完全替代 joint GAE 后，策略仍会沿随机种子分叉到全 no-op 或高交战极端。

因此：

1. 不能选择 `time_pressure/seed9` 单独证明 MCH-PPO 优势；
2. 不能进入 30k/100k 并把算力扩大当作修复；
3. 当前 MCH-PPO 是“已实现但机制未成立”的研究原型，不是已确认创新成果；
4. 下一版必须改变信用接入机制，例如可靠度门控、与 on-policy GAE 的受控组合或在线校准，而不是继续追加随机种子寻找正结果；
5. GNN 仍是后续结构创新方向，但不能建立在当前不稳定的信用目标上。

## 8. 实验产物

```text
rein_learning/algorithms/policy_gradient/mch_ppo.py
scripts/run_air_defense_v1_mch_ppo_stress_test.py
tests/test_mch_ppo.py
results/air_defense_v1/mch_ppo_mechanism_stress_test/
results/air_defense_v1/mch_ppo_mechanism_stress_test/runs.csv
results/air_defense_v1/mch_ppo_mechanism_stress_test/mch_stress_summary.json
results/air_defense_v1/mch_ppo_mechanism_stress_test/mch_stress_summary.md
```
