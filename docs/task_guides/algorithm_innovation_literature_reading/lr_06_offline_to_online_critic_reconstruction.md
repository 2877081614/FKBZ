# LR-06：离线到在线 Critic 重构与受约束微调论文阅读任务

任务状态：`PASSED`  
优先级：P1  
建议用时：3–4 小时  
实验授权：否  
建议前置：LR-05

## 1. 论文身份

标题：*Optimistic Critic Reconstruction and Constrained Fine-Tuning for General Offline-to-Online RL*  
作者：Qin-Wen Luo、Ming-Kun Xie、Ye-Wen Wang、Sheng-Jun Huang  
来源：NeurIPS 2024  
官方页面：<https://openreview.net/forum?id=XVfevb9XFx>

## 2. 选择理由

项目 Task 14 到 SA-RG-MCH 已反复出现：

- 离线 Q-Critic 在固定批次内可排序；
- 在线 actor 访问的状态—前缀大多处于 Critic 支持域之外；
- Critic 集成方向一致仍可能共同犯错；
- 错误信用一旦接入 PPO，会被在线更新放大。

该论文把 offline-to-online 失败分为 evaluation mismatch 与 improvement mismatch，
并用 Critic reconstruction 和 constrained fine-tuning 同时处理。这与项目当前
失败机制高度对应，即使项目不是标准离线 RL，也值得作为安全接入参照。

## 3. 核心阅读问题

1. evaluation mismatch 和 improvement mismatch 分别如何定义？
2. 为什么离线 pessimistic critic 不能直接用于在线微调？
3. optimistic critic reconstruction 使用什么数据和目标？
4. critic alignment 如何依赖可靠离线 actor？
5. constrained fine-tuning 约束的是策略距离、价值还是行为分布？
6. 方法怎样逐步放松约束？
7. 项目的冻结 factorized PPO 是否可充当 reliable actor？
8. 项目问题是 offline-to-online，还是更一般的 auxiliary-critic-to-on-policy mismatch？

## 4. 必读部分

- 问题定义与两类 mismatch；
- Critic reconstruction；
- Critic alignment；
- constrained online fine-tuning；
- 与直接 online fine-tuning、policy constraint 和 value correction 的比较；
- 不同离线算法到不同在线算法的实验；
- 对失败或敏感超参数的分析。

## 5. 必须重建的机制

报告必须给出：

1. offline critic 和 online target 的差异；
2. reconstruction 目标；
3. actor–critic alignment 条件；
4. constrained fine-tuning objective；
5. 约束逐步解除或更新流程。

同时建立项目错配表：

| 论文错配 | 项目证据 | 是否等价 |
| --- | --- | --- |
| offline evaluation mismatch | Q-Critic 排序到在线状态失效 |  |
| offline improvement mismatch | MCH/BPCE 更新分叉 |  |
| behavior support shift | SA-RG 支持度 0.124/0.022 |  |
| policy constraint | joint PPO fallback / KL anchor |  |

## 6. 项目压力测试

至少对照：

- [Task 14 Q-Critic](../../experiments/air_defense_v1_task14_q_critic.md)；
- [跨批次独立确认](../../experiments/air_defense_v1_task14_independent_confirmation.md)；
- [SA-RG-MCH](../../experiments/air_defense_v1_sa_rg_mch_ppo_stress_test.md)；
- [BPCE-PPO](../../experiments/air_defense_v1_bpce_ppo_stress_test.md)。

必须回答：

- 项目是否错误地把“离线预测正确”当成“在线改进方向正确”？
- reconstruction 是否需要项目当前没有的数据？
- strict fallback 与 constrained fine-tuning 有何不同？
- 小 KL 不能防止 argmax 跨界时，论文约束是否仍有效？
- 如果 Critic 共同偏置，optimistic reconstruction 会改善还是放大风险？
- 该论文应作为机制参照、实现基线还是仅作警示？

## 7. 交付物

```text
docs/literature/algorithm_innovation_reading/lr_06_offline_to_online_critic_reconstruction.md
```

必须包含：

- 两类 mismatch 公式/流程卡；
- 论文—项目错配对照表；
- 当前 MCH/BPCE 接入协议的风险清单；
- 最低在线接入证据条件；
- `BASELINE / ADAPT / AVOID / OPEN` 判决。

## 8. 通过条件

- 能区分 critic 估值准确与 policy improvement 安全；
- 能说明方法为何不是简单 uncertainty penalty；
- 明确该论文假设与项目 on-policy PPO 的差异；
- 给出不超过五条、可被未来实验检验的接入条件；
- 不提出具体训练任务。

## 9. 禁止结论

- 不把 offline-to-online 方法原样套入项目；
- 不假设 optimistic critic 天然更安全；
- 不把 ensemble agreement 当成独立不确定性证明；
- 不以更小 KL 代替 deterministic 行为稳定性；
- 不启动 Critic 重训或在线微调。

## 10. 移交

本任务完成后，六篇阅读结果共同进入人工头脑风暴。此处只输出创新边界、强基线
和可证伪问题，不自动建立 N4 或任何在线算法任务。

## 11. 执行结果

完成时间：2026-07-29  
交付物：
[离线到在线 Critic 重构、策略对齐与受约束微调边界](../../literature/algorithm_innovation_reading/lr_06_offline_to_online_critic_reconstruction.md)

验收结果：

- [x] 从 NeurIPS 正式页面、41 页正式 PDF、arXiv HTML 和官方仓库核对论文身份；
- [x] 重建 evaluation mismatch、improvement mismatch、policy
  re-evaluation、三种 value alignment 和 CFT 公式；
- [x] 区分乐观重构、actor–critic alignment、策略距离约束和严格 optimizer
  fallback；
- [x] 核对单策略集中性、可靠离线 actor、完整转移/轨迹和历史参考策略假设；
- [x] 完成 Task 14、独立确认、SA-RG-MCH 和 BPCE-PPO 压力测试；
- [x] 比较冻结 Q、每批 COSAC ridge 和 BPCE/C3 replay 标签的三类漂移；
- [x] 给出五条可检验的最低在线接入证据条件；
- [x] 给出 `BASELINE / ADAPT / AVOID / OPEN` 判决；
- [x] 未下载或运行外部代码，未修改算法、环境或启动训练实验。
