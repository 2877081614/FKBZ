# LR-03：多约束安全 RL 梯度塑形论文阅读任务

任务状态：`PASSED`  
优先级：P0  
建议用时：3 小时  
实验授权：否  
建议前置：LR-02

## 1. 论文身份

标题：*Gradient Shaping for Multi-Constraint Safe Reinforcement Learning*  
作者：Yihang Yao、Zuxin Liu、Zhepeng Cen、Peide Huang、Tingnan Zhang、
Wenhao Yu、Ding Zhao  
来源：L4DC 2024，PMLR 242  
官方页面：<https://proceedings.mlr.press/v242/yao24a.html>

算法名称：GradS。

## 2. 选择理由

项目多次出现：

- 降低高威胁突防却提高资源成本；
- 减少 false-noop 却增加 wasteful engagement；
- 一个场景安全改善而另一个场景方向反转。

这些现象可能不是单一估值器问题，而是多个约束梯度冗余、协同或冲突。GradS
直接从多目标优化角度分析多约束 safe RL，因此可用于判断项目是否应停止把安全
和资源压缩为单一标量。

## 3. 核心阅读问题

1. 论文怎样区分 redundant、aligned 和 conflicting constraints？
2. 原始 Lagrangian 梯度在多约束下为什么可能低效或方向错误？
3. GradS 对梯度做了什么几何变换？
4. 该变换是否保持原 CMDP 可行性语义？
5. 方法依赖准确 cost critic 吗？
6. 若约束只在少数临界状态违反，期望梯度是否会掩盖局部失败？
7. AirDefense 的损伤、突防、资源成本应是三个约束还是目标与约束的组合？

## 4. 必读部分

- Multi-constraint safe RL formulation；
- MOO 统一框架；
- 梯度关系分类；
- GradS 算法与理论性质；
- constraint dimension 扩展实验；
- 与 Lagrangian、CPO 类方法的消融；
- 局限性。

## 5. 必须重建的内容

报告必须重写：

1. 多约束优化目标；
2. reward gradient 与各 constraint gradient；
3. GradS 的梯度变换；
4. 可行性或收敛性质所需条件。

必须绘制 AirDefense 梯度关系假设图：

```text
奖励梯度
├─ 损伤约束梯度
├─ 高威胁突防约束梯度
└─ 资源成本约束梯度
```

对每一对关系标记“可能一致/可能冲突/现有证据不足”，不得伪造实际梯度数据。

## 6. 项目压力测试

至少对照：

- [资源约束交战边界](../../experiments/air_defense_v1_task14_engagement_calibration.md)；
- [状态条件交战价值](../../experiments/air_defense_v1_task14_state_conditioned_value.md)；
- [BPCE 压力测试](../../experiments/air_defense_v1_bpce_ppo_stress_test.md)。

必须回答：

- 现有安全—资源冲突能否被 GradS 问题定义覆盖？
- GradS 解决的是目标语义错误，还是在语义已正确时改善优化？
- 当前项目是否具备可训练的独立 cost critics？
- 若标签跨批次不稳定，梯度塑形是否只会塑造错误梯度？
- GradS 应作为强基线、可迁移组件还是 no-go 邻近工作？

## 7. 交付物

```text
docs/literature/algorithm_innovation_reading/lr_03_gradient_shaping_multi_constraint_safe_rl.md
```

必须包含：

- 多约束公式卡；
- 梯度冲突类型矩阵；
- 项目三类约束的语义候选表；
- 使用前必须满足的前置条件；
- `BASELINE / ADAPT / AVOID / OPEN` 判决。

## 8. 通过条件

- 能区分“约束定义错误”和“正确约束间梯度冲突”；
- 不把 GradS 当作自动修复 all-noop 的工具；
- 明确指出项目缺少哪些 cost-value 证据；
- 给出至少一个该方法在本项目中可能失败的具体机制。

## 9. 禁止结论

- 不把多头 loss 称作多约束优化；
- 不用事后权重调整代替约束定义；
- 不假设平均约束满足能保证每个高威胁状态安全；
- 不从概念映射直接产生在线算法任务。

## 10. 移交

结果与 LR-02 合并，为后续头脑风暴提供“规范目标层”的现有方法边界。

## 11. 执行结果

完成时间：2026-07-29  
交付物：
[GradS 多约束梯度塑形与 AirDefense 规范目标边界](../../literature/algorithm_innovation_reading/lr_03_gradient_shaping_multi_constraint_safe_rl.md)

验收结果：

- [x] 从 PMLR 正式页面、正式 PDF 与 arXiv 核对论文身份；
- [x] 重建多约束 CMDP、Lagrangian、cost gradients 和 GradS 变换；
- [x] 区分冲突、冗余、独立及一般同向约束；
- [x] 说明 Theorem 4 是非零邻域梯度上界，不是逐次可行性保证；
- [x] 完成 AirDefense 梯度关系假设图，未伪造实际梯度数据；
- [x] 对照资源边界、状态条件双价值、BPCE 和跨批次失败证据；
- [x] 明确当前缺少合格的独立 on-policy cost critics；
- [x] 给出使用前置条件及 `BASELINE / ADAPT / AVOID / OPEN` 判决；
- [x] 未修改算法、未启动实验、未下载或运行外部代码。
