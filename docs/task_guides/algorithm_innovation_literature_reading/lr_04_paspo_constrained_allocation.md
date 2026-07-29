# LR-04：约束分配自回归策略优化论文阅读任务

任务状态：`PASSED`  
优先级：P0  
建议用时：3–4 小时  
实验授权：否

## 1. 论文身份

标题：*Autoregressive Policy Optimization for Constrained Allocation Tasks*  
作者：David Winkel、Niklas Alexander Strauß、Maximilian Bernhard、
Zongyue Li、Thomas Seidl、Matthias Schubert  
来源：NeurIPS 2024  
官方页面：<https://openreview.net/forum?id=hRKsahifqj>  
代码入口：<https://github.com/niklasdbs/paspo>

算法名称：PASPO。

## 2. 选择理由

该论文与项目 Task 8–12 的问题结构高度接近：有限资源按顺序分配给多个实体，
每一步必须满足硬约束。项目已经证明自回归采样可以消除冲突，但固定顺序会造成
性能与资源偏置。PASPO 明确提出去除 sequential sampling 初始偏置的机制，是
当前最重要的结构性邻近工作。

## 3. 核心阅读问题

1. PASPO 的 constrained allocation MDP 如何定义？
2. 自回归动作如何保证线性硬约束？
3. initial bias 的精确定义是什么？
4. 去偏机制修正策略分布、梯度还是采样顺序？
5. 去偏是否依赖连续分配或凸多面体结构？
6. 项目的离散 unit-target matching 是否满足其数学条件？
7. Task 10 的固定顺序代价能否被 PASPO 机制直接解决？

## 4. 必读部分

- Problem formulation；
- 自回归可行分配构造；
- de-biasing mechanism；
- policy objective 与梯度；
- 三个 allocation benchmark；
- 约束违反、性能和顺序消融；
- 代码中的 action construction 和 mask 逻辑。

## 5. 必须重建的公式与流程

报告必须给出：

1. 联合分配的自回归分解；
2. 每一步可行集合更新；
3. 初始偏置的来源；
4. 去偏公式或伪代码；
5. 训练和推理时的复杂度。

并与项目动作结构对照：

| PASPO | AirDefense v1 |
| --- | --- |
| allocation entity | 防御单元或目标 |
| resource constraint | 弹药、目标唯一占用 |
| continuous/linear allocation | 离散 unit-target assignment |
| autoregressive order | 012/120/201 |
| debiasing target | 顺序/可行域引起的分布偏置 |

## 6. 项目压力测试

至少对照：

- [Task 8 无冲突联合动作](../../experiments/air_defense_v1_task8_screening.md)；
- [Task 9 自回归策略](../../experiments/air_defense_v1_task9_screening.md)；
- [Task 10 顺序诊断](../../experiments/air_defense_v1_task10_order_diagnostics.md)；
- [Task 11 角色条件策略](../../experiments/air_defense_v1_task11_role_conditioned_screening.md)。

必须判断：

- PASPO 是否已经覆盖“自回归约束分配”这一创新叙事；
- 去偏机制能否处理异质资源的固定顺序影响；
- 它是否处理 all-noop/engage 边界；
- 它能否保证资源成本与安全，而不仅是当前动作合法；
- 若作为强基线，最小可比版本是什么。

## 7. 交付物

```text
docs/literature/algorithm_innovation_reading/lr_04_paspo_constrained_allocation.md
```

必须包含：

- 自回归去偏公式卡；
- PASPO—Task 8/9/10/11 五层差异矩阵；
- 可迁移接口和数学不适用点；
- 强基线判决；
- `BASELINE / ADAPT / AVOID / OPEN` 判决。

## 8. 通过条件

- 能准确说明“硬约束可行”与“未来资源负责”之间的差别；
- 能判断 PASPO 去偏是否针对项目观察到的顺序问题；
- 不因两者都使用 autoregressive 就宣称等价；
- 给出复现所需的最小算法要素，但不启动实现。

## 9. 禁止结论

- 不把自回归动作本身称作项目创新；
- 不把 PASPO 的线性/连续约束结论直接外推到离散 WTA；
- 不假设顺序去偏能解决交战信用；
- 不下载或运行外部代码。

## 10. 移交

结果移交 LR-05。LR-05 必须在理解 PASPO 的顺序偏置语义后，区分“动作生成偏置”
与“顺序信用估计偏差”。

## 11. 执行结果

完成时间：2026-07-29  
交付物：
[PASPO 约束分配、自回归初始化偏置与 AirDefense 适用边界](../../literature/algorithm_innovation_reading/lr_04_paspo_constrained_allocation.md)

验收结果：

- [x] 从官方 OpenReview、NeurIPS 正式 PDF 和官方仓库核对论文身份；
- [x] 重建自回归分解、LP 可行区间、Beta 条件策略和去偏初始化；
- [x] 区分当前硬约束合法、累计资源约束和未来资源责任；
- [x] 完成 PASPO—Task 8/9/10/11 五层差异矩阵；
- [x] 明确 PASPO 不处理离散 WTA、状态依赖合法集和 all-noop；
- [x] 给出初始化-only 的离散可行后缀计数强基线；
- [x] 给出 `BASELINE / ADAPT / AVOID / OPEN` 判决；
- [x] 未下载或运行外部代码，未启动实现或实验。
