# LR-02：可扩展安全多智能体约束优化论文阅读任务

任务状态：`PASSED`  
完成时间：2026-07-29  
交付物：[LR-02 阅读报告](../../literature/algorithm_innovation_reading/lr_02_scalable_constrained_mappo.md)  
优先级：P0  
建议用时：3–4 小时  
实验授权：否

## 1. 论文身份

标题：*Scalable Constrained Policy Optimization for Safe Multi-agent Reinforcement Learning*  
作者：Lijun Zhang、Lin Li、Wei Wei、Huizhong Song、Yaodong Yang、Jiye Liang  
来源：NeurIPS 2024  
官方页面：<https://proceedings.neurips.cc/paper_files/paper/2024/hash/fa76985f05e0a25c66528308dda33de0-Abstract-Conference.html>

算法名称：Scal-MAPPO-L。

## 2. 选择理由

项目 N1 已判定“全局 CMDP 约束”是必须存在的强基线，而非创新。该论文进一步
处理安全 MARL 中全局约束耦合、联合状态动作空间增长和顺序局部更新问题，可以
帮助确定：

- 全局资源预算怎样成为规范目标；
- 局部单元更新怎样与全局安全保证关联；
- 后续算法至少需要对照什么强基线。

## 3. 核心阅读问题

1. 论文中的全局 reward 和 safety constraints 如何定义？
2. 截断 advantage 的上下界如何导出局部更新目标？
3. \(\kappa\)-hop policy 的局部性假设是什么？
4. 顺序更新如何保证联合策略改进和约束满足？
5. Lagrangian 乘子是全局、局部还是按智能体维护？
6. 该理论适用于集中式自回归动作因子，还是只适用于多个独立智能体？
7. 项目把三个防御单元当作 agent 会改变当前集中式问题吗？

## 4. 必读部分

- Problem formulation；
- trust-region 和 truncated advantage bounds；
- sequential update theorem；
- Scal-MAPPO-L 实际算法；
- 复杂度/可扩展性分析；
- 安全约束实验与消融；
- 与 CPO、MAPPO-L、HAPPO/HATRPO 的比较。

## 5. 必须重建的公式

报告必须重写：

1. CMDP/MACPO 的优化目标与约束；
2. 单智能体或局部策略更新的 surrogate；
3. 截断 advantage 上下界；
4. 联合策略安全/改进结论中的关键条件。

每个公式旁标注 AirDefense v1 对应量：

```text
task reward        → 奖励/损伤/截获
safety cost        → 资源成本或高威胁突防
agent              → 防御单元或策略因子
local neighborhood → 哪一种资源—目标关系邻域
```

## 6. 项目压力测试

至少对照：

- [Task 7 正式 100k](../../experiments/air_defense_v1_task7_formal_100k.md)；
- [N1 离线语义审计](../../experiments/air_defense_v1_n1_offline_semantic_audit.md)；
- [RG-MCH 压力测试](../../experiments/air_defense_v1_rg_mch_ppo_stress_test.md)。

必须完成：

| 问题 | 输出 |
| --- | --- |
| Scal-MAPPO-L 能否作为本项目强基线 | 明确判定 |
| 资源成本适合作为期望累计约束吗 | 适用条件与反例 |
| 高威胁突防是目标还是约束 | 两种建模的后果 |
| 局部责任是否为安全保证所必需 | 论文证据与项目证据分别说明 |
| 集中式 factorized PPO 如何公平对照 | 最小接口草图，只读 |

## 7. 交付物

```text
docs/literature/algorithm_innovation_reading/lr_02_scalable_constrained_mappo.md
```

必须包含：

- 全局—局部约束映射图；
- 理论假设清单；
- 与 N1 候选 B 的重合判定；
- 强基线最小接口说明；
- `BASELINE / ADAPT / AVOID / OPEN` 判决。

## 8. 通过条件

- 能说明全局约束与局部责任不是同一问题；
- 能指出顺序更新保证依赖的全部关键条件；
- 能判断该方法作为基线时需要哪些公平性控制；
- 不以“防空领域应用”作为算法差异。

## 9. 禁止结论

- 不把 Lagrangian 资源约束称作项目创新；
- 不假设单元等同独立 agent；
- 不把期望成本满足等同每状态都安全；
- 不提出实现任务或训练预算。

## 10. 移交

结果移交 LR-03，用于判断多个安全/资源约束的梯度冲突是否是项目的新问题，还是
现有多约束 RL 已覆盖的标准问题。

## 11. 执行结果

LR-02 已按 NeurIPS 2024 官方 33 页版本完成全文公式、附录、实验图与局限性
核验，并与 Task 7、N1 和 RG-MCH 三份项目正式证据逐项对照。

验收结论：

- 已纠正“一个全局安全成本被分解为局部责任”的误读；论文实际使用联合奖励和
  每 agent 多个局部期望累计成本约束；
- 已重建 CMDP 目标、顺序 surrogate、截断 advantage 界、奖励/成本界和
  Theorem 3.7 的成立条件；
- 已记录 Proposition 3.3 正文/附录的 \(\eta\) 常数不一致，以及
  \(\zeta<2/\gamma\) 与几何级数收敛条件之间的待澄清点；
- 已区分 Eq. 16 理论更新与实际 PPO 近似 Scal-MAPPO-L；
- 已判定 N1 候选 B 与 CMDP/Lagrangian 强基线高度重合；
- 已给出当前集中式 factorized PPO 的公平强基线最小接口；
- 已给出 `BASELINE / ADAPT / AVOID / OPEN` 判决；
- 未提出算法实现、在线训练或预算扩展。

因此本任务满足第 8 节全部通过条件，状态记为 `PASSED`。
