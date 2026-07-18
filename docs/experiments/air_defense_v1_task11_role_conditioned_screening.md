# AirDefense v1.0 任务十一：角色条件关系动作头筛选

更新时间：2026-07-17  
训练场景：`medium`  
测试场景：`medium / time_pressure / heterogeneity_pressure`  
结论：任务十一完成；不运行条件性 100k

## 1. 研究问题

任务十表明固定顺序会改变单元参与模式，但换序无法同时降低高威胁泄漏并保持资源效率。任务十一使用参数匹配的共享 unit-target scorer 和共享 no-op scorer，检验位置独立关系归纳偏置能否：

1. 消除种子级单元塌缩；
2. 保持资源效率并降低高威胁泄漏；
3. 降低 `012 / 120 / 201` 三个生成顺序之间的性能跨度。

`012` 在实验前固定为主方法，另外两个顺序只用于鲁棒性诊断。

## 2. 实验协议与完整性

```text
训练：30,000 步，种子 0/1/2
最终评估：50 个成对回合/场景/种子
曲线：10k / 20k / 30k
诊断：episode、decision、leak attribution、参数量
统计：Student-t 95% CI
```

| 产物 | 数量 |
| --- | ---: |
| 模型 | 9 |
| 最终运行块 | 27 |
| 最终评估回合 | 1,350 |
| 决策记录 | 164,868 |
| 决策汇总 | 81 |
| 高威胁泄漏归因 | 1,418 |
| 参数记录 | 9 |

schema 7、模型签名、TensorBoard、训练曲线和泛化图均完整。所有结构指标严格为 0。

## 3. 三顺序结果

| 场景 | 顺序 | 奖励 | 总毁伤 | 高威胁泄漏率 | 资源成本 | 决策耗时/ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| medium | 012 | -60.12 | 1.832 | 0.504 | 3.89 | 2.042 |
| medium | 120 | -47.68 | 1.465 | 0.408 | 4.96 | 2.048 |
| medium | 201 | -63.63 | 1.842 | 0.476 | 4.46 | 2.073 |
| time_pressure | 012 | -77.09 | 2.242 | 0.650 | 3.39 | 1.967 |
| time_pressure | 120 | -58.74 | 1.783 | 0.536 | 5.57 | 2.415 |
| time_pressure | 201 | -86.55 | 2.337 | 0.660 | 4.42 | 2.194 |
| heterogeneity_pressure | 012 | -65.92 | 1.925 | 0.542 | 3.91 | 2.373 |
| heterogeneity_pressure | 120 | -48.91 | 1.510 | 0.404 | 6.20 | 2.443 |
| heterogeneity_pressure | 201 | -64.91 | 1.842 | 0.517 | 4.29 | 2.403 |

三种子置信区间较宽。例如异质场景 `012` 奖励 95% CI 为 `[-141.23, 9.38]`，`120` 为 `[-141.40, 43.58]`。本轮结果用于筛选研究分支，不能表述为统计显著优势。

## 4. 相对任务十主基线

主方法定义为 `role-conditioned 012 - task10 autoregressive 012`：

| 指标 | 差异 | 判定 |
| --- | ---: | --- |
| Actor 参数比例 | 0.941 | 通过 |
| medium 奖励 | +12.273 | 通过 |
| medium 总毁伤 | -0.128 | 通过 |
| time_pressure 资源成本 | -5.063 | 通过 |
| heterogeneity 高威胁泄漏下降量 | -0.00335 | **未通过** |
| 高威胁泄漏种子同向 | 1/3 | **未通过** |
| heterogeneity 总毁伤 | -0.068 | 通过 |
| heterogeneity 塌缩单元总数 | 5 | **未通过** |
| 决策耗时 | +73.51% | **未通过** |

相对原始 Maskable PPO 的六项性能/资源非劣效门槛全部通过，但不足以抵消主内部机制门槛失败。

## 5. 顺序鲁棒性

| 指标 | 观测跨度 | 门槛 | 判定 |
| --- | ---: | ---: | --- |
| 异质高威胁泄漏率 | 0.1379 | ≤0.03 | **未通过** |
| 时间压力资源成本 | 2.1767 | ≤1.00 | **未通过** |
| 异质总毁伤 | 0.4141 | ≤0.15 | **未通过** |
| 异质塌缩单元总数 | 14 | 0 | **未通过** |

共享关系评分没有使动作生成对固定顺序稳定。`120` 的均值更好，但按预注册协议不能事后将诊断顺序替换为主方法。

## 6. 单元塌缩诊断

异质场景每个“顺序 × 种子”的塌缩单元数：

| 顺序 | 种子 0 | 种子 1 | 种子 2 |
| --- | ---: | ---: | ---: |
| 012 | 2 | 3 | 0 |
| 120 | 0 | 0 | 3 |
| 201 | 3 | 2 | 1 |

主方法种子 1 的三个单元全部 no-op，异质资源成本为 0，高威胁泄漏率为 0.752。种子 2 则三个单元均参与，泄漏率为 0.390。模型结构共享并没有消除 PPO 的训练种子分叉。

池化后主方法两个导弹单元的分配率为 2.29% 和 1.04%，激光为 3.97%；只看池化结果会掩盖种子 1 的全单元塌缩，因此正式判定使用逐种子 `collapsed_unit`。

当策略实际分配目标时，主方法三个单元的平均匹配效率约为 0.939、0.967 和 0.946，说明关系 scorer 能识别较优匹配；问题主要发生在是否选择行动，而不是行动后选错目标。

## 7. 泄漏归因

异质场景池化高威胁泄漏：

| 顺序 | 泄漏数 | unassigned | prefix_denied | mismatch | attempted_miss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 012 | 173 | 164（94.8%） | 0 | 6（3.5%） | 3（1.7%） |
| 120 | 126 | 117（92.9%） | 1（0.8%） | 0 | 8（6.3%） |
| 201 | 162 | 148（91.4%） | 10（6.2%） | 0 | 4（2.5%） |

绝大多数高威胁泄漏发生在存在合法机会但策略未分配资源时。关系表示并非当前主要限制，no-op 概率和 PPO 优化稳定性才是下一层瓶颈。

## 8. 计算代价

- 单模型平均训练时间：任务十约 92.35 秒，任务十一约 128.41 秒，增加约 39%；
- 主方法平均决策时延相对任务十增加 73.51%，超过 +25% 门槛；
- 参数量没有增加，因此开销来自逐实体编码和 pair/no-op 前向计算。

## 9. 冻结决策

自动判定结果：

```text
main_internal_gate = false
external_noninferiority_gate = true
order_robustness_gate = false
confirmation_100k_eligible = false
```

因此不运行任务十一 100k × 5 独立种子确认实验，也不直接进入 GNN。

## 10. 下一步研究方向

根据任务十一预注册决策树，当前属于“仍出现种子级低交战塌缩”分支。下一阶段应优先：

1. 记录训练过程中的 no-op 概率、动作熵、policy/value loss、KL 和 advantage 分布；
2. 分析 all-no-op 吸引域何时出现，以及是否由确定性评估放大；
3. 设计不修改环境奖励的 no-op 参数化或优化稳定机制；
4. 先用短预算多种子诊断训练分叉，再讨论均衡循环顺序；
5. 暂缓 GNN，因为关系头实际匹配效率已高，主要失败发生在是否交战。

## 11. 复现入口

```powershell
conda run -n rein-learning python scripts\compare_air_defense_v1_methods.py --methods role_conditioned_ar_ppo_order_012 role_conditioned_ar_ppo_order_120 role_conditioned_ar_ppo_order_201 --train-scenario medium --eval-scenarios medium time_pressure heterogeneity_pressure --seeds 0 1 2 --timesteps 30000 --record-decisions --output-dir results\air_defense_v1 --experiment-name task11_role_conditioned_screening_30k_3seeds
conda run -n rein-learning python scripts\analyze_air_defense_v1_task11.py
conda run -n rein-learning python -m pytest tests -q
```

结果目录：`results/air_defense_v1/task11_role_conditioned_screening_30k_3seeds/`。
