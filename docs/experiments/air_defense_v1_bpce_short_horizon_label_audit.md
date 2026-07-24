# AirDefense v1 BPCE 短视窗安全—资源双分量标签审计

更新时间：2026-07-23  
实验状态：阶段 A2 已完成但未通过  
路线决策：暂停 BPCE 在线辅助主线，不进入阶段 B/C

## 1. 研究问题

阶段 A 表明完整回合标签缺少可靠 STOP 方向。本阶段检验：

> 在合法目标精确边缘化和冻结策略随机延续下，目标相关短视窗能否用
> “排除最小安全收益且确认正资源成本”识别 STOP，同时保留 ENGAGE。

短视窗只改变同一反事实轨迹的标签读出，不改变环境、策略、动作语义、
目标概率、随机带或训练参数。

## 2. 冻结协议

| 项目 | 配置 |
| --- | --- |
| 场景 | `time_pressure`、`heterogeneity_pressure` |
| 策略 | 原10k factorized joint PPO |
| 种子 | `8、9、10` |
| 上下文 | 完全复用阶段 A 的72个 |
| 每上下文重复 | 32 |
| 动作语义 | 目标精确边缘化 + stochastic continuation |
| 事件窗 | `min(remaining_steps, ceil(TTI)+1)` |
| 毁伤等效阈值 | `0.05` |
| 高威胁突防等效阈值 | `0.10` |
| Actor更新 | 禁止 |
| transition上限 | 266,198 |

标签为 `ENGAGE / STOP / AMBIGUOUS`。STOP 必须同时满足两个安全分量的
最小收益排除和资源成本95%置信下界大于0。

## 3. 数据完整性

- 72/72个上下文身份匹配；
- observation hash、场景、种子、时间步、单元和合法目标全部一致；
- 目标概率最大重建误差为 `4.98e-13`；
- 2304条上下文—重复记录完整；
- 169条目标事件窗记录完整；
- Actor最大参数差为 `0.0`；
- 完整回归 `255 passed`；
- 实际额外transition为127,700，低于266,198上限。

如果在线实现只运行到事件窗而不保留完整回合对照，预计需要98,684个
transition，可减少29,016个，约22.7%。

## 4. 总体结果

| 视窗 | ENGAGE | STOP | AMBIGUOUS | 可操作 |
| --- | ---: | ---: | ---: | ---: |
| 短视窗 | 15 | 16 | 41 | 31 |
| 完整回合 | 14 | 13 | 45 | 27 |

短视窗只增加4个可操作标签，仍明显低于48/72门槛。标签变化全部是
完整回合 AMBIGUOUS 转为短视窗标签：1个 ENGAGE、3个 STOP，没有发生
ENGAGE 与 STOP 直接反转。

## 5. 场景与种子

### 5.1 场景汇总

| 场景 | ENGAGE | STOP | AMBIGUOUS |
| --- | ---: | ---: | ---: |
| time pressure | 5 | 2 | 29 |
| heterogeneity pressure | 10 | 14 | 12 |

异质场景达到场景双向覆盖；时间压力场景未达到每类至少6个的门槛。

### 5.2 块级汇总

| 场景/种子 | ENGAGE | STOP | AMBIGUOUS | 可操作 |
| --- | ---: | ---: | ---: | ---: |
| time/8 | 4 | 0 | 8 | 4 |
| time/9 | 0 | 2 | 10 | 2 |
| time/10 | 1 | 0 | 11 | 1 |
| heterogeneity/8 | 5 | 4 | 3 | 9 |
| heterogeneity/9 | 0 | 4 | 8 | 4 |
| heterogeneity/10 | 5 | 6 | 1 | 11 |

四个块低于6个可操作标签，且没有任何块全部满足至少2个 ENGAGE 和2个
STOP。跨种子双向覆盖失败。

## 6. 槽位诊断

| 场景/槽位 | ENGAGE | STOP | AMBIGUOUS |
| --- | ---: | ---: | ---: |
| time/safety | 5 | 2 | 11 |
| time/resource | 0 | 0 | 18 |
| heterogeneity/safety | 9 | 4 | 5 |
| heterogeneity/resource | 1 | 10 | 7 |

关键差异来自成本可辨识性：

| 场景/槽位 | 平均短窗成本差 | 成本下界>0 |
| --- | ---: | ---: |
| time/safety | +0.286 | 14/18 |
| time/resource | -0.034 | 0/18 |
| heterogeneity/safety | +0.369 | 12/18 |
| heterogeneity/resource | +1.036 | 12/18 |

`time/resource` 中强制当前交战通常替代随机后续中的其他射击，没有形成
确定的额外成本。即使安全收益较小，也无法满足预注册 STOP 定义。该失败
不是简单增加重复数可以解决的置信区间问题，而是当前干预与场景资源结构
下的局部命题不成立。

异质资源槽具有显著成本差异，因此能形成10个 STOP；这说明短视窗双分量
标签具有场景条件有效性，但不能写成跨场景通用机制。

## 7. 门控结果

| 门控 | 结果 | 状态 |
| --- | ---: | --- |
| 72上下文与32重复完整 | 72/72 | 通过 |
| 上下文身份一致 | 72/72 | 通过 |
| 可操作标签至少48 | 31 | 失败 |
| 每块至少6个可操作标签 | 最差1 | 失败 |
| 每场景ENGAGE/STOP各至少6 | time为5/2 | 失败 |
| 每块ENGAGE/STOP各至少2 | 0/6块 | 失败 |
| ENGAGE安全分量一致 | 15/15 | 通过 |
| Actor冻结 | 最大差0.0 | 通过 |
| transition预算 | 127,700 | 通过 |
| 软件回归 | 255 passed | 通过 |

阶段 A2 总门控失败。

## 8. 可证伪命题判定

| 命题 | 结论 | 证据 |
| --- | --- | --- |
| P-A2-1：短视窗显著提高可操作标签 | 否决 | 仅由27增至31，仍低于48 |
| P-A2-2：STOP可跨场景、跨种子稳定定义 | 否决 | time仅2个STOP，四个块缺少方向 |
| P-A2-3：短视窗保留必须交战方向 | 部分支持 | 15个ENGAGE分量一致，但seed9两场景均为0 |

## 9. 项目决策

按照预注册规则：

1. 阶段 B 辅助剂量审计不启动；
2. 阶段 C 选点覆盖审计不启动；
3. 不实现 coverage-balanced BPCE；
4. 不运行修订版10k；
5. 不增加重复数、候选视窗或训练预算；
6. 暂停 BPCE 在线辅助主线。

保留的研究成果包括：

- strict joint PPO fallback；
- 动态合法目标精确边缘化；
- 冻结策略共同随机数反事实轨迹；
- deterministic continuation 失效证据；
- 完整回合与短视窗三态标签对照；
- 局部 STOP 可辨识性依赖资源异质性的受控证据。

后续论文叙事应收窄为结构化分配中局部反事实标签的可辨识性边界与失败
机制，不能宣称 BPCE 已解决 all-noop 或资源过度交战。若另立研究任务，
应重新审视“当前动作替代未来动作”下的资源机会成本定义，而不是继续修补
本轮标签或 PPO 剂量。

## 10. 产物

```text
rein_learning/common/bpce_short_horizon_labels.py
scripts/run_air_defense_v1_bpce_short_horizon_label_audit.py
tests/test_bpce_short_horizon_labels.py

results/air_defense_v1/bpce_short_horizon_label_audit/
  experiment_config.json
  context_identity_check.csv
  repeat_component_deltas.csv
  context_component_labels.csv
  target_horizons.csv
  horizon_comparison.csv
  block_summary.csv
  gate_summary.json
```
