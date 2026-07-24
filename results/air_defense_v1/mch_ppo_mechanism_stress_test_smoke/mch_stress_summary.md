# AirDefense v1 MCH-PPO 机制压力实验

## 门控结论

- 总门控：`false`
- `structural_zero`：`true`
- `all_noop_noninferiority`：`false`
- `high_threat_improvement`：`false`
- `resource_cost`：`true`
- `reward_damage_safety`：`false`

## time_pressure

- all-noop 非劣种子数：0
- 高威胁突防率均值差：0.000000
- 资源成本比：0.000000
- 奖励均值差：-20.596518
- 损伤均值差：0.583181

| seed | all-noop 差 | 高威胁突防差 | 成本差 | 奖励差 | 损伤差 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1.000000 | 0.000000 | -17.000000 | -20.596518 | 0.583181 |

## 解释边界

本结果是冻结困难场景上的机制筛选。通过门控只允许进入更大预算正式实验；未通过时不得挑选单个优势种子宣称 MCH-PPO 普遍优越。
