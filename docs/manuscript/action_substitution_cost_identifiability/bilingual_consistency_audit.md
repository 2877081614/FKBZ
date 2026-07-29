# W1-09 中英文整稿一致性审计

更新时间：2026-07-28  
中文稿：`manuscript_draft_zh.md`  
英文稿：`manuscript_draft_en.md`  
审计单位：Paragraph ID、冻结公式、决定性数字与边界主张

## 1. 故事范围

| 检查项 | 中文稿 | 英文稿 | 结果 |
| --- | --- | --- | --- |
| 研究对象 | AirDefense v1 动态掩码序列资源分配 | AirDefense v1 dynamically masked sequential allocation | PASS |
| 来源策略 | 冻结 factorized joint PPO | frozen factorized joint PPO | PASS |
| 中心问题 | 回合成本混合直接成本与动作替代 | episode cost mixes direct cost and action substitution | PASS |
| 方法 | N/E、CRN、目标边缘化、三分量账本 | N/E, CRN, target marginalization, three-component ledger | PASS |
| 独立确认 | 新 seeds 17/18/19、9 模型、108 context | new seeds 17/18/19, nine models, 108 contexts | PASS |
| 失败边界 | P-C3 未通过 | P-C3 failed | PASS |
| 非算法定位 | 不主张在线 PPO、BPCE/MCH-PPO 或 GNN 成功 | no validated online PPO, BPCE/MCH-PPO, or GNN claim | PASS |
| 贡献数量 | 3 | 3 | PASS |

一句话范围一致：

> 在 AirDefense v1 冻结 factorized joint PPO 的动态掩码序列分配中，
> 三分量成对反事实账本揭示回合成本中的同一步和未来动作替代，并在新策略种子
> 上确认其存在；成本符号掩盖受场景与资源类型约束，因此当前贡献是测量诊断，
> 而不是已验证的在线算法改进。

英文稿没有超出该句的科学主张。

## 2. 数字、单位和统计对象

| 项目 | 冻结值 | 中文 | 英文 | 结果 |
| --- | ---: | --- | --- | --- |
| 新来源模型 | 9/9 | 9/9 | 9 of 9 | PASS |
| 新上下文 | 108/108 | 108/108 | 108 of 108 | PASS |
| 每 context repeats | 32 | 32 | 32 | PASS |
| context-repeat 记录 | 3,456 | 3,456 | 3,456 | PASS |
| 目标账本记录 | 7,776 | 7,776 | 7,776 | PASS |
| 额外 transitions | 157,485 | 157,485 | 157,485 | PASS |
| 软件回归 | 264 passed | 264 passed | 264 passed | PASS |
| 首轮受影响账本 | 287/7,776 | 287/7,776 | 287 of 7,776 | PASS |
| 首轮最大残差 | 2.0 | 2.0 | 2.0 | PASS |
| 完整恒等式最大误差 | \(8.88\times10^{-16}\) | 一致 | 一致 | PASS |
| R1 正替代 context | 18/18 | 18/18 | all 18 | PASS |
| R1 非正成本解释 | 11/11 | 11/11 | all 11 | PASS |
| R2 正下界 context | 13/18 | 13/18 | 13 of 18 | PASS |
| R2 非正成本解释 | 7/7 | 7/7 | all seven | PASS |
| missile 掩盖 context | 2/9 | 2/9 | 2 of 9 | PASS |
| laser 掩盖 context | 5/9 | 5/9 | 5 of 9 | PASS |

统计单位在两稿中均保持：

```text
ledger row < repeat < context < block < seed
```

目标账本行未被写成独立 context。

## 3. 公式与方向

| 冻结项 | 中英文一致表达 | 结果 |
| --- | --- | --- |
| N 分支 | 当前探针 no-op / probe forced to no-op | PASS |
| E 分支 | 当前探针合法 engage / probe forced to a legal engagement | PASS |
| 累计成本方向 | \(\Delta C_{\mathrm{episode}}=C(E)-C(N)\) | PASS |
| 替代方向 | 所有 substitution 均为 \(N-E\) saving direction | PASS |
| 总替代成本 | same + future probe + future other | PASS |
| 替代射击 | 严格晚于干预步，不含 same-step suffix | PASS |
| 成本恒等式 | \(\Delta C_{\mathrm{episode}}=C_{\mathrm{direct}}-Sub_{\mathrm{cost,total}}\) | PASS |
| 替代比率 | \(Sub_{\mathrm{cost,total}}/C_{\mathrm{direct}}\) | PASS |
| 符号掩盖 | \(C_{\mathrm{direct}}>0\land\Delta C_{\mathrm{episode}}\le0\) | PASS |
| “精确”边界 | 仅指逐账本代数重构 | PASS |

## 4. 场景、资源和门控

| 项目 | 中文稿 | 英文稿 | 结果 |
| --- | --- | --- | --- |
| 场景 | medium / time_pressure / heterogeneity_pressure | 相同 | PASS |
| 资源 | missile / laser | 相同 | PASS |
| P-C1 | 完整成本恒等式 | complete cost identity | PASS |
| P-C2 | 新策略种子独立替代确认 | independent substitution confirmation | PASS |
| P-C3 | 跨资源类型普遍门控失败 | cross-resource gate failed | PASS |
| 机会成本 | 未形成通用 oracle | no broadly applicable oracle | PASS |
| 在线方法 | 尚未形成稳定性能贡献 | no validated stable performance contribution | PASS |
| GNN | 未测试 | not evaluated | PASS |

## 5. 段落与强动词

- 两稿均有 66 个相同且唯一的 Paragraph ID；
- Abstract、Introduction、Related Work、Methods、Results、Discussion、
  Limitations 和 Conclusion 的段落顺序一致；
- 英文 Results 使用 `was observed`、`had`、`produced`、`failed` 等观察性表达；
- 英文 Discussion 使用 `is consistent with`、`could`、`does not establish`
  等与证据相称的解释性表达；
- 未使用未经核验的 `first`、`unprecedented`、`revolutionary`、
  `generalizable` 或 `comprehensive` 主张；
- `exact` 只修饰冻结公式下的代数重构；
- 英文稿没有新增中文稿不存在的性能、泛化或机制结论。

## 6. 审计结论

双语主张、数字、公式、场景、资源类型、门控状态、局限性和贡献数量一致。
英文稿按段落功能重写而非逐句翻译。当前仅剩目标期刊与投稿格式类占位，不存在
未登记的科学证据占位。
