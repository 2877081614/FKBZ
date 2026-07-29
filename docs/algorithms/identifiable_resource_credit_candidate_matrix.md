# 可辨识资源信用候选矩阵

更新时间：2026-07-28。  
状态：N1 候选冻结；没有候选获得在线训练授权。

## 1. 统一语义

令 `N` 为被测单元保持 STOP 的分支，`E` 为该单元执行 ENGAGE 的分支。
直接成本与回合成本差采用 `E-N` 方向，替代量采用 `N-E` 方向：

\[
\Delta C_{\mathrm{episode}}
= C(E)-C(N)
= C_{\mathrm{direct}}
- S_{\mathrm{same\_step\_other}}
- S_{\mathrm{future\_probe}}
- S_{\mathrm{future\_other}}.
\]

四通道分别表示：

- \(C_{\mathrm{direct}}\)：被测单元当前 ENGAGE 的已知直接资源成本；
- \(S_{\mathrm{same\_step\_other}}\)：当前自回归步中其他后缀单元被替代的成本；
- \(S_{\mathrm{future\_probe}}\)：未来被测单元动作被替代的成本；
- \(S_{\mathrm{future\_other}}\)：未来其他单元动作被替代的成本。

若 \(C_{\mathrm{direct}}>0\) 而
\(\Delta C_{\mathrm{episode}}\le 0\)，则回合标量的符号被替代量完全掩盖。
这不说明回合成本差“错误”，只说明它不能唯一解释当前动作的局部成本。

## 2. 候选比较

| 维度 | A：分量保持的约束信用 | B：全局 CMDP 约束 | C：受控延续差异回报 |
| --- | --- | --- | --- |
| 优化对象 | 直接成本与替代通道分开输入 | 保持回合累计成本约束 | 固定/控制后续行为后的局部差值 |
| 主要优点 | 局部语义可读，账本可核验 | 与全局资源预算一致 | 可能减少后续动作混叠 |
| 关键风险 | “直接成本不可补偿”可能改变原目标 | 算法创新已被现有 CMDP 覆盖 | 干预路径偏离当前政策分布 |
| 最近工作 | 因果效应分解、模块化信用 | CPO、LP3、安全 MARL | CCA、DAE、COCOA |
| 动态掩码适配 | 可显式记录同一步后缀 | 不需要局部归因 | 必须重新定义合法 continuation |
| fallback | 可实现系数零严格退化 | 标准基线可实现 | 尚未建立 |
| N1 判决 | 方法组件，不独立命名 | 必要强基线 | 否决 |

## 3. 候选 A 的软件契约

已实现的最小接口位于
`rein_learning/common/identifiable_resource_credit.py`：

```python
ResourceCreditComponents(
    direct_cost,
    same_step_other_substitution,
    future_probe_substitution,
    future_other_substitution,
)
```

该接口只承担三项职责：

1. 强制分量方向和有限值；
2. 重建 \(\Delta C_{\mathrm{episode}}\)；
3. 检测正直接成本被非正回合差掩盖的情形。

辅助项的组合契约为：

\[
\mathcal{L}
= \mathcal{L}_{\mathrm{joint\ PPO}}
+ \alpha\mathcal{L}_{\mathrm{component}},
\qquad \alpha\ge 0.
\]

当 \(\alpha=0\) 时函数直接返回原始 joint PPO loss 对象。这是必要但不充分
的 fallback；若未来重新获准实现在线算法，还必须验证采样、ratio、clipping、
优化器状态和单步参数更新完全一致。

## 4. 人工轨迹验收

测试文件 `tests/test_identifiable_resource_credit.py` 覆盖：

- 零替代下回合差等于直接成本；
- 同一步后缀替代与未来替代可独立出现；
- 替代量等于直接成本时回合差为零；
- 替代量超过直接成本时回合差为负；
- 非法直接成本与非法系数被拒绝；
- 系数为零时 loss 数值与梯度严格恢复基线。

与既有动作替代测试合并执行结果为 `12 passed`。

## 5. 不通过原因

候选 A 的账本是可辨识的，但“用哪个分量优化”仍是规范性选择：

- 若优化回合总成本，则替代是实际政策后果，不应被任意删除；
- 若单独惩罚直接成本，则相当于新增局部责任目标；
- 若同时优化两者，则需要说明双层目标、约束资格和 Pareto 语义。

N1 没有为这一选择给出足够的新原理，也没有证明其相对因果分解和约束 RL
的算法差异。因此不创建“已成立算法”文档
`substitution_decomposed_resource_credit.md`。

## 6. 重新进入算法阶段的条件

新的候选必须先提供：

1. 明确的规范目标：全局预算、局部责任或有定义的双层关系；
2. 与 CPO/安全 MARL 和 2025 因果效应分解的公式级差异；
3. 可证伪的机制预测，而不是“多通道更好”；
4. 候选 B 强基线和等参数非分解对照；
5. 冻结的新种子、新状态、预算、指标和停止门槛；
6. 完整的零系数单步更新等价测试。

