# W1-07：Discussion 与 Limitations 中文稿

更新时间：2026-07-24  
任务状态：NOT_STARTED  
前置任务：W1-02 定位非 L4；W1-04 通过 T04；W1-06 通过 T06  
后续任务：W1-08、W1-09  
允许并行：无  
任务性质：证据解释、文献对话、竞争解释和边界陈述

## 1. 目标

解释已观察结果意味着什么、为什么重要、与既有工作有何关系，以及结论在何处
停止。Discussion 不逐图复述 Results；Limitations 不写通用免责声明。

## 2. 输入

- W1-02 的文献矩阵与定位决策；
- W1-04 的 Results 中文证据稿；
- W1-06 的图表和图注；
- W1-03 的段落工作表和追溯矩阵。

## 3. Discussion 架构

### 3.1 中心推进

用一段说明：

- 累计回合成本不是局部成本的稳定读出；
- 偏差来自同一步和未来动作替代；
- 该机制跨新策略种子复现；
- 符号改变具有场景和资源类型条件。

不得把恒等式本身写成全部科学贡献。

### 3.2 结构性混叠

解释：

- 为什么序列联合动作使当前决策改变后缀单元；
- 为什么当前动作还会改变未来策略分工；
- 为什么增加 rollout 只降低方差，不能改变被估计量的组成。

“为什么”若没有直接机制证据，使用 `suggests`、`is consistent with` 等受限表述。

### 3.3 与既有工作的关系

按技术主题组织：

- counterfactual credit 与 difference rewards；
- temporal/delayed credit；
- action masking 与 sequential joint actions；
- resource value/opportunity cost；
- simulation CRN 与 measurement validity。

每段结尾明确本项目差异，不按作者年份逐篇罗列。

### 3.4 场景和资源类型边界

必须同时写：

- missile 和 laser 均有正替代；
- missile 没有达到稳定符号掩盖门槛；
- laser 较低直接成本更易被替代抵消是数据支持的条件性解释；
- 三场景属于 AirDefense v1 内部边界，不是跨环境泛化。

### 3.5 对后续方法的含义

允许：

- 在线信用方法应先区分当前直接成本和策略诱导替代；
- 监督标签设计不能只靠增加 rollout；
- 测量诊断应先于模型复杂度扩展。

禁止：

- 声称已提出修复算法；
- 声称 GNN、BPCE 或机会成本网络必然有效；
- 将设计含义写成已经验证的性能提升。

## 4. 竞争解释清单

逐项回答：

| 竞争解释 | 所需证据 | 当前处理 |
| --- | --- | --- |
| 只是随机 rollout 方差 | 新种子块下界与 CRN | 支持排除到何种程度 |
| 只是 all-noop 副作用 | 来源模型无条件保留、跨模型结果 | 限定说明 |
| 只是 laser 直接成本较低 | missile/laser 分层 | 保留资源类型边界 |
| 只是单元执行顺序 | 同一步后缀替代 | 作为机制组成，不声称完全排除 |
| 只是目标冲突处理 | 无冲突动作结构 | 写明依赖 |
| 只是成本定义特例 | AirDefense v1 成本范围 | 不跨环境外推 |
| CRN 人为制造相关性 | 分支边缘过程与随机带定义 | 解释其方差控制角色 |

无法排除的解释进入 Limitations，不用修辞掩盖。

## 5. Limitations 必含范围

- 单一 AirDefense v1 环境；
- 冻结 factorized PPO；
- 三个场景；
- missile/laser 资源定义；
- N/E 局部干预与 CRN；
- 回合累计成本定义；
- P-C3 未通过；
- 未验证在线算法修复；
- 未完成跨环境或跨算法泛化。

未来工作只写由当前边界直接产生的问题，不写泛化营销。

## 6. 交付物

```text
discussion_draft_zh.md
limitations_draft_zh.md
rival_explanations_matrix.md
```

并更新追溯矩阵中的解释段落、文献来源和边界句。

## 7. 验收门控 T07

- Discussion 不逐图复述 Results；
- 中心推进与冻结主张一致；
- 每个文献比较来自 W1-02 已核验来源；
- 竞争解释逐项处理；
- P-C3 和环境范围写入 Limitations；
- 方法含义没有冒充已验证算法；
- 强动词与证据等级匹配；
- 每段只有一个解释功能。

## 8. 移交

通过 T07 后向 W1-08 提供用于 Introduction/Conclusion 的中心意义和边界；
向 W1-09 提供三个交付文件。

