# W1-03 稿件章节架构

更新时间：2026-07-24  
状态：T03 架构冻结  
定位：L2 方法论文组成模块  
组织原则：证据顺序优先，不按项目时间顺序组织

## 1. 全稿论证链

```text
动态掩码序列资源分配
→ 回合成本不是纯局部信用读出
→ N/E + CRN + 三分量成本账本
→ 逐账本恒等式与独立策略种子确认
→ 场景/资源类型符号边界
→ 机会成本和在线算法主张停止边界
→ 为后续在线信用方法提供测量基础，而非宣称算法已完成
```

## 2. 推荐章节

### 1. Introduction

功能：建立资源信用测量问题、与既有反事实信用研究的关系，以及 L2 定位。

计划内容：

1. 动态资源分配需要同时评估任务效果和局部资源消耗；
2. 团队/回合回报到局部动作信用的困难已被既有工作研究；
3. 动态掩码自回归动作带来同一步后缀不能固定的具体测量缺口；
4. 本文给出资源成本操作账本和独立机制确认；
5. 明确贡献是测量与边界，不是新 PPO 性能主张。

### 2. Related Work

按技术主题组织，不按年代或作者逐篇罗列：

1. multi-agent counterfactual credit 与 difference rewards；
2. temporal/hindsight credit 与因果效应传播；
3. sequential/autoregressive MARL 与 invalid action masking；
4. constrained MARL、资源成本和 CRN 仿真比较；
5. 当前工作与最近效应分解/CAPO 的实质差异和 L2 边界。

### 3. Problem Formulation and Evaluation Scope

#### 3.1 AirDefense v1 dynamic resource assignment

定义环境、资源单元、目标、状态、动态合法动作掩码和资源成本，不展开无关环境细节。

#### 3.2 Frozen factorized joint policy

定义按固定单元顺序条件分解的联合策略，说明 factorized 不等于单元独立。

#### 3.3 Local resource-credit estimand

区分：

- 当前探针动作的直接成本；
- 回合累计成本差；
- 同一步和未来动作替代；
- 冻结策略下的局部反事实测量范围。

#### 3.4 Scope and non-claims

声明不评估在线学习增益、不证明跨环境泛化、不把 GNN 或 BPCE 写成已验证方案。

### 4. Paired Counterfactual Resource-Cost Decomposition

#### 4.1 N/E paired intervention

定义 no-engage 和 engage 分支、冻结上下文、目标动作和延续策略。

#### 4.2 Common random numbers and exact target marginalization

分别说明：

- CRN 用于降低配对方差；
- 目标边缘化用于消除目标采样误差；
- 二者均不自行定义结构分量。

#### 4.3 Same-step and future cost ledger

给出：

\[
Sub_{\mathrm{cost,total}}
=Sub_{\mathrm{cost,same}}
+Sub_{\mathrm{cost,future,probe}}
+Sub_{\mathrm{cost,future,other}}.
\]

#### 4.4 Episode-cost identity and sign masking

给出：

\[
\Delta C_{\mathrm{episode}}
=C_{\mathrm{direct}}-Sub_{\mathrm{cost,total}},
\]

以及 \(\rho_{\mathrm{sub}}\) 和 \(I_{\mathrm{mask}}\) 的定义与解释边界。

#### 4.5 Assumptions and identifiability boundary

明确冻结策略、共同随机带、动态分支合法性、目标条件边缘化和统计单位。

### 5. Experimental Protocol

#### 5.1 Discovery and independent-confirmation roles

R1 只负责机制发现；R2 使用新策略种子和新上下文承担独立确认。

#### 5.2 Source policies and context independence

报告 9 个新来源模型、seeds 17/18/19、108 个上下文和旧 hash 重叠为零。

#### 5.3 Paired sampling and statistical units

报告每上下文 32 次、目标精确边缘化、block/context/repeat/ledger row 的区别。

#### 5.4 Integrity and preregistered gates

报告 Actor 参数差、目标概率误差、transition 上限、恒等式阈值和支持/否决门槛。

### 6. Results

结果采用“问题—证据—边界”梯度，不按 R1/R2 执行日期排列。

#### 6.1 Local resource-credit ambiguity

- 冻结短视窗审计中，time/resource 的 18 个上下文全部为 AMBIGUOUS；
- 将问题收窄为回合累计成本能否代表当前动作局部成本；
- 不在本节展开 BPCE 历史变体或在线性能。

#### 6.2 Paired counterfactual action substitution

- 定义 N/E 当前步差异和共同随机数配对；
- R1 time/resource 中 18/18 个上下文具有可靠正未来替代射击；
- 11/11 个非正累计成本差可由正未来替代成本解释；
- 明确 R1 是旧种子机制发现，独立性由 R2 提供。

#### 6.3 Same-step suffix actions are required for exact decomposition

- future-only 账本在 287/7,776 行出现残差，最大为 2.0；
- 加入同一步其他单元替代后最大误差为 \(8.88\times10^{-16}\)；
- 同一步项是联合动作内部必要分量，不是事后美化。

#### 6.4 Action substitution replicates across new policy seeds

- 9 个新模型、108 个新上下文、旧 hash 零重叠；
- time/resource 中 13/18 个上下文 95% 下界为正；
- seeds 17/18/19 的块级下界均为正。

#### 6.5 Scenario and resource-type boundaries

- resource 槽位场景 \(\rho_{\mathrm{sub}}\)：0.747/0.873/0.972；
- time/resource 的 missile/laser：0.571/1.175；
- 符号掩盖上下文为 2/9 与 5/9；
- C4 普遍性主张被否决。

#### 6.6 Resource restoration does not yield a universal opportunity label

- 报告 time 与 heterogeneity 的可靠机会计数；
- 行动集合扩大不等于稳定安全收益；
- C5 被否决，不训练通用 opportunity oracle。

### 7. Discussion

#### 7.1 Measurement meaning

解释结构性混合为何不能靠增加 rollout 解除。

#### 7.2 Relation to counterfactual credit and causal effect decomposition

说明当前工作是资源成本操作化，与 COMA、CCA、效应传播分解互补，而非替代。

#### 7.3 Why dynamic masking changes the comparison

解释当前动作会改变同一步后缀合法集，因此“固定其他动作”不总是有效对照。

#### 7.4 Conditional sign masking

解释动作替代存在与成本符号被掩盖是两个不同强度的问题。

#### 7.5 Implications for future online methods

仅提出未来方法必须尊重账本分量和边界，不提出未经验证的 MCH-PPO/GNN 性能结论。

### 8. Limitations

1. 单一 AirDefense v1 环境；
2. 冻结 factorized PPO 和局部 N/E 干预；
3. 配对 CRN 不能证明所有潜在因果路径均可辨识；
4. 资源恢复安全价值未稳定识别；
5. 在线算法未通过门控；该证据只用于限制算法主张，不作为本 Results 的正面结果；
6. GNN 修复未验证。

### 9. Conclusion

压缩为“测量问题—账本方法—决定性证据—条件边界”四部分，不引入新数字或新机制。

## 3. L2 稿件形态落实

- 本架构把第 3-6 节设计为较大方法论文的核心测量模块；
- 若后续在线方法通过独立门控，可在第 4 节后增加“Online credit estimator”，
  但当前 W1 不预留虚构结果；
- 在没有新算法证据时，可形成完整学位论文章节或方法论文的机制部分；
- 不以当前结果单独宣称通用反事实信用算法。

## 4. 标题与摘要顺序

标题和摘要只做占位，不在 W1-03 定稿：

```text
Title: [DEFERRED TO W1-08]
Abstract: [DEFERRED TO W1-08 AFTER W1-04/W1-05/W1-07]
```

原因：标题和摘要必须反映最终 Results、Methods 和 Limitations，不能提前扩大主张。

## 5. 章节修改控制

W1-04、W1-05、W1-06 可在本架构内写作和制图，不得自行改变章节功能。
若发现证据无法支撑某段，应在追溯矩阵标记缺口；若必须改变架构，先更新本文件和
`manuscript_traceability_matrix.md`，再同步三个下游任务。
