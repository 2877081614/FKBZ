# LR-03 阅读报告：GradS 多约束梯度塑形与 AirDefense 规范目标边界

任务状态：`PASSED`  
完成时间：2026-07-29  
实验授权：否  
总体判决：方法为 `BASELINE`；梯度诊断可 `ADAPT`；当前在线接入为 `AVOID`；约束语义与可靠 cost critic 为 `OPEN`

## 1. 论文身份

| 项目 | 内容 |
| --- | --- |
| 标题 | *Gradient Shaping for Multi-Constraint Safe Reinforcement Learning* |
| 作者 | Yihang Yao、Zuxin Liu、Zhepeng Cen、Peide Huang、Tingnan Zhang、Wenhao Yu、Ding Zhao |
| 会议 | 6th Annual Learning for Dynamics & Control Conference（L4DC 2024） |
| 正式出版 | PMLR 242:25–39 |
| 算法 | Gradient Shaping（GradS） |
| 官方页面 | <https://proceedings.mlr.press/v242/yao24a.html> |
| 正式 PDF | <https://proceedings.mlr.press/v242/yao24a/yao24a.pdf> |
| arXiv | <https://arxiv.org/abs/2312.15127> |
| 项目页 | <https://sites.google.com/view/mc-grads/home> |

本报告以 PMLR 正式 15 页论文为公式和页码基准，并与 arXiv v1 核对。正式
PDF 在 Theorem 4 后写有“证明见附录”，但 PMLR 和 arXiv 当前可检索 PDF
均在参考文献结束，没有附录正文；项目页在本次工具环境中无法打开。因此本
报告能核对定理陈述与假设，不能独立复核其完整证明和附录实现细节。

官方 PMLR 页面没有列出代码仓库，本次公开检索也没有定位到可确认的官方
GradS 实现。没有下载或运行第三方代码。

## 2. 一句话结论

GradS 已经覆盖“把多约束安全 RL 写成多目标梯度组合，并按约束梯度余弦关系
删除冗余/冲突方向后随机选择更新”的主叙事。它解决的是：**成本、阈值和
cost gradients 已经正确时，如何改善多约束在线优化效率**。

它不决定 AirDefense 应把损伤、高威胁突防和资源消耗定义成什么约束，不修复
错误或跨批次不稳定的 cost critic，也不保证稀有高威胁状态、每条轨迹或每次
更新都安全。项目目前还没有通过验收的独立在线 cost critics，因此 GradS
只能作为后续强基线和当前离线梯度诊断参考，不能直接接入 PPO。

## 3. 首要概念纠偏

### 3.1 GradS 不是梯度投影

GradS 没有像 PCGrad、MGDA 或二次规划方法那样寻找所有目标的共同下降方向。
它执行的是：

```text
按随机顺序扫描 cost gradients
        ↓
与已选梯度过于同向：删除
与已选梯度过于反向：删除
        ↓
从剩余集合随机抽一个梯度
        ↓
缩放后与 reward gradient 相加
```

所以它是“过滤 + 随机路由”，不是“把冲突梯度几何修正为共同可行方向”。

### 3.2 论文没有单独定义 aligned constraints

论文正式定义三类关系：

- \(\sigma\)-conflicting：余弦相似度不大于 \(-\sigma\)；
- \(\kappa\)-redundant：余弦相似度不小于 \(\kappa\)；
- \(\eta\)-independent：余弦相似度接近 0。

“aligned”只能作为一般描述：正余弦表示同向；当相似度达到 \(\kappa\) 时，
论文把它归为 redundant。GradS 实际接受区间是
\((-\sigma,\kappa)\)，并不要求候选梯度严格落入
\([-\eta,\eta]\) 的 independent 区间。

### 3.3 它只比较 cost–cost，不比较 reward–cost

论文的筛选矩阵只包含各约束梯度之间的余弦相似度。reward gradient 最后直接
与选中的 cost gradient 相加。因此：

- reward 与资源约束冲突时，GradS 没有显式消解；
- reward 已含损伤/资源惩罚时，目标与约束的重复也不会被筛掉；
- 多约束内部优化改善不代表 reward–safety 权衡已经正确。

## 4. Problem–Method–Insight

| 层 | 论文内容 |
| --- | --- |
| Problem | 多个期望累计成本约束同时激活时，简单求和可能因同向重复而过度保守，或因反向冲突而陷入局部最优/探索不稳 |
| Method | 把 Lagrangian 多约束更新统一为非负 cost-gradient 权重；按余弦相似度构造过滤后的候选集合，再随机抽取一个约束梯度 |
| Theory | 在 Slater 可行性、梯度有界光滑等条件下，给出 shaped constraint gradient 的非零邻域上界 |
| Insight | 多约束算法的差异不仅是乘子数目，还包括如何组合、选择或删除由约束产生的策略梯度 |

## 5. 多约束公式卡

### 5.1 多约束 CMDP

令成本向量为：

\[
c(s,a,s')=(c_1,\ldots,c_N)\in\mathbb R_{\ge0}^{N}.
\]

奖励和第 \(i\) 个成本的期望折扣回报为（PDF p.3）：

\[
V_r^\pi(\mu_0)
=
\mathbb E_{\tau\sim\pi,s_0\sim\mu_0}
\left[
\sum_{t=0}^{\infty}\gamma^t r_t
\right],
\]

\[
V_{c_i}^\pi(\mu_0)
=
\mathbb E_{\tau\sim\pi,s_0\sim\mu_0}
\left[
\sum_{t=0}^{\infty}\gamma^t c_{i,t}
\right].
\]

多约束安全 RL 的规范问题是：

\[
\boxed{
\pi^\star
=
\arg\max_\pi V_r^\pi,
\quad
\text{s.t.}\quad
V_c^\pi\preceq\epsilon
}
\]

其中 \(V_c^\pi=(V_{c_1}^\pi,\ldots,V_{c_N}^\pi)\)，
\(\epsilon=(\epsilon_1,\ldots,\epsilon_N)\)，偏序逐元素成立。

### 5.2 Lagrangian

采用最小化记号：

\[
\min_\pi\max_{\lambda\succeq0}
J(\pi,\lambda),
\]

\[
J(\pi,\lambda)
=
-V_r^\pi
+\lambda^\top(V_c^\pi-\epsilon).
\]

论文在策略更新公式中省略了与 \(\pi\) 无关的
\(-\lambda^\top\epsilon\)，写为：

\[
J(\pi,\lambda)
=
-V_r^\pi+\lambda^\top V_c^\pi.
\]

阈值仍应进入 multiplier 更新；省略它只不影响固定 \(\lambda\) 时的
policy gradient。

### 5.3 reward gradient 与 constraint gradients

定义 reward loss gradient：

\[
g_r=-\nabla_\theta V_r^\pi,
\]

以及带乘子的第 \(i\) 个约束梯度：

\[
g_i
=
\lambda_i\nabla_\theta V_{c_i}^\pi.
\]

令 \(G=[g_1,\ldots,g_N]\)，则统一框架为（PDF p.4, Eq. 4）：

\[
\nabla_\theta J
=
g_r+\nabla J_c,
\qquad
\nabla J_c=w^\top G,
\qquad
w\succeq0.
\]

三类基线只是选择不同 \(w\)：

| 方法 | 权重规则 |
| --- | --- |
| Vanilla | \(w=\mathbf 1\)，聚合全部约束梯度 |
| Lagrangian-CRPO | 每步从全部约束中均匀随机选一个，\(\|w\|_0=1\) |
| Min-Max | 只选当前归一化违反最严重的约束 |
| GradS | 先过滤相似/冲突梯度，再从候选集合随机选一个 |

### 5.4 梯度关系

论文使用：

\[
\operatorname{sim}(g_i,g_j)
=
\frac{g_i^\top g_j}
{\|g_i\|_2\|g_j\|_2}.
\]

当 \(\lambda_i,\lambda_j>0\) 时，乘子缩放不改变余弦方向；若某个乘子或
估计梯度接近 0，余弦会数值不稳定，论文正文没有规定零梯度处理。

正式定义为：

\[
\operatorname{sim}
(\nabla V_{c_i},\nabla V_{c_j})
\le-\sigma
\quad\Longrightarrow\quad
\sigma\text{-conflicting},
\]

\[
\operatorname{sim}
(\nabla V_{c_i},\nabla V_{c_j})
\ge\kappa
\quad\Longrightarrow\quad
\kappa\text{-redundant},
\]

\[
-\eta
\le
\operatorname{sim}
(\nabla V_{c_i},\nabla V_{c_j})
\le
\eta
\quad\Longrightarrow\quad
\eta\text{-independent}.
\]

### 5.5 GradS 变换

令随机打乱后的索引为 \(\rho=(\rho_1,\ldots,\rho_N)\)，候选集合初始为：

\[
\mathcal G=\{g_{\rho_1}\}.
\]

依次检查 \(g_{\rho_i}\)。只有当它与所有已选梯度满足：

\[
-\sigma
<
\operatorname{sim}(g_{\rho_i},g_j)
<
\kappa,
\qquad
\forall g_j\in\mathcal G,
\]

才加入 \(\mathcal G\)。完成后均匀抽取：

\[
\tilde i\sim\operatorname{Uniform}(\operatorname{index}(\mathcal G)),
\]

并返回：

\[
\boxed{
\nabla J_c^{\mathrm{GradS}}
=
\frac{|\mathcal G|}{N}g_{\tilde i}.
}
\]

最终策略梯度为：

\[
\nabla J^{\mathrm{GradS}}
=
-\nabla V_r
+
\nabla J_c^{\mathrm{GradS}}.
\]

条件于候选集合，其期望为：

\[
\mathbb E[
\nabla J_c^{\mathrm{GradS}}
\mid\mathcal G]
=
\frac1N\sum_{g_i\in\mathcal G}g_i.
\]

它不是未筛选总梯度 \(\sum_i g_i\) 的无偏估计。GradS 同时改变方向选择和
约束梯度量级；\(|\mathcal G|/N\) 也会随删除数变化。因此实验增益不能只解释
为“冲突方向被去除”，还混合了随机探索和梯度缩放效应。

## 6. 梯度冲突类型矩阵

| 类型 | 余弦范围 | 几何含义 | GradS 处理 | 风险 |
| --- | --- | --- | --- | --- |
| 强冲突 | \(\cos\le-\sigma\) | 降低一个成本可能提高另一个成本 | 后扫描者被删除 | 真实不可兼得或预算不可行时，删除不等于解决 |
| 中度反向 | \(-\sigma<\cos<-\eta\) | 有冲突但未达阈值 | 可保留 | 候选集合内仍可能互相干扰 |
| 近独立 | \(|\cos|\le\eta\) | 两个约束局部影响近正交 | 保留 | 小批次噪声也可能造成假正交 |
| 一般同向 | \(\eta<\cos<\kappa\) | 方向一致但未达重复阈值 | 保留 | 多个近同向梯度仍可能过度保守 |
| 强冗余 | \(\cos\ge\kappa\) | 两个约束给出近相同方向 | 后扫描者被删除 | 重复语义可能本应合并，而不是仅在优化器中隐藏 |
| 零/微小梯度 | 分母接近 0 | 约束未激活、critic 失真或样本不足 | 正文未定义 | 余弦分类不稳定，稀有约束容易被误判 |

### 6.1 随机顺序不是无害细节

筛选是随机顺序的贪心集合构造。候选集合可能随 shuffle 改变；冲突梯度中谁被
保留由扫描先后决定。若一个语义族被复制成很多相似约束，其成员更可能成为
候选代表，算法并不严格对“约束复制”不变。

### 6.2 冲突可能有三种完全不同的来源

| 来源 | 正确处理 |
| --- | --- |
| 两个正确且可同时满足的约束在当前局部梯度冲突 | GradS 所针对的优化效率问题 |
| 两个预算根本不可同时满足 | 重新审查 feasibility/Slater；不能靠删除梯度 |
| 成本语义、critic 或采样错误 | 修正定义/估计器；塑形只会处理错误梯度 |

## 7. 理论性质：保证与不保证

### 7.1 假设

论文列出：

1. **Slater condition**：存在严格可行策略，使全部期望累计成本满足阈值；
2. **有界梯度**：

\[
\|\nabla V_{c_i}\|\le G;
\]

3. **光滑性**：

\[
|u^\top\nabla^2V_{c_i}u|
\le
L\|u\|^2;
\]

4. 有限学习率 \(\alpha\) 和迭代数 \(T\)；
5. cost gradients 和乘子能够被足够准确地估计。

第五项没有作为单独编号假设写出，却是深度 actor-critic 实现成立的必要统计
条件。

### 7.2 Theorem 4

令第 \(t\) 步删除的冗余、冲突约束数分别为
\(N_R(\kappa,t)\)、\(N_C(\sigma,t)\)。论文给出：

\[
\mathbb E_t
\left[
\|\nabla V_c^{\mathcal G}(\theta_t)\|^2
\right]
\le
\frac{V_c(\theta_0)-V_c^\star}{T\alpha}
+
G^2
\left(
\mathbb E_t[N_R]
+
\mathbb E_t[N_C]
\right)
+
\frac{\alpha G^2L}{2}.
\]

第一项随 \(T\) 衰减；删除数和随机采样形成作者所称的 noise-ball terms。
即使 \(T\rightarrow\infty\)，右侧一般也不趋于 0。

### 7.3 该定理不保证

- 每次策略更新后所有 \(V_{c_i}\le\epsilon_i\)；
- 训练过程中的零违反或 anytime safety；
- 每条轨迹、每个状态或每个高威胁事件安全；
- shaped gradient 是原 Lagrangian 梯度的无偏估计；
- reward–constraint Pareto 改进；
- 不可行预算下仍能收敛；
- PPO、TRPO、SAC、DDPG 深度近似中的完整 saddle-point 收敛。

定理只分析 constraint-gradient 部分的范数上界，而且允许停在非零邻域。把它
称为“GradS 保持 CMDP 可行性保证”是不准确的。

## 8. 复杂度

若 policy 参数维数为 \(d\)、约束数为 \(N\)：

- 需要分别获得 \(N\) 个 cost-policy gradients；
- 两两相似度与贪心筛选最坏为 \(O(N^2d)\)；
- 保存全部扁平梯度需要 \(O(Nd)\) 内存；
- 每步仍需 reward critic、\(N\) 个 cost critics 和 \(N\) 个乘子更新；
- 随机选择只减少最终聚合，不免除计算相似度所需的各约束反向传播。

论文唯一明确局限也是计算梯度相似度的额外负担。

## 9. 论文证据

### 9.1 实验设置

论文从 Bullet-Safety-Gym 和 Safety-Gymnasium 构造连续控制任务：

- Circle 与 Goal；
- Point、Ball、Car、Drone 等机器人；
- boundary/collision、high-velocity、low-velocity 三类二元成本；
- `-v2` 使用前两项，作者把它们视为可能冗余；
- `-v3` 加入 low-velocity，作者把 high/low velocity 视为可能冲突。

底层安全 RL 包括 PPO-Lag、TRPO-Lag、SAC-Lag、DDPG-Lag；梯度处理比较
Vanilla、修改为 Lagrangian 版本的 CRPO、Min-Max 与 GradS。

主表为 5 个训练种子、每种子 20 个评估回合；尺度实验为 5 个种子、每种子
10 条轨迹。成本报告为最坏归一化约束：

\[
\operatorname{Cost\text{-}N}
=
\max_i\frac{c_i}{\epsilon_i},
\]

阈值为 1。

### 9.2 证据支持到哪里

- Vanilla 在作者构造的 `-v2` 冗余任务中常出现低 reward，在 `-v3` 冲突任务
  中常出现探索失败；
- Min-Max 在冗余任务较强，但在冲突任务容易陷入局部最优；
- CRPO 在冲突任务 reward 较强，却可能因同类约束数量不平衡而忽略其他约束；
- GradS 在多个组合上同时取得较高 reward 和较低 Cost-N；
- 增加相似阈值/边界位置生成更多约束时，GradS 的均值曲线比三类基线稳定。

### 9.3 证据边界

1. 表格报告 mean ± standard deviation，没有给出“显著改善”所需的假设检验或
   置信区间；作者的 “significantly” 应读作定性表述。
2. GradS 并非每个设置都满足 `Cost-N≤1`，例如若干表项仍略高于阈值；实验
   不支持绝对可行保证。
3. “冗余/冲突”主要由成本定义直觉和训练行为支持，正文没有报告实际余弦矩阵
   与环境标签逐步一致的统计验证。
4. 尺度实验通过复制相似速度阈值和边界位置增加约束，主要增加结构相似的
   constraints，不等于扩展到大量语义独立、稀有或尾部风险约束。
5. 随机筛选、梯度缩放和删除策略没有完整独立消融，性能增益无法唯一归因。
6. 任务都是连续控制和期望二元成本，不包含离散 AR mask、deterministic
   all-noop 或关键状态 chance/CVaR 约束。
7. 正式 PDF 所指附录与实现入口未能从官方出版物直接复核，复现细节不完整。

## 10. AirDefense 梯度关系假设图

以下只是**待测假设**，不是项目已经记录的梯度数据：

```text
奖励损失梯度 g_reward = -∇V_reward
├─ 损伤成本梯度 g_damage
│  └─ 可能一致：当前 reward 已包含安全损失；但是否重复及尺度未知
├─ 高威胁突防梯度 g_leak
│  └─ 可能一致：reward 倾向降低突防；稀有状态下支持不足
└─ 资源成本梯度 g_resource
   └─ 可能一致或冲突：reward 含资源惩罚，但更积极交战可提高 reward 并增加成本
```

约束间假设：

| 梯度对 | 假设关系 | 项目现有间接证据 | 正式判定 |
| --- | --- | --- | --- |
| damage–high-threat leak | 可能一致/冗余 | 高威胁泄漏通常增加损伤，但两指标并非同一事件 | 证据不足 |
| damage–resource | 可能冲突 | Task 8、BPCE 均出现安全改善伴随成本增加 | 只有行为权衡证据，无梯度证据 |
| high-threat leak–resource | 可能冲突 | Task 10 的 201 降低 unassigned 泄漏同时显著增加成本 | 只有行为权衡证据，无梯度证据 |
| reward–damage | 可能一致/重复 | reward 已含任务安全后果 | 需核对 reward 分量和实际梯度 |
| reward–leak | 可能一致 | 异质场景 reward 与泄漏趋势常同向但种子可反转 | 证据不足 |
| reward–resource | 可能一致或冲突 | reward 同时交易安全与成本，固定标量边界无可行解 | 证据不足 |

结果指标相关、动作行为权衡和 gradient cosine 是三个不同对象。现有实验不能
把上表任一关系标记为“已冲突”或“已冗余”。

## 11. 三类约束的语义候选

| 量 | 可选身份 | 优点 | 关键风险 | 当前建议 |
| --- | --- | --- | --- | --- |
| 任务 reward | 主目标 | 保持现有综合任务效用 | 已含安全/资源项时与显式约束重复 | 作为主目标保留，但先登记分量与尺度 |
| 防区/资产损伤 | 期望累计 cost | 非负、逐步或终局可记账 | 平均约束允许少数严重回合 | 可作为独立期望约束候选 |
| 高威胁突防 | 期望 cost、chance/CVaR、轨迹约束或硬规则 | 可突出关键目标安全 | 不同定义的规范含义完全不同 | `OPEN`；不能由 GradS 选择 |
| 资源消耗 | 期望累计 cost | 非负、可加、适合 CMDP 预算 | all-noop 可低成本却不安全；局部责任仍不明 | 可作为团队期望预算候选 |
| false-noop | 诊断指标或安全 surrogate | 直接定位漏交战 | 依赖 oracle/标签，跨批次不稳 | 暂不作为在线约束 |
| wasteful-engage | 诊断指标或资源 surrogate | 直接定位过度交战 | 依赖反事实标签与阈值 | 暂不作为在线约束 |

### 11.1 推荐的最小规范结构

若未来人工冻结为期望 CMDP，最小结构应是：

\[
\max_\pi V_r^\pi
\]

\[
\text{s.t.}\quad
V_{\mathrm{damage}}^\pi\le\epsilon_D,
\qquad
V_{\mathrm{resource}}^\pi\le\epsilon_R.
\]

高威胁突防是否另列期望约束，取决于它与 damage 的语义重合；若真正目标是
控制低概率严重泄漏，则应使用 chance/CVaR/轨迹语义，而不是为了套 GradS
强行写成普通期望成本。

## 12. 与项目证据的压力测试

### 12.1 资源约束交战边界

[资源约束交战边界](../../experiments/air_defense_v1_task14_engagement_calibration.md)
显示固定全局阈值或资源乘子在 validation 已是 `0/3` 可行，正式 test 的
no-op recall 全部低于 0.65。资源压力方向平均正确，但 engage/no-op 范围高度
重叠。

这首先说明单元成本与当前弹药不足以定义正确状态条件资源边界，不是“多个
正确 cost gradients 已知但组合方式不好”。GradS 不能从一个信息不足的标量
压力中恢复未来风险、替代单元或剩余任务预算。

### 12.2 状态条件双价值

[状态条件交战价值](../../experiments/air_defense_v1_task14_state_conditioned_value.md)
显式预测安全收益与增量成本后，总体性能改善，但三种子逐场景完整通过仍为
`0/3`。安全收益相关约 0.49–0.53，成本相关仅
`-0.044 / 0.034 / 0.128`。

这些 head 是冻结反事实数据上的监督模型，不是 on-policy
\(V_{c_i}^{\pi}\) 和 cost advantage。弱成本相关意味着即使把它们改写成 loss
并计算 cosine，方向也可能主要反映估计误差。

后续跨场景实验出现异质 engage recall 从 `1.0` 翻转到 `0.182–0.273`；
多批次 leave-one-batch-out 只有 `1/3` 可行，最终 test `0/3`；统一跨批次
校准也是 `0/3`。这已经直接否定“当前 cost-value 足够稳定，只需更好的梯度
组合”这一前提。

### 12.3 BPCE

[BPCE 压力测试](../../experiments/air_defense_v1_bpce_ppo_stress_test.md)保留
joint PPO 主干，反事实标签只形成 engagement 排序辅助。它仍有 `2/6`
all-noop，异质安全改善伴随 1.93 倍成本，边界选点不稳定优于随机选点。

BPCE 没有独立 damage/leak/resource cost critics 或各自 Lagrange multiplier。
其正负标签在 seed 9 单边缺失，说明梯度来源的支持域先于几何冲突成为瓶颈。
GradS 也不会自动修复 deterministic all-noop：如果安全约束梯度在该种子中
为零、错误或未激活，筛选器没有正确信号可选。

## 13. 当前是否具备可训练的独立 cost critics

**判定：不具备。**

当前项目拥有：

- reward Critic；
- 多类离线反事实 Q/value/oracle；
- safety gain 与 incremental cost 的监督预测器；
- BPCE 的稀疏 engagement 排序标签。

当前项目缺少：

1. 与冻结约束语义一一对应的
   \(V_{\mathrm{damage}}^\pi,V_{\mathrm{leak}}^\pi,
   V_{\mathrm{resource}}^\pi\)；
2. 每个 cost critic 的 on-policy return、GAE/advantage 和校准审计；
3. 全部预算 \(\epsilon_i\) 与乘子更新；
4. 跨场景、跨批次稳定的 cost-gradient 符号和方向；
5. cost-gradient cosine 的 minibatch/seed 置信度；
6. 至少一个满足全部预算的可行策略证据。

“有多个 value head”不等于“具备多约束优化”。只有每个 head 对应一个冻结
cost return、独立 critic、advantage、预算和 multiplier，才进入 GradS 的问题
设定。

## 14. 稀有临界状态为何会被期望梯度掩盖

普通约束是 occupancy 加权的期望：

\[
\nabla V_{c_i}^\pi
=
\mathbb E_{s,a\sim d^\pi}
\left[
\nabla\log\pi(a\mid s)A_{c_i}^\pi(s,a)
\right].
\]

若高威胁临界状态只占极小概率质量，则：

- minibatch 常常没有正例，梯度接近零；
- 普通状态的梯度方向主导 cosine；
- 平均成本可满足而关键状态持续违反；
- 两个 tail constraints 可能在总体余弦上看似 independent；
- GradS 随机抽取进一步减少稀有约束实际更新频率。

因此 GradS 的期望约束框架不能替代：

- 分层/重要性采样；
- chance/CVaR 或最坏组约束；
- 逐场景、逐批次、逐高威胁状态门控；
- shield 或硬合法性规则。

## 15. 使用前必须满足的前置条件

以下条件全部满足前，不得进入在线 GradS：

| 前置 | 最低验收 |
| --- | --- |
| 约束语义 | damage、leak、resource 的时间范围、折扣、聚合与尾部语义冻结 |
| 非重复性 | 说明 reward 与各 cost 是否重复；damage 与 leak 是否应合并 |
| 可行预算 | 至少一个冻结策略或策略混合同时满足全部预算，支持 Slater 可行性 |
| 独立 critic | 每个约束独立 on-policy cost critic 与 advantage |
| 估值可靠性 | 新批次上 return MAE/相关/符号/校准达到预注册门槛 |
| 激活功效 | 每个约束在所有场景和种子都有足够违反与非违反样本 |
| 梯度稳定性 | cosine 在 minibatch、epoch、seed 间有置信区间，不由零梯度主导 |
| joint PPO fallback | 所有乘子/GradS 关闭时参数更新严格退化为原 factorized PPO |
| 强基线 | 同时比较 Vanilla、CRPO、Min-Max 和 GradS，不只比较 reward penalty |
| 安全报告 | 每个约束单独报告平均、最坏组、尾部和逐场景违反 |
| 计算预算 | 记录 \(N\) 个反向和 \(O(N^2d)\) similarity 开销 |

## 16. 最小强基线与当前可做诊断

### 16.1 后续强基线

在 LR-02 定义的
`centralized constrained factorized PPO` 上，GradS 应作为第四种
constraint-gradient aggregation：

```text
同一 actor / joint PPO / reward critic / cost critics / multipliers
├─ Vanilla sum
├─ CRPO random-one
├─ Min-Max most-violated
└─ GradS filter-and-random-one
```

这样才能把“显式多约束优化”的收益与“GradS 选择规则”的增量分开。

### 16.2 当前只允许的 ADAPT

如果后续单独授权，第一步只能做**只读梯度诊断**：

1. 冻结策略和一批 on-policy trajectories；
2. 对每个候选成本分别估计 policy gradient；
3. 不更新 actor，只保存 norm、cosine、零梯度率和 bootstrap 区间；
4. 按场景、临界/普通状态、seed 分层；
5. 检验间接行为权衡是否真的对应稳定 gradient conflict。

若关系在分层、批次或种子间翻转，则停止 GradS 分支，先修 cost semantics 和
critic；不得用总体池化余弦掩盖局部失败。

本 LR-03 不授权上述诊断实现。

## 17. 具体失败机制

GradS 在 AirDefense 中至少可能以下列方式失败：

1. **错误冲突**：cost critic 的跨批次误差使两个真实同向约束呈负 cosine，
   GradS 删除其中一个；
2. **稀有约束饥饿**：高威胁状态梯度在多数 minibatch 为零，资源梯度持续被
   选中，策略走向低成本 all-noop；
3. **真实不可行被误作探索问题**：安全和资源预算不可同时满足，GradS 轮流
   删除冲突梯度，产生乘子振荡而不是可行策略；
4. **目标—约束重复**：reward 已含损伤/资源惩罚，GradS 只去除 cost–cost
   冗余，仍可能双重施压；
5. **随机路由放大种子分叉**：候选集合与抽样本身有随机性，项目已有的
   PPO seed 分叉可能加剧；
6. **行为边界不连续**：小梯度变化跨过 deterministic engage/no-op argmax，
   平均梯度平滑性不能保证行为稳定；
7. **复制敏感**：把一个安全语义拆成多个近似成本可能改变候选代表概率和
   \(|\mathcal G|/N\) 缩放。

## 18. 创新覆盖压力测试

### 18.1 已被覆盖或距离过近

| 候选主张 | 创新距离 | 判定 |
| --- | --- | --- |
| 把资源和安全写成多个 Lagrangian constraints | 极低 | LR-02/GradS 均已覆盖 |
| 为每个成本增加独立 critic/乘子 | 极低 | 标准多约束安全 RL |
| 用余弦区分约束梯度冲突与冗余 | 极低 | GradS 核心定义 |
| 删除冲突/冗余后随机选一个约束更新 | 已覆盖 | GradS 算法本身 |
| 多头 loss 后调权重 | 不构成 | 没有预算与 cost semantics 时不是多约束优化 |
| 在防空任务应用 GradS | 弱 | 领域迁移不能单独作为方法创新 |

### 18.2 仍开放但尚未形成算法

| 开放问题 | 为什么 GradS 未覆盖 | 当前状态 |
| --- | --- | --- |
| 稀有高威胁状态与期望资源约束的分层梯度关系 | GradS 使用总体期望 cost gradients | 有失败现象，无稳定梯度证据 |
| 动态 AR 合法集下的离散 engage/no-op 边界 | 论文为连续控制，不约束 argmax 跃迁 | 有项目证据，尚无规范方法 |
| 跨批次不确定 cost gradients 的可靠筛选 | GradS 假定梯度可用，不建模置信度 | 当前 critic 门控失败 |
| 不可行预算与局部梯度冲突的区分 | Slater 直接假设可行 | 预算尚未冻结 |

全文阅读与创新压力测试把项目叙事收窄为：先证明约束和梯度对象有效，再讨论
梯度组合。不能把当前安全—资源结果反转直接命名为“多约束梯度冲突创新”。

## 19. `BASELINE / ADAPT / AVOID / OPEN` 判决

| 标签 | 判决 |
| --- | --- |
| `BASELINE` | 当项目建立显式多约束 PPO 后，GradS 必须与 Vanilla、CRPO、Min-Max 一起作为 constraint-gradient aggregation 强基线 |
| `ADAPT` | 可适配其 gradient cosine taxonomy 做冻结、分层、带置信度的只读诊断；不更新 actor |
| `AVOID` | 当前不把 GradS 接入 BPCE/MCH/PPO，不用离线监督 head 冒充 on-policy cost critics，不把删除冲突梯度称为同时满足约束 |
| `OPEN` | damage/leak/resource 的最终规范身份、可行预算、跨批次可靠 cost critics、尾部安全与期望资源的接口仍未解决 |

总体路线：

```text
LR-02：先定义显式全局多约束强基线
                    ↓
LR-03：GradS 只解决正确梯度之间的组合效率
                    ↓
当前项目先冻结语义、预算并验收独立 cost critics
                    ↓
未满足前置条件：不进入在线梯度塑形
```

## 20. 与 LR-02 的合并移交

供后续头脑风暴使用的“规范目标层”边界为：

1. 资源成本与高威胁安全不能再压成一个事后标量；
2. 但把它们拆成两个 head 也不自动成为多约束优化；
3. 必须先冻结 cost、threshold、risk semantics 和可行策略；
4. centralized constrained factorized PPO 是第一强基线；
5. GradS 是该基线之上的 aggregation 消融，不是替代基线；
6. 平均期望成本不能保证稀有高威胁状态安全；
7. 当前 cost-value 跨批次证据不足，在线任务继续不授权；
8. 后续若提出新机制，必须相对
   `Vanilla / CRPO / Min-Max / GradS` 说明差异，而不能只说“协调安全与资源”。

## 21. 术语表与来源锚点

| 英文 | 本报告译法 | 原文锚点 |
| --- | --- | --- |
| multi-constraint safe RL | 多约束安全强化学习 | PDF pp.1–3 |
| multi-objective optimization, MOO | 多目标优化 | PDF pp.2–3 |
| constraint gradient | 约束梯度 | PDF p.4, Eq. 4 |
| conflicting constraints | 冲突约束 | PDF p.4, Def. 1 |
| redundant constraints | 冗余约束 | PDF p.5, Def. 2 |
| independent constraints | 独立约束 | PDF p.5, Def. 3 |
| candidate gradient set | 候选梯度集合 | PDF pp.5–6, Algorithm 1 |
| gradient shaping, GradS | 梯度塑形 | PDF pp.5–6 |
| Slater's condition | Slater 可行性条件 | PDF p.6, Assumption 1 |
| bounded and smooth gradients | 有界且光滑的梯度 | PDF p.6, Assumption 2 |
| noise ball | 噪声球/非零收敛邻域 | PDF p.7 |
| normalized cost, Cost-N | 最坏归一化成本 | PDF p.8, Eq. 14 |

### 关键来源索引

- 摘要、问题与贡献：PDF pp.1–2；
- 多约束 CMDP 和 Lagrangian：PDF p.3, Eqs. 1–3；
- MOO 梯度统一框架与三类基线：PDF p.4, Eqs. 4–7；
- 冲突、冗余、独立定义：PDF pp.4–5, Eqs. 8–10；
- GradS 伪代码与权重：PDF pp.5–6, Algorithm 1, Eq. 11；
- 假设和 Theorem 4：PDF pp.6–7, Eqs. 12–13；
- 任务、成本、基线和 Cost-N：PDF pp.7–8；
- 主结果、机制解释与尺度实验：PDF pp.8–10；
- 局限：PDF p.10。
