# LR-04 阅读报告：PASPO 约束分配、自回归初始化偏置与 AirDefense 适用边界

任务状态：`PASSED`  
完成时间：2026-07-29  
实验授权：否  
总体判决：`BASELINE + ADAPT`；原算法直接移植为 `AVOID`；训练后顺序问题为 `OPEN`

## 1. 论文身份

| 项目 | 内容 |
| --- | --- |
| 标题 | *Autoregressive Policy Optimization for Constrained Allocation Tasks* |
| 作者 | David Winkel、Niklas Alexander Strauß、Maximilian Bernhard、Zongyue Li、Thomas Seidl、Matthias Schubert |
| 会议 | NeurIPS 2024 |
| 算法 | Polytope Action Space Policy Optimization（PASPO） |
| 官方页面 | <https://openreview.net/forum?id=hRKsahifqj> |
| 正式论文 | <https://proceedings.neurips.cc/paper_files/paper/2024/file/d79c1390baa2e4835586b094d82e5ffb-Paper-Conference.pdf> |
| arXiv | <https://arxiv.org/abs/2409.18735> |
| 官方代码 | <https://github.com/niklasdbs/paspo> |

官方 OpenReview 页面将其列为 NeurIPS 2024 Poster，论文页面许可为 CC BY 4.0。
本报告以 24 页 NeurIPS 正式 PDF 为公式与页码基准。官方仓库仅做在线静态
阅读，没有下载、安装依赖或运行代码。

截至 2026-07-29，仓库主页显示 2 次提交、无 release，根目录未见独立
`LICENSE` 文件；README 还要求手工覆盖两个 Ray/RLlib 文件。因此代码可用于
核对实现，不应在未审计版本和许可证前直接 vendoring 到项目。

## 2. 一句话结论

PASPO 已经覆盖“在连续凸多面体上，以自回归策略直接生成满足线性硬约束的
资源分配，并校正顺序采样造成的初始联合分布偏置”这一主叙事；AirDefense
不能再把“自回归约束分配”本身称作创新。

但 PASPO 的去偏只是**固定约束下的策略初始化**，不改变 PPO 梯度、不改变
采样顺序，也不处理离散 unit-target matching、状态依赖合法集、每单元 no-op、
未来弹药责任或训练后的信用分叉。它对项目的正确身份是强邻近基线和初始化
思想来源，而不是可直接运行的等价算法。

## 3. Problem–Method–Insight

| 层 | 论文内容 |
| --- | --- |
| Problem | 每一步必须把总量为 1 的连续资源分配给多个实体，且分配向量必须落在由线性不等式定义的凸多面体内 |
| Method | 按固定实体顺序逐维采样；每一步用两个线性规划求当前坐标的完整可行区间，再用缩放 Beta 分布采样并更新剩余多面体 |
| Initialization | 从完整多面体近似均匀采样，拟合每个顺序位置的条件 Beta 参数，并写入输出层初始 bias |
| Optimization | 使用标准 PPO 联合优化自回归策略；联合 log-probability 是各条件分布 log-probability 之和 |
| Insight | 直接从可行域采样可避免事后投影的分布偏差；但逐条件“看似均匀”不等于联合空间均匀，初始化必须按完整可行域校准 |

## 4. 一页公式卡

### 4.1 约束分配 MDP

论文动作是连续分配向量：

\[
a=(a_1,\ldots,a_n),\qquad
a_i\ge 0,\qquad
\sum_{i=1}^{n}a_i=1.
\]

额外线性硬约束写为：

\[
Ca\le b.
\]

把非负、单纯形约束和业务线性约束合并后，可行动作空间是：

\[
\mathcal P
=
\{a\in\mathbb R_{\ge 0}^{n}\mid Ca\le b,\ \mathbf 1^\top a=1\}.
\]

其中 \(a_i\) 是分给实体 \(i\) 的资源比例，\(C\in\mathbb R^{m\times n}\)
和 \(b\in\mathbb R^m\) 定义凸多面体。论文也说明可以把违反量写成 CMDP
cost：

\[
\operatorname{CF}_k(s,a)
=
\max\{0,(Ca)_k-b_k\},
\]

但 PASPO 本身不靠累计 cost 学会守约束，而是把策略支持集直接限制为
\(\mathcal P\)。

### 4.2 联合策略的自回归分解

给定固定实体顺序，联合策略为（PDF p.4）：

\[
\pi_\theta(a\mid s)
=
\prod_{i=1}^{n}
\pi_\theta^i(a_i\mid s,a_{1:i-1}).
\]

因此：

\[
\log\pi_\theta(a\mid s)
=
\sum_{i=1}^{n}
\log\pi_\theta^i(a_i\mid s,a_{1:i-1}).
\]

这意味着 PPO 的概率比仍是完整联合动作的概率比：

\[
r_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
=
\exp\left(
\sum_i
\log\pi_\theta^i
-
\sum_i
\log\pi_{\theta_{\mathrm{old}}}^i
\right).
\]

PASPO 没有提出逐实体独立 PPO 更新或新的 credit estimator。

### 4.3 每一步可行区间

在已固定前缀 \(a_{1:i-1}\) 后，剩余变量满足：

\[
C^{(i)}a^{(i)}\le b^{(i)}.
\]

当前坐标的完整可行范围由两个 LP 给出（PDF pp.3–4）：

\[
a_i^{\min}
=
\min_{a^{(i)}} a_i
\quad
\text{s.t.}\quad
C^{(i)}a^{(i)}\le b^{(i)},
\]

\[
a_i^{\max}
=
\max_{a^{(i)}} a_i
\quad
\text{s.t.}\quad
C^{(i)}a^{(i)}\le b^{(i)}.
\]

采样 \(a_i\in[a_i^{\min},a_i^{\max}]\) 后，删除 \(C^{(i)}\) 的首列，并更新：

\[
b^{(i+1)}
=
b^{(i)}-a_i\,c_{:,i}.
\]

最后一个维度由单纯形剩余量确定：

\[
a_n=1-\sum_{i=1}^{n-1}a_i.
\]

附录 D 的 Theorem 1 证明：只要初始多面体非空，每一步从上述完整区间中
取值都保留至少一个可行后缀；PASPO 能生成的点集恰好等于整个
\(\mathcal P\)，既不会生成域外动作，也没有漏掉域内点。

### 4.4 缩放 Beta 条件策略

在可行区间 \([l,u]\) 上，论文使用四参数 Beta 密度：

\[
p(x;\alpha,\beta,l,u)
=
\frac{(x-l)^{\alpha-1}(u-x)^{\beta-1}}
{(u-l)^{\alpha+\beta-1}B(\alpha,\beta)}.
\]

神经网络为每个实体输出 \(\alpha_i,\beta_i\)，区间 \(l_i,u_i\) 则由 LP
确定。Beta 不是理论必要条件；论文只要求使用有界、可微的条件分布。

### 4.5 初始偏置的精确定义

若在无额外约束的单纯形上，每一步都从当前剩余区间使用相同“均匀”条件
分布，则：

\[
\mathbb E[a_1]=\frac12,\qquad
\mathbb E[a_2]=\frac14,\qquad
\mathbb E[a_3]=\frac18,\ldots
\]

直到最后两个坐标因剩余和约束具有相同均值。越晚采样的实体越难获得大比例
分配。这是**自回归参数化的初始联合分布偏置**，不是环境奖励偏置、策略梯度
信用偏差或训练后角色偏置。

### 4.6 PASPO 去偏

令 \(k\) 为完整多面体样本数。Algorithm 1 的过程是：

1. 通过“先从标准单纯形采样、再拒绝多面体外点”得到
   \(a^{(j)}\sim\operatorname{Unif}(\mathcal P)\)；
2. 对每个样本 \(j\) 和位置 \(i\)，沿该样本真实前缀重新求
   \([a_{ji}^{\min},a_{ji}^{\max}]\)；
3. 把样本在条件区间中的位置归一化：

\[
z_{ji}
=
\frac{a_{ji}-a_{ji}^{\min}}
{a_{ji}^{\max}-a_{ji}^{\min}};
\]

4. 对每个位置 \(i\) 的 \(\{z_{ji}\}_{j=1}^{k}\) 做 Beta 最大似然拟合：

\[
(\hat\alpha_i,\hat\beta_i)
=
\arg\max_{\alpha,\beta}
\sum_{j=1}^{k}\log
\operatorname{Beta}(z_{ji};\alpha,\beta);
\]

5. 把拟合值变换后加到相应策略头输出层的初始 bias。

它只让训练开始时的联合样本**近似**多面体均匀分布。有限拒绝样本、按位置
拟合单一 Beta 以及网络状态输入都会带来近似误差，因此论文没有证明初始化
后联合分布严格等于均匀分布。

### 4.7 熵

联合策略熵没有正文使用的简单闭式，论文用当前策略样本估计：

\[
H_{\mathrm{emp}}(\pi_\theta(\cdot\mid s))
=
\mathbb E_{a\sim\pi_\theta}
\left[
\sum_{i=1}^{n}
H\left(
\pi_\theta^i(\cdot\mid s,a_{1:i-1})
\right)
\right].
\]

若用于 off-policy 算法，必须从当前策略重采样动作；旧行为策略样本不能直接
给出当前联合熵。

## 5. 算法流程与复杂度

### 5.1 推理/rollout

```text
输入：状态 s、固定多面体 (C,b)、固定实体顺序 σ
编码：h = state_encoder(s)
初始化：剩余资源 = 1，C(1)=C，b(1)=b
for i = 1,...,n-1:
    用 LP 求当前坐标的最小值和最大值
    策略头读取 h、已采样前缀（实现还可读取旧参数和区间）
    构造缩放 Beta(αi,βi,amin,amax)
    采样 ai 或取其确定性代表值
    更新 b(i+1)，删除已固定坐标
an = 1 - sum(a1,...,a(n-1))
返回完整联合动作与各条件 log-probability 之和
```

设动作维数为 \(n\)、约束数为 \(m\)。每个联合动作最多需要
\(2(n-1)\) 个逐渐缩维的 LP，再加 \(n-1\) 个策略头前向。论文没有给出
统一 wall-clock 复杂度，因为 LP 代价依赖求解器、约束几何和并行方式。

官方实现的
[`generic_autoregressive_distribution.py`](https://github.com/niklasdbs/paspo/blob/main/src/action_distributions/generic_autoregressive_distribution.py)
确实逐头调用 `solve_min_max`，并提供多进程 LP 分支；最后一维用
`1 - sum(prefix)` 构造。每次新建动作分布或按给定动作重算 log-probability
都需要沿前缀重建区间，因此 PPO rollout 和更新均会承受该开销。

### 5.2 初始化

去偏额外包含：

- 从单纯形拒绝采样，期望提议次数与
  \(\operatorname{Vol}(\mathcal P)/\operatorname{Vol}(\text{simplex})\)
  的倒数相关；狭窄多面体可能非常昂贵；
- 对 \(k\) 个样本和 \(n-1\) 个位置求区间，约
  \(2k(n-1)\) 次 LP；
- \(n-1\) 组 Beta MLE。

该成本只发生在初始化，但不能解决论文已承认的每个在线动作都要串行解 LP
的问题。

### 5.3 代码与论文一致性

静态代码核对结果：

- [`autoregressive_model.py`](https://github.com/niklasdbs/paspo/blob/main/src/models/autoregressive_model.py)
  在模型初始化时一次性加载固定 \(A,b\)；
- 创建 \(n-1\) 个独立策略 head，最后一维不设 head；
- 开启 `uniform_bias_init` 时，把拟合参数的对数加到最后层 bias；
- 动作分布按前缀更新 \(A,b\)，联合 log-probability 由列表分布求和；
- 代码可以把前缀动作、旧分布参数和可行边界输入后续 head；
- 实现中没有离散 action mask；它的“mask 逻辑”等价物是 LP 求出的连续
  可行区间，区间外点在条件分布中没有支持；
- 代码结构印证论文的限制：约束不是由当前状态动态生成。

## 6. 机制成立的关键假设

| 假设 | PASPO 需要什么 | AirDefense v1 判定 |
| --- | --- | --- |
| 连续动作 | 每个坐标可在区间内取任意实数 | 不满足；每单元选离散目标或 no-op |
| 凸可行域 | \(Ca\le b\) 的解集是非空凸多面体 | 原始联合指派集是有限非凸集合 |
| 完整分配 | 总资源满足 \(\sum_i a_i=1\) | 不满足；多个单元可分别 no-op，弹药可保留到未来 |
| 固定维数与语义 | 每个位置始终对应同一分配实体 | 单元数虽固定，但合法目标和角色关系随状态变化 |
| 固定约束 | 官方实现启动时加载同一个 \(A,b\) | 不满足；存活目标、占用前缀、弹药和冷却共同改变合法集 |
| 可解 LP | 每个前缀的 min/max 可及时求解 | 离散指派不应以连续 LP 区间代替 |
| 可近似均匀采样 | 从完整多面体取得足够均匀样本 | 离散可行联合动作应做计数/枚举，不是体积采样 |
| 初始化是主要顺序源 | 早期覆盖不足导致最终次优 | 项目只部分满足；Task 10–12 还观察到训练后 no-op 与种子分叉 |

即使把二元 assignment 写成线性约束，PASPO 的 Beta 策略仍会生成分数指派。
若再取整或投影，便失去其“直接从策略支持集生成全部且仅有可行动作”的性质，
也重新引入论文试图避免的校正偏差。

## 7. 论文证据

### 7.1 实验协议

论文使用三个连续分配任务：

1. 13 资产的 portfolio optimization；
2. 9 台服务器的 compute load distribution；
3. 7 维、611 个线性约束的 synthetic benchmark。

PASPO-PPO 与 CPO、CUP、IPO、P3O、Lagrangian PPO 和 OptLayer 比较。除
portfolio 使用 250k 步、10 个种子外，主要任务使用 150k 步、5 个种子；
网络为小型两层 MLP，训练在 CPU 上进行。

### 7.2 作者证据支持到哪里

- Figure 4 显示 PASPO 在三个任务上获得作者报告的更快收敛和更高平均回报；
- 软约束方法在训练/评估中仍有违反，PASPO 与 OptLayer 保持零违反；
- Figure 5a 在仅有单纯形约束的 synthetic setting 中显示，去偏初始化相对
  随机初始化收敛更快且避免较早停在次优解；
- Figure 5b 把实体顺序反转后没有观察到显著性能差，作者据此认为去偏提高了
  顺序鲁棒性。

### 7.3 证据边界

Figure 5 的顺序结论只来自：

- 连续分配；
- 固定维数；
- 除单纯形外无额外约束的 synthetic benchmark；
- 正序与完全反序两种顺序；
- 小规模 MLP/PPO。

它没有检验异质资源角色、状态依赖 mask、离散目标唯一占用、每单元 no-op、
deterministic argmax 或训练种子级单元塌缩。因此“反序无显著差异”不能外推
为 PASPO 已解决一般自回归顺序偏置。

论文明确承认两个限制：

1. 每个动作需要一串 LP，高维实体数下显著昂贵；
2. 当前不支持 state-dependent constraints，作者把它列为未来工作。

论文只保证**当前动作满足硬约束**。软累计约束仍需与 Safe RL 方法组合，
论文没有给出未来弹药预算、安全责任或跨时域覆盖的保证。

## 8. PASPO—Task 8/9/10/11 五层差异矩阵

| 层 | PASPO | Task 8 | Task 9 | Task 10 | Task 11 | 项目判定 |
| --- | --- | --- | --- | --- | --- | --- |
| Problem | 连续总量按比例分配 | 枚举离散无冲突联合指派 | 固定顺序离散指派 | 诊断 012/120/201 顺序 | 共享 unit-target/no-op scorer | 都属于约束分配邻域，但任务语义不等价 |
| Feasible set | 固定凸多面体 \(Ca\le b\) | 状态依赖的 136 类联合动作子集 | 前缀动态屏蔽已占目标 | 同 Task 9，改变单元顺序 | 同 Task 9，另加角色条件评分 | AirDefense 是有限组合集合，不是连续凸域 |
| Action distribution | 区间缩放 Beta，最后一维为剩余量 | 单个联合 categorical | 条件 categorical 乘积 | 三个固定排列的条件 categorical | 共享关系分数与独立 no-op 分数 | PASPO 不能直接替换现有动作头 |
| Bias/optimization | 只校正初始联合分布；标准 PPO | 联合枚举导致资源使用增加 | 训练后仍有种子敏感性 | 顺序改变单元参与和资源成本 | 表示共享仍未消除塌缩和顺序跨度 | 项目剩余问题不只是初始化边际偏置 |
| Evidence/failure | 连续 synthetic 反序消融；高维与状态依赖约束未覆盖 | 零冲突但 time 成本门槛失败 | 零冲突、成本恢复，高威胁门槛差 0.00517 | 201 降泄漏但 time 成本增加 4.253 | 012 仍有 5 个塌缩单元，时延增加 73.51% | PASPO 证据不足以解释或消除项目失效 |

### 8.1 对 Task 8

Task 8 的 `Discrete(136)` 与 PASPO 都把结构约束放进动作生成而不是奖励惩罚，
并都能保证当前动作合法。但前者完整枚举离散匹配，后者覆盖连续凸多面体。
两者共同否定“事后仅靠软惩罚即可等价保证硬合法”的表述。

### 8.2 对 Task 9

Task 9 已经实现 PASPO 最一般的结构思想：按前缀生成条件动作、动态删除已占
目标、联合 log-probability 求和，并取得零冲突。项目不需要 PASPO 才能证明
该工程性质。PASPO 新增的可借鉴部分仅是“初始联合分布应相对完整可行域校准”。

### 8.3 对 Task 10

Task 10 支持的顺序效应包括：

- `012` 第二枚导弹接近塌缩；
- `120` 最后一枚导弹参与明显下降；
- `201` 增加两枚导弹参与，同时显著增加资源成本；
- 不同顺序改变 `unassigned`、`prefix_denied` 和 `attempted_miss` 构成。

这些是学习后、异质角色和 no-op 共同形成的行为差异。PASPO 的初始化校准最多
能排除其中“初始条件分布让先决策位置天然获得更多质量”的一部分，不能直接
解决角色—目标价值、交战强度或长期成本权衡。

### 8.4 对 Task 11 与 all-noop

PASPO 的单纯形强制总资源全部分配，最后一维吸收剩余量；它没有每个资源各自
选择 no-op 的决策边界。把 no-op 当成一个连续分配实体，只能表示“总量的一部分
不分配”，不能表达三个异质单元分别交战或保留弹药。

Task 11 的主要失败发生在“是否交战”，不是“交战后选哪个目标”。Task 12
进一步显示 deterministic argmax 会放大 no-op，且 PPO 训练会形成不开火/高成本
开火的种子分叉。PASPO 的 Beta bias 初始化和 Figure 5 均没有覆盖这一边界。

## 9. 硬约束可行与未来资源负责

必须区分三件事：

\[
\text{当前合法}
\neq
\text{当前高效}
\neq
\text{对未来资源负责}.
\]

PASPO 保证：

\[
a_t\in\mathcal P
\quad\text{对每个当前时刻 }t.
\]

它不保证：

\[
\mathbb E_\pi\left[\sum_t c_t\right]\le d,
\]

也不保证当前分配不会消耗未来高威胁目标所需资源。若把弹药、冷却或安全风险
写成累计约束，仍需 CMDP/Safe RL、规划或其他跨时域机制。论文自己也只称其
硬约束模块可与处理软累计约束的方法组合。

因此 PASPO 不能替代：

- 项目的 resource cost 门槛；
- all-noop/engage 稳定性诊断；
- N1 的全局成本与局部解释边界；
- FCRC 一类未来可覆盖性分析；
- 顺序信用估计。

## 10. 可迁移接口与数学不适用点

### 10.1 可迁移

1. **可行支持集内直接采样**：继续保留 Task 9 的条件 mask，而不是事后修复；
2. **联合概率契约**：PPO ratio 使用全部条件 categorical 的联合概率；
3. **初始化分布审计**：不能只看每个条件头“近似均匀”，必须看完整可行联合
   动作的边际和顺序差；
4. **初始化与训练机制分离**：若适配 PASPO，去偏只能在 PPO 前发生，训练中
   不增加辅助损失，才能判定它是否解释早期探索问题；
5. **顺序反事实对照**：同一联合目标分布应在 012/120/201 三种分解下具有相同
   初始边际，再观察学习后是否重新分叉。

### 10.2 不可直接迁移

- 连续区间 Beta 不能表示离散目标选择；
- 凸多面体体积均匀不等于有限匹配集合计数均匀；
- 固定 \(C,b\) 不能表达状态依赖目标存活、射程、冷却和弹药；
- 总量为 1 的完整分配不能表达每单元 no-op；
- LP 区间与最后剩余量不能保证二元匹配的积分性；
- 初始化 bias 不能替代 credit assignment、价值估计或长期预算；
- Figure 5 的反序结果不能当作异质顺序鲁棒性的理论保证。

## 11. 最小强基线：离散可行后缀计数均匀初始化

### 11.1 为什么不用连续 PASPO

在 AirDefense 主环境直接运行原 PASPO，需要把离散指派放松成连续变量，再
取整、投影或重定义环境动作。这会同时改变动作语义、可行性保证和资源行为，
无法与 Task 9 公平比较。因此原 PASPO 不应作为直接 runnable baseline。

### 11.2 PASPO 思想的离散严格对应

给定状态 \(s\)、顺序 \(\sigma\) 和合法联合指派集合
\(\mathcal F(s)\)。令前缀为 \(h_i\)，定义：

\[
N_\sigma(s,h_i)
=
\left|
\left\{
a\in\mathcal F(s):
a_{\sigma_{1:i-1}}=h_i
\right\}
\right|,
\]

即该前缀仍有多少个合法联合后缀。则初始条件分布设为：

\[
q_i^\sigma(x\mid s,h_i)
=
\frac{
N_\sigma(s,h_i\cup\{a_{\sigma_i}=x\})
}{
N_\sigma(s,h_i)
}.
\]

对任意 \(a\in\mathcal F(s)\)，条件概率连乘会望远镜消去：

\[
\prod_i q_i^\sigma(a_{\sigma_i}\mid s,h_i)
=
\frac{1}{|\mathcal F(s)|}.
\]

所以它在**有限可行联合动作集合上严格均匀**，并且所得联合分布与分解顺序
无关。它是 PASPO“完整可行域均匀初始化”思想在离散组合空间中的数学对应，
不需要把匹配松弛为连续多面体。

### 11.3 最小可比实现定义

若后续单独授权实现，最小基线应满足：

```text
环境、奖励、Critic、PPO、网络宽度、训练预算全部冻结
保留 Task 9 的冲突自由自回归 categorical 与联合 PPO ratio
在不读取 reward/outcome 的冻结状态语料上枚举可行后缀
PPO 开始前，把策略条件分布蒸馏到 q_i^σ
蒸馏完成后关闭全部初始化损失
分别训练 012 / 120 / 201
比较 naive init 与 completion-count init
```

当前 3 单元、最多 5 目标的规模允许精确枚举。若以后规模扩大，可以用动态
规划或近似 completion count，但近似版必须另行审计。

### 11.4 必须保留 no-op，但不能误读均匀

no-op 应作为每个单元的合法离散动作参与后缀计数。可是“联合动作计数均匀”
不是规范上最优的交战先验：可行组合的数量本身可能让 engage 或 no-op 的边际
概率偏高。因此该基线只用于消除**顺序参数化初始化偏置**，不能声称提供正确
交战率。报告时必须同时给出：

- 初始联合分布到计数均匀目标的 KL/TV；
- 各单元 engage/no-op 边际；
- 012/120/201 的初始边际跨度；
- deterministic 与 stochastic all-noop；
- 训练后奖励、毁伤、泄漏、成本、塌缩单元和决策时延。

### 11.5 可证伪解释

| 结果 | 能支持的解释 |
| --- | --- |
| 初始顺序跨度消失，训练后仍重新出现 | PASPO 式初始化偏置存在，但不是 Task 10–12 的主要瓶颈 |
| 初始与训练后跨度都下降，且成本/安全门槛通过 | 初始化覆盖是顺序分叉的重要来源；可保留为强基线 |
| 初始分布无法拟合计数目标 | 网络/状态表示不足，不能把失败归因于 PPO |
| all-noop 仍分叉 | 去偏不处理 engage/no-op 优化稳定性 |
| 只改善某一顺序或某一资源类型 | 仍是异质角色或状态分布问题，不是通用顺序去偏 |

这一定义是 `ADAPT` 候选基线，不是本任务已经实现或验证的算法。

## 12. 创新覆盖压力测试

### 12.1 已被 PASPO 覆盖的主张

以下表述必须删除或收窄：

1. “首次用自回归策略生成满足硬约束的资源分配”；
2. “首次发现顺序采样会造成早期资源偏置”；
3. “首次用完整可行域的均匀目标校正自回归初始化”；
4. “自回归分解天然对顺序鲁棒”；
5. “当前动作合法即可保证资源安全”。

### 12.2 项目仍可能保留的差异

| 差异 | 当前身份 | 需要什么证据 |
| --- | --- | --- |
| 状态依赖离散匹配的 completion-count 初始化 | 强基线候选；与 PASPO 高度相邻 | 先做查新，再以初始化-only 对照验证 |
| 训练后异质角色与 no-op 顺序分叉 | 项目已有现象证据，尚无解决方法 | 区分初始化覆盖、优化稳定和顺序信用 |
| 当前合法与未来资源责任的接口 | `OPEN`；PASPO 不处理 | 必须另行冻结规范目标并与 CMDP 强基线比较 |

不能把“从连续改成离散”或“应用到防空”单独作为创新。可辩护的新方法至少
需要解决 PASPO 没有定义的状态依赖组合支持、no-op 决策或跨时域责任之一，
并通过相应强基线。

## 13. `BASELINE / ADAPT / AVOID / OPEN` 判决

| 标签 | 判决 |
| --- | --- |
| `BASELINE` | PASPO 是“自回归硬约束分配 + 初始化去偏”的最近强邻近工作，后续论文与算法设计必须引用和比较 |
| `ADAPT` | 只适配“相对完整可行联合动作校准初始化”的原则；离散版用可行后缀计数，不使用连续 Beta/LP |
| `AVOID` | 不把连续 PASPO 直接移植到离散 WTA，不经松弛取整，不把其初始化参数加入 reward、advantage、loss、mask 或 shield |
| `OPEN` | 初始化校准能否缓解 Task 10–12 的训练后顺序/塌缩问题尚未验证；它也不回答顺序信用和未来资源责任 |

总体决策：

```text
把 PASPO 作为自回归约束分配强基线
                    ↓
删除“自回归本身是创新”的叙事
                    ↓
如后续获准，只实现离散 completion-count 初始化消融
                    ↓
不改 PPO，不启动本任务中的任何实验
```

## 14. 移交 LR-05

LR-05 必须携带以下语义边界：

1. PASPO 的 bias 是**动作生成初始化偏置**；
2. 它来自条件分布连乘与剩余可行域，不是 advantage/critic 估计偏差；
3. 去偏不随机化顺序，不做顺序智能体的逐步策略改进；
4. Task 10–12 的训练后角色参与、no-op 和种子分叉不能自动归因于该偏置；
5. CAPO/COSAC 若讨论 sequential credit，必须说明估计对象、联合概率与更新
   顺序，不能把 credit correction 与 PASPO initialization 混为同一机制；
6. LR-05 的强压力测试应包含：即使初始联合分布已顺序无关，顺序信用方法是否
   仍能提供独立增益。

## 15. 术语表与来源锚点

| 英文 | 本报告译法 | 原文锚点 |
| --- | --- | --- |
| constrained allocation task | 约束分配任务 | PDF pp.2–3 |
| convex polytope action space | 凸多面体动作空间 | PDF pp.1, 3 |
| autoregressive policy | 自回归策略 | PDF pp.3–4 |
| feasible interval | 可行区间 | PDF pp.3–4, Eq. 2 |
| four-parameter beta distribution | 四参数 Beta 分布 | PDF p.4 |
| initialization bias | 初始化偏置 | PDF pp.5–6 |
| de-biasing term | 去偏项 | PDF p.6, Algorithm 1 |
| empirical entropy | 经验联合熵 | PDF p.5, Eq. 3 |
| constraint violation | 约束违反 | PDF pp.7–8 |
| allocation order | 分配顺序 | PDF p.8, Figure 5 |

### 关键来源索引

- 摘要、贡献与方法定位：PDF pp.1–2；
- MDP、线性约束和 CMDP 对照：PDF p.3；
- 可行区间、自回归分解、Beta 策略：PDF pp.3–4；
- Figure 2 可行分配过程：PDF p.4；
- Figure 3、Algorithm 1 与去偏：PDF pp.5–6；
- 三个环境与实验协议：PDF pp.6–7；
- 性能、约束违反、初始化与顺序消融：PDF pp.7–8；
- 局限与结论：PDF pp.8–9；
- 环境细节：PDF pp.12–14；
- 网络结构与超参数：PDF pp.14–15；
- 可行性与完整支持证明：PDF pp.15–17；
- 官方实现动作构造：
  `src/action_distributions/generic_autoregressive_distribution.py`；
- 官方实现策略头和去偏 bias：
  `src/models/autoregressive_model.py`。
