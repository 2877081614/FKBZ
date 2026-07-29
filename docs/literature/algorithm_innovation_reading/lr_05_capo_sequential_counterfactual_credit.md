# LR-05 阅读报告：COSAC 顺序反事实信用与 AirDefense 动态支持边界

任务状态：`PASSED`  
完成时间：2026-07-29  
实验授权：否  
论文身份：2026 年 arXiv 预印本，未按同行评审结论处理  
总体判决：SeqAU/COSAC 为 `BASELINE`；前缀条件 estimand 与理论审计可
`ADAPT`；当前直接接入 MCH/BPCE 为 `AVOID`；完整 MDP、动态支持和
非加性交互下的可靠估计为 `OPEN`

## 1. 版本核验记录

| 项目 | v1 | 当前 v2 |
| --- | --- | --- |
| arXiv 标识 | `2604.17693v1` | `2604.17693v2` |
| 日期 | 2026-04-20 | 2026-05-09 |
| 标题 | *CAPO: Counterfactual Credit Assignment in Sequential Cooperative Teams* | *COSAC: Counterfactual Credit Assignment in Sequential Cooperative Teams* |
| 算法全名 | Counterfactual Advantage Policy Optimization | COunterfactual Sequential Credit Assignment in Cooperative Teams |
| 核心估计量 | SeqAU | SeqAU |
| 主要实验 | 顺序合作 bandit | bandit，并新增四个 Qwen3-0.6B 代理的 ARC 实验 |
| 理论编号 | SeqAU Thm. 1；bias Thm. 2；variance Thm. 3；gradient-MSE Thm. 4 | SeqAU Thm. 2；bias Thm. 3；variance Thm. 4 |

本报告以当前 v2 为主版本。CAPO 和 COSAC 不是两个独立算法，也不能把改名
当作方法差异。后文仅在讨论历史版本时使用 CAPO。

v1 的 gradient-MSE Theorem 4 在当前 v2 中不再出现。该旧定理分析的是从
当前联合策略取得理想 on-policy 样本的估计器，并明确说实际 CAPO 梯度使用
旧批次 \(\mu\) rollout 与虚拟后缀，其精确有限样本分析留待未来。当前 v2
不能再被引用为已经给出实际 COSAC PPO 梯度 MSE 保证。

官方入口：

- 当前摘要页：<https://arxiv.org/abs/2604.17693>；
- 当前 v2 HTML：<https://arxiv.org/html/2604.17693v2>；
- 当前 v2 PDF：<https://arxiv.org/pdf/2604.17693>；
- v1 摘要页：<https://arxiv.org/abs/2604.17693v1>。

## 2. 一句话结论

SeqAU 定义了固定前缀下，当前动作对当前策略后缀期望团队回报的增量：

\[
A_k^{\mathrm{SeqAU}}(a_{\le k})
=
\mathbb E_\pi[R\mid a_{\le k}]
-
\mathbb E_\pi[R\mid a_{<k}].
\]

COSAC 用批内加性回报 ridge 拟合和无需环境调用的虚拟策略后缀近似该量。
这直接覆盖了“顺序前缀条件反事实 advantage、上游抵消和下游间接效应”的
核心叙事，但只在单轮 sequential bandit、离散固定动作槽和可用回报代理的
范围内成立。

AirDefense 的同一步后缀占用与 COSAC 高度相邻；跨环境时间的命中、状态演化、
弹药替代和终止效应不在论文模型中。项目不能继续把普通 MCH 顺序反事实分解
写成独立创新，但也不能把 COSAC 的 action-only additive surrogate 直接
当成 AirDefense 的完整回报 Critic。

## 3. 论文问题设定

给定上下文 \(x\sim\mathcal D\)，\(K\) 个 agent 按固定顺序行动：

\[
a_k\sim\pi_k(\cdot\mid a_{<k},x),\qquad
\pi(a\mid x)=\prod_{k=1}^{K}\pi_k(a_k\mid a_{<k},x).
\]

联合动作产生一个有界标量团队回报：

\[
R(a,x)\in[-R_{\max},R_{\max}],\qquad
f(a,x)=\mathbb E[R\mid a,x].
\]

论文同时固定两种顺序：

1. **execution order**：谁先行动、后续 agent 能看到什么前缀；
2. **update order**：同一批数据内谁先更新策略。

正文令二者都是自然顺序 \(1,\ldots,K\)。批次由旧联合策略 \(\mu\) 采集。
更新 agent \(k\) 时，上游策略已变成新策略，而当前及下游策略仍等于旧策略：

\[
\pi_{\ge k}^{(k-1)}=\mu_{\ge k}.
\]

这条等式是 COSAC 不使用累计前缀重要性采样的关键，不是任意更新顺序下都成立。

论文采用 contextual bandit：每个 agent 在一个 episode 中只行动一次，
不存在 agent 轮次之间的环境状态转移。正文明确把完整 sequential MDP 扩展
留给后续工作。

## 4. Sequential learnability 公式卡

### 4.1 前缀条件差异效用

论文考虑：

\[
U_k(a_{<k},a_k,a_{>k})
=R(a_{<k},a_k,a_{>k})-b_k(a_{<k}).
\]

baseline 只能依赖 agent \(k\) 行动时已经观察到的前缀，不能把尚未产生的
后缀当成自由变量。

定义后缀平均效用：

\[
\bar U_k(a_k;a_{<k})
=
\mathbb E_{a_{>k}\sim\pi_{>k}(\cdot\mid a_{\le k})}
[U_k(a_{<k},a_k,a_{>k})].
\]

### 4.2 顺序可学习性

对参考 kernel \(\rho(\cdot\mid a_{<k})\) 和两个焦点动作
\(a_k^1,a_k^2\)，v2 Definition 1 定义：

\[
\Lambda_k^{\mathrm{seq}}
=
\frac{
\mathbb E_{a_{<k}}
[\bar U_k(a_k^1;a_{<k})-\bar U_k(a_k^2;a_{<k})]
}{
\sqrt{
\mathbb E_{a_{<k},a_k\sim\rho}
[
\operatorname{Var}_{(a_{<k},a_{>k})\mid a_k}[U_k]
]
}
}.
\]

分子表示焦点动作变化带来的信号，分母表示固定焦点动作后剩余的队友噪声。
“learnability 最大”是信噪比结论，不是策略性能、约束满足或全局最优结论。

### 4.3 唯一性边界

v2 Theorem 2 给出：

\[
b_k^*(a_{<k})
=
\mathbb E_{
a_k'\sim\rho(\cdot\mid a_{<k}),
a_{>k}'\sim\pi_{>k}(\cdot\mid a_{<k},a_k')
}
[R(a_{<k},a_k',a_{>k}')].
\]

当 \(\rho=\pi_k\) 时：

\[
b_k^*(a_{<k})=\mathbb E_\pi[R\mid a_{<k}].
\]

SeqAU utility 为：

\[
g_k^{\mathrm{SeqAU}}
=R(a_{<k},a_k,a_{>k})-b_k^*(a_{<k}).
\]

唯一性只成立于：

```text
所有 prefix-conditional difference utilities
{R - b_k(a_<k)}
```

并只在前缀边际分布的支持上唯一。它不是在所有可学习 baseline、所有 Critic
架构或所有控制变量中全局唯一。

### 4.4 SeqAU advantage

\[
A_k^{\mathrm{SeqAU}}(a_{\le k})
=
\mathbb E_\pi[R\mid a_{\le k}]
-
\mathbb E_\pi[R\mid a_{<k}].
\]

这个 estimand 同时包括：

- 当前动作对自己回报槽位的直接影响；
- 当前动作改变后续 agent 行为分布产生的间接影响。

它不等于 LR-01 的轨迹条件 TCFE/ASE。SeqAU 是目标策略分布下的期望
advantage；TCFE/ASE 是给定事实轨迹和结构因果自然干预下的回顾性解释量。

## 5. 为什么该 baseline 最大化 learnability

对任意 \(b_k(a_{<k})\)，动作对之间的效用差中 baseline 消去，所以
learnability 分子不依赖 \(b_k\)。分母经全方差分解后，唯一依赖 baseline
的部分等价于：

\[
\min_{b_k}
\mathbb E_{a_{<k},a_k\sim\rho}
\left[
(\bar R_k(a_k;a_{<k})-b_k(a_{<k}))^2
\right].
\]

该条件平方损失在支持上的唯一最优解就是条件均值 \(b_k^*\)。因此，唯一性
来自条件均方误差最小化，不来自新的策略梯度定理。

## 6. COSAC 的加性回报模型

论文假设或近似：

\[
f(a)=\mathbb E[R\mid a]
\approx
\hat f(a)=\sum_{k=1}^{K}\hat\varphi_k(a_k).
\]

每个 agent 动作用 one-hot 编码，拼接为
\(\psi(a)\in\{0,1\}^{d}\)，其中 \(d=KA\)。批次矩阵
\(\Psi\in\mathbb R^{N\times d}\)，回报向量 \(r\in\mathbb R^N\)，ridge 解为：

\[
\hat\varphi
=
(\lambda I+\Psi^\top\Psi)^{-1}\Psi^\top r.
\]

该步骤：

- 不使用 Bellman bootstrap；
- 不维护长期 \(V/Q\) 网络；
- 但仍然拟合了一个**批内团队回报预测模型**；
- “critic-free”不等于“不学习任何回报模型”。

在 ARC 中，连续文本输出先被离散成
`A/B/C/D/no-answer` 五类，之后才使用该加性模型。论文没有对一般连续动作
或不定长结构动作给出同样构造。

## 7. 上游抵消与 direct/indirect advantage

令：

\[
Q_k^{\mathrm{LS}}(a_{\le k})
=
\mathbb E_{\pi^{(k-1)}}[\hat f\mid a_{\le k}],
\]

\[
V_k^{\mathrm{LS}}(a_{<k})
=
\mathbb E_{\pi^{(k-1)}}[\hat f\mid a_{<k}].
\]

加性模型下：

\[
Q_k^{\mathrm{LS}}
=
\sum_{j<k}\hat\varphi_j(a_j)
+\hat\varphi_k(a_k)
+\sum_{j>k}
\mathbb E[\hat\varphi_j(a_j)\mid a_{\le k}],
\]

\[
V_k^{\mathrm{LS}}
=
\sum_{j<k}\hat\varphi_j(a_j)
+\mathbb E[\hat\varphi_k(a_k)\mid a_{<k}]
+\sum_{j>k}
\mathbb E[\hat\varphi_j(a_j)\mid a_{<k}].
\]

两式相减，上游项逐项消去：

\[
\hat A_k^{\mathrm{LS}}=D_k+I_k,
\]

\[
D_k
=
\hat\varphi_k(a_k)
-
\mathbb E_{a_k'\sim\pi_k(\cdot\mid a_{<k})}
[\hat\varphi_k(a_k')],
\]

\[
I_k
=
\sum_{j>k}
\left(
\mathbb E[\hat\varphi_j(a_j)\mid a_{\le k}]
-
\mathbb E[\hat\varphi_j(a_j)\mid a_{<k}]
\right).
\]

解释：

- \(D_k\)：焦点 agent 自己回报槽位的动作差；
- \(I_k\)：焦点动作改变后续策略分布，进而改变下游回报槽位的效应；
- 若策略完全因子化且下游不读取前缀，\(I_k=0\)；
- 上游抵消依赖加性回报项只读取各自 \(a_j\)，不是一般非加性回报恒等式。

## 8. Fictitious continuation

对实际前缀和焦点动作采样后缀：

\[
\tilde a_{>k}^{(n,\ell)}
\sim
\pi_{>k}^{(k-1)}(\cdot\mid a_{\le k}^{(n)}).
\]

对同一前缀重新采样焦点动作及后缀：

\[
\tilde a_k^{(n,\ell)}
\sim\pi_k^{(k-1)}(\cdot\mid a_{<k}^{(n)}),
\]

\[
\tilde a_{>k}^{\prime(n,\ell)}
\sim
\pi_{>k}^{(k-1)}
(\cdot\mid a_{<k}^{(n)},\tilde a_k^{(n,\ell)}).
\]

Monte Carlo advantage：

\[
\hat A_k^{(n)}
=
\frac1L\sum_{\ell=1}^{L}
\left[
\hat f(a_{\le k}^{(n)},\tilde a_{>k}^{(n,\ell)})
-
\hat f(a_{<k}^{(n)},\tilde a_k^{(n,\ell)},
\tilde a_{>k}^{\prime(n,\ell)})
\right].
\]

虚拟后续只调用当前策略前向和 \(\hat f\) 表查找：

- 不调用真实环境；
- 不重新求真实 reward；
- 不保持环境随机带；
- 不模拟下一环境状态；
- 不等于 BPCE 的环境快照共同随机数 rollout；
- 不等于 C3 的真实下游 pipeline replay。

它之所以便宜，是因为把环境与长期回报替换成了已拟合的批内 surrogate。

## 9. COSAC 一轮算法

```text
用旧联合策略 μ 收集 N 个联合动作和团队回报
                    ↓
用 one-hot joint action 对回报做一次 ridge 拟合
                    ↓
按自然 execution order 依次处理 k=1,...,K
                    ↓
从仍未更新的 π_{>=k}=μ_{>=k} 采样 L 个虚拟后缀
                    ↓
计算每个 rollout 的 SeqAU 近似 advantage
                    ↓
对 agent k 做 M 个 PPO-clipped 内步
```

每轮额外成本：

\[
O((KA)^3)+NKL\text{ 次策略 continuation}+K M\text{ 次更新}.
\]

论文通过复用前一 agent 的 Q-set 作为后一 agent 的 V-set，把朴素
\(2NKL\) 降为 \(NKL\)。

## 10. 非加性残差与 bias bound

定义：

\[
\varepsilon(a)=\hat f(a)-f(a).
\]

对焦点动作的残差敏感度：

\[
\delta_k^\varepsilon(a_{-k})
=
\max_{a_k,a_k'}
\left|
\varepsilon(a_{<k},a_k,a_{>k})
-
\varepsilon(a_{<k},a_k',a_{>k})
\right|.
\]

v2 Theorem 3：

\[
|\Delta_k(a_{\le k})|
\le
\mathbb E_{\pi_{\ge k}\mid a_{<k}}
[\delta_k^\varepsilon(a_{-k})]
+
\|\varepsilon\|_\infty
\max_{a_k,a_k'}
\left\|
\pi_{>k}(\cdot\mid a_{<k},a_k)
-
\pi_{>k}(\cdot\mid a_{<k},a_k')
\right\|_1.
\]

两个 bias 通道：

1. **residual sensitivity**：非加性交互是否随焦点动作改变；
2. **downstream-policy drift**：焦点动作是否显著改变后缀分布。

严格可加时两项为零。该界可以在形式上“容纳”AirDefense 交互，但不能证明
偏差小：目标占用会改变后缀合法支持，第二项的 \(L_1\) 距离可接近最大值 2；
同目标竞争、过杀、命中、终止和未来弹药替代会使第一项和
\(\|\varepsilon\|_\infty\) 同时变大。

所以“存在 bias bound”不等于“AirDefense 的交互回报适合 additive COSAC”。

## 11. 方差公式与三类方法差异

论文定义 population Gram：

\[
G_\mu=\mathbb E_\mu[\psi\psi^\top],\qquad
\kappa_\mu=\lambda_{\min}(G_\mu).
\]

v2 Theorem 4 在 population-Gram 近似下给出：

\[
\operatorname{Var}(D_k)
\le
\frac{2R_{\max}^2}{\lambda+N\kappa_\mu},
\]

\[
\operatorname{Var}(I_k)=O(K/L),
\]

\[
\operatorname{Var}(\hat A_k^{\mathrm{LS}})
\le
2\{\operatorname{Var}(D_k)+\operatorname{Var}(I_k)\}.
\]

论文对照：

\[
\operatorname{Var}(\hat A_k^{\mathrm{shared}})
=O(K\sigma_\varphi^2/N),
\]

\[
\operatorname{Var}(\hat A_k^{\mathrm{HA}})
=
O\left(
\prod_{j<k}\mathbb E_\mu[Y_j^2]\,
\frac{R_{\max}^2}{N}
\right).
\]

| 方法 | 信号来源 | 团队规模/位置噪声来源 |
| --- | --- | --- |
| 共享 baseline / MA-GRPO | 所有 agent 共用团队回报中心化值 | 团队回报含全部 agent 随机性，方差随 \(K\) 线性增长 |
| HA-GRPO/HAPPO 类顺序更新 | 用累计上游 likelihood ratio 修正旧批次 | 比率二阶矩相乘，越靠后位置越容易指数放大 |
| COSAC | 单 agent 直接项 + 当前策略虚拟下游项 | direct 项在论文条件下不随 \(K\)；indirect Monte Carlo 最坏随下游规模线性增长 |

必须保留两点：

1. 论文实验基线是 **HA-GRPO**，只是受 HAPPO 启发；不能把 COSAC 的
   方差对照写成已推翻 HAPPO 的完整单调改进理论。
2. v2 的方差结果分析 advantage estimator，不是深度 PPO 的收敛保证。

## 12. Gram 覆盖假设的内部一致性审计

论文写道 \(\kappa_\mu>0\) 等价于每个 per-agent action 都有正边际概率。
按其明示的 \(d=KA\) 全 one-hot 拼接，该说法并不成立。

对任意 agent block \(k\)：

\[
\sum_{a\in\mathcal A_k}\psi_{k,a}(a_{1:K})=1.
\]

因此对两个 agent block \(k\ne j\)，构造向量 \(v\)：

```text
block k 的所有坐标为 +1
block j 的所有坐标为 -1
其他坐标为 0
```

则对每条联合动作样本都有：

\[
\psi(a)^\top v=1-1=0.
\]

所以 \(G_\mu v=0\)。当 \(K>1\) 时至少存在 \(K-1\) 个此类线性依赖，
完整 \(G_\mu\) 必然奇异：

\[
\lambda_{\min}(G_\mu)=0.
\]

结论：

- 正边际概率只提供 action coverage，不消除 block-intercept 不可辨识；
- \(\lambda>0\) 让 ridge 矩阵可逆，但不能使 \(G_\mu\) 的
  \(\kappa_\mu\) 变正；
- 按论文原式，direct variance 上界退化为 \(2R_{\max}^2/\lambda\)，
  不再展示 \(N\) 带来的 coverage 收缩；
- 若要恢复有意义的 \(N\)-scaling，应删除每个 block 的参考列、采用
  effect coding，或把最小特征值限制在 action-contrast 子空间；
- 论文没有在 v2 中给出这一重参数化。

这是本报告的线性代数审计结论，不表示 SeqAU estimand 错误；它限制的是
当前 ridge variance theorem 的书写和可直接引用程度。

## 13. 理论适用性审计

| 假设/范围 | COSAC v2 | AirDefense 判定 |
| --- | --- | --- |
| 任务时域 | sequential contextual bandit | 不满足；完整 MDP、最多 50 步 |
| 每 agent 行动次数 | 每 episode 一次 | 不满足；同一单元跨环境步反复决策 |
| 回报 | 一个有界标量团队回报 | 有综合 reward，但安全/资源规范不应继续压成一个标量 |
| 动作 | 每 agent 固定离散槽 | 部分满足；合法目标集合随状态和前缀变化 |
| 更新顺序 | 与 execution 自然顺序一致 | 当前 joint PPO 不等于逐单元自然顺序更新 |
| 回报模型 | action-only additive ridge，按上下文点态理解 | 不满足；价值依赖状态、前缀、命中和未来状态 |
| 加性 | 近似 \(\sum_k\varphi_k(a_k)\) | 风险高；占用、过杀、同目标竞争和终止强交互 |
| coverage | 正动作概率、population Gram | 动态非法动作有结构零概率；完整 Gram 还天然奇异 |
| downstream sample | 当前策略前向即可 | 同一步后缀可以；跨时间后续需要环境或动力学模型 |
| 优化保证 | advantage bias/variance | 不保证 joint PPO fallback、MDP 改进或约束满足 |
| 安全约束 | 无 | 不处理 damage/leak/resource 多约束 |

## 14. 动态 mask 是工程差异还是数学差异

### 14.1 仅使用 mask 不是新数学

COSAC 已允许 \(\pi_j(a_j\mid a_{<j},x)\) 读取前缀。把非法动作 logit 置为
\(-\infty\)，并从条件 masked categorical 采样，本身仍是该自回归策略族的
直接实例。项目不能用“COSAC 没写 mask”建立创新主张。

### 14.2 以下组合才形成实质理论缺口

AirDefense 的一个焦点动作会：

1. 占用一个目标，使后缀 agent 的动作支持集合改变；
2. 让同一 target ID 在不同前缀中合法性和含义不同；
3. 改变同一步其他单元是否射击；
4. 经命中和目标存活改变后续环境状态；
5. 改变未来各单元的弹药、冷却与可行动作；
6. 产生非加性安全—资源交互。

COSAC 的 SeqAU estimand 能表示第 1–3 项的**期望后缀影响**，但 action-only
ridge 与 bandit 理论没有为第 4–6 项提供可靠估计。动态 mask 的研究价值不在
“存在掩码”，而在：

```text
替换焦点动作
→ 后缀支持集合改变
→ 交互残差与 coverage 同时改变
→ 如何仍获得可验证的 full-MDP advantage
```

只有围绕这个链条给出新 estimand、受控偏差或一致性结果，动态 mask 才不只是
工程适配。目前项目尚未完成。

## 15. 团队回报加性近似不等于成本账本恒等式

COSAC：

\[
\hat f(a)=\sum_k\hat\varphi_k(a_k)
\]

是从团队回报数据拟合出的统计 surrogate，允许残差。

AirDefense R2：

\[
\Delta C_{\mathrm{episode}}
=
C_{\mathrm{direct}}
-S_{\mathrm{same\ step\ other}}
-S_{\mathrm{future\ probe}}
-S_{\mathrm{future\ other}}
\]

是按资源事件身份和时间位置建立的精确账本，修正后 7,776 条记录的最大误差
为 \(8.88\times10^{-16}\)。

两者不可互换：

- COSAC 的槽位是 agent action contribution；
- R2 的槽位是跨时间资源事件及其替代；
- COSAC 近似一个标量团队回报；
- R2 解释一次动作替换后的累计资源成本差；
- R2 可精确不代表综合 reward 对各单元动作可加；
- COSAC bias bound 也不能替代 R2 的路径和账本证据。

## 16. SeqAU 能否避开冻结 Critic OOD

判定：**只能避开一种 Critic，不能消除支持域问题。**

它避开：

- 冻结的神经 \(Q(s,h_k,a_k)\)；
- Bellman bootstrap；
- 从旧 joint action 直接累计重要性采样。

它引入：

- 每批重新拟合的 action-only ridge surrogate；
- 批内稀有动作和组合覆盖；
- 对未在 batch 中充分出现的条件后缀作模型求值；
- additive residual bias；
- context/state 不同质时的模型错配。

虚拟后缀来自当前未更新的 \(\mu_{\ge k}\)，因此是 policy-side 条件 on-support；
但 \(\hat f\) 是否在这些组合上可靠仍取决于 batch coverage。ridge regularization
提供数值收缩，不生成缺失的真实 reward 信息。

AirDefense 若把 \(R\) 定义为完整 episode return，则虚拟跨时间 continuation
必须调用环境或学习动力学/回报模型；若只把 \(R\) 定义为当前步 reward，
又会丢失已经确认的未来动作替代。因此 COSAC 不能直接绕过项目冻结 Critic
的 OOD 问题。

## 17. 与 COMA/HAPPO/MCH/BPCE 的差异矩阵

| 方法 | Estimand / advantage | Counterfactual | 更新与约束 | 主要风险 |
| --- | --- | --- | --- | --- |
| COMA | \(Q(s,a)-\sum_{a_k'}\pi_k(a_k')Q(s,a_{-k},a_k')\) | 固定其他同时 agent 动作，只边缘化焦点动作 | centralized critic + decentralized actors | joint-action critic 规模、非顺序前缀 |
| HAPPO/HA-GRPO | 联合 advantage 的顺序分解/累计前缀 ratio | 不以回报模型虚拟后缀构造 SeqAU | 逐 agent 更新；HA 类用前缀比率处理旧批次 | ratio 二阶矩随上游位置乘积放大 |
| COSAC | \(\mathbb E[R\mid a_{\le k}]-\mathbb E[R\mid a_{<k}]\) | 当前策略下重新采样焦点动作和政策后缀，用 additive surrogate 求值 | 自然顺序逐 agent PPO；无安全约束 | 非加性 bias、coverage、bandit 范围 |
| MCH-PPO | 冻结 \(Q(s,h_k,a_k)\) 上的 engagement/target 条件反事实差 | 枚举动态合法目标，不运行环境 continuation | engagement/target 独立 ratio/clip | Critic OOD；3/6 塌缩；fallback 不等价 joint PPO |
| RG-MCH | joint GAE 主信用 + 冻结 Critic 有界残差 | 同 MCH | 仍使用层级独立优化 | 2/6 塌缩、共同错误门控、成本失控 |
| BPCE-PPO | 完整回合 N/E 成对差形成 engagement 排序辅助 | 环境快照、共同随机数、真实 deterministic/stochastic continuation | 标准 joint PPO 严格主干 + auxiliary | 计算约 1.94x；标签单边、2/6 塌缩 |
| R2 动作替代账本 | 一次动作替换的累计成本测量分解 | 真实环境分支和随机带 | 只读，不训练 | 是解释/可辨识性证据，不是 advantage |

## 18. 指导文件要求的五层差异表

| 层 | CAPO/COSAC | MCH/BPCE | 剩余差异是否充分 |
| --- | --- | --- | --- |
| Problem | 固定顺序 agent 的团队回报信用与批内逐 agent 更新陈旧性 | 动态掩码 WTA 的 engage/target 信用、冻结 Critic OOD、边界探测 | “顺序信用”已被覆盖；完整 MDP 动态支持仍有差异 |
| Advantage | 前缀条件 SeqAU；additive direct + downstream indirect | MCH 为层级 Q 差；BPCE 为完整回合 engage/no-op 排序标签 | MCH 广义叙事不足；BPCE estimand 与计算协议仍不同 |
| Counterfactual | 当前策略虚拟后缀，无环境/reward 调用 | MCH 枚举 Critic；BPCE 环境快照 CRN rollout | 真实 continuation 与 surrogate continuation 是实质差异 |
| Constraints | 无安全/资源约束；固定离散槽 | 动态合法集、目标占用、弹药/冷却；但在线安全—资源约束尚未成立 | 当前合法性是真实结构差异，多约束仍未解决 |
| Evidence | bandit 30/300 seeds；ARC 10 问题、单模型族；预印本 | 多轮独立压力实验，大量负结果；R2 有新种子独立确认 | 项目有领域失败边界，但没有新算法正证据 |

## 19. 对现有 MCH 创新叙事的覆盖判决

### 19.1 已被直接覆盖

以下主张必须删除：

1. 首次为固定顺序协作决策定义前缀条件反事实 advantage；
2. 首次把焦点动作贡献分成直接项和下游间接项；
3. 首次通过重采样同一步后缀估计顺序动作信用；
4. critic-free 顺序信用天然比共享 advantage 方差更低；
5. 逐单元反事实 advantage 本身构成 MCH 的核心创新。

### 19.2 仍有差异但尚不构成算法

可以保留为待研究问题：

- 前缀动作改变后缀动态合法支持；
- engage/no-op 与 conditional target 两层动作语义；
- 同一步替代与跨时间环境替代同时存在；
- full-MDP return 不满足 action-only additive reward；
- 严格 joint PPO fallback 下如何接入局部信用；
- 安全与资源需要独立规范目标。

这些差异目前缺少统一 estimand、理论保证和通过门控的在线证据，不能写成
“COSAC 未覆盖，所以 MCH 已创新”。

## 20. 对第一算法候选的保留、重写与放弃

| 对象 | 建议 | 理由 |
| --- | --- | --- |
| 第一项 R2 测量贡献 | **保留** | COSAC 的同一步 indirect effect 与其相邻，但不分解跨时间资源事件，不能替代精确账本 |
| MCH-PPO v0 | **放弃为主算法** | 核心顺序反事实叙事被 COSAC 覆盖，且 3/6 在线塌缩 |
| 独立 engagement/target ratio 与 clip | **放弃为安全 fallback** | SA-RG-MCH 已以 5/6 塌缩证伪 |
| RG-MCH 的 GAE 锚定 | **保留为机制证据** | 证明局部信用不能完全替代 on-policy GAE；不等于算法成立 |
| BPCE-PPO 当前候选 | **保持暂停** | 与 C3 类真实 replay 相邻；标签双向覆盖和资源语义未成立 |
| 新算法问题 | **重写但不实现** | 聚焦 full-MDP、动态支持、非加性交互下的顺序 advantage，并保持 joint PPO 严格 fallback |

重写后的 Problem–Method–Insight 只能暂记为：

| 层 | 当前草案 |
| --- | --- |
| Problem | 在动作改变后缀合法支持且影响跨时间状态的自回归 MDP 中，bandit SeqAU 的批内 additive surrogate 无法提供可靠全回报信用 |
| Method | 尚未冻结；至少需要动态支持条件 estimand、交互残差诊断和 joint PPO 零系数等价 |
| Insight | 顺序前缀条件化不足以保证可靠信用；支持变化与时间交互必须同时进入可证伪误差边界 |

这是一条待讨论的研究问题，不是本 LR-05 已提出的新算法。

## 21. 创新压力测试与可证伪命题

### 21.1 伪创新风险

| 风险 | 判决 |
| --- | --- |
| 把 CAPO 改称 COSAC 当作两个邻近方法 | 禁止 |
| 把动态 mask 作为全部方法创新 | 不充分 |
| 把 additive ridge 改成神经网络就称新方法 | 工程替换 |
| 把 R2 精确账本当成 additive reward decomposition | 概念错误 |
| 只在 AirDefense 应用 SeqAU | 领域移植 |
| 选择 RG-MCH/BPCE 正结果种子证明差异 | 禁止 |
| 引用预印本的“first”而不独立查新 | 禁止 |

### 21.2 后续若重开算法必须证伪

| 命题 | 支持条件 | 否决条件 | 最小测试 |
| --- | --- | --- | --- |
| H1 动态支持引入超出普通 SeqAU 的估计误差 | 同前缀的合法支持变化能预测 COSAC surrogate 残差 | mask 统计无增量解释力 | 冻结状态上按支持变化分层的 advantage error |
| H2 非加性交互残差是主要失效源 | interaction-aware estimator 在独立批次降低误差 | 等容量 action-only/联合模型同样有效 | additive、pairwise、等容量非机制三方对照 |
| H3 局部信用可在 joint PPO fallback 下提供稳定增量 | 零系数完全等价且多种子安全/成本门控通过 | 再现 all-noop 或高成本分叉 | 先离线误差门控，再 10k 预注册压力实验 |

当前三项均未满足在线授权条件。

## 22. 论文实验证据及边界

### 22.1 顺序 bandit

设置：

\[
f(a)
=
\sum_k\varphi_k(a_k)
+
\lambda_{\mathrm{int}}
\sum_{k<\ell}g_{k\ell}(a_k,a_\ell).
\]

- 每 agent 动作数 \(A=4\)；
- \(\lambda_{\mathrm{int}}\in[0,1]\) 控制非加性交互；
- \(\rho\) 控制下游策略依赖上游动作的强度；
- advantage MSE 使用 30 seeds、\(N=16\)；
- 优化和 direct-effect 消融使用 300 seeds/cell；
- 团队规模最大到 \(K=16\) 的 MSE、\(K=10\) 的优化。

作者结果：

- \(K=16\) 时 COSAC advantage MSE 约比 C3、MA-GRPO、HA-GRPO 分别低
  `2x / 30x / 46x`；
- \(K\ge4\) 时平均 regret 最优；
- \(\rho=0\) 时 direct-only 不劣，\(\rho\) 增大后 indirect term 才产生增益；
- HA-GRPO 的位置误差随 agent index 增长，COSAC 近似平坦。

边界：

- 环境 reward 完全可枚举，ground-truth advantage 可精确计算；
- interaction 是人工 pairwise 项；
- 没有状态转移、动态 mask、累计资源约束或 deterministic all-noop；
- C3 因真实交互预算匹配而获得更少 outer iterations，性能差含成本协议影响。

### 22.2 ARC 多代理推理

- 四个 Qwen3-0.6B agent，分别使用 LoRA；
- 10 个经过筛选、初始未全体一致且未达 ceiling 的 ARC 问题；
- 回报混合 independence 与 correct-consensus，\(\alpha\in\{0,0.5,1\}\)；
- COSAC 三种 \(\alpha\) 的 AUC 均最高；
- \(\alpha=1\) 时最终回报与 MA-GRPO 接近/统计并列；
- 单一模型族、10 个问题，不能证明一般 MARL 或防空控制有效性。

### 22.3 v2 证据不能支持

- 完整 MDP 的 SeqAU 估计；
- 动态合法集下的 coverage 保证；
- additive surrogate 在强交互任务中偏差小；
- 深度 PPO 的收敛或单调改进；
- 安全—资源多约束满足；
- 跨资源类型、跨场景和跨策略架构泛化。

## 23. 独立邻近工作核对

本报告没有采用 COSAC 的“to our knowledge first”作为结论。独立核对至少包括：

- COMA：<https://arxiv.org/abs/1705.08926>；
- HATRPO/HAPPO：<https://arxiv.org/abs/2109.11251>；
- C3：<https://arxiv.org/abs/2603.06859>；
- CCPO：<https://arxiv.org/abs/2603.21563>；
- LR-01 已核对的 Counterfactual Effect Decomposition；
- 项目已有 difference reward、CCA、COCOA、DAE 和 factorized baseline
  查新记录。

检索支持“顺序反事实、下游 replay、差异回报、顺序更新”均已有密集邻近工作。
本 LR-05 不作“完整系统检索已经证明新颖”的声明；投稿前仍需更新 2026 年后续
版本、同行评审状态和中文数据库。

## 24. `BASELINE / ADAPT / AVOID / OPEN` 判决

| 标签 | 判决 |
| --- | --- |
| `BASELINE` | SeqAU/COSAC 是固定顺序团队反事实 credit 的最近强基线；后续任何 MCH 类方法必须与 shared、HA、COMA/C3、COSAC 对照 |
| `ADAPT` | 可适配 prefix-conditional SeqAU estimand、direct/indirect 审计和 natural-order 无前缀 IS 的思路；仅先做只读误差诊断 |
| `AVOID` | 不直接把 action-only ridge 和虚拟 policy suffix 接入 AirDefense PPO；不恢复独立层级 clipping；不把 critic-free 写成无模型或无 OOD |
| `OPEN` | full-MDP 动态支持、非加性交互、跨时间动作替代、多约束规范与严格 joint PPO 接口仍未解决 |

总体路线：

```text
COSAC 覆盖普通顺序反事实 advantage
                    ↓
删除 MCH 的宽泛创新叙事
                    ↓
保留 R2 跨时间动作替代测量贡献
                    ↓
只把动态支持 + full MDP + 非加性交互定义为开放问题
                    ↓
LR-06 审计批内 surrogate / 离线 Critic 到在线策略的分布漂移
```

## 25. 移交 LR-06

LR-06 必须同时审计三种模型，而不能只讨论冻结神经 Critic：

1. 冻结离线 \(Q(s,h,a)\)；
2. 每批重建的 COSAC additive ridge \(\hat f(a)\)；
3. BPCE/C3 类真实 replay 标签。

必须区分三类漂移：

| 漂移 | LR-05 已发现的接口 |
| --- | --- |
| policy drift | \(\mu\) rollout 与批内逐 agent 更新后的策略不同 |
| context/state drift | bandit pointwise \(x\) 与 AirDefense 连续变化状态不同 |
| feasible-support drift | 焦点动作改变下游 mask，合法组合支持随前缀变化 |

LR-06 还必须保留：

- v2 没有实际 COSAC 深度 PPO gradient-MSE theorem；
- ridge 正则化不等于真实数据覆盖；
- 完整 one-hot Gram 的 \(\kappa_\mu>0\) 假设需要在 contrast subspace 重述；
- 零辅助系数时必须严格恢复 factorized joint PPO 的完整更新；
- 所有离线/批内估值在进入在线更新前先通过独立状态、跨批次、双向信用和
  安全—资源分层门控。

## 26. 术语表与来源锚点

| 英文 | 本报告译法 | v2 来源锚点 |
| --- | --- | --- |
| sequential cooperative team | 顺序合作团队 | PDF pp.1–3 |
| execution order | 执行顺序 | PDF p.3 |
| update order | 更新顺序 | PDF p.3 |
| sequential learnability | 顺序可学习性 | PDF pp.3–4, Def. 1 |
| Sequential Aristocrat Utility | 顺序贵族效用 | PDF p.4, Thm. 2 |
| prefix-conditional baseline | 前缀条件 baseline | PDF pp.3–4 |
| upstream cancellation | 上游抵消 | PDF p.5, App. D |
| direct effect | 直接效应 | PDF p.5, Eq. 4 |
| indirect effect | 间接效应 | PDF p.5, Eq. 4 |
| additive reward decomposition | 加性回报分解 | PDF p.5 |
| fictitious continuation | 虚拟后续 | PDF pp.5–6, Eq. 5 |
| non-additive residual | 非加性残差 | PDF p.6 |
| residual sensitivity | 残差敏感度 | PDF p.6, Thm. 3 |
| downstream-policy drift | 下游策略漂移 | PDF p.6, Thm. 3 |
| coverage eigenvalue | 覆盖特征值 | PDF p.6 |

### 关键来源索引

- 版本、摘要和贡献：v2 PDF pp.1–2；v1/v2 arXiv history；
- contextual-bandit 设定、execution/update order：v2 PDF p.3；
- learnability、SeqAU 和唯一性：v2 PDF pp.3–4；Appendix C；
- prior approaches、additive ridge：v2 PDF pp.4–5；
- upstream cancellation、direct/indirect：v2 PDF p.5；Appendix D；
- fictitious sampling 和算法：v2 PDF pp.5–6；Appendix E；
- bias/variance：v2 PDF pp.6–7；Appendix F；
- bandit 实验：v2 PDF pp.7–8；Appendices G–K；
- ARC 实验：v2 PDF pp.8–9；Appendix L；
- 局限：v2 PDF p.10；
- v1 删除的 gradient-MSE 结果：v1 Section 5, Theorem 4；Appendix C.4。
