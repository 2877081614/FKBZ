# LR-06 阅读报告：离线到在线 Critic 重构、策略对齐与受约束微调边界

任务状态：`PASSED`  
完成时间：2026-07-29  
实验授权：否  
总体判决：`BASELINE（机制参照） + ADAPT（错配审计框架）`；直接移植为 `AVOID`；动态支持下的在线接入证书为 `OPEN`

## 1. 论文身份与版本

| 项目 | 内容 |
| --- | --- |
| 标题 | *Optimistic Critic Reconstruction and Constrained Fine-Tuning for General Offline-to-Online RL* |
| 作者 | Qin-Wen Luo、Ming-Kun Xie、Ye-Wen Wang、Sheng-Jun Huang |
| 会议 | NeurIPS 2024，Main Conference Track |
| 方法 | Optimistic Critic Reconstruction and Constrained Fine-Tuning（OCR-CFT） |
| OpenReview | <https://openreview.net/forum?id=XVfevb9XFx> |
| NeurIPS 页面 | <https://papers.nips.cc/paper_files/paper/2024/hash/c3b3c5f9297ad0012cfe7188c34cea0e-Abstract-Conference.html> |
| 正式论文 | <https://proceedings.neurips.cc/paper_files/paper/2024/file/c3b3c5f9297ad0012cfe7188c34cea0e-Paper-Conference.pdf> |
| arXiv | <https://arxiv.org/abs/2412.18855> |
| 官方代码 | <https://github.com/QinwenLuo/OCR-CFT> |
| DOI | <https://doi.org/10.52202/079017-3435> |

本报告以 41 页 NeurIPS 正式 PDF（正文、附录、证明、实现细节和伪代码）为
公式与页码基准，并用 arXiv HTML 和官方仓库 README 交叉核对。官方代码只做
在线静态阅读，没有下载、安装依赖或运行。

## 2. 一句话结论

OCR-CFT 的关键贡献不是“乐观 Critic 天然更安全”，而是把离线到在线失败拆成
三个连续接口：

```text
离线评价规则与在线 Bellman 评价不一致
        ↓  policy re-evaluation
新 Critic 与可靠离线 actor 的偏好不一致
        ↓  value alignment
在线探索继续产生状态—动作分布漂移
        ↓  constrained fine-tuning
```

这套错配分层对 AirDefense 很有价值，但论文依赖可靠离线 actor、单策略覆盖
集中性、完整离线转移/轨迹和可用的参考策略评价。项目当前不满足这些前提，
而且是更一般的 **auxiliary-estimator-to-on-policy mismatch**，不是标准
offline-to-online RL。因此 OCR-CFT 应作为机制参照和风险警示，不能原样成为
当前实现基线。

## 3. Problem–Method–Insight

| 层 | 论文内容 |
| --- | --- |
| Problem | 任意离线 RL 方法产生的 actor/critic，怎样稳定过渡到 SAC、TD3 或 PPO 在线微调 |
| Mismatch 1 | 离线价值正则或非目标策略评价与在线 Bellman 评价不同，导致初始 Q 值突变 |
| Mismatch 2 | 离线 actor 的动作概率还受行为分布约束，未必与 critic 的 Q 排序一致 |
| Method 1 | 固定离线 actor，用在线算法对应的 OPE/FQE 目标重新训练一个随机初始化 Critic |
| Method 2 | 以离线 actor 的高概率/确定性动作为价值锚，压低会误导策略的过高 Q 值 |
| Method 3 | 以历史参考策略为中心，用 KL、MSE 或辅助 advantage 约束在线更新，再逐步放松 |
| Insight | “估值尺度适合在线更新”与“估值诱导的策略改进方向正确”是两个独立条件；二者满足后仍需处理在线分布漂移 |

## 4. 两类 mismatch 公式卡

### 4.1 在线 RL 目标

标准在线 MDP 的目标为：

\[
J(\pi)
=
\mathbb E_{\pi}
\left[
\sum_{t=0}^{\infty}\gamma^t r_t
\right]
=
\mathbb E_{s_0\sim\mu,a_0\sim\pi}
\left[Q^\pi(s_0,a_0)\right].
\tag{1}
\]

这里 \(s_t,a_t,r_t\) 分别是状态、动作和回报，\(\gamma\) 是折扣因子，
\(\mu\) 是初始状态分布。在线 actor 的改进主要由 \(Q^\pi\) 或由它产生的
advantage 决定。

### 4.2 离线行为正则目标

论文用 behavior-regularized MDP 统一表示一类离线 RL：

\[
J_{\mathrm{off}}(\pi)
=
\mathbb E_{\pi}
\left[
\sum_{t=0}^{\infty}\gamma^t
\left(
r_t-\alpha
f\!\left(
\frac{\pi(a_t\mid s_t)}
{\mu_b(a_t\mid s_t)}
\right)
\right)
\right],
\tag{2}
\]

其中 \(\mu_b\) 是产生离线数据的行为策略，\(f\) 是分布偏离惩罚，
\(\alpha\) 是正则强度。于是离线 actor 的偏好同时取决于回报和数据支持，
而在线目标只要求最大化环境回报。

### 4.3 evaluation mismatch

令 \(\mathcal T_{\mathrm{off}}\) 表示带悲观正则、行为策略评价或特殊
expectile/one-step 结构的离线评价算子，令 \(\mathcal T_{\mathrm{on}}^\pi\)
表示目标在线算法的 Bellman 评价算子。evaluation mismatch 可以写成：

\[
\Delta_{\mathrm{eval}}(s,a)
=
\mathcal T_{\mathrm{on}}^\pi Q(s,a)
-
\mathcal T_{\mathrm{off}} Q(s,a).
\tag{3}
\]

例如 CQL 的离线 Critic 会压低 OOD 动作；直接切换到没有该悲观项的 SAC
目标后，Q 值可能在微调初期剧烈上升。IQL 等方法还可能评价行为策略或未知
策略，而不是将要在线更新的目标策略。

注意：evaluation mismatch 指的是**评价规则/目标策略不一致**，不是简单的
“离线 Q 偏低”。如果离线 Critic 已使用与在线相同的评价规则，policy
re-evaluation 可能可以省略。

### 4.4 improvement mismatch

定义离线 actor 的锚动作和新 Critic 的贪心动作：

\[
a_\pi(s)=\arg\max_a \pi_{\mathrm{off}}(a\mid s),\qquad
a_Q(s)=\arg\max_a \widehat Q_{\mathrm{on}}(s,a).
\tag{4}
\]

当

\[
a_\pi(s)\ne a_Q(s)
\tag{5}
\]

或更一般地，

\[
\operatorname{sign}
\left[
\log\pi_{\mathrm{off}}(a\mid s)
-
\log\pi_{\mathrm{off}}(a'\mid s)
\right]
\ne
\operatorname{sign}
\left[
\widehat Q_{\mathrm{on}}(s,a)
-
\widehat Q_{\mathrm{on}}(s,a')
\right],
\tag{6}
\]

新 Critic 诱导的在线改进方向就与可靠离线 actor 的偏好不一致。即使
\(\widehat Q_{\mathrm{on}}\) 的总体 MAE 较小，这种**同状态动作排序错误**
仍可能在第一轮 actor 更新中造成不可逆性能下降。

## 5. Optimistic Critic Reconstruction

### 5.1 输入与目标

OCR 不继承原离线 Critic，而是：

1. 保留离线 actor \(\pi_{\mathrm{off}}\)；
2. 随机初始化在线 Critic；
3. 在离线数据集 \(\mathcal D\) 上，使用目标在线算法的评价目标重估
   \(\pi_{\mathrm{off}}\)。

对 SAC，重构目标是：

\[
y_{\mathrm{SAC}}
=
r+\gamma
\mathbb E_{a'\sim\pi_{\mathrm{off}}(\cdot\mid s')}
\left[
\min_{i=1,2}\bar Q_i(s',a')
-
\alpha\log\pi_{\mathrm{off}}(a'\mid s')
\right],
\tag{7}
\]

\[
\mathcal L_Q
=
\mathbb E_{(s,a,r,s')\sim\mathcal D}
\left[
\left(Q_i(s,a)-y_{\mathrm{SAC}}\right)^2
\right].
\tag{8}
\]

对 TD3，使用固定离线确定性 actor、target policy smoothing 和 clipped
double Q 的标准在线 Critic 目标。对 PPO，论文不能从 off-policy Bellman
目标直接得到在线 \(V^\pi\)，因此只用完整离线轨迹的 return 拟合一个
\(V(s)\)。

### 5.2 “optimistic”的准确含义

论文所说的 optimistic 主要是：

- 去掉离线悲观/value regularization；
- 用在线算法的评价规则重建 Critic；
- 避免上线后从悲观 Q 突然跳到在线尺度。

它不是上置信界、posterior optimism 或独立不确定性估计，也没有给每个
状态—动作提供 epistemic uncertainty。其安全性不能由“optimistic”一词推出。

### 5.3 覆盖假设

论文要求单策略集中性：

\[
\max_{(s,a)}
\frac{d^{\pi_{\mathrm{off}}}(s,a)}
{d^{\mu_b}(s,a)}
\le C,
\tag{9}
\]

其中 \(d^\pi\) 是策略 \(\pi\) 的 occupancy measure，\(C\) 越小表示离线
数据越能覆盖该策略。论文引用 FQE 结果给出误差上界：

\[
\left|Q^\pi-\widehat Q^\pi\right|
\le
\frac{1-\gamma^K}{1-\gamma}\sqrt{C\epsilon}
+
\gamma^K\bar V,
\tag{10}
\]

其中 \(K\) 是 FQE 迭代次数，\(\epsilon\) 包含有限数据误差和 inherent
Bellman evaluation error，\(\bar V\le R_{\max}/(1-\gamma)\)。

该式的正确解释是：只有在 \(\pi_{\mathrm{off}}\) 被数据覆盖、数据足够、
函数类误差小且迭代充分时，重构误差才可能受控。它不保证 OOD 动作正确，
更不保证策略在线漂移后仍被覆盖。

## 6. Actor–Critic value alignment

### 6.1 O2SAC

最大熵最优策略满足：

\[
Q(s,a)=V(s)+\alpha\log\pi(a\mid s).
\tag{11}
\]

以离线 actor 的最可能动作
\(\dot a=\arg\max_a\pi_{\mathrm{off}}(a\mid s)\) 为锚，论文对其他动作构造：

\[
Q'_\mu(s,a)
=
\min\left\{
\bar Q_\mu(s,\dot a)
-
\alpha
\left[
\log\pi_{\mathrm{off}}(\dot a\mid s)
-
\log\pi_{\mathrm{off}}(a\mid s)
\right],
\bar Q_\mu(s,a)
\right\}.
\tag{12}
\]

第一个分支把 Q 排序压回离线 actor 概率排序；`min` 保留本来没有超过该
上界的 FQE 值。Critic 同时优化：

\[
\mathcal L_{\mathrm{critic}}
=
\mathbb E_{s,a}
\left[
\left(Q_\mu(s,a)-Q'_\mu(s,a)\right)^2
\right]
+
\mathbb E_s
\left[
\left(
Q_\mu(s,\dot a)-\bar Q_\mu(s,\dot a)
\right)^2
\right].
\tag{13}
\]

第二项保留锚动作的重构 Q，防止整体下压重新变成悲观估计。论文证明的只是
\(V_{\mathrm{fqe}}(s)\le V_{\mathrm{align}}(s)\le V_{\dot a}(s)\)，即相对
内部锚点的区间，不是相对真实价值或安全回报的上、下界。

### 6.2 O2TD3

TD3 没有显式概率—Q 能量关系。论文假设离线确定性动作 \(\dot a\) 周围的
归一化 Q 形状近似 Gaussian，并用：

\[
Q'_\mu(s,\tilde a)
=
\min\left\{
\bar Q_\mu(s,\tilde a),
\frac{Q(s,\dot a)}
{1+k\max\!\left(d(\tilde a,\dot a)^2,\sigma^2\right)}
\right\},
\tag{14}
\]

校准扰动动作 \(\tilde a\)。\(d\) 是按动作维数归一化的欧氏距离，
\(\sigma\) 对应 policy smoothing noise，\(k=1\)。该构造依赖连续动作的
局部平滑性，不能直接替换离散 masked categorical 的动作排序。

### 6.3 O2PPO

PPO 只有 \(V(s)\) 和 GAE。论文承认离线轨迹 return 拟合得到的只是接近
\(V^\mu(s)\) 的量，因此不把一个动作 Q-Critic 直接塞进 PPO，而是定义：

\[
A_\alpha(s,a)
=
\alpha\log\pi_{\mathrm{ref}}(a\mid s)
+
\alpha\mathcal H
\left(\pi_{\mathrm{ref}}(\cdot\mid s)\right),
\tag{15}
\]

\[
A'(s,a)
=
A_{\mathrm{GAE}}(s,a)
+
\beta A_\alpha(s,a).
\tag{16}
\]

在线开始时 \(\pi_{\mathrm{ref}}=\pi_{\mathrm{off}}\)，之后更新为历史参考
策略。\(\beta\) 从 1 逐步退火。附录证明该项在 PPO surrogate 中等价于
相对参考策略的 cross-entropy regularization。

必须修正正文的一处表述：\(A_\alpha(s,a)\) 在开始时并非对每个动作逐点为零，
而是

\[
\mathbb E_{a\sim\pi_{\mathrm{ref}}}
\left[A_\alpha(s,a)\right]=0.
\tag{17}
\]

不同采样动作仍有正负值。论文实现还对该项做 SoftPlus 截断、按 batch GAE
标准差缩放，并限制部分 IQL policy 的标准差，说明 O2PPO 不是只有式（16）
一个无敏感性的通用插件。

## 7. Constrained Fine-Tuning

### 7.1 约束目标

令 \(\pi_{\mathrm{ref}}\) 为在线评估中最好的历史策略，论文约束当前策略留在
其可信邻域：

\[
\max_\pi J(\pi)
\quad
\text{s.t.}\quad
\mathbb E_\pi
\left[
f\!\left(\pi(\cdot\mid s),\pi_{\mathrm{ref}}(\cdot\mid s)\right)
\right]
<\tau.
\tag{18}
\]

其中 SAC 使用 KL divergence，TD3 使用动作 MSE，\(\tau\) 是允许的策略距离。
拉格朗日形式同时改变 actor、critic target 和乘子：

\[
\max_\theta
\mathbb E
\left[
Q_\mu^{\pi_\theta}(s,a)
-
\lambda f(\pi_\theta,\pi_{\mathrm{ref}})
\right],
\tag{19}
\]

\[
y
=
r+\gamma\mathbb E_{a'\sim\pi_\theta}
\left[
\bar Q_\mu^{\pi_\theta}(s',a')
-
\lambda f(\pi_\theta,\pi_{\mathrm{ref}})
\right],
\tag{20}
\]

\[
\min_{\lambda\ge0}
-
\lambda
\left[
\mathbb E f(\pi_\theta,\pi_{\mathrm{ref}})
-\tau
\right].
\tag{21}
\]

这不是只在 actor loss 外加 uncertainty penalty：它没有显式不确定性，
并把参考策略距离写入 actor 和 off-policy critic target。

### 7.2 约束怎样逐步解除

论文采用三种松弛机制：

1. **阈值扩张**：O2SAC/O2TD3 的 \(\tau\) 随在线步数线性增大；
2. **辅助项退火**：O2PPO 的 \(\beta\) 从 1 向 0 线性衰减；
3. **参考策略前移**：若在线评估发现 \(\pi_{\mathrm{on}}\) 优于
   \(\pi_{\mathrm{ref}}\)，就用当前策略替换参考策略；不能评估时按固定间隔替换。

论文进一步声称收敛时 \(\lambda^\star=0\)，恢复无约束在线最优。但证明依赖
RCPO 的收敛前提，并额外假设历史参考策略最终就是 \(\pi^\star\)。这不是有限
步安全保证，也不能证明每次策略更新单调改善。

### 7.3 关键实现敏感性

| 组件 | 论文实现 |
| --- | --- |
| O2SAC \(\tau\) | medium/replay 从 0.125 线性增至 2.0；expert 从 0.005 增至 0.125 |
| O2TD3 \(\tau\) | 依据 exploration noise 设定，medium/replay 从 0.0025 增至 0.01 |
| O2PPO \(\beta\) | 普通数据在 250k 步从 1 衰减到 0；expert 设 500k 衰减，因此实验末仍约 0.5 |
| \(\pi_{\mathrm{ref}}\) | 主实验使用在线评估得到的历史最优；替代方案按 1k/10k 步固定更新 |
| replay | 测试 All/Half/Part/Null；正式 off-policy 采用 offline/online 对称采样 Half |
| \(\lambda\) | 采用加权乘子更新，防止一个 batch 使 \(\lambda\) 突降 |

因此所谓“逐步放松”不是一个无参数结论，而是依赖数据质量、探索噪声、在线
评估和任务特定 schedule。

## 8. 完整算法流程

```text
输入：离线策略 π_off、离线数据 D、目标在线算法
  │
  ├─ 1. 随机初始化在线 Critic
  │      SAC/TD3：在 D 上按目标在线 Bellman loss 评价固定 π_off
  │      PPO：拟合完整离线轨迹 return
  │
  ├─ 2. actor–critic alignment
  │      SAC：按 π_off 概率关系压低过估动作，保留锚动作 Q
  │      TD3：按动作距离的局部 Gaussian 形状压低过估动作
  │      PPO：保留 GAE，加入 π_ref 对数概率辅助 advantage
  │
  ├─ 3. π_ref ← 对齐后的 π_on
  │
  └─ 4. 在线微调
         更新 actor / critic / λ
         周期性评价并前移 π_ref，或按固定间隔前移
         增大 τ 或衰减 β
```

## 9. 方法成立的主要假设

1. **可靠离线 actor**：高概率动作在上线初期确实比低概率动作好；差 actor
   不属于论文重点，作者把这类情况归给 hybrid learning/from-scratch。
2. **单策略覆盖集中性**：离线数据能覆盖
   \(d^{\pi_{\mathrm{off}}}(s,a)\)，否则 FQE/OPE 的外推误差不受控。
3. **完整数据接口**：SAC/TD3 需要 \((s,a,r,s')\) 转移；PPO 需要能拟合
   return 的完整离线轨迹。
4. **函数逼近和 Bellman 误差足够小**：论文的误差界显式包含 inherent
   Bellman evaluation error。
5. **actor 可以提供可信锚**：SAC 的概率排序、TD3 的局部平滑/高斯形状、
   PPO 的高概率动作更优假设必须分别成立。
6. **策略距离代表可信邻域**：KL 或动作 MSE 小的策略在环境中也应具有相近
   行为后果。
7. **参考策略可以可靠更新**：主方案需要在线评估识别历史最优策略；固定
   间隔方案则假设足够长间隔后策略大概率改善。
8. **连续动作基准可代表目标应用**：论文实验没有动态 masked categorical
   联合动作，也没有 deterministic argmax 的组合边界。

## 10. 实验证据与作者主张分离

### 10.1 实验协议

- 环境：D4RL MuJoCo locomotion 与 AntMaze；
- 在线目标算法：SAC、TD3、PPO；
- 离线来源包括 CQL、IQL、TD3+BC、ODT 等；
- off-policy 主实验通常为 100k 在线交互，O2PPO 为 250k；
- 每个设置 5 个随机种子，报告均值和标准差；
- 对照包括 AWAC、IQL、PEX、Off2On、Cal-QL 和 ACA。

### 10.2 论文实际支持的结论

1. AntMaze umaze/umaze-diverse 表中，O2SAC、O2TD3、O2PPO 均能在离线
   初始化上取得提升，其中 O2PPO 总分从 133.7 增至 184.3。
2. 更难的 AntMaze medium/large 追加实验中，O2SAC 和 O2PPO 总提升分别为
   61.2 和 77.16，但没有覆盖全部主文对照的统一超参数比较。
3. 消融显示：
   - O2SAC 不做 optimistic critic reconstruction 时可在初期突降；
   - O2TD3 直接微调可因 actor–critic 错配严重退化；
   - 只做 value alignment 而无 CFT，后期仍可受 OOD 状态影响；
   - O2PPO 在 value 不准时依赖辅助 advantage 保持稳定。
4. 参考策略按固定间隔更新在多数 MuJoCo 设置可接近“在线评估历史最优”
   版本，但窄数据集需要从 1k 改为 10k 的手工间隔。

### 10.3 不能由证据推出的结论

1. “从任意 offline 到任意 online”实际只在若干离线方法和
   SAC/TD3/PPO 上验证；非标准 policy 还需 behavior cloning。
2. 没有证明 poor offline actor 也能稳定过渡；作者明确不研究这一情况。
3. 没有证明 value alignment 后的 Q 对真实值乐观且准确；命题只给内部
   锚点范围。
4. 没有证明 KL/MSE 约束提供逐步、逐轨迹或稀有事件安全。
5. 没有离散动态掩码、顺序匹配、联合 argmax 或资源—安全多约束实验。
6. 主参考策略选择读取在线评估结果，论文自己承认这引入额外测试信息。

## 11. 论文自身边界与额外审计

| 边界 | 影响 |
| --- | --- |
| 高概率动作更好是隐含假设 | actor 自身若 all-noop/高交战，alignment 会把错误偏好写回 Critic |
| OCR 去掉悲观性但不估计 epistemic uncertainty | 共同函数偏置可能被放大，而不是被识别 |
| Proposition 4.3 只给内部 Q 区间 | 不能解释为安全上界或真实价值校准 |
| O2PPO 辅助 advantage 只在期望上为零 | 不能声称开始时每个样本完全无扰动 |
| Corollary 4.5 是渐近固定点论证 | 不等于有限样本或有限步策略改进保证 |
| 历史最优 \(\pi_{\mathrm{ref}}\) 依赖在线评价 | 高风险部署时评价本身可能不可接受 |
| 固定间隔替代需要任务特定间隔 | 窄数据更易 OOD，不能称无条件通用 |
| 连续策略距离与行为后果之间未校准 | 对 deterministic argmax、组合动作尤其危险 |

## 12. AirDefense 的问题究竟是哪类

AirDefense 当前不是标准 offline-to-online RL：

- 主 actor 是在线 factorized joint PPO；
- Task 14 Critic 是在冻结反事实批次上训练的辅助估值器；
- COSAC 类模型是在当前 batch 重建的 action-only surrogate；
- BPCE/C3 类标签来自在线环境快照和成对 replay；
- 这些估值或标签只作为 PPO 辅助信用，不是一个完整离线 RL actor/critic
  切换到 SAC/TD3/PPO。

因此更准确的总问题是：

\[
\boxed{
\text{auxiliary-estimator-to-on-policy improvement mismatch}
}
\tag{22}
\]

OCR-CFT 的三阶段框架仍适用作审计语言，但论文的具体 OPE、连续动作 alignment
和 policy-distance constraint 不能直接套用。

## 13. 论文—项目错配对照表

| 论文错配 | 项目证据 | 是否等价 | 判定 |
| --- | --- | --- | --- |
| offline evaluation mismatch | 普通 Task 14 Q-Critic 的 Q MAE 比 \(V(s)\) 改善 36.4%–40.1%，但总体排序仅 0.25–0.375；后续 hierarchical target 排序才达 0.830–0.870 | 部分等价 | 项目首先存在动作差异估计不足，不只是悲观到乐观的尺度切换 |
| offline improvement mismatch | MCH 完全替换 GAE 后 3/6 all-noop；RG-MCH 保留 GAE 后改善但仍 2/6 塌缩 | 高度相似 | 冻结估值即使含信息，也未证明能给当前 actor 提供正确更新方向 |
| cross-batch calibration shift | 冻结独立确认中 safety sign 为 0.740–0.753，但 BA 仅 0.625–0.646，三个冻结阈值均未通过，cost correlation 仅 0.195–0.235 | 是项目特有的评价与决策边界漂移 | “方向含信息”仍不足以形成跨批次可用的停止/交战规则 |
| behavior support shift | SA-RG 在线 engagement/target support 仅 0.1244/0.0218 | 高度相似且更强 | 项目还有 prefix 和 legal-mask 支持随动作变化，超出普通 occupancy shift |
| policy constraint | SA-RG 初始 actor Bernoulli KL 均值 0.0171、penalty 为 0，仍 5/6 塌缩 | 形式相似，行为语义不等价 | 小分布距离不能控制 0.5 argmax 跨界 |
| reliable offline actor | factorized PPO 是严格 joint optimizer 主干，但 5 个 10k seed 中有 3 个 all-noop | 不满足 | “安全主干”不等于“可靠行为 teacher” |
| historical best reference | 项目正式协议禁止结果后选 seed/阈值，且部署评价有安全与资源多指标 | 不满足 | 单一 return 的历史最优不能替代预注册多指标门控 |
| optimistic OPE data | Task 14 只有 90 个状态、571 个候选、338 条训练样本；BPCE 标签稀疏且方向失衡 | 不满足 | 当前没有覆盖在线 occupancy 的完整离线转移/轨迹语料 |
| smooth action neighborhood | 动作是 no-op/engage + masked target 的离散顺序联合动作 | 不等价 | TD3 的欧氏距离 Gaussian Q 形状没有直接对应物 |

## 14. 是否把“离线预测正确”误当成“在线改进正确”

结论是：**机制决策上曾发生过这种过度外推，但不能把历史事实简化成“离线
Critic 已经正确”。**

更准确的证据链是：

1. 普通 Task 14 Q-Critic 只证明总体回报 MAE 优于 \(V(s)\)，并未通过动作
   排序、engage 符号、target top-1 或跨场景门槛；
2. 后续模型只在部分接口变强：hierarchical target 排序较好、state-conditioned
   critic 总体 BA 较高、ensemble agreement 较高；
3. MCH/RG-MCH 仍允许这些固定分布证据改变在线 actor；
4. SA-RG 才直接测得在线状态—前缀已大多离开训练支持；
5. BPCE 改用在线成对标签后，仍因双向覆盖和辅助剂量分叉而失败。

所以项目真正犯过的错误是：

> 把固定数据上的局部预测能力、总体指标或模型一致性，外推成当前策略分布上
> 的 policy-improvement certificate。

OCR-CFT 对此提供了正确的概念切分：prediction/evaluation accuracy 与
improvement alignment 必须分别验收。

## 15. Reconstruction 是否需要项目当前没有的数据

需要。论文 OCR 至少需要：

| 目标在线算法 | 所需数据 |
| --- | --- |
| SAC/TD3 | 覆盖 \(\pi_{\mathrm{off}}\) occupancy 的完整 \((s,a,r,s')\) 转移 |
| PPO | 能按状态拟合 return 的完整离线轨迹 |
| value alignment | 每个状态能可靠求 \(\pi_{\mathrm{off}}\) 的高概率/确定性锚动作，并采样邻近或过估动作 |
| CFT | 持续在线转移，以及可可靠更新的历史参考策略 |

项目当前 Task 14 是少量状态上的候选分支 Monte Carlo 标签，不是对 factorized
actor occupancy 的完整轨迹覆盖。BPCE 虽生成真实在线分支 transition，但：

- 只探测少量 engagement 临界上下文；
- 正负标签按 seed 和场景失衡；
- continuation 被冻结，不能自动代表更新后的 actor；
- 没有覆盖完整 state–prefix–mask 可行支持；
- 数据量与方向覆盖不足以支持全局 FQE。

因此当前数据不能合法地执行论文意义上的 critic reconstruction。

## 16. 三类估值器的漂移审计

| 估值器 | policy drift | context/state drift | feasible-support drift | 当前风险 |
| --- | --- | --- | --- | --- |
| 冻结 \(Q(s,h,a)\) | 随 PPO 更新持续累积 | 冻结 90 状态/有限批次到在线状态 | 动作改变后缀 mask，离线候选支持不闭合 | 最高；SA-RG 已直接测得低支持 |
| 每批 COSAC ridge \(\hat f(a)\) | 同一 batch 内顺序更新后也会漂移 | 每批重建减轻跨批漂移，但 action-only surrogate 忽略状态交互 | 虚拟后缀可能落入当前策略/动态合法支持之外 | ridge 可逆不等于覆盖，且加性残差未受控 |
| BPCE/C3 replay 标签 | 生成时接近当前 policy | 在线快照减少状态漂移 | 快照恢复可保持当前合法前缀，但标签只覆盖被探测方向 | 最接近在线证据；仍受双向覆盖、continuation、类别与剂量漂移影响 |

三者都需要 improvement-direction admission，而不应只比较离线 MAE。

## 17. strict fallback 与 CFT 的本质区别

### 17.1 strict fallback

项目要求：

\[
\lambda_{\mathrm{aux}}=0
\quad\Longrightarrow\quad
\Delta\theta_{\mathrm{candidate}}
=
\Delta\theta_{\mathrm{factorized\ joint\ PPO}}
\tag{23}
\]

在同 batch、同随机数、同初始化下，joint log-prob ratio、单个 joint clipping、
GAE、value loss、entropy、optimizer step 和参数更新都必须数值一致。BPCE
已用最大参数差不超过 \(10^{-6}\) 验证过这一软件契约。

### 17.2 constrained fine-tuning

CFT 只保证当前策略在某种距离上接近 \(\pi_{\mathrm{ref}}\)。即使
\(f(\pi,\pi_{\mathrm{ref}})=0\)：

- Critic 已经被 OCR 重构；
- actor–critic alignment 已改变 actor/critic 初始化；
- off-policy critic target 可能含 \(\lambda f\)；
- O2PPO 使用的是辅助 advantage 与参考策略更新；
- 算法并不自动等于项目冻结的 factorized joint PPO。

因此：

\[
\boxed{
\text{bounded deviation} \ne \text{optimizer identity fallback}
}
\tag{24}
\]

CFT 可作为外层稳定机制参照，不能代替 strict fallback。

## 18. 小 KL 能否防止 deterministic argmax 跨界

不能。对 Bernoulli engagement，若概率从 \(0.499\) 变为 \(0.501\)，KL 极小，
但 deterministic rule 已从 no-op 变为 engage。对多个单元共同使用阈值时，
小变化可以同时改变多个联合动作。

项目 SA-RG 的直接证据是：

- initial-anchor engagement KL 均值仅 0.0171；
- 阈值 0.10 从未激活，anchor penalty 为 0；
- 候选仍有 5/6 all-noop。

OCR-CFT 的 KL/MSE 约束在连续动作 benchmark 中可减缓分布漂移，但没有证明：

\[
\arg\max \pi_\theta(\cdot\mid s)
=
\arg\max \pi_{\mathrm{ref}}(\cdot\mid s)
\tag{25}
\]

也没有处理动态 mask 重归一化后联合动作的边界。项目若未来借鉴 CFT，必须
另外审计 deterministic engagement margin、actionable engagement coverage、
联合 argmax 变化率和 all-noop/high-engagement 极端；更小 KL 不能替代这些量。

## 19. Critic 共同偏置下 optimism 的风险

当多个 Critic 因共享数据、结构或标签语义而共同偏置时：

1. 去掉悲观项会提高 OOD 动作 Q；
2. ensemble agreement 仍可能很高，因为模型共享同一错误；
3. value alignment 只把 Q 对齐到 actor；如果 actor 也偏向 all-noop 或高交战，
   锚点本身就是错误；
4. CFT 再把新策略限制在该 actor 附近，只能减缓漂移，不能创造独立真值证据。

所以 optimism 可能放大共同偏置。只有式（9）的覆盖、独立 improvement-direction
证据和可靠 actor 锚同时成立时，重构才可能改善上线尺度错配。项目现有
ensemble agreement 已被 RG-MCH 的共同 OOD 错误否决，不能用 OCR 再包装成
“乐观校准”。

## 20. 当前 MCH/BPCE 接入协议风险清单

| 风险 | MCH/RG-MCH | BPCE/C3 | OCR-CFT 给出的警示 |
| --- | --- | --- | --- |
| 估值正确与改进正确混淆 | 高 | 中 | 必须单独做 actor–critic alignment |
| actor 锚不可靠 | factorized seed 可 all-noop | 当前 actor 也可能已偏 | 高概率动作更好不是自动成立 |
| 在线支持不足 | 已实测 0.124/0.022 | 只覆盖 top-K 临界点 | 覆盖是 OPE 前提，不是事后权重 |
| 动态合法支持变化 | 冻结 Q 不闭合 | 单次快照闭合但窗口不足 | 普通 KL/OPE 未覆盖 prefix-dependent support |
| 共同偏置被放大 | ensemble 一致仍会错 | 标签可单边 | optimism 不是 uncertainty correction |
| scalar credit 混合安全与资源 | 明显 | 仍存在高成本改善 | 参考 return 不能代替多约束语义 |
| deterministic 边界失稳 | 多次 all-noop | all-noop 与 0.959 交战极端并存 | smooth policy distance 不保证行为稳定 |
| fallback 不同于约束 | SA-RG 低门控仍独立 clipping | BPCE 已严格验证 | CFT 不能替代 joint PPO identity |
| 参考策略选择污染 | 未使用历史最优 | 若结果后选 seed/模型会污染 | OCR-CFT 主方案使用在线评估信息 |
| 辅助剂量放大稀疏证据 | 残差多 epoch 使用 | 已实测标签/剂量分叉 | 约束 schedule 也需独立冻结与验证 |

## 21. 最低在线接入证据条件

以下只定义未来任何离线/批内/在线辅助估值进入 actor 更新前的 admission
conditions，不构成训练任务。最多五条：

1. **覆盖条件**：在独立批次上分别报告 state、prefix、legal-mask 和
   engage/target 支持；关键层不得只依赖 ensemble agreement 或 ridge
   可逆性，并预注册拒绝域。
2. **改进方向条件**：同状态动作差、当前 policy 的更新方向和真实成对
   continuation 在正/负两个方向均有独立证据；MAE、总体 BA 或 Q 尺度正确
   不能单独放行。
3. **参考 actor 条件**：候选参考策略必须在冻结种子和核心场景中排除
   all-noop 与高交战极端，并同时通过安全、资源和 deterministic margin
   审计；“标准 PPO”身份不等于可靠 teacher。
4. **严格退化条件**：辅助系数为零、覆盖拒绝或证据不足时，完整 joint PPO
   更新必须在固定容差内逐参数一致；不得保留独立层级 ratio/clipping。
5. **行为与规范条件**：除 KL/MSE 外，必须验收联合 argmax 变化、交战覆盖和
   安全—资源分层非劣；单一 return、单一 Q 差或小 KL 不得作为上线门槛。

## 22. 强基线、可迁移部分与 no-go

### 22.1 `BASELINE`

OCR-CFT 是以下主张的强机制基线：

- 离线到在线需要分别处理 evaluation mismatch、improvement mismatch 和
  online distribution shift；
- 不能只继承离线 Critic，也不能只加 policy constraint；
- O2PPO 的最邻近基线应保留 GAE，再加入参考策略 regularization，而不是
  用离线 Q 完全替换 GAE。

但它目前只是**机制参照基线**。只有未来真的建立“可靠离线 actor + 完整离线
轨迹/转移 + 在线 PPO/SAC/TD3”的 O2O 任务时，才应成为可运行实现基线。

### 22.2 `ADAPT`

可迁移的是审计框架，而不是原公式：

```text
Estimator reconstruction / refresh
        → improvement-direction alignment
        → online support and behavior constraint
        → exact joint-PPO fallback
```

可以把“高概率动作锚”重写为只读的 actor–critic 排序一致性诊断，并把
\(\pi_{\mathrm{ref}}\) 改成满足多指标预注册门控的参考策略；当前不授权接入
actor loss。

### 22.3 `AVOID`

- 不把乐观 Q 当成安全 Q；
- 不在当前 338 条训练候选上运行 FQE/OCR 并上线；
- 不把 factorized PPO 自动视为 reliable offline actor；
- 不直接移植 SAC/TD3 的连续动作 value alignment；
- 不用小 KL、动作 MSE 或 ensemble agreement 替代 deterministic 行为证据；
- 不用历史最优 return 结果后选择参考策略；
- 不把 CFT 当成 strict fallback；
- 不启动 Critic 重训、在线微调或新的 10k/30k/100k 实验。

### 22.4 `OPEN`

仍开放但尚未形成算法的命题是：

> 在 state–prefix–mask 支持随动作和策略共同变化、最终行为由联合
> deterministic argmax 产生的 on-policy PPO 中，怎样构造一个可拒绝的
> improvement-direction certificate，使辅助估值只在双向、跨批次、规范
> 目标一致的区域影响 actor，并在证据不足时严格恢复 joint PPO。

该问题没有被 OCR-CFT 的固定离线支持、连续动作距离或可靠 actor 假设覆盖；
但目前只有问题差异，没有新估计量、理论或在线证据，不得建立创新声明。

## 23. 创新压力测试

| 候选叙事 | 最近工作覆盖 | 判定 |
| --- | --- | --- |
| 重训离线 Critic 以适合在线更新 | OCR 已直接覆盖 | 不能作为项目创新 |
| 用离线 actor 对齐重构 Critic | OCR 核心 value alignment | 不能换名复用 |
| 用 KL/MSE 约束在线微调并逐步放松 | CFT 已直接覆盖 | 只能作为基线 |
| PPO 保留 GAE 并加参考策略 log-prob advantage | O2PPO 已直接覆盖 | RG/BPCE 需说明不同接口 |
| 用历史最好策略作为可信中心 | CFT 已覆盖，且有评价信息边界 | 不能作为新机制 |
| 动态 mask 下同时审计 support、argmax 和 exact fallback | OCR 未处理该组合 | 研究问题仍开放，尚非算法 |
| 安全—资源多约束下的 improvement admission | OCR 只优化单一 return | 与 LR-02/03 合并后开放 |

本轮阅读进一步否决了“只要把 Critic 从 pessimistic 改为 optimistic，再用 KL
保护 actor 就能恢复 MCH”的直觉。它没有解决项目最关键的可靠 actor、
动态可行支持、离散行为边界和规范多约束问题。

## 24. 与 LR-01 至 LR-05 的合并边界

| 前序任务 | 给 LR-06 的边界 | LR-06 结论 |
| --- | --- | --- |
| LR-01 | 解释效应不自动成为优化目标 | OCR alignment 也不能把 actor 偏好变成规范真值 |
| LR-02 | 全局约束需有明确 cost 与 threshold | 单一 return 的历史最优参考不够 |
| LR-03 | 正确约束梯度的组合晚于语义和估值验收 | CFT 的 λ 不能修复错误 Critic/cost |
| LR-04 | 自回归合法生成与顺序偏置已有强基线 | OCR 未处理动态离散可行后缀 |
| LR-05 | 冻结 Q、batch ridge、replay 标签均有不同漂移 | 三者都需 reconstruction/refresh、alignment、support、fallback 四层审计 |

六篇合并后，当前最稳健的总结构是：

```text
规范目标：全局安全/资源约束先冻结
                    ↓
动作结构：动态合法、无冲突 joint AR policy
                    ↓
估值接口：覆盖与 improvement direction 独立验收
                    ↓
在线约束：行为边界 + 多约束，而非仅平均 KL
                    ↓
失败退化：完整 joint PPO 数值同一
```

这是一张头脑风暴输入图，不是新算法任务。

## 25. 最终判决

| 标签 | 判决 |
| --- | --- |
| `BASELINE` | OCR-CFT 是 offline-to-online 三类错配和 O2PPO“GAE + 参考策略辅助”的强机制参照 |
| `ADAPT` | 适配 reconstruction/refresh → improvement alignment → support/behavior constraint → strict fallback 的四层审计流程 |
| `AVOID` | 当前不重构 Critic、不做乐观上线、不移植连续 value alignment、不以小 KL 或历史最优 return 放行 |
| `OPEN` | 动态 state–prefix–mask 支持、联合 argmax 行为、多约束规范和 exact joint-PPO fallback 下的可拒绝改进证书 |

回答指导文件最后一个问题：本论文当前应作为**机制参照，兼具强警示作用**，
而不是立即实现的算法基线。其最重要的正面启示是分开验收评价与改进；最重要
的警示是可靠 actor、覆盖和策略距离都不能在本项目中被默认满足。

## 26. 术语表与来源锚点

| 英文 | 本报告译法 | 原文锚点 |
| --- | --- | --- |
| offline-to-online RL, O2O RL | 离线到在线强化学习 | PDF pp.1–2 |
| evaluation mismatch | 评价错配 | PDF pp.2–3 |
| improvement mismatch | 改进错配 | PDF pp.2–3 |
| policy re-evaluation | 策略重评价 | PDF pp.4–5 |
| optimistic critic reconstruction | 乐观 Critic 重构 | PDF pp.4–5, App. J |
| single-policy concentrability | 单策略覆盖集中性 | PDF p.4, Assumption 4.1 |
| fitted Q evaluation, FQE | 拟合 Q 评价 | PDF p.4, Corollary 4.2 |
| value alignment | 价值对齐 | PDF pp.5–7 |
| anchor action | 锚动作 | PDF pp.5–6 |
| auxiliary advantage | 辅助优势 | PDF p.7, Eq. 17–18 |
| constrained fine-tuning, CFT | 受约束微调 | PDF pp.7–8 |
| reference policy | 参考策略 | PDF pp.7–8, App. C.3 |
| constraint threshold | 约束阈值 | PDF App. H.3–H.4 |
| strict fallback | 严格退化/同一优化器回退 | 项目术语，非论文原词 |
| improvement-direction certificate | 改进方向证书 | 本报告问题重写，非论文原词 |

### 关键来源索引

- 论文身份、摘要与贡献：PDF pp.1–2；NeurIPS 官方页面；
- 两类 mismatch 与 behavior-regularized MDP：PDF pp.3–4, Eq. 9；
- 单策略集中性与 FQE 上界：PDF p.4, Assumption 4.1, Corollary 4.2；
- O2SAC value alignment：PDF pp.5–6, Eqs. 10–14；
- O2TD3 value alignment：PDF p.6, Eqs. 15–16；
- O2PPO auxiliary advantage：PDF pp.6–7, Eqs. 17–18；App. F/G.3/H.5；
- CFT 目标、乘子与渐近结论：PDF pp.7–8, Eqs. 19–20, Corollary 4.5；
- 主实验与消融：PDF pp.8–16, Figures 2–5, Tables 1–3；
- 参考策略固定间隔替代：PDF App. C.3；
- 实现阈值、replay 与 schedule：PDF App. H；
- 三个算法伪代码：PDF App. J, Algorithms 1–3；
- 官方实现入口：<https://github.com/QinwenLuo/OCR-CFT>。
