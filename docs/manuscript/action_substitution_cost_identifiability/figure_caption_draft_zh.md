# 主图中文图注草稿

更新时间：2026-07-28  
方向约定：\(N\) 为当前探针 no-op，\(E\) 为指定合法 engage，
\(\Delta C_{\mathrm{episode}}=C(E)-C(N)\)

## Figure 1 | 动态掩码序列分配混合局部直接成本与策略介导资源消耗

**a，** 三个防御单元按固定顺序 0、1、2 形成条件联合动作；前序单元占用的目标
从后续单元合法掩码中移除。**b，** 在同一冻结状态下，\(N\) 分支强制探针
no-op，\(E\) 分支强制指定合法 engage；两个分支共享随机带，但使用各自实时
合法掩码生成同一步后缀和未来动作。**c，** 当前正直接成本减去同一步与未来
总替代成本后得到回合累计成本差，因此累计差可以为零或负。本图是机制示意，
对应 AirDefense v1 的 3 单元、5 目标动态动作结构；正式定量评估覆盖三个场景、
9 个来源模型和 108 个 context，但本图不表示经验发生率，也不支持 PPO 性能
提升或跨环境普遍性。

## Figure 2 | 成对反事实协议与三分量成本恒等式

**a，** 每个 repeat 从同一环境快照开始，以共同环境随机带和策略均匀随机带
回放 N/E 分支，并由冻结 Actor 执行 stochastic continuation。**b，** \(E\)
分支对全部合法目标按条件于 engage 的目标概率精确边缘化，因而目标账本行不是
独立样本。**c，** 总替代成本由同一步其他单元、未来探针和未来其他单元三项
组成。**d，** future-only 公式在 7,776 条目标条件账本中影响 287 条，最大
残差为 2.0；加入同一步项后最大残差为 \(8.88\times10^{-16}\)，低于
\(10^{-6}\) 容限。这里的“精确”只表示逐账本代数重构；CRN 只降低成对方差，
不自行证明结构可辨识或统计无偏。完整性结果来自三个场景、9 个来源模型和
108 个 context；每个 context 有 32 个 paired repeat，目标账本行为精确边缘化
构成项。

## Figure 3 | R1 动作替代发现及 R2 新策略种子独立确认

所有面板限定于 AirDefense v1 的 `time_pressure/resource` 切片；R1 和 R2
在该切片各含 3 个来源模型。
**a，** R1 使用旧来源策略 seeds 8/9/10；点为 18 个预设 resource context 的
\(Sub_{\mathrm{shot}}\) 均值，误差线为每 context 32 个 N/E paired repeat 的
\(\bar{x}\pm1.96s/\sqrt n\)。**b，** R2 使用新策略 seeds 17/18/19 和零旧
观测 hash 重叠的新 context，以相同统计口径显示 18 个 context。空心/实心点
分别区分发现与确认，不表示样本权重不同。**c，** 每个点为一个新 seed block
内 6 个 context 的均值和 95% 区间；三个下界均高于零。**d，** 对
\(\Delta C_{\mathrm{episode}}\le0\) 的 context，R1 的 11/11 和 R2 的 7/7
均有正 \(Sub_{\mathrm{cost,total}}\)。R1 只承担发现，独立确认由 R2 承担；
结果不外推到任意种子、算法或环境。

## Figure 4 | 同一步与未来动作共同构成总替代成本

所有面板使用 R2 `time_pressure/resource` 的 3 个来源模型和 18 个 context，
每个 context 有 32 个 paired repeat，并对 context 等权。**a，** 当前探针
直接成本、由同一步其他单元/未来探针/未来其他单元组成
的总替代成本，以及二者相减后的回合累计成本差；颜色编码对应冻结三分量账本。
**b，** 同一步分量约占总替代成本 17%，严格未来分量约占 83%。
**c，** 每个点为一个 context，横轴为
\(\rho_{\mathrm{sub}}=Sub_{\mathrm{cost,total}}/C_{\mathrm{direct}}\)，纵轴为
\(\Delta C_{\mathrm{episode}}\)，颜色表示 missile/laser，点大小表示 32 次
repeat 中的成本符号掩盖率。虚线为 \(\rho_{\mathrm{sub}}=1\)，实线为累计成本
差零点。本图使用成本量，不能以 \(Sub_{\mathrm{shot}}\) 代替总替代成本，也不
证明两类资源具有相同掩盖强度。

## Figure 5 | 场景与资源类型限定动作替代的符号掩盖强度

**a，** R2 三个场景、共 9 个来源模型的 resource 槽
\(\rho_{\mathrm{sub}}\)；每场景 \(n=18\) 个 context，点和误差线为 context
均值及
\(\bar{x}\pm1.96s/\sqrt n\)。**b，** `time_pressure/resource` 中 missile
与 laser 的 \(Sub_{\mathrm{shot}}\)；每类型 \(n=9\) 个 context，两类 95%
下界均为正；该切片来自 3 个 `time_pressure` 来源模型。**c，** 平均累计成本
发生符号掩盖的 context 数；冻结 P-C3 要求
每种资源至少 3 个，missile 为 2/9、laser 为 5/9。**d，** P-C1 成本分解和
P-C2 独立确认通过，P-C3 跨资源类型普遍门控失败。因此只支持资源类型条件
主张；不支持“missile 无动作替代”，也不支持跨资源类型同强度或跨环境泛化。
