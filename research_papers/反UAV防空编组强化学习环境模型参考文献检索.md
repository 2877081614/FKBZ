# 反 UAV 防空编组强化学习环境模型参考文献检索

检索日期：2026-07-09  
检索目标：为“防空编组中反 UAV 场景的强化学习环境模型搭建”寻找参考文献，重点关注状态空间、agent 指代对象、动作空间、奖励函数、约束、部分可观测和任务分配基线。

## 1. 检索结论

目前直接以“反 UAV 防空编组 + 强化学习环境建模”为主题的成熟论文不多，最接近的文献分布在四条线：

1. **直接反 UAV / drone swarm defense 强化学习**：适合借鉴状态特征、动作定义、拦截优先级和高保真仿真环境。
2. **perimeter defense / pursuit-evasion / UAV swarm MARL**：适合借鉴入侵者-防御者结构、多目标追踪和部分可观测建模。
3. **C-UAS survey 与探测/干扰系统论文**：适合支撑 Sensor、Jammer、Interceptor 等 agent 类型的现实合理性。
4. **MARL 环境与算法基础文献**：适合支撑 Dec-POMDP/CTDE、局部观测、离散/连续动作、PettingZoo API 和 MAPPO/HAPPO/MADDPG 基线。

对你下一步最有用的直接启发是：

- **状态空间**可包含：目标位置/速度/类别/威胁等级、保护目标价值、传感器探测状态、拦截器剩余弹药/冷却/射程、干扰器可用频段/作用半径、通信邻接、目标预计到达时间、任务分配关系。
- **agent 类型**建议至少区分：Sensor/Tracker、Jammer、Interceptor/Hard-kill effector、Command/Allocator，也可以把每个具体平台建成 agent，并用 type embedding 表示资源类型。
- **动作空间**可从简单离散分配起步：选择目标、等待、跟踪/干扰/拦截；随后扩展到混合动作：目标选择 + 功率/频段/航向/速度/发射时机。
- **奖励函数**应同时覆盖：保护目标损伤、拦截成功、威胁延迟或压制、资源消耗、误击/重复分配、通信与重分配代价。

## 2. 最优先阅读文献

### 2.1 Reinforcement Learning for Decision-Level Interception Prioritization in Drone Swarm Defense

- 链接：[arXiv:2508.00641](https://arxiv.org/abs/2508.00641)
- 作者：Alessandro Palmas
- 年份：2025
- 类型：直接反无人机 / drone swarm defense / 拦截优先级 RL

**为什么优先读：**  
这是目前检索到的最贴近“反 UAV 防空编组环境建模”的强化学习论文之一。论文构建高保真仿真环境，让决策层 RL agent 协调多个 effector 的拦截优先级。其动作空间是离散动作，即每个 effector 选择要打击的 drone；状态特征包含 drone 的位置、类别和 effector 状态。

**可借鉴点：**

- 状态空间：目标位置、目标类别、目标威胁、effector 状态。
- 动作空间：每个 effector 选择一个目标进行 engagement。
- 奖励/评价：保护高价值区域、降低损伤、提高防御效率。
- 基线：与手工规则策略比较。

**对本课题的价值：**  
可作为“第一版环境”的直接模板：多个来袭 UAV + 多个拦截资源 + 离散目标选择动作。

### 2.2 Delay-Aware Active Triangulation with Uncertainty-Driven Multi-Agent Reinforcement Learning for Counter-UAS

- 链接：[arXiv:2607.05957](https://arxiv.org/abs/2607.05957)
- 作者：Seungwook Lee, David Hyunchul Shim
- 年份：2026
- 类型：Counter-UAS / 多智能体主动定位 / Dec-POMDP / MAPPO

**为什么优先读：**  
这是非常新的 Counter-UAS MARL 论文，关注多个移动观察者和可控相机协同定位空中目标。论文显式把问题建成 Dec-POMDP，并把 Age-of-Information、观测延迟、像素/位姿/云台/内参不确定性纳入观测与奖励。

**可借鉴点：**

- 状态/观测：局部观测、延迟信息、AoI、观测不确定性。
- agent：移动传感器/观察平台，而非拦截武器。
- 动作：移动观察者或控制相机姿态。
- 奖励：定位误差、track loss、triangulation validity。

**对本课题的价值：**  
适合支撑 Sensor/Tracker 类 agent 的建模，尤其是“防空编组中感知链并非完美状态输入”的设定。

### 2.3 A Decentralized Multi-UAV Spatio-Temporal Multi-Task Allocation Approach for Perimeter Defense

- 链接：[arXiv:2102.07381](https://arxiv.org/abs/2102.07381)
- 作者：Shridhar Velhal, Suresh Sundaram, Narasimhan Sundararajan
- 年份：2021
- 类型：perimeter defense / 多 UAV 防御任务分配

**为什么优先读：**  
论文把入侵者视为即将在特定时间、特定位置到达边界的 spatio-temporal task，防御者需要在入侵者进入保护区域前完成捕获。这个建模与“来袭 UAV 突防保护区”的结构高度相似。

**可借鉴点：**

- 威胁建模：把 UAV 转化为带预计到达时间和到达位置的任务。
- 状态空间：入侵者位置、速度、预计边界交点、预计到达时间、防御者位置。
- 动作空间：防御者选择任务/目标并执行拦截轨迹。
- 基线：去中心化任务分配，可作为非学习基线。

**对本课题的价值：**  
适合支撑“反 UAV 防御不是纯追踪，而是带时间窗口的动态任务分配问题”。

### 2.4 MAGNNET: Multi-Agent Graph Neural Network-based Efficient Task Allocation for Autonomous Vehicles with Deep Reinforcement Learning

- 链接：[arXiv:2502.02311](https://arxiv.org/abs/2502.02311)
- 作者：Lavanya Ratnabala, Aleksey Fedoseev, Robinroy Peter, Dzmitry Tsetserukou
- 年份：2025
- 类型：异构多智能体 / GNN / PPO / 动态任务分配

**为什么优先读：**  
论文处理异构 UAV/UGV 在通信约束下的去中心化任务分配，将 GNN、CTDE 和 PPO 结合起来。虽然不是反 UAV，但“异构 agent + 动态任务 + 冲突避免 + 去中心化执行”与防空编组资源分配非常贴近。

**可借鉴点：**

- 将 agent 和 task 建成图。
- 用 GNN 表达资源-目标关系。
- 动作是任务分配选择。
- 评价指标包括 conflict-free success rate、任务处理时间、扩展性。

**对本课题的价值：**  
适合支撑“资源-目标二部图”或“防空资源图 + 威胁任务图”的环境设计。

## 3. C-UAS 场景与资源类型支撑文献

### 3.1 Counter-Unmanned Aircraft System(s): State of the Art, Challenges and Future Trends

- 链接：[arXiv:2008.12461](https://arxiv.org/abs/2008.12461)
- 作者：Jian Wang, Yongxin Liu, Houbing Song
- 年份：2020
- 类型：C-UAS survey

**可借鉴点：**

- C-UAS 功能链条：detect、track、identify、mitigate。
- 探测技术：acoustic、vision、passive RF、radar、data fusion。
- 处置技术：jamming、capture、hard kill 等。
- 场景挑战：小型目标、低空、蜂群、误警、法规约束。

**对环境设计的作用：**  
支撑你把 agent 类型拆成 Sensor/Tracker、Jammer、Interceptor，并解释为什么反 UAV 环境应包含软杀伤和硬杀伤资源。

### 3.2 A Survey on Detection, Tracking, and Classification of Aerial Threats using Radars and Communications Systems

- 链接：[arXiv:2211.10038](https://arxiv.org/abs/2211.10038)
- 作者：Wahab Khawaja et al.
- 年份：2022
- 类型：雷达/通信感知综述

**可借鉴点：**

- UAV 相比有人机更难探测，原因包括尺寸小、形状复杂、贴地飞行、蜂群和自主机动。
- 雷达、joint communication-radar、passive communication signal monitoring 都可用于 UAV 探测、跟踪和分类。

**对环境设计的作用：**  
支撑 Sensor agent 的状态不是“全知全能”，而应有探测概率、跟踪质量、分类置信度、观测延迟等变量。

### 3.3 An Autonomous Drone System with Jamming and Relative Positioning Capabilities

- 链接：[arXiv:2206.04307](https://arxiv.org/abs/2206.04307)
- 作者：Nicolas Souli, Panayiotis Kolios, Georgios Ellinas
- 年份：2022
- 类型：反无人机干扰与相对定位系统

**可借鉴点：**

- 使用 SDR 在 jamming transmission 与 spectrum sweeping 之间切换。
- 将 GPS disruption、wireless interception 和 self-localization 结合。
- 通过 field experiments 验证实际环境中的干扰效果。

**对环境设计的作用：**  
支撑 Jammer agent 的动作空间，例如选择干扰目标、干扰频段、功率档位、持续时间，或在“搜索/定位”和“干扰/压制”之间切换。

## 4. 多 UAV / 部分可观测 / 追逃类建模参考

### 4.1 Multi-Target Pursuit by a Decentralized Heterogeneous UAV Swarm using Deep Multi-Agent Reinforcement Learning

- 链接：[arXiv:2303.01799](https://arxiv.org/abs/2303.01799)
- 作者：Maryam Kouzeghar, Youngbin Song, Malika Meghjani, Roland Bouffanais
- 年份：2023
- 类型：异构 UAV swarm / 多目标追踪 / MADDPG

**可借鉴点：**

- 异构 pursuer 角色设计。
- 同时处理 exploration 和 tracking。
- 多目标、随机障碍、非平稳未知环境。
- 通过仿真和 Crazyflie 实验验证。

**对环境设计的作用：**  
适合借鉴 UAV 目标追踪、巡逻搜索、目标再发现、异构角色分工等状态与奖励设计。

### 4.2 Deep Decentralized Multi-task Multi-Agent Reinforcement Learning under Partial Observability

- 链接：[arXiv:1703.06182](https://arxiv.org/abs/1703.06182)
- 作者：Shayegan Omidshafiei, Jason Pazis, Christopher Amato, Jonathan P. How, John Vian
- 年份：2017
- 类型：部分可观测 / 多任务 / Decentralized MARL

**可借鉴点：**

- 多智能体局部观测导致环境非平稳。
- 任务身份可能不可观测。
- 需要能在多个相关任务间泛化的策略。

**对环境设计的作用：**  
支撑你把防空编组建成 Dec-POMDP 或 partially observable stochastic game，而不是默认每个 agent 都看到全局真实状态。

### 4.3 Learning Approach to Efficient Vision-based Active Tracking of a Flying Target by an Unmanned Aerial Vehicle

- 链接：[arXiv:2506.18264](https://arxiv.org/abs/2506.18264)
- 年份：2025
- 类型：飞行目标主动跟踪 / AirSim / RL 状态动作奖励设计

**可借鉴点：**

- 针对飞行目标跟踪提出 state space、action space 和 reward formulation。
- 在 AirSim 中训练速度机动策略。
- 强调让目标保持在未来视场内，而不只是当前检测。

**对环境设计的作用：**  
如果你后续把 Sensor/Interceptor 建成可机动平台，这篇适合借鉴单平台主动跟踪状态与动作。

## 5. 任务分配与规则基线文献

### 5.1 A Formal Analysis and Taxonomy of Task Allocation in Multi-Robot Systems

- 链接：[DOI: 10.1177/0278364904045564](https://doi.org/10.1177/0278364904045564)
- 作者：Brian P. Gerkey, Maja J. Matarić
- 年份：2004
- 类型：MRTA 经典分类

**可借鉴点：**

- 机器人任务分配分类：single-task/multi-task robot、single-robot/multi-robot task、instantaneous/time-extended assignment。
- 为“防空资源-威胁目标分配”提供任务分配术语体系。

**对环境设计的作用：**  
帮助你明确问题属于动态、时间扩展、异构能力约束下的多资源任务分配。

### 5.2 Consensus-Based Decentralized Auctions for Robust Task Allocation

- 链接：[DOI: 10.1109/TRO.2009.2022423](https://doi.org/10.1109/TRO.2009.2022423)
- 作者：Han-Lim Choi, Luc Brunet, Jonathan P. How
- 年份：2009
- 类型：CBBA / 去中心化拍卖任务分配

**可借鉴点：**

- Bundle construction。
- Consensus phase。
- Conflict-free assignment。
- 分布式任务分配与通信需求。

**对环境设计的作用：**  
适合作为规则基线，和 MARL 策略比较。

### 5.3 Partial Replanning for Decentralized Dynamic Task Allocation

- 链接：[arXiv:1806.04836](https://arxiv.org/abs/1806.04836)
- 作者：Noam Buckman, Han-Lim Choi, Jonathan P. How
- 年份：2018
- 类型：CBBA-PR / 动态任务到达 / 局部重规划

**可借鉴点：**

- 新任务在线到达时，不必完全重分配所有任务。
- 可在收敛时间、通信负担和协调质量之间折中。

**对环境设计的作用：**  
可支撑动态 UAV 威胁到达、局部重编组、切换代价和重分配代价设计。

## 6. MARL 基础与环境 API 参考

### 6.1 Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments

- 链接：[arXiv:1706.02275](https://arxiv.org/abs/1706.02275)
- 作者：Ryan Lowe et al.
- 年份：2017
- 类型：MADDPG / CTDE

**可借鉴点：**  
集中训练、分散执行；critic 可考虑其他 agent 的动作策略；适合连续动作或混合协作-竞争场景。

### 6.2 The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games

- 链接：[arXiv:2103.01955](https://arxiv.org/abs/2103.01955)
- 作者：Chao Yu et al.
- 年份：2021
- 类型：MAPPO / cooperative MARL baseline

**可借鉴点：**  
MAPPO 是合作 MARL 中很强的基线，适合作为第一版防空编组环境的 baseline。

### 6.3 PettingZoo: Gym for Multi-Agent Reinforcement Learning

- 链接：[arXiv:2009.14471](https://arxiv.org/abs/2009.14471)
- 作者：J. K. Terry et al.
- 年份：2020
- 类型：MARL 环境 API

**可借鉴点：**

- 多智能体环境统一 API。
- Agent Environment Cycle, AEC 模型。
- 有利于复现实验、替换算法和接入 MARL 框架。

**对环境设计的作用：**  
建议你的反 UAV 环境从一开始就按 PettingZoo ParallelEnv 或 AEC 风格设计。

### 6.4 MAgent: A Many-Agent Reinforcement Learning Platform for Artificial Collective Intelligence

- 链接：[arXiv:1712.00600](https://arxiv.org/abs/1712.00600)
- 作者：Lianmin Zheng et al.
- 年份：2017
- 类型：大规模多智能体平台

**可借鉴点：**  
适合大规模蜂群、网格化战场、局部观测和群体行为涌现类环境设计。

## 7. 建议的环境模型设计框架

### 7.1 状态空间设计

建议将全局状态分成五类：

1. **威胁目标状态**：位置、速度、航向、高度、类型、载荷/威胁等级、剩余到达时间、是否被发现、是否被跟踪、是否被压制、是否被拦截。
2. **防御资源状态**：资源类型、位置、朝向、射程/探测范围、剩余弹药、冷却时间、干扰功率、可用频段、健康状态。
3. **保护目标状态**：位置、价值权重、剩余生命值/可承受损伤、区域边界。
4. **信息状态**：目标分类置信度、跟踪误差、观测延迟、通信邻接、AoI、是否丢失目标。
5. **任务分配状态**：资源-目标当前匹配、重复分配数量、任务切换次数、未覆盖威胁数量。

### 7.2 Agent 类型设计

第一版建议采用三类 agent：

1. **Sensor/Tracker agent**：负责探测、跟踪、分类和信息更新。
2. **Jammer agent**：负责软杀伤、压制导航/通信、延迟或降低目标突防能力。
3. **Interceptor agent**：负责硬杀伤、发射拦截器、目标打击。

第二版可加入：

4. **Command/Allocator agent**：负责资源-目标分配或重分配。
5. **Mobile platform agent**：如果资源可机动，则负责位移、阵位选择和保持通信。

### 7.3 动作空间设计

可以分三阶段设计：

**阶段 1：离散任务分配动作**

- wait/no-op；
- track target j；
- jam target j；
- intercept target j；
- reassign/hold current assignment。

**阶段 2：参数化离散动作**

- 目标选择 + 干扰功率档位；
- 目标选择 + 传感器模式；
- 目标选择 + 发射窗口；
- 目标选择 + 拦截弹类型。

**阶段 3：混合动作**

- 离散目标选择；
- 连续机动速度/航向；
- 连续干扰功率；
- 连续雷达波束/搜索区域。

### 7.4 奖励函数设计

建议从团队共享 reward 起步：

```text
R = + 拦截成功奖励
    + 威胁延迟/压制奖励
    - 保护目标损伤惩罚
    - 弹药/能量/干扰资源消耗
    - 重复分配惩罚
    - 误击/误判/规则违反惩罚
    - 任务切换或重编组代价
```

如果使用异质 agent，可再加少量局部 shaping reward，但最终评价指标应以团队防御效果为主。

## 8. 推荐阅读顺序

1. C-UAS survey：先理解反无人机功能链条和资源类型。
2. Interception prioritization in drone swarm defense：直接看状态/动作/奖励如何建。
3. Perimeter defense DMUST-MTA：理解来袭 UAV 如何转成时空任务。
4. MAGNNET：理解资源-任务图和异构任务分配。
5. Delay-aware active triangulation：补充 Sensor/Tracker 的部分可观测、延迟、不确定性。
6. HAPPO/HARL：连接到异质 agent 策略学习。
7. PettingZoo/MAPPO/MADDPG：用于环境 API 和基线算法实现。

## 9. 可直接进入论文 related work 的文献组合

如果篇幅有限，建议正文至少引用以下 8 篇：

1. Wang et al., 2020, C-UAS survey。
2. Palmas, 2025, drone swarm interception prioritization RL。
3. Velhal et al., 2021, perimeter defense DMUST-MTA。
4. Ratnabala et al., 2025, MAGNNET。
5. Kouzeghar et al., 2023, heterogeneous UAV swarm pursuit。
6. Omidshafiei et al., 2017, partially observable decentralized multi-task MARL。
7. Yu et al., 2021, MAPPO。
8. Terry et al., 2020, PettingZoo。

如果要突出防空传感器/干扰资源，再补：

9. Khawaja et al., 2022, radar/communication aerial threat survey。
10. Souli et al., 2022, jamming and relative positioning counter-drone system。

