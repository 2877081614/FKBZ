# Dynamic Resource Target Assignment Problem for Laser Systems' Defense Against Malicious UAV Swarms Based on MADDPG-IA

## 论文信息 / Paper Metadata

| Field                         | Value                                                                                                                 |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Title**                     | Dynamic Resource Target Assignment Problem for Laser Systems' Defense Against Malicious UAV Swarms Based on MADDPG-IA |
| **中文标题**                      | 基于MADDPG-IA的激光系统防御恶意无人机蜂群的动态资源目标分配问题                                                                                  |
| **Authors**                   | Wei Liu, Lin Zhang, Wenfeng Wang, Haobai Fang, Jingyi Zhang, Bo Zhang                                                 |
| **Affiliation**               | Air and Missile Defense College, Air Force Engineering University, Xi'an; Unit of 95972, PLA                          |
| **Journal**                   | Aerospace 2025, 12, 729 (MDPI)                                                                                        |
| **DOI**                       | [10.3390/aerospace12080729](https://doi.org/10.3390/aerospace12080729)                                                |
| **Received/Revised/Accepted** | 14 July 2025 / 12 August 2025 / 15 August 2025                                                                        |
| **License**                   | CC BY 4.0 Open Access                                                                                                 |
| **Keywords**                  | HELS; UAV swarms; DRTA; combinatorial optimization; MARL; attention mechanism; intrinsic reward                       |

---

## 目录 / Page Index

| Section                                               | Pages   |
| ----------------------------------------------------- | ------- |
| Abstract                                              | p.1     |
| 1. Introduction                                       | p.2–4   |
| 2. Modeling the HELS–UAV–DRTA Problem                 | p.4–17  |
| 2.1 Laser Damage Model                                | p.4–11  |
| 2.2 Malicious UAV Swarm Density Model                 | p.12–13 |
| 2.3 HELS–UAV–DRTA Model Formulation                   | p.14–17 |
| 3. MADDPG-IA Algorithm Design                         | p.17–24 |
| 3.1 MADDPG Framework                                  | p.18–20 |
| 3.2 Enhanced MADDPG with Attention & Intrinsic Reward | p.20–24 |
| 4. Simulation and Analysis                            | p.25–33 |
| 4.1 Experimental Setup                                | p.25–26 |
| 4.2 Typical Scenario Experiments                      | p.26–30 |
| 4.3 Algorithm Performance Verification                | p.30–33 |
| 5. Conclusions and Outlook                            | p.33–34 |
| References                                            | p.34–36 |

---

## 术语表 / Terminology

| English          | 中文            | 说明                                                |
| ---------------- | ------------- | ------------------------------------------------- |
| HELS             | 高能激光系统        | High-Energy Laser System                          |
| UAV              | 无人机           | Unmanned Aerial Vehicle                           |
| DRTA             | 动态资源目标分配      | Dynamic Resource Target Assignment                |
| MADDPG           | 多智能体深度确定性策略梯度 | Multi-Agent Deep Deterministic Policy Gradient    |
| MADDPG-IA        | MADDPG-IA算法   | I = Intrinsic reward, A = Attention               |
| RND              | 随机网络蒸馏        | Random Network Distillation                       |
| MARL             | 多智能体强化学习      | Multi-Agent Reinforcement Learning                |
| CTDE             | 集中训练分散执行      | Centralized Training with Decentralized Execution |
| ATP              | 捕获跟踪瞄准(系统)    | Acquisition Tracking Pointing                     |
| FSM              | 快速转向镜         | Fast Steering Mirror                              |
| thermal blooming | 热晕效应          | 激光大气传输热畸变                                         |
| battery magazine | 电池弹匣 / 能量储备   | HELS continuous irradiation capacity              |

---

## Abstract / 摘要

<a id="S000"></a>
**Source:** p.1

**Original:**

The widespread adoption of Unmanned Aerial Vehicles (UAVs) in civilian domains, such as airport security and critical infrastructure protection, has introduced significant safety risks that necessitate effective countermeasures. High-Energy Laser Systems (HELSs) offer a promising defensive solution; however, when confronting large-scale malicious UAV swarms, the Dynamic Resource Target Assignment (DRTA) problem becomes critical.

**中文：**

无人机在民用领域的广泛应用（如机场安保和关键基础设施保护）带来了重大安全风险，亟需有效的反制措施。高能激光系统是一种有前景的防御方案；然而，面对大规模恶意无人机蜂群时，动态资源目标分配问题变得至关重要。

**Original:**

To address the challenges of complex combinatorial optimization problems, a method combining precise physical models with multi-agent reinforcement learning (MARL) is proposed. Firstly, an environment-dependent HELS damage model was developed. This model integrates atmospheric transmission effects and thermal effects to precisely quantify the required irradiation time to achieve the desired damage effect on a target. This forms the foundation of the HELS–UAV–DRTA model, which employs a two-stage dynamic assignment structure designed to maximize the target priority and defense benefit.

**中文：**

为解决复杂的组合优化问题，本文提出了一种将精确物理模型与多智能体强化学习相结合的方法。首先，建立了环境依赖的HELS毁伤模型，该模型综合了大气传输效应和热效应，以精确量化对目标达到期望毁伤效果所需的照射时间。这构成了HELS–UAV–DRTA模型的基础，该模型采用两阶段动态分配结构，旨在最大化目标优先级和防御收益。

**Original:**

An innovative MADDPG-IA (I: intrinsic reward, and A: attention mechanism) algorithm is proposed to meet the MARL challenges in the HELS–UAV–DRTA problem: an attention mechanism compresses variable-length target states into fixed-size encodings, while a Random Network Distillation (RND)-based intrinsic reward module delivers dense rewards that alleviate the extreme reward sparsity.

**中文：**

提出了创新的MADDPG-IA算法（I：内在奖励，A：注意力机制）来解决HELS–UAV–DRTA问题中的MARL挑战：注意力机制将可变长度的目标状态压缩为固定大小的编码，而基于随机网络蒸馏的内在奖励模块提供密集奖励，缓解了极度稀疏的奖励问题。

**Original:**

Large-scale scenario simulations (100 independent runs per scenario) involving 50 UAVs and 5 HELS across diverse environments demonstrate the method's superiority, achieving mean damage rates of 99.65% ± 0.32% vs. 72.64% ± 3.21% (rural), 79.37% ± 2.15% vs. 51.29% ± 4.87% (desert), and 91.25% ± 1.78% vs. 67.38% ± 3.95% (coastal). The method autonomously evolved effective strategies such as delaying decision-making to await the optimal timing and cross-region coordination.

**中文：**

涵盖50架无人机和5台HELS的多样化环境大规模场景仿真（每个场景100次独立运行）证明了该方法的优越性，平均毁伤率分别达到99.65%±0.32% vs. 72.64%±3.21%（乡村）、79.37%±2.15% vs. 51.29%±4.87%（沙漠）和91.25%±1.78% vs. 67.38%±3.95%（沿海）。该方法自主演化出了延迟决策以等待最佳时机和跨区域协调等有效策略。

---

## 1. Introduction / 引言

<a id="S001"></a>
**Source:** p.2 S001
**Original:**

Rapid advances in unmanned aerial vehicle (UAV) technology have expanded its applications across diverse civilian domains, including logistics, surveying and mapping, and agriculture [1]. However, the misuse or malfunction of UAVs also poses substantial threats to public safety. This is particularly critical in sensitive areas such as airport clear zone security, the protection of critical infrastructure (e.g., power plants and oil depots), security for large-scale public events, and border surveillance. Unauthorized or malicious UAV operations in these contexts can potentially lead to severe consequences. Consequently, there is a pressing practical need to develop efficient and reliable UAV defense systems [2]. High-Energy Laser Systems (HELSs) are considered a promising technological solution for addressing such threats due to their inherent advantages: light-speed damage capability, high precision, and low cost-per-shot. However, existing HELS-based defensive approaches face formidable challenges when confronted with large-scale, highly dynamic UAV swarm incursions. The core challenge is to intelligently and dynamically assign limited HELS resources (including the number of available systems, and their energy reserves) to a multitude of rapidly moving targets in real time to maximize overall defensive effectiveness.

**中文：**

无人机技术的快速发展使其在物流、测绘和农业等民用领域的应用不断扩展[1]。然而，无人机的误用或故障也对公共安全构成重大威胁，这在机场净空区安全、关键基础设施（如发电厂和油库）保护、大型公共活动安保和边境监视等敏感区域尤为关键。未经授权或恶意的无人机操作可能导致严重后果。因此，迫切需要开发高效可靠的无人机防御系统[2]。高能激光系统因其光速毁伤能力、高精度和低单次发射成本等固有优势，被认为是应对此类威胁的有前途的技术方案。然而，面对大规模、高动态的无人机蜂群入侵时，现有的基于HELS的防御方法面临巨大挑战，核心挑战在于将有限的HELS资源（包括可用系统数量及其能量储备）实时智能、动态地分配给大量快速移动的目标，以最大化整体防御效能。

<a id="S002"></a>
**Source:** p.2 S002
**Original:**

The Dynamic Resource Target Assignment (DRTA) is a complex combinatorial optimization problem, mainly focusing on how to assign targets to various defense units in a planning manner, to optimally achieve the defense intention [3–5]. However, the unique characteristics of HELS defense UAV swarm scenarios introduce significant complexities: (1) Individual HELSs possess inherent limitations in the damage range. Consequently, multi-HELS networks are essential for an effective coverage against a low-altitude UAV swarm... (2) Countering large-scale UAV swarms operating at low altitudes and close ranges, and from multiple directions creates a highly uncertain environment... (3) The near-instantaneous effect capability of HELSs enables multiple decisions within short timeframes... (4) The damage efficacy of HELSs diminishes significantly with distance, making long-range damage resource-intensive and inefficient.

**中文：**

动态资源目标分配是一个复杂的组合优化问题，主要关注如何以规划方式将目标分配给各个防御单元，以最佳地实现防御意图[3-5]。然而，HELS防御无人机蜂群场景的独特特征引入了显著的复杂性：(1)单个HELS的毁伤范围有限，因此多HELS组网对有效覆盖低空无人机蜂群至关重要；(2)对抗大规模低空近距离多方向入侵的无人机蜂群造成高度不确定的环境；(3)HELS的近瞬时效应能力允许在短时间内做出多次决策；(4)HELS的毁伤效能随距离显著降低，使远程毁伤资源消耗大且低效。

<a id="S003"></a>
**Source:** p.3 S003
**Original:**

Traditional dynamic target assignment methods assume instantaneous interception with fixed resource constraints, while high-energy laser systems (HELSs) introduce unique challenges: (1) continuous irradiation requirements (targets must be irradiated until energy thresholds are met), (2) weather-dependent damage efficiency (atmospheric turbulence and thermal blooming degrade laser beam quality), and (3) asynchronous decision-making (irradiation time varies with target distance and material properties). These constraints render conventional models inadequate for real-time decisions.

**中文：**

传统的动态目标分配方法假设瞬时拦截和固定资源约束，而高能激光系统带来了独特的挑战：(1)持续照射要求（目标必须被照射直到达到能量阈值），(2)依赖天气的毁伤效率（大气湍流和热晕效应降低激光束质量），(3)异步决策（照射时间随目标距离和材料属性变化）。这些约束使得传统模型无法胜任实时决策。

<a id="S004"></a>
**Source:** p.3 S004
**Original:**

In recent years, deep reinforcement learning (DRL) has demonstrated strong learning and decision optimization capabilities in issues such as ATARI games and AlphaGo, and has received extensive attention. DRL's "action-selection" naturally parallels the optimization of discrete decision variables in combinatorial optimization. Its characteristics of "offline training and online decision-making" endow it with the potential to solve combinatorial optimization problems online in real time, and it is currently a better solution method for combinatorial optimization [21].

**中文：**

近年来，深度强化学习在ATARI游戏和AlphaGo等任务中展示了强大的学习和决策优化能力，受到广泛关注。DRL的"行动-选择"机制天然对应组合优化中离散决策变量的优化过程。其"离线训练、在线决策"的特点使其具备实时在线求解组合优化问题的潜力，是目前求解组合优化的较好方法[21]。

<a id="S005"></a>
**Source:** p.3 S005
**Original:**

However, due to the different numbers and spatial distributions of unmanned aerial vehicle swarms, existing RL frameworks rely on fixed-length state representations, resulting in information loss or filling noise. The traditional exploration strategy does not work well in the HELS scenario because the reward (target destruction) only appears after prolonged exposure [25,26].

**中文：**

然而，由于无人机蜂群的数量和空间分布差异显著，现有RL框架依赖固定长度的状态表示，导致信息丢失或填充噪声。传统的探索策略在HELS场景中效果不佳，因为奖励（目标摧毁）仅在长时间的照射后才出现[25,26]。

<a id="S006"></a>
**Source:** p.3-4 S006
**Original:**

Based on the above discussion, this paper will propose an HELS–UAV–DRTA model and a solution method based on DRL, providing a scientifically sound decision-making tool for real-world HELS deployment. The main contributions are summarized as follows:
(1) By analyzing the thermal damage mechanism of HELSs and considering the impact of various factors such as spatial situation and weather conditions, we construct an HELS damage-capability model that incorporates atmospheric-transmission and thermal-damage effects...
(2) To tackle the challenges of dynamically varying state dimensions, sparse extrinsic rewards, and limited resources when solving the HELS–UAV–DRTA problem via DRL, we proposed the MADDPG-IA algorithm. An attention-based encoder aggregates variable-length target states into fixed-size representations, while a Random Network Distillation (RND)-based intrinsic reward module provides dense exploration rewards...
(3) Taking the defense problems of small-scale and large-scale UAV swarms as examples... a typical HELS–UAV–DRTA scenario with the background of rural, desert, and coastal regions was established, providing ideas for solving actual problems.

**中文：**

基于以上讨论，本文将提出一个HELS–UAV–DRTA模型和基于DRL的求解方法，为实际HELS部署提供科学合理的决策工具。主要贡献总结如下：(1)通过分析HELS的热毁伤机理，综合考虑空间态势和天气条件等因素，构建了融合大气传输和热毁伤效应的HELS毁伤能力模型；(2)为应对动态变化的状态维度、稀疏的外在奖励和有限资源等挑战，提出了MADDPG-IA算法——基于注意力的编码器将可变长度目标状态聚合为固定大小表示，基于RND的内在奖励模块提供密集的探索奖励；(3)以小规模和大规模无人机蜂群的防御问题为例，建立了乡村、沙漠和沿海等典型HELS–UAV–DRTA场景，为解决实际问题提供了思路。

---

## 2. Modeling the HELS–UAV–DRTA Problem / HELS–UAV–DRTA问题建模

<a id="S007"></a>
**Source:** p.4 S007
**Original:**

This section constructs a mathematical model used to describe its decision-making process. Given that the damage range of HELSs is usually only at the kilometer level and the time for UAV swarms to reach the defense area is extremely short, the damage time has become a key indicator for measuring defense effectiveness. Firstly, starting from the damage mechanism of HELSs, through in-depth research on the effects of laser atmospheric transmission and thermal damage, a quantitative model of the damage time of HELSs was constructed. Secondly, taking a malicious UAV swarm as an example, a target flow description method for the swarm patterns is constructed in order to determine its spatial distribution characteristics. On this basis, considering the target threat and defense benefit factors, the HELS–UAV–DRTA model is constructed to achieve the dynamic and optimal assignment of HELS resources. The framework of the model is shown in Figure 1.

**中文：**

本节构建了描述其决策过程的数学模型。鉴于HELS的毁伤范围通常仅为公里级别，且无人机蜂群到达防御区域的时间极短，毁伤时间已成为衡量防御效能的关键指标。首先，从HELS的毁伤机理出发，通过对激光大气传输效应和热毁伤效应的深入研究，构建了HELS毁伤时间的量化模型。其次，以恶意无人机蜂群为例，构建了蜂群模式的目标流描述方法，以确定其空间分布特征。在此基础上，综合考虑目标威胁和防御收益因素，构建了HELS–UAV–DRTA模型，以实现HELS资源的动态最优分配。模型框架如图1所示。

[Figure 1: Framework of the HELS–UAV–DRTA model — p.5, assets/page05_img*]

### 2.1 Laser Damage Model: Transmission and Thermal Effects / 激光毁伤模型：传输与热效应

<a id="S008"></a>
**Source:** p.5 S008
**Original:**

The most significant constraint on laser transmission in the atmosphere is the Fraunhofer diffraction limit, which determines the size of the laser spot that can be formed at a distance [27]. Ideally, the radius rspot of the far-field laser spot is measured in proportion to the wavelength λ and the aperture D: rspot = 1.22(λ/D)Lβ (Eq.1). Due to the existence of atmospheric turbulence effects, thermal blooming effects, and absorption and scattering effects... the beam quality factor after atmospheric transmission can be expressed as: β² = β²₀ + β²_T + β²_B + β²_J (Eq.2).

**中文：**

激光在大气中传输最关键的约束是夫琅禾费衍射极限，它决定了远距离处激光光斑的大小[27]。大气湍流效应、热晕效应、吸收和散射效应的存在使激光束质量逐步恶化。大气传输后的光束质量因子可表示为：β² = β²₀ + β²_T + β²_B + β²_J（式2），其中β₀、β_T、β_B、β_J分别表征衍射、湍流、热晕和跟踪抖动引起的光束扩展。

[Figure 2: The process of laser atmospheric transmission — p.6, assets/page06_img*]
[Figure 3: Schematic diagram of HELS irradiating UAV — p.6, assets/page06_img*]
[Figure 4: Schematic diagram of the laser thermal damage effect — p.9, assets/page09_img*]

<a id="S009"></a>
**Source:** p.7-8 S009
**Original:**

Key equations: (3) C²_n,HV (slant-path H-V model for atmospheric turbulence); (4) r₀ (atmospheric coherence length); (5) β²_T (beam quality factor from turbulence); (6) N_D (thermal distortion parameter); (7) β_B (beam quality factor from thermal blooming); (8) β_J = 6.93(σᵢ/σ_d)² (beam quality factor from tracking jitter). Finally, Equation (9) yields the average laser spot radius r_spot. Equation (10) gives atmospheric transmittance τ(L,κ,ν) using the Beer-Lambert law. Equation (11): P_e(λ,θ,L) = η₀·τ(L,κ,ν)·P₀.

**中文：**

关键方程：(3) 斜程H-V修正模型的C²_n（折射率结构常数）；(4) r₀大气相干长度；(5) β²_T湍流光束质量因子；(6) N_D热畸变参数（衡量热晕强度）；(7) β_B热晕光束质量因子；(8) β_J = 6.93(σᵢ/σ_d)² 跟踪抖动光束质量因子。最终得到式(9)激光光斑平均半径r_spot，式(10)斜程大气透过率τ，式(11)衰减后激光功率P_e。

<a id="S010"></a>
**Source:** p.9-11 S010
**Original:**

Section 2.1.2 Laser Thermal Damage Model. When a laser irradiates a target, the energy absorbed by the target causes the thermal damage effect [14]. The temperature field T is characterized by the heat conduction equation: ρc·∂T/∂t = ∇(k∇T) + (1−R)·I_target·e^{−αz} (Eq.13). The analytical formula for the temperature field under continuous-wave laser irradiation is given by Eq.14. The time t_m of melting penetration is expressed as: t_m = z_d·ρ[c_s(T_m−T₀)+L_m]/[(1−R)·I_target(x,y)] (Eq.17). The damage threshold e_th is defined as: e_th = min E_m = z_m·ρ[C_s(T_m−T₀)+L_m] (Eq.18). Finally, the damage time of the HELS can be defined as Eq.20.

**中文：**

2.1.2 激光热毁伤模型。激光照射目标时，被目标吸收的能量引发热毁伤效应。温度场T由热传导方程表征（式13）。在连续波激光照射下，温度场的解析解由式14给出。熔化穿透时间t_m由式17给出。毁伤阈值e_th定义为穿透单位厚度目标材料所需的最小能量密度（式18）。最终，HELS的毁伤时间可由式20确定，其主要取决于激光在大气中传输的特性以及热毁伤效应。具体来说，毁伤时间与激光功率P₀近似线性递减，与光束质量因子β近似二次方关系，与距离L近似平方关系。

[Figure 5: Laser damage time vs. distance under different atmospheric conditions — p.11, assets/page11_img*]
[Figure 6: Laser damage time vs. various materials — p.12, assets/page12_img*]

<a id="S011"></a>
**Source:** p.12 S011
**Original:**

The irradiation transfer time t_trans refers to the time required for an HELS to irradiate and destroy a target, and then adjust the direction to the next target. In HELS with FSM, t_trans can reach millisecond response (<50 ms), but the maximum slew angle is limited to ±10°~±30°. When exceeding this angle, mechanical turntable rotation is required. The total time for one attack period: t_period,i = t_trans,i + t_damage,i (Eq.22).

**中文：**

照射转移时间t_trans指HELS完成一个目标毁伤后调整方向到下一个目标所需的时间。采用快速转向镜(FSM)的HELS可达毫秒级响应（<50ms），但最大转动角受限（±10°~±30°）；超出该范围需要机械转台物理旋转。一个攻击周期的总时间：t_period,i = t_trans,i + t_damage,i（式22）。

### 2.2 Malicious UAV Swarm Density Model / 恶意无人机蜂群密度模型

<a id="S012"></a>
**Source:** p.13 S012
**Original:**

Two cluster modes: (1) multi-direction and multi-batch intrusion — time intervals follow a lognormal distribution (Eq.23); (2) large-scale simultaneous intrusion — time intervals follow a uniform distribution (Eq.24). [Figure 7]

**中文：**

两种集群模式：(1)多方向多批次入侵——发现时间间隔服从对数正态分布（式23）；(2)大规模同时入侵——发现时间间隔服从均匀分布（式24）。[图7]

### 2.3 HELS–UAV–DRTA Model Formulation / HELS–UAV–DRTA模型建立

<a id="S013"></a>
**Source:** p.14-15 S013
**Original:**

Five factors are established: (1) height threat factor T_rh (Eq.25, exponential decay), (2) velocity threat factor T_rv (Eq.26, exponential), (3) safe distance threat factor T_rL (Eq.27, linear), (4) resource consumption benefit factor B_rc (Eq.28, sigmoid), (5) HELS application value benefit factor B_rs (Eq.29, linear weighted).

**中文：**

建立了五个因素模型：(1)高度威胁因子T_rh（式25，指数衰减型）——越低越具威胁；(2)速度威胁因子T_rv（式26，指数型）——越接近预设速度v₀越具威胁；(3)安全距离威胁因子T_rL（式27，线性）——剩余飞行距离越短威胁越大；(4)资源消耗收益因子B_rc（式28，sigmoid函数）——优先选择毁伤时间短的目标；(5)HELS应用价值收益因子B_rs（式29，线性加权）——优先使用低功率、大剩余电量的HELS。

<a id="S014"></a>
**Source:** p.15-17 S014
**Original:**

The HELS–UAV–DRTA model based on the optimal energy criterion (Eq.30-31). The objective function maximizes the weighted sum of threat factors and benefit factors. [λ₁, λ₂] are the weights of threat and benefit. Seven constraints (a–h) govern: target set evolution, battery consumption, energy feasibility, time step update, mutual exclusion, irradiation exclusivity, and binary decision variables. The worst-case computational complexity is O(m·n·T).

**中文：**

基于最优能量准则的HELS–UAV–DRTA模型（式30-31）：目标函数最大化威胁因子和防御收益因子的加权和。[λ₁,λ₂]∈[0,1]分别为威胁和收益权重，λ₁+λ₂=1。七个约束(a–h)控制：目标集演进、电量消耗、能量可行性、时间步长更新、互斥性、照射独占性和二元决策变量。最坏情况计算复杂度为O(m·n·T)。模型适用边界：(1)环境边界——中等湍流以下、轻霾以上能见度；(2)目标边界——小型商用无人机、距离≤10km；(3)系统边界——HELS电池弹匣须能支持累计照射时间。

---

## 3. MADDPG-IA Algorithm Design and Implementation / MADDPG-IA算法设计与实现

<a id="S015"></a>
**Source:** p.17 S015
**Original:**

When solving the HELS–UAV–DRTA problem with DRL, it can be regarded as a multi-step RL process in the continuous state space and the discrete action space. The task of learning is to find an optimal target assignment strategy to maximize the global benefit. Treating each HELS as an agent, through centralized training, all HELS agents can access global information, which is conducive to learning effective coordination strategies. When implementing decisions, each HELS agent makes decisions only based on local information, thereby reducing the computational burden and improving the real-time decision-making ability. An improved MADDPG algorithm based on the state coding of the attention mechanism and the sparse reward exploration strategy driven by an RND-based intrinsic reward module is proposed.

**中文：**

用DRL求解HELS–UAV–DRTA问题时，可视为连续状态空间和离散动作空间中的多步强化学习过程。学习任务是找到最优的目标分配策略以最大化全局收益。将每个HELS视为一个智能体，采用集中训练分散执行（CTDE）范式——训练时共享全局信息学习协作策略，决策时仅基于局部信息，降低计算负担并提高实时性。本文提出基于注意力机制状态编码和RND内在奖励模块驱动的稀疏奖励探索策略来改进MADDPG算法。

### 3.1 MADDPG Framework in HELS–UAV–DRTA / MADDPG框架

<a id="S016"></a>
**Source:** p.18-20 S016
**Original:**

MADDPG extends DDPG to multi-agent domain with Actor-Critic architecture. Each agent has two networks: Actor (θ^π) for action selection and Critic (θ^Q) for value evaluation. The policy gradient: ∇_{θ^π_i} J(π_i) = E_{s,a∼D}[∇ log π_i(a_i|s_i) Q^π_i(s,a₁,...,a_n)] (Eq.34). The Critic is updated by minimizing the TD error (Eq.36). Each agent updates target networks via soft update (Eq.37).

**中文：**

MADDPG将DDPG扩展到多智能体领域，采用Actor-Critic架构。每个智能体包含两个网络：Actor网络（θ^π）选择动作，Critic网络（θ^Q）评估动作价值。策略梯度由式34给出。Critic通过最小化TD误差（式36）实现价值评估。各智能体通过软更新（式37）更新目标网络参数。

**State space design (Eq.38-40):** Each agent observes its own state s^{LaSW}_i (position, remaining battery, remaining duration, rotation axis direction) ∈ R⁴ and target state s^{UAV}_i (number of UAVs, distance, height, velocity, atmospheric environment) ∈ R^{4m_i+1}. The observation state is s_i = concat(s^{LaSW}_i, s^{UAV}_i) ∈ R^{4m_i+5}.

**中文（状态空间）：**每个智能体观测自身状态s^{LaSW}_i（部署位置、剩余电量、连续照射剩余时间、激光转轴方向）和观测到的目标状态s^{UAV}_i（无人机数量、距离、高度、速度、大气环境参数）。

**Discrete action space (Eq.41):** One-Hot encoded: m+1 actions (irradiate each target or wait). Gumbel-Softmax enables gradient backpropagation for discrete actions.

**中文（动作空间）：**One-Hot编码：m+1维离散动作（m个目标各对应一个照射动作 + 1个等待动作）。采用Gumbel-Softmax方法实现离散动作的梯度反向传播。

<a id="S017"></a>
**Source:** p.20 S017
**Original:**

**Reward function (Eq.42):** Three cases — (1) if irradiating and conditions met: reward based on threat + benefit factors; (2) if irradiating but conditions not met: −1 penalty; (3) if waiting: 0.1·exp(−β_d·t) decaying positive reward to encourage delayed-damage strategies.

**中文（奖励函数）：**奖励函数（式42）分三种情况：(1)满足照射条件且执行照射→基于威胁+收益因子的正奖励；(2)不满足照射条件却执行照射→−1惩罚；(3)选择等待→0.1·exp(−β_d·t) 随时间衰减的正奖励，鼓励早期延迟决策。

### 3.2 Enhanced MADDPG with Attention and Intrinsic Reward / 注意力机制与内在奖励增强

<a id="S018"></a>
**Source:** p.21-22 S018
**Original:**

3.2.1 Attention-Based State Encoding. The attention mechanism converts variable-length UAV state information into fixed-dimension feature vectors. The agent's own state s^{LaWS}_i serves as Query, each UAV's state s^{UAV}_i serves as Key and Value. Q_i = W_q·s^{LaWS}_i, K_j = W_k·s^{UAV}_i, V_j = W_v·s^{UAV}_i (Eq.43). Attention weights: a_ij = softmax(Q_i·K_j^T/√d_k) (Eq.44). The output state encoding: ẽ_i = Attn(W_q·s^{LaWS}_i, {W_k·s^{UAV}_i}, {W_v·s^{UAV}_i}) ∈ R^{d_attn} (Eq.45). [Figure 8]

**中文：**

3.2.1 基于注意力的状态编码。注意力机制将可变长度的无人机状态信息转换为固定维度的特征向量。以智能体自身状态s^{LaWS}_i为Query，以各无人机状态为Key和Value（式43）。注意力权重a_ij通过点积相似度和Softmax计算（式44）——更近距离的UAV将获得显著更高的注意力权重。最终输出固定维度的编码特征向量（式45）。[图8]

[Figure 8: State coding network based on attention mechanism — p.21, assets/page21_img*]

<a id="S019"></a>
**Source:** p.22-23 S019
**Original:**

3.2.2 Intrinsic Reward-Driven Exploration. The RND-based intrinsic reward module contains two networks: a fixed randomly-initialized target network φ and a trainable prediction network φ̃. The intrinsic reward r_c = β_r||φ̃(s̃^{t+1}_i) − φ(s̃^{t+1}_i)|| (Eq.46). The hybrid reward: r_h = γ·r_e + (1−γ)·r_c (Eq.47), where γ is dynamically adjusted based on environment reward sparsity. [Figure 9]

**中文：**

3.2.2 内在奖励驱动的探索。基于RND的内在奖励模块包含两个网络：固定随机初始化的目标网络φ和可训练的预测网络φ̃。内在奖励r_c = β_r||φ̃ − φ||（式46），衡量状态的认知不确定性。混合奖励r_h = γ·r_e + (1−γ)·r_c（式47），γ根据环境奖励稀疏度自动调整。内在奖励鼓励智能体探索未见过或少访问的状态，促进了"延迟决策等待最佳时机"这一关键策略的学习。[图9]

[Figure 9: Architecture of the RND-based intrinsic reward module — p.23, assets/page23_img*]

### 3.3 MADDPG-IA Algorithm Workflow / MADDPG-IA算法流程

<a id="S020"></a>
**Source:** p.23-24 S020
**Original:**

Algorithm 1 (Pseudocode): The MADDPG-IA algorithm uses experience replay, dual network architecture, and CTDE. All agents share networks and experience pool. The algorithm loops over episodes, at each step: attention-encode observed states, select actions via Gumbel-Softmax, execute joint action, calculate hybrid reward, store experience, then update Critic, Actor, target networks, and RND prediction network. [Figure 10, Algorithm 1]

**中文：**

算法1（伪代码）：MADDPG-IA采用经验回放、双网络架构和CTDE思想。所有智能体共享网络和经验池。每轮迭代：注意力编码观测状态→Gumbel-Softmax选择动作→执行联合动作→计算混合奖励→存储经验→更新Critic/Actor/目标网络/RND预测网络。[图10，算法1]

[Figure 10: Architecture of MADDPG-IA — p.24, assets/page24_img*]

---

## 4. Simulation and Analysis / 仿真与分析

### 4.1 Experimental Environment and Parameter Settings / 实验环境与参数设置

<a id="S021"></a>
**Source:** p.25-26 S021
**Original:**

Platform: PyCharm 2024.3.1.1, Anaconda3 (2024.10), Python 3.12.9, torch 2.6.0+cu126. Hardware: AMD Ryzen 7 5800H, 32 GB RAM, NVIDIA RTX 3060 Laptop. Two parts: (1) typical scenario analysis with rational decision-making; (2) performance comparison via ablation and algorithm comparison experiments.

**中文：**

实验平台：PyCharm+Anaconda3+Python 3.12.9+torch 2.6.0，配备AMD Ryzen 7 5800H CPU、32GB RAM、NVIDIA RTX 3060 Laptop GPU。实验分两部分：(1)典型场景分析与合理决策验证；(2)消融实验与算法对比实验评估性能。

[Table 1: Simulation parameter setting — p.25-26]

**Key parameters:** HELS: 2 (small-scale)/5 (large-scale), P₀ = 20~50 kW, λ = 1.064 µm, D = 0.6 m. UAV: 10/50, multi-direction multi-batch intrusion, flight speed 0.02~0.03 km/s, height 0.2~0.8 km, 2024 Al alloy (5 mm). Atmosphere: rural/desert/coastal, C²_n(0) = 1×10⁻¹⁷ (weak) or 1×10⁻¹⁵ (medium), ν = 5~10 km. MADDPG-IA: max 2000 episodes, buffer size 1×10⁶, batch 2048, lr = 1×10⁻³, γ = 0.95. 100 independent runs per scenario with randomized initial conditions.

**中文（关键参数）：**HELS 2/5台，功率20~50kW，波长1.064µm，孔径0.6m。UAV 10/50架，多方向多批次入侵，速度0.02~0.03 km/s，高度0.2~0.8 km，2024铝合金(5mm)。大气：乡村/沙漠/沿海，C²_n(0) 1×10⁻¹⁷或1×10⁻¹⁵，能见度5~10 km。MADDPG-IA：2000轮训练，经验池1×10⁶，批量2048，学习率1×10⁻³，折扣因子0.95。每场景100次独立运行，随机初始化条件。

### 4.2 Typical Scenario Experiments / 典型场景实验

<a id="S022"></a>
**Source:** p.26-27 S022
**Original:**

Large-scale scenario (rural, sunshine, weak turbulence, 5 HELSs vs 50 UAVs). [Figure 11: front & top view of spatial situation after decision-making; Figure 12: irradiation timing of each HELS agent]. Key observations: In early stages, HELS agents 2,4,5 started irradiation first (at 7.32km, 7.27km, 7.25km). HELS agents 1,3 evolved "delaying decision-making to await optimal timing" strategy. In later stages, agents concentrated on the 0°~15° direction targets. The last target was 1.68 km from the protected asset. Battery magazines remained at 64.39s, 0s, 36.78s, 7.88s, 9.55s. Cross-region coordination observed.

**中文：**

大规模场景（乡村/晴空/弱湍流/5 HELS vs 50 UAV）。[图11：决策后空间态势正视图和俯视图；图12：各HELS智能体照射时序]。关键观察：早期阶段，HELS 2、4、5率先在7.32km/7.27km/7.25km处开始照射；HELS 1和3因其方向目标较少，自主演化出"延迟决策等待最佳时机"策略。中后期集中处理0°~15°方向目标。最终目标距离保护资产1.68km时被毁伤。剩余电量：64.39s/0s/36.78s/7.88s/9.55s。观察到跨区域协调策略。

[Figure 11: Spatial situation after decision-making — p.26, assets/page26_img*]
[Figure 12: Irradiation timing of each HELS agent — p.27, assets/page27_img*]

<a id="S023"></a>
**Source:** p.27-29 S023
**Original:**

Model Generalization Experiment — 6 scenarios (3 environments × 2 scales). [Figure 13 a–f] Key findings: (1) Swarm scale matters — small-scale allows more "delayed" decisions; first irradiation distances: 3.79 km (small) vs 7.18 km (large) in desert. (2) Atmosphere affects decisions — small-scale first irradiation: 3.79 km (desert), 6.69 km (rural), 4.21 km (coastal). In coastal environments, agents prioritize sea-surface targets for higher efficiency. [Table 2] Large-scale damage rates: rural 99.65%±0.32% (MADDPG-IA) vs 72.64%±3.21% (Traditional); desert 79.37%±2.15% vs 51.29%±4.87%; coastal 91.25%±1.78% vs 67.38%±3.95%.

**中文：**

模型泛化实验——6个场景（3种环境×2种规模）。[图13 a–f] 关键发现：(1)蜂群规模影响决策——小规模场景因目标少允许更多"延迟"策略；沙漠环境中小规模首次照射距离3.79 km，大规模则为7.18 km。(2)大气环境影响决策——小规模首次照射距离：沙漠3.79km/乡村6.69km/沿海4.21km。沿海环境中智能体优先照射海面目标以提高效率。[表2]大规模场景毁伤率：乡村99.65%±0.32%(MADDPG-IA) vs 72.64%±3.21%(传统)；沙漠79.37%±2.15% vs 51.29%±4.87%；沿海91.25%±1.78% vs 67.38%±3.95%。

[Figure 13: Solving HELS–UAV–DRTA in different scenarios — p.28, assets/page28_img*]
[Table 2: Full statistical analysis of damage rate in large-scale scenarios — p.29]

<a id="S024"></a>
**Source:** p.29-30 S024
**Original:**

Parameter-variation experiment [Table 3]: With 5 HELSs, damage rates saturate at 99.6% (5 HELS) and 99.7% (6 HELS of 40kW). 5 HELSs can effectively handle ≤70 UAVs (89.3%). At turbulence C²_n ≥ 5×10⁻¹⁴, damage rate drops to 58.7%. Practical significance: (1) High damage rates → massive decrease in UAV breach probability. (2) Longer-range interception + delayed strategy → shorter irradiation per kill → boosts battery depth and turnover rate. (3) Superior energy efficiency → fewer HELS needed. (4) Autonomous strategy adaptation without pre-defined rules.

**中文：**

参数变化实验[表3]：5台HELS毁伤率99.6%——再增加HELS收益递减；40kW功率已满足需求。5台HELS能有效处理不超过70架UAV（毁伤率89.3%）。湍流C²_n ≥ 5×10⁻¹⁴时毁伤率降至58.7%。实际意义：(1)高毁伤率→大幅降低UAV突防概率；(2)更远拦截+延迟策略→缩短单次杀伤照射时间→提升电池弹匣深度和周转率；(3)优异的能源效率→减少所需HELS数量；(4)无需预定义复杂规则即可自主适应决策。

### 4.3 Algorithm Performance Verification / 算法性能验证

<a id="S025"></a>
**Source:** p.30-31 S025
**Original:**

Training (2000 episodes, 100 runs): [Figure 14] After ~500 episodes, the algorithm stabilizes and converges. Wall-clock times: 4.2±0.3 h (small-scale), 11.8±0.9 h (large-scale) on consumer hardware. HELS agent 4 (high-performance, early-deployed) has highest average return due to supplementing damage tasks in all directions.

**中文：**

训练过程（2000轮，100次训练）：[图14]约500轮后算法趋于稳定收敛。训练时间：小规模4.2±0.3 h，大规模11.8±0.9 h（消费级硬件即可完成）。HELS智能体4（高性能、前置部署）因在各方向补充毁伤任务而获得最高平均回报。

[Figure 14: Rewards for each HELS agent and sum rewards — p.31, assets/page31_img*]

<a id="S026"></a>
**Source:** p.31-32 S026
**Original:**

Ablation experiment [Figure 15]: MADDPG-IA vs MADDPG-RND vs MADDPG-Attn vs MADDPG-Basic. MADDPG-Attn focuses on high-threat targets but has ~1× lower exploration efficiency than MADDPG-IA. MADDPG-RND converges faster than Basic but cannot handle variable-length input — global efficiency ~21.8% lower than MADDPG-IA. MADDPG-IA produces a super-additive effect: global revenue +39.6% vs traditional MADDPG.

**中文：**

消融实验[图15]：MADDPG-IA vs MADDPG-RND vs MADDPG-Attn vs MADDPG-Basic。MADDPG-Attn能始终关注高威胁目标但探索效率约低1倍；MADDPG-RND收敛更快但无法处理变长输入——全局效率低约21.8%。MADDPG-IA联合两个模块产生超加和效应，相比传统MADDPG全局收益平均提升39.6%。

[Figure 15: Results of ablation experiments — p.32, assets/page32_img*]

<a id="S027"></a>
**Source:** p.32-33 S027
**Original:**

Algorithm comparison [Figure 16]: DQN, QMIX, MAPPO vs MADDPG-IA. Small-scale: MADDPG-IA global average revenue is 48.2%, 20.1%, 14.7% higher than DQN, QMIX, MAPPO. Large-scale: DQN (single agent, no cooperation) and QMIX (fixed-scene, "0" padding noise) fail to converge. MAPPO has robust policy-gradient architecture but suffers from sparse rewards. MADDPG-IA converges rapidly with strong exploration. Execution latencies (5 HELS-50 UAVs): DQN 5.2±0.4ms, QMIX 8.7±0.6ms, MAPPO 12.1±1.1ms, MADDPG-IA 15.3±1.3ms — still suitable for millisecond-level real-time decisions.

**中文：**

算法对比[图16]：DQN将所有HELS视为同一智能体，无协作，大规模难收敛；QMIX在变目标数量场景中"0"填充法引入严重噪声；MAPPO策略梯度架构相对稳健但稀疏奖励和动态输入限制其收敛效率。MADDPG-IA在小规模场景中全局平均收益分别高出DQN/QMIX/MAPPO 48.2%/20.1%/14.7%。推理延迟（5 HELS-50 UAVs）：DQN 5.2ms / QMIX 8.7ms / MAPPO 12.1ms / MADDPG-IA 15.3ms——仍在毫秒级实时决策要求范围内。

[Figure 16: Algorithm comparison results — p.33, assets/page33_img*]

---

## 5. Conclusions and Outlook / 结论与展望

<a id="S028"></a>
**Source:** p.33-34 S028
**Original:**

(1) Based on the effects of laser atmospheric transmission and thermal damage, a quantitative characterization of the damage capability of HELSs with the damage time as the core index was constructed. Considering the spatio-temporal characteristics of malicious UAV swarms comprehensively, an HELS–UAV–DRTA model with the threat factor of UAVs and the damage benefit factor of HELSs as the objective functions was proposed. In the typical scene experiments, the model can dynamically optimize HELS resource allocation according to weather conditions and real-time information, evolving strategies such as delaying decision-making to await optimal timing and cross-region coordination.

**中文：**

(1)基于激光大气传输和热毁伤效应，构建了以毁伤时间为核心指标的HELS毁伤能力量化表征。综合考虑恶意无人机蜂群的时空特性，提出了以UAV威胁因子和HELS毁伤收益因子为目标函数的HELS–UAV–DRTA模型。实验证明模型能根据天气条件和实时信息动态优化HELS资源配置，自主演化出延迟决策等待最佳时机和跨区域协调等策略。

(2) An MADDPG-IA algorithm for the HELS–UAV–DRTA problem is proposed. The problem of dynamic changes in the state dimension is solved by designing the state coding network based on the attention mechanism, and the exploration predicament under sparse rewards is cracked by using an RND-based intrinsic reward module. Large-scale damage rates: 99.65%±0.32% (rural), 79.37%±2.15% (desert), 91.25%±1.78% (coastal). Global average returns exceed DQN, QMIX, and MAPPO by 48.2%, 20.1%, and 14.7%.

**中文：**

(2)提出了MADDPG-IA算法——注意力机制状态编码网络解决状态维度动态变化，RND内在奖励模块破解稀疏奖励下的探索困境。大规模毁伤率：乡村99.65%±0.32%/沙漠79.37%±2.15%/沿海91.25%±1.78%。全局平均收益超越DQN/QMIX/MAPPO 48.2%/20.1%/14.7%。

(3) In future research, we will further study the dynamic combinatorial optimization problem, the soft/hard damage mode and probability model of laser destruction, the intelligent path planning of UAV swarms, and the game confrontation between the two sides.

**中文：**

(3)未来研究将进一步探讨动态组合优化问题、软/硬毁伤模式和激光毁伤概率模型、无人机蜂群智能路径规划以及双方博弈对抗。

---

## References / 参考文献

**Source:** p.34-36

1. Javed, S. et al. State-of-the-Art and Future Research Challenges in UAV Swarms. *IEEE Internet Things J.* 2024, 11, 19023–19045.
2. Extance, A. Military Technology: Laser Weapons Get Real. *Nature* 2015, 521, 408–411.
3. Manne, A. A Target-Assignment Problem. *Oper. Res.* 1958.
   4-5. Andersen et al. (2022); Peng et al. (2024). Weapon-Target Assignment & Multi-Ship Dynamic WTA.
4. Yang, R.; Li, C. PSO in anti-UAV fire allocation of laser weapon. *Command. Inf. Syst. Technol.* 2021.
   7-12. Shi et al. (2023); Hemani & Georges (2017); Karr & Trebes (2024); Li et al. (2023); Chang et al. (2018).
5. Hanák, J. et al. Cross-Entropy Method for Laser Defense Applications. *J. Aerosp. Inf. Syst.* 2025.
6. Taylor, A.B. Counter-UAV Study: Shipboard Laser Weapon System. PhD Thesis, Naval Postgraduate School, 2021.
   15-20. Gong et al. (2023/2024); Xu et al. (2020); Guo et al. (2019); Davis et al. (2017); Wang et al. (2024); Xin et al. (2019).
7. Chen, L. et al. Human-Machine Agent Based on Active RL for Target Classification. *IEEE TNNLS* 2024.
8. Liu, J. et al. Task Assignment in Ground-to-Air Confrontation Based on Multiagent DRL. *Def. Technol.* 2023.
9. Huang, T. et al. Task assignment method of compound anti-drone based on DQN. *Control Decis.* 2022.
10. Hu, T. et al. Dynamic Target Assignment by USVs Based on RL. *Mathematics* 2024.
    25-26. Hua et al. (2022); Shojaeifard et al. (2019).
11. Bahman, Z. *Directed Energy Weapons Physics of High Energy Lasers*. Springer, 2016.
    28-29. Sun et al. (2022); Qiao et al. (2010). Scaling laws of high energy laser propagation.
12. Jabczyński, J.K.; Gontar, P. Impact of Atmospheric Turbulence on Coherent Beam Combining. *Def. Technol.* 2021.
    31-37. Additional atmospheric, thermal, and materials references.
13. Liu, W. et al. Damage Capability of Laser System in Ground-Air Defense Environments. *Chin. J. Aeronaut.* 2025.
    39-45. Heat conduction, vulnerability assessment, laser weapons survey, UAV cluster interception, WTA problem survey.
    46-48. Li et al. (2024); Chen & Nie (2023); Cai et al. (2024). MADDPG-based multi-agent algorithms.
14. Tilbury, C.R. et al. Revisiting the Gumbel-Softmax in MADDPG. *arXiv* 2023.
    50-51. Fu et al. (2025); Hu, K. et al. Attention Mechanisms in MARL. *Neurocomputing* 2024.
    52-54. Hu, M. et al. (2024); Chen, W. et al. (2021); Song, C. et al. (2025). RND and MADDPG-related.
15. Li, Q. Numerical simulation of laser thermal ablation effect. Master's Thesis, Xidian University, 2019.
16. Guo, W. et al. Enhancing the Robustness of QMIX. *Neurocomputing* 2024.
17. Liu, X. et al. Multi-UCAV Cooperative Decision-Making Based on MAPPO. *Aerospace* 2022.

---

## 阅读提示 / Critical Reading Notes

1. **核心创新点:** MADDPG-IA 的核心创新在于将注意力机制（解决变长状态输入）和 RND 内在奖励（解决稀疏奖励）集成到 MADDPG 框架中，产生了"超加和效应"（super-additive effect）——联合使用的效果超出单模块之和。

2. **物理模型深度:** 该论文最突出的特点是其物理模型的严谨性。第 2 章建立了从激光大气传输（湍流/热晕/抖动）到热毁伤（熔化穿透）的完整定量模型，这在已有 MARL 应用于武器-目标分配的研究中是罕见的。

3. **奖励设计的巧妙之处:** 奖励函数（式 42）中的退火等待奖励 0.1·exp(−β_d·t) 是关键设计——它鼓励"延迟决策等待最佳时机"而非立即开火，这是 HELS 场景中最具价值的策略。

4. **实验严谨性:** 每场景 100 次独立运行 + bootstrap 置信区间 + 三种大气环境 + 两种规模 + 参数变化实验 + 消融 + 对比实验，实验设计相当全面。

5. **可复现性:** 论文使用了消费级硬件（RTX 3060 Laptop），训练时间 4.2-11.8 小时，具备良好的可复现性。数据可用性声明仅为"数据包含在文章中"，未提供代码仓库链接。

6. **局限性:** (1) UAV 路径假设为固定轨迹（不考虑智能规避）；(2) HELS 材料假设为简单金属（对复合材料分析不足）；(3) 未涉及博弈对抗层面（UAV 蜂群主动规避策略）；(4) 软毁伤（仅致盲传感器）模式未建模。

7. **应用价值:** 该方法对机场安保、重要设施防护、大型活动安保等民用场景具有直接转化价值。5 台 HELS 即可有效应对 50 架无人机蜂群的结论具有实际部署参考意义。

---

*Reader built by nature-reader v2.0.0 | Generated 2025 | Source: MADDPG-IA_paper.pdf | Aerospace 2025, 12, 729 (CC BY 4.0)*