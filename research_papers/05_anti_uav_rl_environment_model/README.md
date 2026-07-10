# 反 UAV 防空编组强化学习环境模型论文包

创建日期：2026-07-09

来源文档：`research_papers/反UAV防空编组强化学习环境模型参考文献检索.md`

## 已下载 PDF

1. `P01_2025_RL_Decision_Level_Interception_Prioritization_Drone_Swarm_Defense.pdf`
   - 来源：https://arxiv.org/abs/2508.00641
2. `P02_2026_Delay_Aware_Active_Triangulation_Uncertainty_Driven_MARL_Counter_UAS.pdf`
   - 来源：https://arxiv.org/abs/2607.05957
3. `P03_2021_Decentralized_Multi_UAV_Spatio_Temporal_Multi_Task_Allocation_Perimeter_Defense.pdf`
   - 来源：https://arxiv.org/abs/2102.07381
4. `P04_2025_MAGNNET_GNN_Task_Allocation_Autonomous_Vehicles_DRL.pdf`
   - 来源：https://arxiv.org/abs/2502.02311
5. `P05_2020_Counter_UAS_State_of_the_Art_Challenges_Future_Trends.pdf`
   - 来源：https://arxiv.org/abs/2008.12461
6. `P06_2022_Aerial_Threats_Radar_Communications_Survey.pdf`
   - 来源：https://arxiv.org/abs/2211.10038
7. `P07_2022_Autonomous_Drone_System_Jamming_Relative_Positioning.pdf`
   - 来源：https://arxiv.org/abs/2206.04307
8. `P08_2023_Multi_Target_Pursuit_Heterogeneous_UAV_Swarm_DMARL.pdf`
   - 来源：https://arxiv.org/abs/2303.01799
9. `P09_2017_Deep_Decentralized_Multitask_MARL_Partial_Observability.pdf`
   - 来源：https://arxiv.org/abs/1703.06182
10. `P10_2025_Vision_Based_Active_Tracking_Flying_Target_UAV.pdf`
    - 来源：https://arxiv.org/abs/2506.18264
11. `P11_2018_Partial_Replanning_Decentralized_Dynamic_Task_Allocation.pdf`
    - 来源：https://arxiv.org/abs/1806.04836
12. `P12_2017_MADDPG_Mixed_Cooperative_Competitive_Environments.pdf`
    - 来源：https://arxiv.org/abs/1706.02275
13. `P13_2021_MAPPO_Surprising_Effectiveness_PPO_Cooperative_Multi_Agent_Games.pdf`
    - 来源：https://arxiv.org/abs/2103.01955
14. `P14_2020_PettingZoo_Gym_for_Multi_Agent_RL.pdf`
    - 来源：https://arxiv.org/abs/2009.14471
15. `P15_2017_MAgent_Many_Agent_RL_Platform.pdf`
    - 来源：https://arxiv.org/abs/1712.00600

## 未下载的 DOI 文献

以下两篇是任务分配经典文献，检索文档中保留为 DOI 链接。已尝试访问出版社 PDF，但未获得开放 PDF：

1. Brian P. Gerkey and Maja J. Mataric, 2004, A Formal Analysis and Taxonomy of Task Allocation in Multi-Robot Systems.
   - DOI：https://doi.org/10.1177/0278364904045564
   - 状态：SAGE 页面显示 restricted access，直接 PDF 请求失败。
2. Han-Lim Choi, Luc Brunet, Jonathan P. How, 2009, Consensus-Based Decentralized Auctions for Robust Task Allocation.
   - DOI：https://doi.org/10.1109/TRO.2009.2022423
   - 状态：IEEE 页面未返回真实 PDF 文件，可能需要机构权限。

## 建议精读顺序

1. `P05`：先建立 C-UAS 功能链条和资源类型。
2. `P01`：直接看反 UAV 拦截优先级的状态、动作和奖励设计。
3. `P03`：学习 perimeter defense 中的时空任务建模。
4. `P04`：学习异构资源-任务图和 GNN/PPO 分配结构。
5. `P02`：补充感知延迟、AoI、不确定性和 Dec-POMDP 建模。
6. `P13`、`P14`：衔接 MAPPO baseline 与多智能体环境 API。
