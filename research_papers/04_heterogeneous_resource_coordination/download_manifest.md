# 异构防御资源协同编组补充文献下载清单

下载日期：2026-07-01

## 已成功下载的开放 PDF

| 优先级   | 文件名                                                                        | 来源                                                                  |
| ----- | -------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| P0-1  | `P0_01_2021_HATRPO_HAPPO_Trust_Region_MARL.pdf`                            | https://arxiv.org/pdf/2109.11251                                    |
| P0-2  | `P0_02_2023_HARL_Heterogeneous_Agent_RL.pdf`                               | https://arxiv.org/pdf/2304.09870                                    |
| P0-3  | `P0_03_2025_MAGNNET_GNN_Heterogeneous_Task_Allocation.pdf`                 | https://arxiv.org/pdf/2502.02311                                    |
| P0-4  | `P0_04_2021_Decentralized_Multi_UAV_Perimeter_Defense_Task_Allocation.pdf` | https://arxiv.org/pdf/2102.07381                                    |
| P0-5  | `P0_05_2020_CUAS_State_of_the_Art_Challenges_Future_Trends.pdf`            | https://arxiv.org/pdf/2008.12461                                    |
| P0-6  | `P0_06_2018_CBBA_PR_Partial_Replanning_Dynamic_Task_Allocation.pdf`        | https://arxiv.org/pdf/1806.04836                                    |
| P1-9  | `P1_09_2017_Deep_Decentralized_Multitask_MARL_POMDP.pdf`                   | https://proceedings.mlr.press/v70/omidshafiei17a/omidshafiei17a.pdf |
| P1-10 | `P1_10_2023_Heterogeneous_UAV_Swarm_Multi_Target_Pursuit_MARL.pdf`         | https://arxiv.org/pdf/2303.01799                                    |
| P2-11 | `P2_11_2022_Aerial_Threats_Radar_Communications_Survey.pdf`                | https://arxiv.org/pdf/2211.10038                                    |
| P2-12 | `P2_12_2022_Autonomous_Drone_System_Jamming_Relative_Positioning.pdf`      | https://arxiv.org/pdf/2206.04307                                    |

## 经典文献：官方 PDF 受限

以下两篇为经典任务分配文献。已尝试从官方 DOI/出版社入口下载 PDF，但官方入口返回访问限制，因此未保存非授权副本。

| 优先级  | 文献                                                                       | DOI / 官方入口                               | 状态                |
| ---- | ------------------------------------------------------------------------ | ---------------------------------------- | ----------------- |
| P1-7 | A Formal Analysis and Taxonomy of Task Allocation in Multi-Robot Systems | https://doi.org/10.1177/0278364904045564 | Sage PDF 入口返回 403 |
| P1-8 | Consensus-Based Decentralized Auctions for Robust Task Allocation        | https://doi.org/10.1109/TRO.2009.2022423 | IEEE PDF 入口不可直接下载 |

建议处理：

- 可通过学校/机构图书馆访问 DOI 页面下载。
- 可在 Google Scholar、作者主页或 ResearchGate 查找作者自存档版本。
- 写论文引用时保留 DOI，不依赖本地 PDF。

## 已下载的可开放替代参考

为避免 CBBA/动态任务分配阅读链中断，已额外下载一篇开放的 CBPA/CBBA 扩展文献：

| 文件名                                                                     | 来源                               | 用途                          |
| ----------------------------------------------------------------------- | -------------------------------- | --------------------------- |
| `P1_08b_2024_CBPA_Consensus_Based_Dynamic_Task_Allocation_Payloads.pdf` | https://arxiv.org/pdf/2412.10087 | 作为 CBBA 类动态任务分配与资源消耗建模的补充阅读 |

该文献不能替代 Choi et al. 2009 经典 CBBA 的历史地位，但可用于理解近期 consensus-based dynamic task allocation 如何处理任务更新和载荷/资源约束。

## 建议本地阅读顺序

1. `P0_01_2021_HATRPO_HAPPO_Trust_Region_MARL.pdf`
2. `P0_02_2023_HARL_Heterogeneous_Agent_RL.pdf`
3. `P0_03_2025_MAGNNET_GNN_Heterogeneous_Task_Allocation.pdf`
4. `P0_04_2021_Decentralized_Multi_UAV_Perimeter_Defense_Task_Allocation.pdf`
5. `P0_05_2020_CUAS_State_of_the_Art_Challenges_Future_Trends.pdf`
6. `P0_06_2018_CBBA_PR_Partial_Replanning_Dynamic_Task_Allocation.pdf`
7. `P1_08b_2024_CBPA_Consensus_Based_Dynamic_Task_Allocation_Payloads.pdf`
8. `P1_09_2017_Deep_Decentralized_Multitask_MARL_POMDP.pdf`
9. `P1_10_2023_Heterogeneous_UAV_Swarm_Multi_Target_Pursuit_MARL.pdf`
10. `P2_11_2022_Aerial_Threats_Radar_Communications_Survey.pdf`
11. `P2_12_2022_Autonomous_Drone_System_Jamming_Relative_Positioning.pdf`

同时补读两篇受限经典文献的摘要、引言和方法：

- MRTA taxonomy：用于任务分配问题分类。
- CBBA：用于去中心化拍卖分配基线。
