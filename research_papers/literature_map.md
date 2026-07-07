# Literature Map for HELS-UAV-DRTA Research

This folder collects papers beyond the currently reproduced MADDPG-IA paper. The selection is organized into three layers:

1. foundational algorithms,
2. innovation references,
3. direction-aligned papers.

Selection criteria: strong academic venue or recognized preprint, clear methodological relevance, and reproducibility through public code, standard benchmarks, or sufficiently specified algorithms.

## 01 Foundational Algorithms

### Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments

- File: `01_foundational_algorithms/2017_NeurIPS_MADDPG_Multi-Agent_Actor_Critic.pdf`
- Source: NeurIPS 2017.
- Source value: NeurIPS is a top-tier machine learning conference. This is one of the canonical papers for centralized-training decentralized-execution multi-agent reinforcement learning.
- Paper value: It is the direct algorithmic root of MADDPG-IA. The current project's Actor-Critic structure, centralized critic, target networks, and replay-buffer training are all inherited from this framework.
- Reproducibility: OpenAI released reference multi-agent particle-environment and MADDPG implementations, and many later MARL baselines reuse this setting.
- How to use it here: Treat it as the baseline theory for CTDE, critic input design, target-network updates, and multi-agent instability analysis.
- Link: https://proceedings.neurips.cc/paper/2017/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html

### QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning

- File: `01_foundational_algorithms/2018_ICML_QMIX.pdf`
- Source: ICML 2018, PMLR.
- Source value: ICML is a top-tier ML conference. QMIX is a standard cooperative MARL baseline.
- Paper value: Provides a value-decomposition alternative to MADDPG. It is important for understanding why fixed-agent cooperative value decomposition may struggle when UAV target sets are dynamic.
- Reproducibility: Implemented in PyMARL and many MARL benchmark repositories.
- How to use it here: Use as a comparison baseline, especially for small-scale settings. Its limitations under variable target numbers can motivate the attention/state-encoding part of your method.
- Link: https://proceedings.mlr.press/v80/rashid18a.html

### The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games

- File: `01_foundational_algorithms/2022_NeurIPS_MAPPO_Surprising_Effectiveness_PPO.pdf`
- Source: NeurIPS 2022.
- Source value: NeurIPS is top-tier; MAPPO has become a strong default baseline in cooperative MARL.
- Paper value: Gives a stable policy-gradient baseline against which MADDPG-IA-style methods should be compared.
- Reproducibility: Public benchmark code is available through the MAPPO / marlbenchmark on-policy ecosystem.
- How to use it here: Use MAPPO as a serious baseline rather than only DQN/QMIX. If the proposed method cannot beat or explain differences from MAPPO, the paper's algorithmic claim will be weak.
- Link: https://arxiv.org/abs/2103.01955

### Categorical Reparameterization with Gumbel-Softmax

- File: `01_foundational_algorithms/2017_ICLR_Gumbel_Softmax.pdf`
- Source: ICLR 2017.
- Source value: ICLR is a top-tier representation learning conference. This paper is the standard reference for differentiable sampling from categorical variables.
- Paper value: The HELS-UAV-DRTA action is discrete: choose one target or wait. Gumbel-Softmax explains how discrete choices can be trained with gradient methods.
- Reproducibility: The method is simple, widely implemented in PyTorch and TensorFlow examples.
- How to use it here: Cite it for the discrete-action bridge in MADDPG-IA and for any future actor that outputs target-selection logits.
- Link: https://openreview.net/forum?id=rkE3y85ee

## 02 Innovation References

### Exploration by Random Network Distillation

- File: `02_innovation_references/2019_ICLR_Random_Network_Distillation.pdf`
- Source: ICLR 2019.
- Source value: ICLR is top-tier; RND is a widely recognized intrinsic-motivation method.
- Paper value: It is the direct source of the "I" module in MADDPG-IA. It addresses sparse or delayed rewards, which is exactly the issue in laser damage tasks where successful kill feedback appears after sustained irradiation.
- Reproducibility: OpenAI released implementations; the algorithm is concise enough to reproduce from the paper.
- How to use it here: Use it to redesign intrinsic reward schedules, compare RND against ICM/NGU-style exploration, and justify sparse-reward handling.
- Link: https://openreview.net/forum?id=H1lJJnR5Ym

### Actor-Attention-Critic for Multi-Agent Reinforcement Learning

- File: `02_innovation_references/2019_ICML_Actor_Attention_Critic_MAAC.pdf`
- Source: ICML 2019, PMLR.
- Source value: ICML is top-tier. The paper is a core attention-based MARL reference.
- Paper value: It uses attention inside the critic to select relevant information from other agents. This is highly relevant if you want to improve MADDPG-IA from target-attention to agent-target relational attention.
- Reproducibility: Public code exists and the experiments use standard multi-agent benchmarks.
- How to use it here: Use it as the main reference for attention-based critic design and for explaining why not all agents/targets should contribute equally to value estimation.
- Link: https://proceedings.mlr.press/v97/iqbal19a.html

### Graph Attention Networks

- File: `02_innovation_references/2018_ICLR_Graph_Attention_Networks.pdf`
- Source: ICLR 2018.
- Source value: ICLR is top-tier; GAT is a classic graph neural network architecture.
- Paper value: It provides the foundation for a stronger innovation path: model HELS and UAVs as a graph, then learn attention over weapon-target or resource-target edges.
- Reproducibility: Official and community implementations are widely available.
- How to use it here: Use it to move from simple target attention to graph attention over heterogeneous nodes: HELS, UAVs, protected assets, and possibly jammer/interceptor resources.
- Link: https://arxiv.org/abs/1710.10903

### Qatten: A General Framework for Cooperative Multiagent Reinforcement Learning

- File: `02_innovation_references/2020_AAAI_Qatten_Attention_MARL.pdf`
- Source: AAAI 2020.
- Source value: AAAI is a recognized top AI conference. The paper targets cooperative MARL with attention.
- Paper value: Offers an attention-based value factorization framework, useful for thinking about global value decomposition under cooperative HELS defense.
- Reproducibility: The method is benchmarked on standard cooperative MARL tasks and is implemented in community MARL libraries.
- How to use it here: Useful if you want an alternative to centralized critic MADDPG: attention-weighted value decomposition for multiple HELS resources.
- Link: https://arxiv.org/abs/2002.03939

## 03 Direction-Aligned Papers

### Deep Reinforcement Learning for Weapons to Targets Assignment in a Hypersonic Strike

- File: `03_direction_aligned/2023_WTA_Deep_RL_Hypersonic_Strike.pdf`
- Source: arXiv preprint.
- Source value: Not as authoritative as NeurIPS/ICML/IEEE journal papers, but directly targets WTA-style decision-making with deep RL.
- Paper value: Closest to the weapon-target-assignment side of this project. It is valuable for state/action/reward modeling and for comparing classical optimization framing with RL framing.
- Reproducibility: The model and algorithmic setup are explicitly described; if code is not available, it is still reproducible as a simulation baseline because the WTA abstraction is compact.
- How to use it here: Use it to strengthen the problem-definition section: why target assignment is sequential, dynamic, and hard for static combinatorial optimization.
- Link: https://arxiv.org/abs/2310.18509

### Multi-Target Pursuit by a Decentralized Heterogeneous UAV Swarm Using Deep Multi-Agent Reinforcement Learning

- File: `03_direction_aligned/2023_Role_Based_MARL_UAV_Swarm.pdf`
- Source: arXiv preprint.
- Source value: Directionally relevant to UAV-swarm MARL and heterogeneous roles; academic authority is weaker than top conferences, so use it as a design reference rather than as the main theoretical pillar.
- Paper value: Useful for heterogeneous-agent modeling, pursuit-evasion dynamics, and decentralized execution in spatial UAV tasks.
- Reproducibility: The setup is simulation-based and can be recreated; it also connects to standard MARL pursuit-evasion environments.
- How to use it here: Use it for adversarial or semi-adversarial UAV behavior design when moving beyond passive incoming targets.
- Link: https://arxiv.org/abs/2303.01799

### MAGNNET: Multi-Agent Graph Neural Network-Based Efficient Task Allocation for Autonomous Vehicles with Deep Reinforcement Learning

- File: `03_direction_aligned/2025_MAGNNET_GNN_Heterogeneous_Task_Assignment.pdf`
- Source: arXiv preprint.
- Source value: Recent preprint, not a replacement for peer-reviewed baselines. Its value is mainly in current methodology: GNN + MARL + heterogeneous task allocation.
- Paper value: Very relevant to the proposed next step: graph-based resource-target allocation. It provides a template for representing agents and tasks as a graph.
- Reproducibility: Simulation task-allocation setup is specified; treat as a reference architecture to reproduce and adapt, not as a final authority.
- How to use it here: Use it to justify a graph-attention extension of MADDPG-IA, especially for heterogeneous resources such as laser, jammer, interceptor, and sensor nodes.
- Link: https://arxiv.org/abs/2502.02311

## Recommended Reading Order

1. MADDPG, Gumbel-Softmax, and RND: understand the exact architecture of the current reproduced algorithm.
2. MAAC and GAT: understand the most promising innovation direction for a new paper.
3. QMIX, MAPPO, and Qatten: prepare stronger baselines and comparison logic.
4. WTA / UAV / MAGNNET papers: extract task modeling ideas for the next research stage.

## Suggested Research Direction After This Collection

The most promising direction is:

**Graph-attention enhanced heterogeneous dynamic resource-target assignment for UAV-swarm defense.**

The incremental path should be:

1. reproduce MADDPG-IA faithfully,
2. implement MAPPO/QMIX/MADDPG baselines,
3. replace simple target attention with graph attention,
4. extend HELS-only resources to heterogeneous defensive resources,
5. evaluate scalability, resource consumption, response delay, and generalization across environments.

Important caution: the direction-aligned papers include arXiv preprints because directly matching HELS-UAV-DRTA literature is sparse. For manuscript claims, rely primarily on the top-tier foundational and innovation papers, and use the direction-aligned preprints as engineering inspiration rather than as the main proof of academic consensus.
