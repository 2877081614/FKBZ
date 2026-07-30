# Task Guides

本目录用于存放项目后续阶段的任务指导文档，重点记录：

- 当前阶段目标与研究问题；
- 工作任务、依赖关系和执行顺序；
- 代码、实验与文档交付物；
- 可检查、可复现的验收标准；
- 进入下一研究阶段的前置条件；
- 明确暂缓内容，控制环境与算法复杂度。

## 当前任务

- [DS-TR 拆分执行包：动态支持域诊断、最小算法与增量控制](dynamic_support_trust_region/README.md)
- [已完成 N3：FCRC 冻结成对预测验证与 N3-E3 否决](next_research_phase_fcrc_paired_predictive_validation.md)
- [已完成 N2：未来可覆盖性责任证书与 N2-E1 静态门控](next_research_phase_future_coverability_certificate.md)
- [已完成 N1：可辨识资源信用定义、离线证伪与 N1-E4 退出](next_research_phase_identifiable_resource_credit.md)
- [下一研究阶段：场景难度、优化基线与泛化诊断](next_research_phase_difficulty_generalization.md)
- [下一研究阶段：无冲突联合动作机制](next_research_phase_conflict_free_joint_action.md)
- [下一研究阶段：顺序式/自回归无冲突联合动作生成](next_research_phase_autoregressive_joint_action.md)
- [下一研究阶段：单元顺序偏置与异质目标优先级诊断](next_research_phase_order_bias_diagnostics.md)
- [下一研究阶段：角色条件化关系动作头与顺序鲁棒性](next_research_phase_role_conditioned_action_head.md)
- [下一研究阶段：no-op 塌缩机理与 PPO 优化稳定性](next_research_phase_noop_optimization_stability.md)
- [下一研究阶段：交战概率校准与反事实分层信用分配](next_research_phase_counterfactual_credit_assignment.md)
- [下一研究阶段：非图结构掩码条件动作价值 Critic](next_research_phase_action_conditioned_q_critic.md)
- [下一研究阶段：动作差异数据与组内排序监督](next_research_phase_q_critic_ranking_refinement.md)
- [下一研究阶段：MCH-PPO 机制压力实验](next_research_phase_mch_ppo_mechanism_stress_test.md)
- [下一研究阶段：可靠度门控 MCH-PPO 核心机制验证](next_research_phase_reliability_gated_mch_ppo.md)
- [下一研究阶段：支持感知与累计漂移约束 RG-MCH-PPO](next_research_phase_support_anchored_rg_mch_ppo.md)
- [下一研究阶段：BPCE-PPO v0 语义实现与机制证伪](next_research_phase_bpce_ppo_v0.md)
- [下一研究阶段：BPCE 标签语义、辅助剂量与选点覆盖审计](next_research_phase_bpce_label_semantics_and_dose_audit.md)
- [下一研究阶段：BPCE 短视窗安全—资源双分量标签审计](next_research_phase_bpce_short_horizon_component_label_audit.md)
- [下一研究阶段：动作替代与弹药机会成本可辨识性审计](next_research_phase_action_substitution_resource_opportunity_cost_audit.md)
- [下一研究阶段：动作替代测量失真独立确认与适用边界](next_research_phase_action_substitution_independent_confirmation.md)
- [下一研究阶段：W1 主张—证据冻结与论文正文写作](next_research_phase_claim_evidence_freeze_and_manuscript_drafting.md)
- [W1 拆分执行包：10 项协调写作任务](w1_claim_evidence_manuscript/README.md)
- [下一研究阶段：显式交战与目标分层 Q 诊断](next_research_phase_hierarchical_q_diagnostics.md)
- [下一研究阶段：风险与约束感知的交战效用诊断](next_research_phase_risk_aware_engagement_utility.md)
- [下一研究阶段：安全临界状态与类别平衡交战估值](next_research_phase_critical_state_balanced_engagement.md)
- [下一研究阶段：资源约束交战边界校准](next_research_phase_resource_constrained_engagement_calibration.md)
- [下一研究阶段：状态条件资源预算与显式约束价值](next_research_phase_state_conditioned_constrained_value.md)
- [下一研究阶段：跨场景鲁棒预算与可靠成本差监督](next_research_phase_cross_scenario_robust_budget.md)
- [下一研究阶段：多批次临界状态语料与留一批次泛化](next_research_phase_multibatch_leave_one_out.md)
- [下一研究阶段：OOB 安全-停止 Pareto 可行性审计](next_research_phase_oob_pareto_feasibility.md)
- [下一研究阶段：冻结 OOB 校准协议的独立批次确认](next_research_phase_independent_calibration_confirmation.md)
- [下一研究阶段：跨批次统一概率校准与不确定性约束](next_research_phase_cross_batch_uncertainty_calibration.md)

其中，任务一至任务十四的当前门控阶段均已完成。任务十二证明 all-no-op 同时包含 deterministic argmax 概率碎片化和 PPO 种子分叉；任务十三进一步否决了“统一阈值即可修复”，并确认现有 `V(s)` Critic 不能提供动作条件反事实价值。

BPCE-PPO v0 已完成。joint PPO 严格 fallback、环境快照和索引共同随机带通过软件验收；正式10k三种子双场景实验有2/6个运行all-noop，异质场景安全改善但资源成本达到baseline的1.928倍，边界探测也只在一个场景优于等预算随机探测。当前候选未通过30k门控。现阶段先执行标签语义审计，依次验证 argmax target、target 边缘化和 stochastic continuation；尚未允许直接实现 coverage-balanced loss。30k/100k、target辅助与GNN继续冻结。

后续每个主要研究阶段应新增独立任务文档，不直接覆盖已经完成阶段的验收记录。

BPCE阶段A2已完成但未通过。短视窗标签只将可操作上下文由27提高到31；
异质场景形成10个ENGAGE和14个STOP，但time-pressure只有5/2，且其18个
资源槽全部AMBIGUOUS。阶段B/C和修订版10k不启动，BPCE在线辅助主线暂停。

动作替代与弹药机会成本审计已完成。P-R1通过：time/resource的18个上下文
均确认当前交战替代后续射击，非正累计成本差可被未来成本替代解释。
P-R2/P-R3失败：可靠资源机会价值仅为time 5/18、heterogeneity 2/18，
且只覆盖missile。通用机会成本oracle和在线辅助路线停止。

R2独立确认已完成。9个全新来源模型和108个零重叠上下文通过完整性门控；
P-C1/P-C2通过，动作替代测量失真跨新种子复现。P-C3因time/missile仅
2个符号掩盖上下文而失败，贡献冻结为资源类型与场景条件结论并转入写作。

W1-01至W1-10已经完成，阶段出口为L2/M2。N1以N1-E4否决回报责任候选；
N2的FCRC通过静态门控，但N3冻结成对验证未通过因果方向、增量预测与安全
一致性门槛，出口为N3-E3。FCRC只保留为静态解释组件，不进入reward、loss、
mask或shield。下一主线入口重新回到规范性算法问题定义；BPCE/MCH-PPO不
恢复，GNN仍需独立的关系估值或跨规模泛化瓶颈证据。
