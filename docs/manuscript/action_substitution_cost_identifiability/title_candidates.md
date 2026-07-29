# W1-08 标题候选与主标题决策

更新时间：2026-07-28  
定位：L2 测量、诊断与资源信用分解模块  
禁用词：`first`、`universal`、`general`、`solves`

## 1. 候选标题

| ID | 类型 | 英文标题 | 中文释义 | 证据锚点 | 过度承诺风险 |
| --- | --- | --- | --- | --- | --- |
| T-A | measurement-led | **Paired Counterfactual Auditing of Resource-Cost Measurement in Dynamically Masked Sequential Allocation** | 动态掩码序列分配中的资源成本成对反事实审计 | N/E、CRN、三分量账本、动态掩码 | 最低；没有宣称新算法或普遍偏差 |
| T-B | finding-led | Action Substitution Can Mask Local Resource Cost in Dynamically Masked Sequential Allocation | 动作替代可在动态掩码序列分配中掩盖局部资源成本 | C1、C3、C4 | 中；“can”保留条件，但标题未显式写 AirDefense v1 |
| T-C | object-and-consequence | Same-Step and Future Action Substitution in Episode-Level Cost Measurement for AirDefense Resource Allocation | 防空资源分配回合成本测量中的同一步与未来动作替代 | 三分量账本、AirDefense v1 | 低；场景最明确，但标题较长 |
| T-D | method-led | Three-Component Counterfactual Cost Decomposition for Autoregressive Resource Allocation | 自回归资源分配的三分量反事实成本分解 | C2、P-C1 | 中；弱化动态掩码和资源类型边界 |
| T-E | hook-led | When Episode Cost Misreads Local Action Cost: Counterfactual Auditing in AirDefense v1 | 当回合成本误读局部动作成本：AirDefense v1 反事实审计 | C1-C4、环境范围 | 中；“misreads”较强，但 AirDefense v1 范围清楚 |

## 2. 主标题

> **Paired Counterfactual Auditing of Resource-Cost Measurement in Dynamically Masked Sequential Allocation**

中文工作标题：

> **动态掩码序列分配中的资源成本成对反事实审计**

选择理由：

1. 同时包含可检索的 `counterfactual`、`resource-cost measurement` 和
   `dynamically masked sequential allocation`；
2. 以审计和测量为中心，符合 W1-02 的 L2 定位；
3. 不暗示优于 PPO、跨环境泛化或跨资源类型普遍成立；
4. AirDefense v1 的具体范围在摘要、Methods 和 Limitations 中明确，不必用
   场景名牺牲标题的技术可检索性。

## 3. 标题证据审计

| 标题短语 | 证据 | 状态 |
| --- | --- | --- |
| Paired counterfactual | N/E + CRN + exact target marginalization | 支持 |
| Auditing | 逐账本恒等式、完整性门控、独立确认 | 支持 |
| Resource-cost measurement | missile/laser 物理成本账本 | 支持 |
| Dynamically masked | 分支合法集和同一步后缀替代 | 支持 |
| Sequential allocation | order 0-1-2 factorized joint action | 支持，限当前实现 |

