# 证据与定义冲突日志

更新时间：2026-07-24  
状态：W1-01 已核查  
核查范围：R1、R2、BPCE 标签审计、BPCE 短视窗审计、BPCE 在线压力测试、第一创新 Claim–Evidence 矩阵及 R2 冻结结果目录

## 1. 冲突处理记录

| Conflict ID | 问题 | 核查证据 | 处理 | 状态 |
| --- | --- | --- | --- | --- |
| CF-01 | 首轮成本公式只含未来替代，遗漏同一步其他单元替代 | 首轮 `7,776` 条账本中 `287` 条 `abs(protocol_residual)>1e-6`，最大残差 `2.0` | 主公式加入 \(Sub_{\mathrm{cost,same}}\)；最终最大误差为 `8.88e-16`；首轮结果移入 `pre_ledger_correction/` 仅作审计 | RESOLVED |
| CF-02 | R1 报告中的 `sub_cost` 是 future-only，而 R2 最终 `sub_cost` 含同一步与未来，名称相同但范围不同 | R1 结果字段与 R2 `_cost_ledger`、最终账本对照 | 论文冻结为 \(Sub_{\mathrm{cost,future}}\) 和 \(Sub_{\mathrm{cost,total}}\) 两个长名称；代码映射写入术语账本 | RESOLVED |
| CF-03 | `Sub_shot` 只统计未来射击，而总替代成本还含同一步替代，二者可能被误当成同一口径 | R2 `repeat_cost_ledger.csv` 字段定义和完整恒等式 | 明确 `Sub_shot` 为 future-only；主恒等式只使用 \(Sub_{\mathrm{cost,total}}\) | RESOLVED |
| CF-04 | R1 的“机会价值仅 missile 可辨识”和 R2 的“laser 符号掩盖更强”表面不一致 | R1 `reliable_resource_unit_types`；R2 time/resource 的 missile `2/9`、laser `5/9` | 两者估计对象不同：前者是安全机会价值标签，后者是成本符号掩盖；不得互相替代 | RESOLVED |
| CF-05 | R2 的 P-C2 通过，但 P-C3 失败，容易被误写为“跨资源类型没有动作替代” | R2 两类资源的 \(Sub_{\mathrm{shot}}\) 下界均为正，但掩盖上下文 missile `2/9`、laser `5/9` | 冻结结论为“替代机制跨类型存在，改变标签符号的强度具有资源类型条件” | RESOLVED |
| CF-06 | BPCE 个别运行改善可能被挑选为算法有效证据，与机制总门控失败冲突 | BPCE 压力测试有 `2` 个塌缩运行，`mechanism_gate_passed=false` | 只保留为失败机制和研究动机；禁止声称 BPCE/MCH-PPO 已稳定优于 PPO | RESOLVED |
| CF-07 | “动作替代”可能与 difference reward、counterfactual baseline、opportunity cost 或 shadow price 文献术语重叠 | 当前项目内部报告尚未完成系统术语检索 | 作为工作术语保留；交由 W1-02 系统检索确认，不作“首次发现/首次提出”主张 | ESCALATED_W1-02 |
| CF-08 | 同一关键数字同时出现在 JSON、正式报告和 Claim–Evidence 矩阵，可能形成多个权威版本 | W1-01 全部输入文件 | 数值唯一绑定 `evidence_source_index.md` 中的机器结果文件；正式报告只叙事，矩阵只控制主张 | RESOLVED |
| CF-09 | “累计成本不是无偏读出”可能被理解为已证明任意策略和环境下的统计无偏性命题 | R1/R2 均为冻结策略、选定场景的配对反事实审计 | 正文使用“在冻结 AirDefense v1 协议下会被动作替代系统性混合/偏置”的条件性表述 | RESOLVED |

## 2. 停止条件核查

| 核查项 | 结果 | 依据 |
| --- | --- | --- |
| 修正恒等式可由冻结账本复核 | PASS | 完整公式最大误差 `8.88e-16` |
| 新旧上下文 hash 零重叠 | PASS | R2 `integrity_gates.old_hash_overlap_zero=true` |
| Actor 冻结记录一致 | PASS | R2 最大参数差 `0.0` |
| 代码、数据和报告中的 N/E 身份一致 | PASS | `N=no-op`、`E=engage`，累计成本方向固定为 E−N |

## 3. 未解决项

没有阻塞 W1-01 的证据或定义冲突。

CF-07 已明确升级到 W1-02，属于文献命名与优先权核验，不改变现有数值、N/E
身份或成本恒等式，因此不阻塞 T01。W1-02 完成前禁止使用“首次提出动作替代”
及任何同义优先权表述。

## 4. T01 判定

**PASS**

- 关键数字均绑定唯一机器结果来源；
- N/E 方向只保留 E−N 的累计成本写法；
- 三类替代成本均进入主恒等式；
- `Sub_shot` 与 \(Sub_{\mathrm{cost,total}}\) 的统计范围已分离；
- 术语账本包含中文、英文、符号、定义和使用边界；
- 未创建第二份 Claim–Evidence 权威矩阵；
- 所有冲突均已解决或明确升级。
