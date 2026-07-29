# W1-06 图表计划与制图契约

更新时间：2026-07-28  
状态：Python/matplotlib 正式生成  
目标版式：183 mm 双栏宽度；白底；SVG/PDF 可编辑文字；TIFF 600 dpi

## 1. 编号说明

W1-03 的四图草案将“审计协议/恒等式”和“替代成本组成”放在同一张 Fig. 2。
W1-06 按任务门控将其拆成两张图，形成 Fig. 1-Fig. 5。该拆分只降低单图信息
负荷，不改变 C1-C8、公式方向、统计单位或证据来源。原边界图由 Fig. 4 顺延为
Fig. 5。

## 2. 主图契约

| Figure | 核心结论 | 图型 | Panel 证据链 | 主要审稿风险 |
| --- | --- | --- | --- | --- |
| Fig. 1 | 动态掩码自回归后缀使回合累计成本不能被直接视为当前局部动作成本 | schematic-led composite | a：联合动作与动态占用；b：N/E 分支；c：直接成本被替代成本抵消 | 示意图不得伪装成定量结果；N/E 方向不得反转 |
| Fig. 2 | 只有包含同一步与两类未来替代的完整账本才能逐行重构累计成本差 | asymmetric mixed-modality | a：快照与 CRN；b：目标精确边缘化；c：三分量恒等式；d：future-only 与完整残差 | CRN 不能写成结构识别；“精确”只指代数误差 |
| Fig. 3 | R1 发现的正替代射击在 R2 新策略种子和新上下文中复现 | quantitative grid | a：R1 18 个 context；b：R2 18 个 context；c：R2 seed-block 区间；d：非正成本 context 的替代解释 | R1/R2 必须视觉区分；ledger row 不得当作独立样本 |
| Fig. 4 | 总替代成本由不可省略的同一步项和占主导的未来项共同构成，并可掩盖直接成本符号 | asymmetric quantitative | a：直接成本、三分量替代与累计差；b：same/future 组成；c：\(\rho_{\mathrm{sub}}\) 与累计差；d：资源类型 context 分布 | \(Sub_{\mathrm{shot}}\) 与 \(Sub_{\mathrm{cost,total}}\) 不得互换 |
| Fig. 5 | 替代机制跨场景存在，但成本符号掩盖强度受资源类型约束，P-C3 未通过 | quantitative grid | a：三场景 \(\rho_{\mathrm{sub}}\)；b：missile/laser \(Sub_{\mathrm{shot}}\)；c：掩盖 context 与门槛；d：P-C1/P-C2/P-C3 | 必须同时显示 missile 和 laser；失败门控不得弱化 |

## 3. 主表与补充表

| Table | 功能 | 数据权威 | 输出 |
| --- | --- | --- | --- |
| Table 1 | 任务、来源策略与反事实协议 | R2 `experiment_config.json` | CSV + Markdown |
| Table 2 | R1/R2 独立性与完整性 | R1/R2 `gate_summary.json`、R2 manifest | CSV + Markdown |
| Table 3 | P-C1/P-C2/P-C3 判据与结果 | R2 `gate_summary.json` 和冻结代码判据 | CSV + Markdown |
| Table 4 | 场景与资源类型边界 | R2 scenario/resource CSV | CSV + Markdown |
| Table S1 | 标签语义与短视窗前置审计 | label/short-horizon gate JSON | CSV + Markdown |
| Table S2 | 资源恢复负结果 | R1 gate JSON | CSV + Markdown |
| Table S3 | 首轮账本修正与完整性 | R2 gate JSON、pre-correction 账本 | CSV + Markdown |

## 4. 视觉编码冻结

- \(N\)：中性灰；\(E\)：蓝色；
- 当前直接成本：深灰；
- 同一步其他单元替代：橙色；
- 未来 probe 替代：绿色；
- 未来 other 替代：紫色；
- missile：蓝灰；laser：橙褐；
- PASS：绿色；FAIL：红色，并同时使用文字/符号，避免只依赖颜色；
- R1 使用空心圆和浅灰背景；R2 使用实心圆和蓝色背景提示；
- 所有定量误差线均为 \(\bar{x}\pm1.96s/\sqrt n\)，图注明确样本单位。

## 5. 导出与完整性

生成脚本：

```text
figures/source/generate_figures.py
```

脚本只读 `results/air_defense_v1/` 中冻结文件，确定性写入：

```text
figures/source/figure_*_data.csv
figures/exported/figure_*.svg
figures/exported/figure_*.pdf
figures/exported/figure_*.tiff
figures/exported/figure_*_preview.png
figures/metadata/figure_*_metadata.json
tables/exported/*.csv
tables/exported/*.md
tables/metadata/table_manifest.json
```

不新增 rollout，不按绘图效果删除 context，不从报告正文手工抄数。示意图面板
使用冻结公式和实现结构，不承载新的经验数值。

