# Translation and Extraction Notes

Status: draft precision reader.

What is complete:
- Extracted the full selectable text layer from all 67 PDF pages.
- Built `source_map.json` with 597 stable source block IDs.
- Rendered selected evidence pages into `assets/`: p.7, p.10, p.18, p.21, p.22, p.23, p.25, p.27, p.28, p.29, p.30, p.54.
- Created `paper.md` with bilingual source-aligned translations for the paper's core argument, theoretical framework, algorithms, experiments, ablations, and defense-air-grouping implications.

Known limitations:
- This first reader is not a paragraph-by-paragraph full translation of all 597 extracted blocks.
- Figure assets are page-level renders, not tight panel crops. This preserves captions and traceability but may be denser than a final presentation figure crop.
- Formula extraction comes from the PDF text layer; inspect the original PDF/rendered page for exact superscript/subscript layout when a proof step matters.
- Long appendix proofs and hyperparameter tables are indexed but not fully translated in this first pass.

Terminology decisions:
- Heterogeneous-Agent Reinforcement Learning (HARL): 异质智能体强化学习算法族
- Heterogeneous-Agent Trust Region Learning (HATRL): 异质智能体信任域学习
- Heterogeneous-Agent Mirror Learning (HAML): 异质智能体镜像学习
- heterogeneous-agent drift functional (HADF): 异质智能体漂移泛函
- heterogeneous-agent mirror operator (HAMO): 异质智能体镜像算子
- neighbourhood operator: 邻域算子
- sequential update scheme: 顺序更新机制
- parameter sharing: 参数共享
- monotonic improvement: 单调改进
- Nash Equilibrium: Nash 均衡

Suggested next expansion:
- Expand p.14-p.17 HAML definitions into a formula-by-formula bilingual proof guide.
- Expand Appendix D/E/F for HAML and HAPPO-as-HAML proof details.
- Crop Figure 3, Figure 11, and Figure 12 into presentation-ready panels if a follow-up PPT is needed.
