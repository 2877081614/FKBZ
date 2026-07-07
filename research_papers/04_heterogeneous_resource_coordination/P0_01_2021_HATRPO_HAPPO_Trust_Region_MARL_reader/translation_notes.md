# Translation and Extraction Notes

Status: draft precision reader.

What is complete:
- Extracted the full text layer from all 27 PDF pages.
- Built `source_map.json` with 290 stable text block IDs.
- Rendered visual assets for pages containing the main figures: p.4, p.8, p.9, p.26, p.27.
- Created `paper.md` with bilingual source-aligned translations for the paper's core argument, theory, algorithms, experiments, and conclusion.

Known limitations:
- This first reader is not yet a paragraph-by-paragraph Chinese translation of all 290 extracted blocks.
- Figure assets are page-level renders, not tight crops, because this pass prioritised traceable precision reading over image-crop polishing.
- Formula extraction uses the PDF text layer and may lose some superscript/subscript layout. When a formula matters, inspect the original PDF or rendered page alongside the note.
- References are not translated block by block.

Terminology decisions:
- trust region: 信任域
- monotonic improvement: 单调改进
- joint policy: 联合策略
- parameter sharing: 参数共享
- advantage decomposition: 优势函数分解
- sequential policy update: 顺序策略更新
- centralized training with decentralized execution: 集中训练、分散执行
- heterogeneous agents: 异质智能体

Suggested next expansion:
- If a fully bilingual artifact is required, expand `paper.md` from `source_map.json` in batches: p.1-p.3, p.4-p.7, p.8-p.9, p.14-p.24 appendices.
