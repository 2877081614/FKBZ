# Extraction and Reading Notes

Status: precision reader, first edition.

Completed:

- Extracted the selectable text layer from all 20 PDF pages into `raw_pages.txt`.
- Built `source_map.json` with page-level source anchors.
- Rendered evidence pages p.9, p.11, p.12, p.13, p.15 and p.16 into `assets/`.
- Created `paper.md` covering the paper's argument, detection and mitigation taxonomies, limitations, and a detailed mapping to a heterogeneous MARL environment.

Reading choices:

- The paper is a survey rather than an algorithm paper, so the reader emphasizes taxonomy, engineering assumptions and modeling implications instead of translating every cited study.
- Claims attributed to the paper are tied to its 2020 scope. Proposed state/action/reward designs are marked as project-oriented deductions rather than original claims of the authors.
- Tables IV and V and Figure 13 are retained as full-page renders to preserve captions and traceability.

Known limitations:

- The PDF text layer occasionally contains ligature and encoding artifacts.
- `source_map.json` is page-granular because the two-column PDF extraction does not reliably preserve paragraph boundaries.
- Exact numerical sensor and effector parameters still require later papers, datasets, equipment assumptions or simulation calibration.

Terminology:

- Counter-Unmanned Aircraft System(s), C-UAS: 反无人机系统/反无人航空系统
- detection: 探测；in the full chain, 探测、跟踪与识别
- mitigation / negation: 处置/抗击，不简单等同于摧毁
- physical capture: 物理捕获
- jamming: 干扰
- spoofing: 欺骗
- vulnerabilities exploitation: 漏洞利用
- capture and retrieve: 捕获并安全回收/接管
- disable and drop: 使目标失效并坠落
