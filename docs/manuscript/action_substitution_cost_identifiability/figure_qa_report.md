# W1-06 图表 QA 记录

更新时间：2026-07-28  
后端：Python 3 / matplotlib  
检查对象：Fig. 1-Fig. 5 全部 SVG、PDF、TIFF 和 PNG preview

| 检查 | 结果 |
| --- | --- |
| 每图有单一核心结论和 panel 证据链 | PASS |
| 183 mm 双栏宽度、白底、统一 sans-serif | PASS |
| SVG/PDF 可编辑文字；SVG 含 `<text>` | PASS |
| TIFF 600 dpi、LZW 压缩；PNG 220 dpi preview | PASS |
| 面板标签、标题、轴标、图例无不可读重叠 | PASS |
| R1 空心点、R2 实心点视觉区分 | PASS |
| missile/laser 同时显示且颜色/文字双编码 | PASS |
| P-C3 FAIL 以红色与文字共同显示 | PASS |
| \(Sub_{\mathrm{shot}}\) 与 \(Sub_{\mathrm{cost,total}}\) 分图分轴 | PASS |
| context、block、repeat 和 ledger row 未混用 | PASS |
| 五张图均有 source CSV/JSON 或冻结公式来源 | PASS |
| 无缺失图、空导出、删点或新增 rollout | PASS |

视觉 QA 逐一查看五张 PNG 原尺寸预览。Fig. 5c 初版门槛文字与柱体相交，已在
生成脚本中移至空白区域并全量重导；其他面板未发现遮挡、裁切或符号冲突。

