const pptxgen = require('pptxgenjs');
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'OpenAI Codex';
pptx.subject = 'Counter-UAS survey and anti-UAV MARL environment modeling';
pptx.title = 'Counter-UAS：技术体系与反UAV强化学习环境建模';
pptx.company = 'Research presentation';
pptx.lang = 'zh-CN';
pptx.theme = {
  headFontFace: 'Microsoft YaHei',
  bodyFontFace: 'Microsoft YaHei',
  lang: 'zh-CN',
};
pptx.defineSlideMaster({
  title: 'MASTER',
  background: { color: 'F5F7FA' },
  objects: [
    { rect: { x: 0, y: 0, w: 13.333, h: 0.08, fill: { color: '17A6A1' }, line: { color: '17A6A1' } } },
  ],
});

const W = 13.333;
const H = 7.5;
const C = {
  ink: '17212B',
  muted: '5D6975',
  soft: 'E7ECF1',
  bg: 'F5F7FA',
  white: 'FFFFFF',
  navy: '1D3557',
  cyan: '17A6A1',
  cyanSoft: 'DDF4F2',
  blue: '3B7DDD',
  blueSoft: 'E3EEFC',
  red: 'D95D5D',
  redSoft: 'FBE7E7',
  amber: 'E0A33A',
  amberSoft: 'FFF1D5',
  green: '4C9A6A',
  greenSoft: 'E4F3E9',
  violet: '7B61A8',
  violetSoft: 'EEE8F6',
  graphite: '101820',
};

const outDir = __dirname;
const readerDir = path.resolve(__dirname, '..');
const sourceAssets = path.join(readerDir, 'assets');
const pptAssets = path.join(outDir, 'assets');
fs.mkdirSync(pptAssets, { recursive: true });

function addText(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    x, y, w, h,
    fontFace: opts.fontFace || 'Microsoft YaHei',
    fontSize: opts.fontSize || 18,
    color: opts.color || C.ink,
    bold: !!opts.bold,
    align: opts.align || 'left',
    valign: opts.valign || 'mid',
    margin: opts.margin === undefined ? 0 : opts.margin,
    breakLine: false,
    fit: opts.fit || 'shrink',
    isTextBox: true,
    ...opts,
  });
}

function rect(slide, x, y, w, h, fill, line = fill, radius = 0.08) {
  slide.addShape(radius > 0 ? pptx.ShapeType.roundRect : pptx.ShapeType.rect, {
    x, y, w, h,
    rectRadius: radius,
    fill: { color: fill },
    line: { color: line, width: line === fill ? 0.6 : 1 },
  });
}

function line(slide, x, y, w, h, color = C.muted, width = 1.5, arrow = false) {
  const flipH = w < 0;
  const flipV = h < 0;
  if (flipH) {
    x += w;
    w = Math.abs(w);
  }
  if (flipV) {
    y += h;
    h = Math.abs(h);
  }
  slide.addShape(pptx.ShapeType.line, {
    x, y, w, h,
    flipH,
    flipV,
    line: { color, width, endArrowType: arrow ? 'triangle' : 'none' },
  });
}

function circle(slide, x, y, d, fill, lineColor = fill, lineWidth = 0.8) {
  slide.addShape(pptx.ShapeType.ellipse, {
    x, y, w: d, h: d,
    fill: { color: fill },
    line: { color: lineColor, width: lineWidth },
  });
}

function pill(slide, text, x, y, w, color, fill) {
  rect(slide, x, y, w, 0.32, fill, fill, 0.16);
  addText(slide, text, x, y + 0.01, w, 0.28, { fontSize: 10, color, bold: true, align: 'center' });
}

function title(slide, section, heading, subtitle = '') {
  addText(slide, section.toUpperCase(), 0.58, 0.22, 2.4, 0.22, { fontFace: 'Aptos', fontSize: 9, color: C.cyan, bold: true });
  addText(slide, heading, 0.58, 0.5, 11.9, 0.55, { fontSize: 25, bold: true, color: C.ink });
  if (subtitle) addText(slide, subtitle, 0.6, 1.02, 12.0, 0.32, { fontSize: 11.5, color: C.muted });
}

function footer(slide, source, pageNo) {
  line(slide, 0.58, 7.0, 12.15, 0, 'D6DDE4', 0.7, false);
  addText(slide, source, 0.6, 7.07, 11.3, 0.18, { fontSize: 7.5, color: '7B8794' });
  addText(slide, String(pageNo).padStart(2, '0'), 12.2, 7.04, 0.45, 0.2, { fontFace: 'Aptos', fontSize: 8, color: '7B8794', align: 'right' });
}

function note(slide, text) {
  slide.addNotes(text);
}

function bullet(slide, text, x, y, w, color = C.cyan, fs = 15) {
  circle(slide, x, y + 0.12, 0.1, color, color, 0);
  addText(slide, text, x + 0.18, y, w - 0.18, 0.42, { fontSize: fs, color: C.ink, valign: 'top' });
}

function card(slide, x, y, w, h, heading, body, accent = C.cyan, fill = C.white) {
  rect(slide, x, y, w, h, fill, 'DCE3E9', 0.08);
  rect(slide, x, y, 0.07, h, accent, accent, 0);
  addText(slide, heading, x + 0.22, y + 0.16, w - 0.36, 0.35, { fontSize: 16, bold: true, color: C.ink });
  addText(slide, body, x + 0.22, y + 0.58, w - 0.38, h - 0.72, { fontSize: 11.5, color: C.muted, valign: 'top', breakLine: false });
}

function node(slide, x, y, w, h, tag, heading, sub, color, fill) {
  rect(slide, x, y, w, h, fill, color, 0.08);
  circle(slide, x + 0.16, y + 0.18, 0.38, color, color, 0);
  addText(slide, tag, x + 0.16, y + 0.18, 0.38, 0.38, { fontFace: 'Aptos', fontSize: 10, color: C.white, bold: true, align: 'center' });
  addText(slide, heading, x + 0.62, y + 0.12, w - 0.72, 0.35, { fontSize: 14, bold: true });
  addText(slide, sub, x + 0.16, y + 0.55, w - 0.3, h - 0.64, { fontSize: 9.5, color: C.muted, valign: 'top' });
}

function addIconLabel(slide, x, y, d, label, color, fill) {
  circle(slide, x, y, d, fill, color, 1.3);
  addText(slide, label, x, y, d, d, { fontSize: d > 0.6 ? 15 : 11, color, bold: true, align: 'center' });
}

function imageContain(slide, imgPath, x, y, w, h, border = true) {
  if (border) rect(slide, x, y, w, h, C.white, 'D6DDE4', 0.05);
  slide.addImage({ path: imgPath, x: x + 0.04, y: y + 0.04, w: w - 0.08, h: h - 0.08, sizing: 'contain' });
}

async function cropAssets() {
  const jobs = [
    ['page_13.png', 'table_detection.png', { left: 70, top: 105, width: 1080, height: 620 }],
    ['page_15.png', 'table_mitigation.png', { left: 70, top: 105, width: 1080, height: 540 }],
    ['page_09.png', 'fig_data_fusion.png', { left: 75, top: 75, width: 470, height: 270 }],
    ['page_11.png', 'fig_jamming.png', { left: 70, top: 70, width: 480, height: 250 }],
    ['page_12.png', 'fig_vulnerabilities.png', { left: 75, top: 70, width: 460, height: 430 }],
    ['page_16.png', 'fig_framework.png', { left: 180, top: 70, width: 840, height: 470 }],
  ];
  for (const [src, dst, crop] of jobs) {
    await sharp(path.join(sourceAssets, src)).extract(crop).png().toFile(path.join(pptAssets, dst));
  }
}

function addCover() {
  const slide = pptx.addSlide();
  slide.background = { color: C.graphite };
  slide.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: W, h: H, fill: { color: C.graphite }, line: { color: C.graphite } });
  for (const [d, opacity] of [[5.4, 65], [4.1, 50], [2.8, 35], [1.5, 20]]) {
    slide.addShape(pptx.ShapeType.ellipse, {
      x: 7.85 + (5.4 - d) / 2, y: 1.05 + (5.4 - d) / 2, w: d, h: d,
      fill: { color: C.graphite, transparency: 100 },
      line: { color: C.cyan, transparency: opacity, width: 1.3 },
    });
  }
  line(slide, 11.0, 3.72, 1.7, -1.15, C.cyan, 1.5, false);
  circle(slide, 11.88, 1.88, 0.2, C.red, C.red, 0);
  circle(slide, 9.22, 4.25, 0.14, C.amber, C.amber, 0);
  circle(slide, 10.55, 5.45, 0.12, C.blue, C.blue, 0);
  pill(slide, 'PAPER REVIEW · C-UAS', 0.68, 0.65, 2.35, C.cyan, '163B3B');
  addText(slide, 'Counter-UAS', 0.68, 1.35, 7.0, 0.8, { fontFace: 'Aptos Display', fontSize: 37, bold: true, color: C.white });
  addText(slide, '技术体系与反 UAV 强化学习环境建模', 0.7, 2.15, 7.5, 0.65, { fontSize: 27, bold: true, color: C.white });
  addText(slide, '基于 Wang, Liu & Song (2020) 的精读汇报', 0.72, 3.0, 6.9, 0.35, { fontSize: 14, color: 'B7C4CE' });
  rect(slide, 0.7, 4.2, 6.3, 1.35, '17242E', '31414D', 0.08);
  addText(slide, '核心问题', 0.95, 4.45, 1.1, 0.3, { fontSize: 12, color: C.cyan, bold: true });
  addText(slide, '如何把“探测—识别—分配—处置”技术链，转化为可训练的异构 Dec-POMDP？', 0.95, 4.78, 5.55, 0.62, { fontSize: 18, color: C.white, bold: true });
  addText(slide, '研究方向：反无人机动态编组 · 异构多智能体强化学习', 0.72, 6.78, 8.0, 0.25, { fontSize: 10.5, color: '91A1AD' });
  note(slide, '本次汇报不把这篇综述当成强化学习算法论文，而是把它当作环境需求说明书。重点回答：环境里有哪些实体、能力、失效条件和安全约束，以及如何映射到异构多智能体模型。');
}

function addPositioning() {
  const slide = pptx.addSlide('MASTER');
  title(slide, '01 · PAPER POSITIONING', '这篇论文提供的是技术地图，不是 RL 算法', '综述的价值在于定义“环境里应该有什么”，而非直接给出训练方法');
  rect(slide, 0.65, 1.55, 4.15, 4.75, C.navy, C.navy, 0.08);
  addText(slide, 'C-UAS', 0.98, 1.92, 3.4, 0.62, { fontFace: 'Aptos Display', fontSize: 36, color: C.white, bold: true });
  addText(slide, 'Counter-Unmanned\nAircraft System(s)', 1.0, 2.72, 3.1, 0.9, { fontFace: 'Aptos', fontSize: 19, color: 'DDE7EF', bold: true, valign: 'top' });
  addText(slide, '依法、安全地使无人机失效、受扰或被接管，同时控制附带损伤与单次交战成本。', 1.0, 4.05, 3.0, 1.2, { fontSize: 15, color: C.white, valign: 'top' });
  pill(slide, '综述 · 工程系统视角', 0.98, 5.65, 2.5, C.amber, '3A3122');

  card(slide, 5.15, 1.55, 3.45, 1.35, '① 建立技术分类', '五类探测：声学、被动 RF、视觉、雷达、数据融合。', C.cyan, C.white);
  card(slide, 8.85, 1.55, 3.45, 1.35, '② 建立处置分类', '物理捕获、干扰、漏洞利用，并区分驱离/接管与失效坠落。', C.red, C.white);
  card(slide, 5.15, 3.2, 3.45, 1.35, '③ 比较能力边界', '射程、识别能力、成本、能耗、环境敏感性与附带影响。', C.amber, C.white);
  card(slide, 8.85, 3.2, 3.45, 1.35, '④ 指向系统协同', '单传感器与单效应器均不可靠，需要多源融合和统一协调。', C.green, C.white);
  rect(slide, 5.15, 4.92, 7.15, 1.38, C.cyanSoft, C.cyanSoft, 0.08);
  addText(slide, '对本课题的正确用法', 5.42, 5.15, 2.1, 0.3, { fontSize: 15, bold: true, color: C.cyan });
  addText(slide, '用论文确定实体、能力、失效条件和约束；再用后续论文或仿真假设标定探测概率、处置概率、时延与成本。', 5.42, 5.55, 6.4, 0.46, { fontSize: 14, color: C.ink, bold: true });
  footer(slide, 'Source: Wang, Liu & Song, Counter-UAS (2020), p.1–2；蓝绿色框为本研究解读', 2);
  note(slide, '先明确论文定位。它没有提出新的强化学习算法，也没有统一的概率模型。它更像一个工程技术目录，适合用来确定环境实体、能力边界和约束。');
}

function addThreats() {
  const slide = pptx.addSlide('MASTER');
  title(slide, '02 · THREAT MODEL', '反 UAV 的任务目标不是单一“击落”', '论文将威胁分成公共安全、国家安全与个人隐私三类');
  const cols = [
    [0.7, 'SAFETY', '公共安全', '碰撞、操作失误、受控攻击', '人员伤亡\n设施损坏\n航空运行中断', C.blue, C.blueSoft],
    [4.55, 'SECURITY', '国家安全', '侦察测绘、载荷投送、走私', '敏感信息泄露\n关键设施受损\n任务失败', C.red, C.redSoft],
    [8.4, 'PRIVACY', '个人隐私', '摄像、实时图传、持续监视', '隐私侵害\n信息获取\n行为干扰', C.violet, C.violetSoft],
  ];
  for (const [x, tag, zh, mode, result, color, fill] of cols) {
    rect(slide, x, 1.65, 3.25, 3.75, fill, color, 0.08);
    pill(slide, tag, x + 0.28, 1.92, 1.15, color, C.white);
    addText(slide, zh, x + 0.28, 2.42, 2.5, 0.42, { fontSize: 23, bold: true, color });
    addText(slide, '威胁方式', x + 0.3, 3.08, 0.9, 0.25, { fontSize: 10, color: C.muted, bold: true });
    addText(slide, mode, x + 0.3, 3.36, 2.55, 0.58, { fontSize: 14, bold: true });
    line(slide, x + 0.3, 4.08, 2.55, 0, color, 0.8, false);
    addText(slide, result, x + 0.3, 4.28, 2.5, 0.85, { fontSize: 12, color: C.muted, valign: 'top' });
  }
  rect(slide, 0.7, 5.72, 10.95, 0.75, C.graphite, C.graphite, 0.06);
  addText(slide, '环境建模含义', 0.98, 5.93, 1.35, 0.28, { fontSize: 13, color: C.cyan, bold: true });
  addText(slide, '威胁度应由预计到达时间、保护对象价值、载荷/意图信念、航迹质量和突防概率共同决定。', 2.35, 5.9, 8.9, 0.32, { fontSize: 15, color: C.white, bold: true });
  circle(slide, 11.9, 5.7, 0.8, C.amber, C.amber, 0);
  addText(slide, '≠\n固定标签', 11.9, 5.74, 0.8, 0.68, { fontSize: 11, color: C.graphite, bold: true, align: 'center' });
  footer(slide, 'Source: 论文 Table I–III, p.2–4；底部为强化学习环境推导', 3);
  note(slide, '论文的威胁分类提醒我们：防御目标不是“击落数量最大”，而是保护对象风险最小。因而威胁度应该随时间和观测更新，而不是预先固定。');
}

function addLoop() {
  const slide = pptx.addSlide('MASTER');
  title(slide, '03 · END-TO-END CHAIN', 'C-UAS 是感知与处置耦合的闭环决策', '论文写成“探测子系统 + 处置子系统”，实际至少包含六个决策环节');
  const items = [
    ['01', '发现', 'Detect', C.blue, C.blueSoft],
    ['02', '跟踪', 'Track', C.cyan, C.cyanSoft],
    ['03', '识别', 'Identify', C.violet, C.violetSoft],
    ['04', '威胁评估', 'Assess', C.amber, C.amberSoft],
    ['05', '资源分配', 'Assign', C.green, C.greenSoft],
    ['06', '处置', 'Mitigate', C.red, C.redSoft],
  ];
  let x = 0.55;
  for (let i = 0; i < items.length; i++) {
    const [tag, zh, en, color, fill] = items[i];
    node(slide, x, 2.1, 1.75, 1.25, tag, zh, en, color, fill);
    if (i < items.length - 1) line(slide, x + 1.77, 2.72, 0.28, 0, '82909D', 1.4, true);
    x += 2.08;
  }
  line(slide, 11.8, 3.62, -9.95, 1.15, C.red, 1.2, true);
  addText(slide, '处置失败 / 航迹更新 / 新目标出现', 4.15, 4.53, 4.4, 0.3, { fontSize: 11, color: C.red, bold: true, align: 'center' });
  rect(slide, 0.7, 5.35, 12.0, 1.08, C.white, 'DCE3E9', 0.08);
  addText(slide, '关键建模边界', 0.98, 5.6, 1.5, 0.3, { fontSize: 14, color: C.navy, bold: true });
  addText(slide, 'Track、Identify 与 Assign 不能被折叠掉，否则策略会在“全知状态 + 即时处置”的假想世界中学习。', 2.55, 5.49, 9.45, 0.56, { fontSize: 16, color: C.ink, bold: true });
  footer(slide, 'Source: 论文 p.1–4 的系统定义；流程重构为本研究推导', 4);
  note(slide, '论文只显式区分探测与处置，但实际闭环还包含跟踪、识别、威胁评估和资源分配。后续强化学习环境应保留这些信息阶段，否则任务会被过度简化。');
}

function addDetectionTaxonomy() {
  const slide = pptx.addSlide('MASTER');
  title(slide, '04 · DETECTION TAXONOMY', '五类探测能力：没有一种传感器可以独立解决问题', '能力差异来自物理机理，而不是算法名称');
  const cx = 6.12, cy = 3.65;
  circle(slide, cx - 0.72, cy - 0.72, 1.45, C.navy, C.navy, 0);
  addText(slide, '融合\n航迹', cx - 0.72, cy - 0.67, 1.45, 1.25, { fontSize: 19, color: C.white, bold: true, align: 'center' });
  const sensors = [
    [0.72, 1.65, '声', '声学', '低成本 · 近程\n受风噪影响', C.amber, C.amberSoft],
    [4.0, 1.35, 'RF', '被动 RF', '远程 · 可识别\n依赖通信链', C.cyan, C.cyanSoft],
    [8.2, 1.35, '光', '视觉 / 红外', '分类信息丰富\n受光照遮挡', C.violet, C.violetSoft],
    [10.75, 3.4, '雷', '主动 / 被动雷达', '远程 · 测距测速\n成本与杂波', C.blue, C.blueSoft],
    [8.5, 5.05, '算', '多算法融合', '按状态调度\n平衡精度与开销', C.green, C.greenSoft],
    [2.4, 5.05, '融', '多源数据融合', '一致性与权重\n异步与误差', C.red, C.redSoft],
  ];
  for (const [x, y, tag, heading, sub, color, fill] of sensors) {
    node(slide, x, y, 2.15, 1.18, tag, heading, sub, color, fill);
    const nx = x + 1.08, ny = y + 0.59;
    line(slide, nx, ny, cx - nx, cy - ny, 'B1BCC5', 1.1, true);
  }
  rect(slide, 0.72, 6.42, 11.95, 0.36, C.graphite, C.graphite, 0.04);
  addText(slide, '建模重点：每类传感器都需要独立的覆盖、误差、更新率、环境敏感性和识别能力。', 0.95, 6.46, 11.5, 0.24, { fontSize: 12.5, color: C.white, bold: true, align: 'center' });
  footer(slide, 'Source: 论文 Section III, p.4–10', 5);
  note(slide, '五类探测技术的差异必须体现在环境参数里。否则所谓异构 agent 只是名字不同，策略实际上面对完全相同的观测模型。');
}

function addDetectionTable() {
  const slide = pptx.addSlide('MASTER');
  title(slide, '05 · SENSOR TRADE-OFFS', '探测能力矩阵：射程、识别与鲁棒性不可兼得', '论文 Table IV 给出了构建传感器参数表的直接依据');
  imageContain(slide, path.join(pptAssets, 'table_detection.png'), 0.62, 1.48, 8.55, 4.95, true);
  rect(slide, 9.45, 1.48, 3.23, 4.95, C.graphite, C.graphite, 0.08);
  addText(slide, '读表结论', 9.75, 1.78, 2.35, 0.35, { fontSize: 19, color: C.white, bold: true });
  const bullets = [
    ['雷达', '远程与精度强，但成本、能耗和识别能力受限。', C.blue],
    ['被动 RF', '可识别链路与型号，但静默/加密/跳频会失效。', C.cyan],
    ['视觉', '适合识别和末段跟踪，但受光照、遮挡与鸟类混淆。', C.violet],
    ['数据融合', '可靠性最高，但引入算力、时延和关联错误。', C.green],
  ];
  let y = 2.38;
  for (const [h, b, color] of bullets) {
    addText(slide, h, 9.75, y, 0.9, 0.28, { fontSize: 13, color, bold: true });
    addText(slide, b, 9.75, y + 0.32, 2.55, 0.62, { fontSize: 11, color: 'D9E2E8', valign: 'top' });
    y += 0.98;
  }
  footer(slide, 'Source: 论文 Table IV, p.13；原图裁切保留', 6);
  note(slide, '这张表可以直接转成环境中的能力矩阵。重点不是给每种传感器一个固定好坏分数，而是定义它们在什么环境和目标条件下失效。');
}

function addFusion() {
  const slide = pptx.addSlide('MASTER');
  title(slide, '06 · DATA FUSION', '数据融合不止“多传感器求平均”', '作者区分同型传感器、异型传感器与多算法调度三个层次');
  imageContain(slide, path.join(pptAssets, 'fig_data_fusion.png'), 0.7, 1.55, 4.55, 2.45, true);
  const levels = [
    ['L1', '同型多传感器', '阵列化提高覆盖、测向和定位精度', C.blue, C.blueSoft],
    ['L2', '异型传感器', '雷达/RF 远程预警，光电/声学中近程确认', C.cyan, C.cyanSoft],
    ['L3', '多算法调度', '按当前状态切换算法，平衡精度、时延与算力', C.green, C.greenSoft],
  ];
  let y = 1.55;
  for (const [tag, h, b, color, fill] of levels) {
    node(slide, 5.65, y, 6.75, 0.95, tag, h, b, color, fill);
    y += 1.18;
  }
  rect(slide, 0.7, 4.4, 11.7, 1.75, C.white, 'DCE3E9', 0.08);
  addText(slide, '融合节点真正需要维护的量', 0.98, 4.66, 3.0, 0.48, { fontSize: 14, color: C.navy, bold: true });
  const tags = [
    ['估计状态', C.blueSoft, C.blue], ['协方差', C.cyanSoft, C.cyan], ['类别概率', C.violetSoft, C.violet],
    ['意图概率', C.amberSoft, C.amber], ['信息年龄 AoI', C.greenSoft, C.green], ['来源与关联', C.redSoft, C.red],
  ];
  let tx = 1.0;
  for (const [t, fill, color] of tags) {
    rect(slide, tx, 5.25, 1.62, 0.48, fill, fill, 0.18);
    addText(slide, t, tx, 5.31, 1.62, 0.3, { fontSize: 11.5, color, bold: true, align: 'center' });
    tx += 1.84;
  }
  addText(slide, '难点：异步采样 · 权重选择 · 错误关联 · 通信延迟 · 结果一致性', 1.0, 5.86, 10.9, 0.25, { fontSize: 11.5, color: C.muted, bold: true, align: 'center' });
  footer(slide, 'Source: 论文 Figure 9 与 Section III-E, p.8–10', 7);
  note(slide, '强化学习策略不应该直接读取所有传感器原始数据。第一版环境更适合提供融合航迹及其置信度，同时把融合误差、延迟和错误关联作为环境随机性。');
}

function addMitigationTaxonomy() {
  const slide = pptx.addSlide('MASTER');
  title(slide, '07 · MITIGATION TAXONOMY', '处置不是一个动作，而是一组条件化效应器', '论文将 UAS negation 分为物理捕获、干扰与漏洞利用');
  const branches = [
    [0.7, 'PHYSICAL', '物理捕获 / EMP', '网捕 · 投射网 · 拦截 UAV\n定向电磁脉冲', C.red, C.redSoft],
    [4.65, 'JAMMING', '通信与导航干扰', 'Tone · Sweep · Protocol-aware\nGPS 欺骗 · 网络攻击', C.amber, C.amberSoft],
    [8.6, 'CYBER', '漏洞利用 / 接管', '协议识别 · 命令注入\n传感器欺骗 · 系统漏洞', C.violet, C.violetSoft],
  ];
  for (const [x, tag, h, b, color, fill] of branches) {
    rect(slide, x, 1.62, 3.3, 2.1, fill, color, 0.08);
    pill(slide, tag, x + 0.25, 1.88, 1.15, color, C.white);
    addText(slide, h, x + 0.25, 2.35, 2.75, 0.42, { fontSize: 19, bold: true, color });
    addText(slide, b, x + 0.25, 2.88, 2.75, 0.62, { fontSize: 12, color: C.muted, valign: 'top' });
  }
  addText(slide, '处置结果必须区分', 0.72, 4.18, 2.2, 0.35, { fontSize: 18, bold: true });
  rect(slide, 0.72, 4.7, 5.55, 1.35, C.greenSoft, C.green, 0.08);
  addText(slide, 'CAPTURE & RETRIEVE', 1.0, 4.95, 2.35, 0.28, { fontFace: 'Aptos', fontSize: 12, color: C.green, bold: true });
  addText(slide, '驱离、接管或安全回收', 1.0, 5.3, 3.7, 0.32, { fontSize: 19, color: C.ink, bold: true });
  addText(slide, '低附带损伤，但依赖链路/漏洞与持续控制。', 1.0, 5.67, 4.75, 0.25, { fontSize: 11, color: C.muted });
  rect(slide, 6.55, 4.7, 5.55, 1.35, C.redSoft, C.red, 0.08);
  addText(slide, 'DISABLE & DROP', 6.85, 4.95, 2.1, 0.28, { fontFace: 'Aptos', fontSize: 12, color: C.red, bold: true });
  addText(slide, '失效、摧毁并坠落', 6.85, 5.3, 3.7, 0.32, { fontSize: 19, color: C.ink, bold: true });
  addText(slide, '处置快，但需要计入坠落区、误伤和财产损失。', 6.85, 5.67, 4.75, 0.25, { fontSize: 11, color: C.muted });
  footer(slide, 'Source: 论文 Section IV, Figure 10–12, p.10–12', 8);
  note(slide, '这里最重要的不是术语，而是结果类型。安全驱离、接管、失效和坠落应有不同奖励与风险。单一“击毁成功 +1”会抹掉反无人机场景的关键安全差异。');
}

function addMitigationTable() {
  const slide = pptx.addSlide('MASTER');
  title(slide, '08 · EFFECTOR TRADE-OFFS', '动作是否有效，取决于目标属性与交战条件', '论文 Table V 是设计动作掩码与条件成功率的直接依据');
  imageContain(slide, path.join(pptAssets, 'table_mitigation.png'), 0.62, 1.52, 8.45, 4.75, true);
  rect(slide, 9.35, 1.52, 3.35, 4.75, C.white, 'DCE3E9', 0.08);
  addText(slide, '四个必要条件', 9.65, 1.82, 2.45, 0.35, { fontSize: 18, bold: true, color: C.navy });
  const conds = [
    ['01', '目标可见', '航迹质量与识别置信度达到阈值', C.blue],
    ['02', '物理可达', '距离、射界、波束和到达时间满足', C.red],
    ['03', '机理匹配', '链路、频段、导航模式或漏洞可利用', C.amber],
    ['04', '规则允许', '附带风险、友方频谱与授权满足', C.green],
  ];
  let y = 2.38;
  for (const [tag, h, b, color] of conds) {
    circle(slide, 9.62, y, 0.38, color, color, 0);
    addText(slide, tag, 9.62, y, 0.38, 0.38, { fontFace: 'Aptos', fontSize: 9, color: C.white, bold: true, align: 'center' });
    addText(slide, h, 10.15, y - 0.02, 1.45, 0.28, { fontSize: 13, bold: true });
    addText(slide, b, 10.15, y + 0.28, 2.05, 0.42, { fontSize: 9.7, color: C.muted, valign: 'top' });
    y += 0.91;
  }
  footer(slide, 'Source: 论文 Table V, p.15；右侧为环境建模推导', 9);
  note(slide, '环境中每个动作都应有适用条件。例如目标无线电静默时，通信干扰不能生效；目标不用 GNSS 时，GPS 欺骗不能生效；航迹质量低时，硬杀伤不能直接授权。');
}

function addChallenges() {
  const slide = pptx.addSlide('MASTER');
  title(slide, '09 · CHALLENGES', '作者识别的八类瓶颈，正是环境随机性与约束来源', '成熟 C-UAS 需要同时满足 scalable、modular、affordable 与 low-collateral');
  const items = [
    ['规模', '大区域、多目标、蜂群使覆盖和容量不足', C.blue, C.blueSoft],
    ['模块化', '传感器、平台与效应器难以快速组合', C.cyan, C.cyanSoft],
    ['SWaP', '尺寸、重量、功耗、算力制约部署', C.amber, C.amberSoft],
    ['环境', '天气、杂波、光照、噪声改变性能', C.green, C.greenSoft],
    ['识别', '发现不等于识别类别、协议与意图', C.violet, C.violetSoft],
    ['定向作用', '干扰与 EMP 难以只作用于单目标', C.red, C.redSoft],
    ['附带损伤', '坠落、频谱污染与误接管形成二次风险', 'B86824', 'F7EBDD'],
    ['协同一致性', '多源结果冲突，感知与处置争用资源', '4C6A75', 'E5EEF1'],
  ];
  let idx = 0;
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 4; c++) {
      const [h, b, color, fill] = items[idx++];
      const x = 0.67 + c * 3.08;
      const y = 1.62 + r * 2.28;
      rect(slide, x, y, 2.78, 1.82, fill, color, 0.08);
      addText(slide, String(idx).padStart(2, '0'), x + 0.22, y + 0.18, 0.42, 0.28, { fontFace: 'Aptos', fontSize: 10, color, bold: true });
      addText(slide, h, x + 0.22, y + 0.55, 2.2, 0.35, { fontSize: 18, color, bold: true });
      addText(slide, b, x + 0.22, y + 1.02, 2.32, 0.55, { fontSize: 10.5, color: C.muted, valign: 'top' });
    }
  }
  rect(slide, 0.67, 6.28, 12.02, 0.45, C.graphite, C.graphite, 0.04);
  addText(slide, 'RL 任务不是“忽略这些问题”，而是让策略在这些不确定性与约束中学会动态协调。', 0.9, 6.36, 11.55, 0.23, { fontSize: 13, color: C.white, bold: true, align: 'center' });
  footer(slide, 'Source: 论文 Section V, p.13–15', 10);
  note(slide, '这八类瓶颈可以直接转成环境变量、故障模式、动作约束和评价指标。例如天气影响观测，SWaP 影响能量和算力，定向作用与附带损伤影响安全约束。');
}

function addFutureFramework() {
  const slide = pptx.addSlide('MASTER');
  title(slide, '10 · FUTURE TREND', '从单设备走向“多源感知—统一协调—多手段处置”', '论文的民用安全管理框架，可以迁移为防空编组的指挥闭环');
  imageContain(slide, path.join(pptAssets, 'fig_framework.png'), 0.65, 1.48, 5.9, 3.45, true);
  addText(slide, '论文原框架', 0.82, 5.05, 1.1, 0.28, { fontSize: 12, color: C.muted, bold: true });
  addText(slide, '可信信息、法规库、空域管理与地方协调者共享数据，协同管理 UAS 运行。', 1.85, 5.02, 4.5, 0.48, { fontSize: 11.5, color: C.ink });
  rect(slide, 6.85, 1.48, 5.85, 4.92, C.graphite, C.graphite, 0.08);
  addText(slide, '迁移到防空编组', 7.18, 1.78, 2.4, 0.38, { fontSize: 20, color: C.white, bold: true });
  const flow = [
    ['S', '异构传感器', '雷达 · RF · 光电 · 声学', C.blue],
    ['F', '航迹与威胁融合', '状态估计 · 置信度 · AoI', C.cyan],
    ['C', '指挥与资源分配', '优先级 · 授权 · 动态重编组', C.amber],
    ['E', '软硬杀伤效应器', '干扰 · 欺骗 · 拦截 · 网捕', C.red],
  ];
  let y = 2.38;
  for (let i = 0; i < flow.length; i++) {
    const [tag, h, b, color] = flow[i];
    circle(slide, 7.2, y, 0.46, color, color, 0);
    addText(slide, tag, 7.2, y, 0.46, 0.46, { fontFace: 'Aptos', fontSize: 10, color: C.white, bold: true, align: 'center' });
    addText(slide, h, 7.85, y - 0.04, 2.5, 0.3, { fontSize: 15, color: C.white, bold: true });
    addText(slide, b, 7.85, y + 0.28, 3.75, 0.28, { fontSize: 10.5, color: 'AFC0CC' });
    if (i < flow.length - 1) line(slide, 7.43, y + 0.5, 0, 0.48, '6F8798', 1.2, true);
    y += 0.9;
  }
  rect(slide, 7.18, 5.88, 5.05, 0.38, '173B3B', '173B3B', 0.04);
  addText(slide, '闭环反馈：处置结果重新进入航迹与威胁融合', 7.35, 5.94, 4.7, 0.22, { fontSize: 11, color: C.cyan, bold: true, align: 'center' });
  footer(slide, 'Source: 论文 Figure 13 与 Section VI, p.16–17；右侧为本研究迁移', 11);
  note(slide, '原论文的统一框架偏民用空域治理。迁移到防空编组后，共享数据立方体可以对应融合航迹，地方协调者可以对应指挥与资源分配节点。');
}

function addWhyRL() {
  const slide = pptx.addSlide('MASTER');
  title(slide, '11 · WHY MARL', '为什么静态规则难以覆盖反 UAV 动态编组？', '多目标到达、信息不确定与异构资源耦合，使问题天然具有序贯性');
  const items = [
    ['动态到达', '威胁数量、航向与任务阶段持续变化', C.red],
    ['部分可观测', '真实位置、意图、链路和载荷不可直接获得', C.blue],
    ['条件有效', '同一动作对不同导航/协议目标效果不同', C.amber],
    ['资源耦合', '探测、频谱、弹药、射界和通信相互制约', C.green],
  ];
  let y = 1.62;
  for (const [h, b, color] of items) {
    rect(slide, 0.72, y, 4.7, 0.95, C.white, 'DCE3E9', 0.08);
    circle(slide, 0.95, y + 0.22, 0.5, color, color, 0);
    addText(slide, h.slice(0, 1), 0.95, y + 0.22, 0.5, 0.5, { fontSize: 14, color: C.white, bold: true, align: 'center' });
    addText(slide, h, 1.62, y + 0.14, 1.45, 0.3, { fontSize: 15.5, bold: true });
    addText(slide, b, 1.62, y + 0.49, 3.35, 0.3, { fontSize: 10.5, color: C.muted });
    y += 1.14;
  }
  line(slide, 5.75, 3.58, 0.78, 0, C.cyan, 2, true);
  rect(slide, 6.75, 1.72, 5.65, 3.85, C.navy, C.navy, 0.08);
  pill(slide, 'RECOMMENDED MODEL', 7.08, 2.02, 1.95, C.cyan, '163B3B');
  addText(slide, '异构 Dec-POMDP', 7.08, 2.55, 4.35, 0.56, { fontSize: 29, color: C.white, bold: true });
  addText(slide, 'M = ⟨ I, S, {Oᵢ}, {Aᵢ}, P, R, γ ⟩', 7.1, 3.28, 4.45, 0.42, { fontFace: 'Cambria Math', fontSize: 18, color: C.cyan, bold: true });
  addText(slide, '• 异构 agent 与动作空间\n• 局部观测和信念状态\n• 联合转移与团队奖励\n• 集中训练、分散执行', 7.1, 4.02, 4.2, 1.18, { fontSize: 14, color: 'E3EBF1', valign: 'top', breakLine: false });
  rect(slide, 6.75, 5.88, 5.65, 0.5, C.cyanSoft, C.cyanSoft, 0.04);
  addText(slide, '环境比算法更先决定研究问题是否成立', 6.95, 5.97, 5.25, 0.27, { fontSize: 14, color: C.cyan, bold: true, align: 'center' });
  footer(slide, '本页为论文技术结论向 MARL 问题定义的推导', 12);
  note(slide, '这些工程特征共同指向部分可观测的异构多智能体问题。这里先定义环境，再选择 HAPPO、HARL 或 MAPPO 作为算法基线。');
}

function addDecPomdp() {
  const slide = pptx.addSlide('MASTER');
  title(slide, '12 · DEC-POMDP ARCHITECTURE', '真实状态、局部观测与联合动作必须分开', '仿真器知道真值；策略只能看到带误差、延迟和置信度的航迹');
  rect(slide, 0.62, 1.55, 3.0, 4.9, C.graphite, C.graphite, 0.08);
  addText(slide, 'GLOBAL STATE  Sₜ', 0.92, 1.85, 2.4, 0.3, { fontFace: 'Aptos', fontSize: 12, color: C.cyan, bold: true });
  const stateItems = ['目标真实运动/意图', '通信与导航模式', '环境与电磁条件', '传感器/效应器状态', '保护对象与任务分配'];
  let sy = 2.42;
  for (const t of stateItems) {
    rect(slide, 0.9, sy, 2.38, 0.5, '1B2A35', '344652', 0.05);
    addText(slide, t, 1.08, sy + 0.08, 2.0, 0.28, { fontSize: 11.5, color: C.white, bold: true });
    sy += 0.68;
  }
  line(slide, 3.85, 3.95, 0.65, 0, C.cyan, 1.8, true);
  rect(slide, 4.65, 1.55, 3.75, 4.9, C.white, 'DCE3E9', 0.08);
  addText(slide, 'OBSERVATION  Oᵢ', 4.95, 1.85, 2.6, 0.3, { fontFace: 'Aptos', fontSize: 12, color: C.blue, bold: true });
  const obs = [
    ['测量', '带噪位置/速度/信号'],
    ['信念', '协方差、类别与意图概率'],
    ['时效', 'AoI、更新率、通信时延'],
    ['自身', '覆盖、弹药、能量、冷却'],
    ['协同', '邻居消息与当前任务'],
  ];
  let oy = 2.38;
  for (const [h, b] of obs) {
    addText(slide, h, 4.95, oy, 0.8, 0.28, { fontSize: 12, color: C.blue, bold: true });
    addText(slide, b, 5.7, oy, 2.25, 0.28, { fontSize: 10.5, color: C.muted });
    line(slide, 4.95, oy + 0.38, 2.85, 0, 'E2E7EB', 0.6, false);
    oy += 0.72;
  }
  line(slide, 8.62, 3.95, 0.65, 0, C.cyan, 1.8, true);
  rect(slide, 9.42, 1.55, 3.25, 4.9, C.cyanSoft, C.cyan, 0.08);
  addText(slide, 'JOINT ACTION  Aₜ', 9.72, 1.85, 2.35, 0.3, { fontFace: 'Aptos', fontSize: 12, color: C.cyan, bold: true });
  node(slide, 9.72, 2.36, 2.55, 0.78, 'S', '搜索/跟踪', 'sensor action', C.blue, C.white);
  node(slide, 9.72, 3.31, 2.55, 0.78, 'J', '干扰/欺骗', 'jammer action', C.amber, C.white);
  node(slide, 9.72, 4.26, 2.55, 0.78, 'K', '拦截/中止', 'interceptor action', C.red, C.white);
  node(slide, 9.72, 5.21, 2.55, 0.78, 'C', '分配/授权', 'command action', C.green, C.white);
  footer(slide, '本页为推荐的异构 Dec-POMDP 环境架构', 13);
  note(slide, '这页是建模边界。真实状态用于环境转移和集中 critic；局部观测用于每个 actor。两者不能完全相同，否则部分可观测问题被消失。');
}

function addAgents() {
  const slide = pptx.addSlide('MASTER');
  title(slide, '13 · AGENT DESIGN', '第一版建议：资源节点即 agent', '先学习高层编组与任务分配，底层信号处理和飞行控制由规则模型执行');
  const agents = [
    ['SENSOR', 'Sensor agent', '雷达 / 光电 / 被动 RF\n或复合探测节点', '观测：局部测量、航迹、环境\n动作：搜索、跟踪、模式切换', C.blue, C.blueSoft],
    ['JAMMER', 'Jammer agent', '固定 / 机动干扰设备', '观测：链路信念、频谱、距离\n动作：目标、模式、功率、时长', C.amber, C.amberSoft],
    ['INTERCEPT', 'Interceptor agent', '导弹 / 网捕 UAV / 火力单元', '观测：航迹质量、射界、弹药\n动作：交战、保持、中止', C.red, C.redSoft],
    ['COMMAND', 'Command agent', '可选的编组指挥节点', '观测：融合航迹、资源摘要\n动作：分配、授权、重编组', C.green, C.greenSoft],
  ];
  let x = 0.57;
  for (const [tag, h, entity, detail, color, fill] of agents) {
    rect(slide, x, 1.58, 2.95, 4.55, fill, color, 0.08);
    pill(slide, tag, x + 0.24, 1.86, 1.15, color, C.white);
    addIconLabel(slide, x + 0.24, 2.43, 0.68, tag[0], color, C.white);
    addText(slide, h, x + 1.08, 2.38, 1.7, 0.62, { fontSize: 16, color, bold: true });
    addText(slide, '实体含义', x + 0.25, 3.32, 0.82, 0.24, { fontSize: 10, color: C.muted, bold: true });
    addText(slide, entity, x + 0.25, 3.62, 2.35, 0.66, { fontSize: 13, color: C.ink, bold: true, valign: 'top' });
    line(slide, x + 0.25, 4.48, 2.4, 0, color, 0.8, false);
    addText(slide, detail, x + 0.25, 4.75, 2.35, 0.96, { fontSize: 10.7, color: C.muted, valign: 'top' });
    x += 3.15;
  }
  rect(slide, 0.7, 6.38, 12.0, 0.38, C.graphite, C.graphite, 0.04);
  addText(slide, '若研究重点只是全局分配，可只保留 Command agent；若研究异构协同，则使用前三类资源 agent。', 0.92, 6.45, 11.55, 0.23, { fontSize: 12, color: C.white, bold: true, align: 'center' });
  footer(slide, '本页为基于论文技术分类的 agent 粒度建议', 14);
  note(slide, 'Agent 粒度要与研究问题一致。当前课题更适合把装备资源作为 agent，研究异构资源协同；不建议第一版同时学习底层飞控、波束控制和高层任务分配。');
}

function addStateObservation() {
  const slide = pptx.addSlide('MASTER');
  title(slide, '14 · STATE & OBSERVATION', '状态描述真值，观测描述“我知道什么”', '置信度、信息年龄与误差协方差，是部分可观测性的核心');
  rect(slide, 0.65, 1.55, 5.95, 4.75, C.graphite, C.graphite, 0.08);
  addText(slide, 'GLOBAL STATE', 0.98, 1.83, 2.1, 0.3, { fontFace: 'Aptos', fontSize: 12, color: C.cyan, bold: true });
  const states = [
    ['威胁', '位置、速度、意图、载荷、RCS、链路/导航模式'],
    ['保护对象', '位置、价值、脆弱度、禁射/禁干扰区'],
    ['传感器', '覆盖、模式、误差、更新率、能耗、健康'],
    ['效应器', '射程、弹药、功率、频段、冷却、容量'],
    ['环境/网络', '天气、杂波、遮挡、时延、丢包、邻接'],
  ];
  let y = 2.35;
  for (const [h, b] of states) {
    addText(slide, h, 0.98, y, 1.0, 0.27, { fontSize: 12.5, color: C.cyan, bold: true });
    addText(slide, b, 2.02, y, 4.0, 0.3, { fontSize: 10.3, color: 'D5E0E7' });
    line(slide, 0.98, y + 0.4, 5.15, 0, '354650', 0.6, false);
    y += 0.72;
  }
  rect(slide, 6.9, 1.55, 5.78, 4.75, C.white, 'DCE3E9', 0.08);
  addText(slide, 'LOCAL OBSERVATION', 7.22, 1.83, 2.45, 0.3, { fontFace: 'Aptos', fontSize: 12, color: C.blue, bold: true });
  const obs = [
    ['估计值', '目标位置/速度估计，而非真实状态'],
    ['不确定性', '协方差、类别概率、意图概率'],
    ['信息时效', 'AoI、最近更新时间与通信延迟'],
    ['可见范围', '仅包含覆盖内目标与邻居消息'],
    ['自身资源', '自身弹药、能量、模式和当前任务'],
  ];
  y = 2.35;
  for (const [h, b] of obs) {
    circle(slide, 7.22, y + 0.02, 0.28, C.blueSoft, C.blue, 1);
    addText(slide, '•', 7.22, y, 0.28, 0.28, { fontSize: 12, color: C.blue, bold: true, align: 'center' });
    addText(slide, h, 7.7, y, 1.05, 0.27, { fontSize: 12.5, color: C.blue, bold: true });
    addText(slide, b, 8.85, y, 3.25, 0.3, { fontSize: 10.3, color: C.muted });
    line(slide, 7.22, y + 0.4, 4.95, 0, 'E2E7EB', 0.6, false);
    y += 0.72;
  }
  rect(slide, 2.85, 6.48, 7.65, 0.3, C.redSoft, C.redSoft, 0.04);
  addText(slide, '错误示范：把目标真实坐标、类型、意图和链路模式直接提供给所有 actor。', 3.0, 6.5, 7.35, 0.24, { fontSize: 11.5, color: C.red, bold: true, align: 'center' });
  footer(slide, '本页为推荐的状态/观测字段划分', 15);
  note(slide, '真实目标属性可以存在于仿真器和集中 critic 中，但不能全部泄露给 actor。局部观测需要携带不确定性和信息时效，这样策略才能学习何时继续确认、何时授权处置。');
}

function addActionsMasks() {
  const slide = pptx.addSlide('MASTER');
  title(slide, '15 · ACTION SPACE', '从离散分配开始，再逐步扩展到混合动作', '动作空间复杂度应与研究阶段匹配');
  const stages = [
    [0.7, 'STAGE 1', '离散任务分配', 'search sector k\ntrack / jam / engage target j\nhold / abort / reassign', C.blue, C.blueSoft],
    [4.55, 'STAGE 2', '参数化离散动作', '目标 + 模式\n目标 + 功率档位\n目标 + 发射窗口/弹种', C.cyan, C.cyanSoft],
    [8.4, 'STAGE 3', '离散 + 连续混合', '离散目标选择\n连续波束/功率\n连续速度、航向与欺骗轨迹', C.violet, C.violetSoft],
  ];
  for (const [x, tag, h, b, color, fill] of stages) {
    rect(slide, x, 1.55, 3.25, 2.15, fill, color, 0.08);
    pill(slide, tag, x + 0.25, 1.82, 1.05, color, C.white);
    addText(slide, h, x + 0.25, 2.3, 2.5, 0.4, { fontSize: 19, color, bold: true });
    addText(slide, b, x + 0.25, 2.83, 2.55, 0.66, { fontFace: 'Aptos', fontSize: 11.5, color: C.ink, valign: 'top' });
  }
  addText(slide, '动作掩码示例', 0.72, 4.08, 2.1, 0.36, { fontSize: 19, bold: true });
  rect(slide, 0.72, 4.58, 12.0, 1.7, C.graphite, C.graphite, 0.08);
  addText(slide, 'mask(jam, j)', 1.02, 4.88, 2.05, 0.3, { fontFace: 'Cambria Math', fontSize: 16, color: C.cyan, bold: true });
  addText(slide, '= link_activeⱼ ∧ band_knownⱼ ∧ in_rangeⱼ', 3.0, 4.88, 8.5, 0.3, { fontFace: 'Cambria Math', fontSize: 15, color: C.white });
  addText(slide, 'mask(intercept, j)', 1.02, 5.38, 2.3, 0.3, { fontFace: 'Cambria Math', fontSize: 16, color: C.red, bold: true });
  addText(slide, '= track_qualityⱼ ≥ τ ∧ firing_solutionⱼ ∧ ammo_available', 3.0, 5.38, 8.9, 0.3, { fontFace: 'Cambria Math', fontSize: 15, color: C.white });
  addText(slide, 'mask(spoof, j)', 1.02, 5.88, 2.05, 0.3, { fontFace: 'Cambria Math', fontSize: 16, color: C.amber, bold: true });
  addText(slide, '= uses_GNSSⱼ ∧ spoofableⱼ ∧ collateral_risk ≤ limit', 3.0, 5.88, 8.9, 0.3, { fontFace: 'Cambria Math', fontSize: 15, color: C.white });
  footer(slide, '动作层级与掩码为本研究环境设计建议', 16);
  note(slide, '第一版先做离散目标分配，便于验证场景和奖励是否正确。动作掩码用于屏蔽物理不可能和规则不允许的动作，但情报不足造成的误判可以保留为随机失败。');
}

function addRewardRoadmap() {
  const slide = pptx.addSlide('MASTER');
  title(slide, '16 · REWARD & ROADMAP', '奖励首先保护目标，其次控制误击与附带风险', '过程塑形不能压过任务成败');
  rect(slide, 0.65, 1.52, 6.15, 4.92, C.graphite, C.graphite, 0.08);
  addText(slide, 'TEAM REWARD', 0.98, 1.82, 2.0, 0.3, { fontFace: 'Aptos', fontSize: 12, color: C.cyan, bold: true });
  addText(slide, 'Rₜ =', 0.98, 2.35, 0.8, 0.4, { fontFace: 'Cambria Math', fontSize: 24, color: C.white, bold: true });
  const terms = [
    ['+', '保护对象存活 / 安全处置 / 威胁驱离', C.green],
    ['+', '航迹质量提升与及时发现', C.cyan],
    ['−', '保护对象损伤与漏防', C.red],
    ['−', '误击、坠落与电磁附带影响', C.red],
    ['−', '弹药、能量、频谱和计算消耗', C.amber],
    ['−', '重复分配、切换和决策时延', C.violet],
  ];
  let y = 2.35;
  for (const [sign, t, color] of terms) {
    addText(slide, sign, 1.75, y, 0.35, 0.3, { fontFace: 'Cambria Math', fontSize: 18, color, bold: true, align: 'center' });
    addText(slide, t, 2.18, y, 3.95, 0.3, { fontSize: 12.3, color: C.white, bold: true });
    y += 0.53;
  }
  rect(slide, 0.98, 5.75, 5.45, 0.4, '3B2527', '3B2527', 0.04);
  addText(slide, '优先级：目标损伤 > 附带损伤 > 处置效果 > 资源成本 > 过程奖励', 1.12, 5.82, 5.15, 0.23, { fontSize: 10.7, color: 'F4C9C9', bold: true, align: 'center' });

  addText(slide, '推荐实施路线', 7.15, 1.72, 2.2, 0.38, { fontSize: 20, bold: true });
  const roadmap = [
    ['01', '规则物理 + 离散分配', '先验证威胁、资源、动作掩码与终止条件', C.blue, C.blueSoft],
    ['02', '加入观测误差与融合航迹', '从全局状态过渡到局部观测和信念状态', C.cyan, C.cyanSoft],
    ['03', '加入异构 MARL 与动态重编组', '比较 MAPPO、HAPPO/HARL 和规则基线', C.green, C.greenSoft],
  ];
  y = 2.35;
  for (const [tag, h, b, color, fill] of roadmap) {
    node(slide, 7.15, y, 5.1, 1.05, tag, h, b, color, fill);
    y += 1.3;
  }
  rect(slide, 7.15, 6.08, 5.1, 0.42, C.cyanSoft, C.cyanSoft, 0.04);
  addText(slide, '先做“可信环境”，再追求“复杂算法”', 7.35, 6.15, 4.7, 0.24, { fontSize: 13, color: C.cyan, bold: true, align: 'center' });
  footer(slide, '结论：论文技术分类 + HAPPO/HARL，可形成反 UAV 异构 MARL 环境', 17);
  note(slide, '最后收束到实施路线。第一步先把环境做对，用规则模型执行底层物理；第二步加入部分可观测和融合误差；第三步再比较异构 MARL 算法。');
}

async function main() {
  await cropAssets();
  const builders = [
    addCover,
    addPositioning,
    addThreats,
    addLoop,
    addDetectionTaxonomy,
    addDetectionTable,
    addFusion,
    addMitigationTaxonomy,
    addMitigationTable,
    addChallenges,
    addFutureFramework,
    addWhyRL,
    addDecPomdp,
    addAgents,
    addStateObservation,
    addActionsMasks,
    addRewardRoadmap,
  ];
  const requestedLimit = Number(process.env.SLIDE_LIMIT || builders.length);
  const limit = Math.max(1, Math.min(builders.length, requestedLimit));
  builders.slice(0, limit).forEach((build) => build());

  const outPath = process.env.PPT_OUT
    ? path.resolve(process.env.PPT_OUT)
    : path.join(readerDir, 'P05_Counter_UAS_反UAV环境建模精读汇报.pptx');
  await pptx.writeFile({ fileName: outPath });
  console.log(outPath);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
