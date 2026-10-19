// Renders the award-redesign high-fidelity review frames from ./fixture.json (demo data, never a
// real run) into a self-contained ./index.html. Usage: node build.mjs (needs the app's installed d3
// and fonts; no network). ./index.html?frame=<id> shows one frame at native size for capture.
// Intentional truncation carries data-trunc plus a title with the full text; horizontally
// scrollable code carries data-scroll-x, tabindex=0 and an aria-label (checked by capture).
import { readFileSync, writeFileSync } from 'node:fs'
import * as d3 from '../../../../app/node_modules/d3/src/index.js'

const here = (path) => new URL(path, import.meta.url)
const fixture = JSON.parse(readFileSync(here('./fixture.json'), 'utf8'))
const font = (path) => readFileSync(here(`../../../../app/node_modules/${path}`)).toString('base64')
const INTER = font('@fontsource-variable/inter/files/inter-latin-wght-normal.woff2')
const PLEX400 = font('@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-400-normal.woff2')
const PLEX500 = font('@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-500-normal.woff2')

// ======================= tokens (design.md › 颜色) =======================
const THEMES = {
  dark: { bg: '#0a0a0b', rail: '#0e0e10', raised: '#16161a', hover: '#1c1c21', selected: '#232329', line: 'rgba(255,255,255,.08)', line2: 'rgba(255,255,255,.16)', text: '#ededef', muted: '#a1a1aa', faint: '#8a8a93', signal: '#f2622e', signalG: '#f2622e', stroke: '#6d6d76', grid: '#34343b', ramp0: '#606066', ramp1: '#f1f1f3' },
  light: { bg: '#ffffff', rail: '#f6f6f7', raised: '#ffffff', hover: '#efeff1', selected: '#e7e7ea', line: 'rgba(0,0,0,.08)', line2: 'rgba(0,0,0,.16)', text: '#18181b', muted: '#52525b', faint: '#66666f', signal: '#b83e0b', signalG: '#e3550c', stroke: '#82828b', grid: '#d4d4d8', ramp0: '#8d8d90', ramp1: '#18181b' },
}
const themeCss = (name) => `.f[data-theme="${name}"]{color-scheme:${name};${Object.entries(THEMES[name]).map(([k, v]) => `--${k}:${v}`).join(';')}}`

// ======================= copy (zh / en) =======================
const STATUS = {
  zh: { proposed: '待跑', selected: '已选', running: '运行中', challenged: '待判定', supported: '成立', refuted: '否定', inconclusive: '未定', confirmed: '已确认', unconfirmed: '未确认', dropped: '放弃' },
  en: { proposed: 'queued', selected: 'selected', running: 'running', challenged: 'challenged', supported: 'supported', refuted: 'refuted', inconclusive: 'inconclusive', confirmed: 'confirmed', unconfirmed: 'unconfirmed', dropped: 'dropped' },
}
const STATE = { zh: { active: '进行中', concluded: '已结论', new: '新建', 'over-budget': '超预算' }, en: { active: 'active', concluded: 'concluded', new: 'new', 'over-budget': 'over budget' } }
const EVENT = {
  zh: { scaffold: '搭建实验', run: '运行完成', 'run-failed': '运行失败', skeptic: '质疑检验', verdict: '判词', probe: '试探', confirm: '封存段确认', lockbox: '封存段', hypothesis: '提出假设', note: '备注' },
  en: { scaffold: 'scaffold', run: 'run finished', 'run-failed': 'run failed', skeptic: 'skeptic check', verdict: 'verdict', probe: 'probe', confirm: 'lockbox confirm', lockbox: 'lockbox', hypothesis: 'hypothesis', note: 'note' },
}
const T = {
  zh: {
    directions: '研究方向', runs: '全部运行', library: '资料库', directionsShort: '方向', runsShort: '运行', libraryShort: '资料库', nav: '主导航',
    search: '搜索方向、运行', railDirections: '方向', demoNote: '演示数据，非真实研究', updated: '最后更新', theme: '主题', language: '语言', dark: '深色', light: '浅色',
    trials: '试验', failed: '评估失败', failedShort: '失败', hypotheses: '假设', direction: '方向', evidence: '证据', budget: '预算', lockbox: '封存段',
    best: '最佳', noise: '噪声线', clears: '越过', noTrials: '尚无试验', sealed: '封存', opened: '已开', budgetUnset: '未设预算', activity: '动态', allDirections: '全部方向', log: '日志原文',
    hypothesisKey: '假设状态', scaleKey: (ticks) => `证据各行共用一把刻度（${ticks}）：菱形是最佳记录，竖线是当时的噪声线，橙色段表示最佳记录越过噪声线的部分。`,
    sentence: (n, conf, act, over, fresh) => `${n} 个研究方向：<a href="#d2">${conf} 个已确认结论</a>，<a href="#d1">${act} 个进行中</a>，<a class="is-signal" href="#d4">${over} 个超出预算</a>，<a href="#d3">${fresh} 个尚未开始</a>。`,
    subline: (trials, spend, budget, above, withTrials) => `共 ${trials} 次试验，花费 ${spend}，预算 ${budget}；有试验的 ${withTrials} 个方向中，${above} 个的最佳记录越过噪声线。`,
    kTrials: '试验', kFailed: '评估失败', kSpend: '花费 / 预算', kHyp: '假设', kHypNote: (ch, run) => `待判定 ${ch} · 运行中 ${run}`, kFailNote: (rate) => `占试验 ${rate}`, kDirs: (n) => `${n} 个方向`,
    hypCount: (n) => `${n} 个假设`, trialsN: (n) => `${n} 次试验`,
    created: '创建', claim: '主张', mechanism: '机制', falsification: '证伪条件', trialsAndNoise: '试验与噪声线', ledger: '检查台账', verdict: '判词', experiments: '实验',
    check: '检查', result: '结果', threshold: '阈值', outcome: '结论', pass: '通过', fail: '未通过', reference: '参考', openAnalysis: '进入分析', evaluated: '次评估', probed: '次试探',
    trialAxis: '试验序号', scoreAxis: '样本内得分', thisHyp: '本假设试验', otherHyp: '其他假设试验', record: '最佳记录', failBand: '失败',
    lockboxConfirm: '封存段确认', lockboxInfo: (start, n, dev) => `封存段自 ${start} 起，共 ${n} 根 K 线（开发段 ${dev} 根），只在判词成立后开启一次。`,
    dsr: 'DSR', positiveSub: '验证段为正的子区间', pbo: 'PBO', vsBuyHold: '验证段 Sharpe 对买入持有', vsSeed: '验证段 Sharpe 对种子程序', trialsInDsr: '计入 DSR 的试验', ratio: '验证段 / 样本内', aboveBaseline: '高于基准',
    runSearch: '搜索运行名称或任务', all: '全部', ordinary: '普通', experiment: '实验', sortBy: '排序：最近更新', selectedN: (n) => `已选 ${n} 个运行`, compareSelected: '比较所选',
    run: '运行', kind: '类型', programs: '程序', bestScore: '最佳', cost: '费用', generations: '代', islands: '岛', updatedAt: '更新（UTC）', emptyRun: '空库',
    compareTitle: (n) => `比较 ${n} 个运行`, compareNote: '每行一个指标；右侧刻度只在本行内共用，三个运行用形状区分，数值在左侧列出。', addRun: '添加运行', failRate: '评估失败率', costPerProgram: '每个程序的费用', metric: '指标', scaleNote: '本行刻度',
    lineage: '谱系', progress: '进展', space: '解空间', contest: '对比', island: '岛', generation: '代', gen: (g) => `第 ${g} 代`, programsN: (n) => `${n} 个程序`, failedN: (n) => `${n} 失败`,
    laneNote: (n, f) => `${n} 个程序 · 失败 ${f}`, bestPath: '最佳祖先链', failedMark: '评估失败', scoreRamp: '得分', replay: '回放', scrub: (g, n) => `显示到第 ${g} 代 · ${n} 个程序`, edgesNote: '灰线为同岛父子；跨岛迁移只在选中程序时画出',
    promptTrack: '提示词版本', metaTrack: '元分析更新', metaUpdate: (k, n) => `第 ${k} 次更新 · 已处理 ${n} 个程序`, metaRecs: '建议', metaInsights: '全局洞察',
    inspector: '程序检查器', code: '代码', diff: '差异', evaluation: '评估', llm: 'LLM 记录', meta: '元分析', parent: '父代', vsParent: '较父代', download: '下载纯文本', downloadCode: (file) => `下载 ${file}`, lines: (n) => `${n} 行`, wrap: '换行', off: '关', codeScroll: '代码，可横向滚动',
    cumBest: '累计最佳', median: '中位数', perGenBest: '本代最佳', costSplit: '费用', modelCost: '模型调用', metaCost: '提示词进化', failures: '评估失败', throughput: '吞吐（程序 / 代）',
    models: '按模型', model: '模型', success: '成功率', mean: '平均', diagnostics: '诊断 · 11 张分析小图', expand: '展开', feedback: '评估反馈', reasoning: '模型思路',
    density: '密度等高线', clusters: '聚类', cluster: (k) => `簇 ${k}`, d2: '二维', d3: '三维', leaderboard: '排行榜', pareto: '帕累托前沿', paretoAxis: '费用（对数）', frontier: '前沿', rank: '名次', program: '程序',
    duel: '模型对战', duelNote: '每格是行模型的有效程序严格胜过列模型有效程序的比例，平局不算胜；下方小字为比较对数，失败程序不参与。', duelDist: '有效得分分布（按中位数排序）', ensemble: '集成', ensembleNA: '此演示数据不可用：演示运行没有集成结果。', diversity: '分布',
    posterior: '模型后验', posteriorNA: '此演示数据不可用：演示运行没有记录 model_posteriors；上表是按程序计算的成功率与得分，不是后验。',
    back: '返回', pullUp: '上拉查看完整标签页',
    statesTitle: '空与错误状态', statesNote: '每个状态取自演示数据或服务端真实返回的格式；状态都用形状加文字表达。',
    emptyWorkspace: '还没有研究方向', emptyWorkspaceHint: '在终端创建第一个方向，刷新后出现在这里：', unreadable: '研究记录无法读取（OSError）', unreadableHint: '其余方向照常显示；修复后刷新即可。',
    newDirection: '新方向尚无试验', emptyRunTitle: '运行为空', emptyRunHint: '数据库存在，但还没有任何程序。', unreadableRun: '数据库无法读取', unreadableRunHint: '服务返回 HTTP 500；其他运行不受影响。',
    costUnknown: '费用未记录', costUnknownHint: '只要有一个程序缺少费用字段，合计就显示“—”，不按 0 计算。', evalFailed: '评估失败', hiddenValidation: '判词前隐藏验证段', hiddenValidationHint: '质疑检验已完成但判词未出，验证段数字保持隐藏。', oosSharpe: '验证段 Sharpe',
    cmdDir: '目录', cmdQuestion: '研究问题',
  },
  en: {
    directions: 'Research', runs: 'All runs', library: 'Library', directionsShort: 'Research', runsShort: 'Runs', libraryShort: 'Library', nav: 'Main navigation',
    search: 'Search', railDirections: 'Directions', demoNote: 'Demo data, not real research', updated: 'Updated', theme: 'Theme', language: 'Language', dark: 'Dark', light: 'Light',
    trials: 'Trials', failed: 'Failed evaluations', failedShort: 'failed', hypotheses: 'Hypotheses', direction: 'Direction', evidence: 'Evidence', budget: 'Budget', lockbox: 'Lockbox',
    best: 'best', noise: 'noise', clears: 'clears by', noTrials: 'No trials yet', sealed: 'Sealed', opened: 'Opened', budgetUnset: 'No budget', activity: 'Activity', allDirections: 'All directions', log: 'Log text',
    hypothesisKey: 'Hypothesis status', scaleKey: (ticks) => `Every row shares one evidence scale (${ticks}): the diamond is the best record, the bar is the noise line at that time, and orange marks how far the record clears it.`,
    sentence: (n, conf, act, over, fresh) => `${n} research directions: <a href="#d2">${conf} confirmed</a>, <a href="#d1">${act} in progress</a>, <a class="is-signal" href="#d4">${over} over budget</a>, <a href="#d3">${fresh} not started</a>.`,
    subline: (trials, spend, budget, above, withTrials) => `${trials} trials, ${spend} spent of ${budget} budgeted; ${above} of the ${withTrials} directions with trials have a best record above their noise line.`,
    kTrials: 'Trials', kFailed: 'Failed evaluations', kSpend: 'Spend / budget', kHyp: 'Hypotheses', kHypNote: (ch, run) => `${ch} challenged · ${run} running`, kFailNote: (rate) => `${rate} of trials`, kDirs: (n) => `${n} directions`,
    hypCount: (n) => (n === 1 ? '1 hypothesis' : `${n} hypotheses`), trialsN: (n) => (n === '1' ? '1 trial' : `${n} trials`),
    created: 'Created', claim: 'Claim', mechanism: 'Mechanism', falsification: 'Falsification', trialsAndNoise: 'Trials and noise line', ledger: 'Check ledger', verdict: 'Verdict', experiments: 'Experiments',
    check: 'Check', result: 'Result', threshold: 'Threshold', outcome: 'Outcome', pass: 'Pass', fail: 'Fail', reference: 'Reference', openAnalysis: 'Open analysis', evaluated: 'evaluated', probed: 'probed',
    trialAxis: 'Trial', scoreAxis: 'In-sample score', thisHyp: 'This hypothesis', otherHyp: 'Other hypotheses', record: 'Best record', failBand: 'Failed',
    lockboxConfirm: 'Lockbox confirmation', lockboxInfo: (start, n, dev) => `Lockbox from ${start}: ${n} bars (${dev} development bars), opened once and only after a supported verdict.`,
    dsr: 'DSR', positiveSub: 'Positive OOS sub-periods', pbo: 'PBO', vsBuyHold: 'OOS Sharpe vs buy-and-hold', vsSeed: 'OOS Sharpe vs seed program', trialsInDsr: 'Trials counted in DSR', ratio: 'OOS / in-sample', aboveBaseline: 'above baseline',
    runSearch: 'Search run name or task', all: 'All', ordinary: 'Ordinary', experiment: 'Experiment', sortBy: 'Sort: last updated', selectedN: (n) => `${n} runs selected`, compareSelected: 'Compare selected',
    run: 'Run', kind: 'Kind', programs: 'Programs', bestScore: 'Best', cost: 'Cost', generations: 'Gens', islands: 'Islands', updatedAt: 'Updated (UTC)', emptyRun: 'Empty',
    compareTitle: (n) => `Compare ${n} runs`, compareNote: 'One metric per row; each row has its own scale, runs differ by shape, and values are listed on the left.', addRun: 'Add run', failRate: 'Failure rate', costPerProgram: 'Cost per program', metric: 'Metric', scaleNote: 'Row scale',
    lineage: 'Lineage', progress: 'Progress', space: 'Space', contest: 'Contest', island: 'Island', generation: 'Generation', gen: (g) => `gen ${g}`, programsN: (n) => `${n} programs`, failedN: (n) => `${n} failed`,
    laneNote: (n, f) => `${n} programs · ${f} failed`, bestPath: 'Best ancestry', failedMark: 'Failed evaluation', scoreRamp: 'Score', replay: 'Replay', scrub: (g, n) => `Up to gen ${g} · ${n} programs`, edgesNote: 'Grey lines link parents on the same island; migrations appear for the selected program',
    promptTrack: 'Prompt versions', metaTrack: 'Meta updates', metaUpdate: (k, n) => `Update ${k} · ${n} programs processed`, metaRecs: 'Recommendations', metaInsights: 'Global insights',
    inspector: 'Program inspector', code: 'Code', diff: 'Diff', evaluation: 'Evaluation', llm: 'LLM', meta: 'Meta', parent: 'parent', vsParent: 'vs parent', download: 'Download text', downloadCode: (file) => `Download ${file}`, lines: (n) => `${n} lines`, wrap: 'Wrap', off: 'off', codeScroll: 'Code, scrolls horizontally',
    cumBest: 'Cumulative best', median: 'Median', perGenBest: 'Generation best', costSplit: 'Cost', modelCost: 'Model calls', metaCost: 'Prompt evolution', failures: 'Failed evaluations', throughput: 'Throughput (programs / gen)',
    models: 'By model', model: 'Model', success: 'Success', mean: 'Mean', diagnostics: 'Diagnostics · 11 analysis plots', expand: 'Expand', feedback: 'Feedback', reasoning: 'Model reasoning',
    density: 'Density contours', clusters: 'Clusters', cluster: (k) => `C${k}`, d2: '2D', d3: '3D', leaderboard: 'Leaderboard', pareto: 'Pareto front', paretoAxis: 'Cost (log)', frontier: 'Front', rank: '#', program: 'Program',
    duel: 'Model battle', duelNote: 'Each cell is the share of pairs in which a valid program of the row model scores strictly higher than one of the column model; ties are not wins. The small figure is the number of pairs; failed programs take no part.', duelDist: 'Valid score distribution (by median)', ensemble: 'Ensemble', ensembleNA: 'Not available in this demo data: the demo runs have no ensemble results.', diversity: 'Distribution',
    posterior: 'Model posteriors', posteriorNA: 'Not available in this demo data: the demo runs record no model_posteriors; the table above gives per-program success and scores, not posteriors.',
    back: 'Back', pullUp: 'Pull up for all tabs',
    statesTitle: 'Empty and error states', statesNote: 'Each state uses demo data or the server’s real response format; every state has a shape and a word.',
    emptyWorkspace: 'No research directions yet', emptyWorkspaceHint: 'Create the first direction in a terminal; it appears here after a refresh:', unreadable: 'Research record unreadable (OSError)', unreadableHint: 'Other directions still show; refresh after fixing it.',
    newDirection: 'New direction without trials', emptyRunTitle: 'Empty run', emptyRunHint: 'The database exists but holds no programs yet.', unreadableRun: 'Database unreadable', unreadableRunHint: 'The server returned HTTP 500; other runs are unaffected.',
    costUnknown: 'Cost not recorded', costUnknownHint: 'If any program lacks cost fields the total shows “—” instead of counting it as zero.', evalFailed: 'Evaluation failed', hiddenValidation: 'Validation hidden before the verdict', hiddenValidationHint: 'The skeptic check ran but no verdict is recorded, so validation numbers stay hidden.', oosSharpe: 'OOS Sharpe',
    cmdDir: 'dir', cmdQuestion: 'research question',
  },
}

// ======================= formats (design.md › 字体与数字) =======================
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c])
const int = (n) => (n === null || n === undefined ? '—' : Math.round(n).toLocaleString('en-US'))
const num = (n, d = 2) => (n === null || n === undefined || Number.isNaN(n) ? '—' : (Math.abs(n) < 0.5 * 10 ** -d ? 0 : Number(n)).toFixed(d))
const signed = (n, d = 2) => (n === null || n === undefined ? '—' : `${n >= 0 ? '+' : '−'}${num(Math.abs(n), d)}`)
const money = (n) => (n === null || n === undefined ? '—' : `$${Number(n).toFixed(2)}`)
const pct = (n, d = 0) => (n === null || n === undefined ? '—' : `${(n * 100).toFixed(d)}%`)
const clock = (iso) => (iso ? new Date(iso).toISOString().slice(0, 16).replace('T', ' ') : '—')
const epochClock = (s) => (s ? clock(new Date(s * 1000).toISOString()) : '—')
const hasCjk = (text) => /[\u3400-\u9fff]/.test(text)
const logText = (text, cls = 'log') => `<code class="${cls}"${hasCjk(text) ? '' : ' lang="en"'}${cls === 'log' ? ` data-trunc title="${esc(text)}"` : ''}>${esc(text)}</code>`
const trunc = (text, cls = '') => `<span class="trunc ${cls}" data-trunc title="${esc(text)}">${esc(text)}</span>`
const pathBreaks = (text) => esc(text).replace(/\//g, '/<wbr>')

// ======================= data =======================
const dirs = fixture.directions.filter((d) => !d.error)
const dirById = new Map(dirs.map((d) => [d.id, d]))
const shortTitle = (d) => d.title.replace(/[（(].*?[)）]/g, '').trim()
const UPDATED = d3.max([...fixture.events.map((e) => e.time), ...dirs.map((d) => d.evidence.lastActivity)].filter(Boolean))
const totals = {
  trials: d3.sum(dirs, (d) => d.trials), spend: d3.sum(dirs, (d) => d.spend), budget: d3.sum(dirs, (d) => d.budget ?? 0), failed: d3.sum(dirs, (d) => d.evidence.failed),
  hypotheses: d3.sum(dirs, (d) => d.hypotheses.length),
  challenged: d3.sum(dirs, (d) => d.hypotheses.filter((h) => h.status === 'challenged').length),
  running: d3.sum(dirs, (d) => d.hypotheses.filter((h) => h.status === 'running').length),
  withTrials: dirs.filter((d) => d.trials > 0).length,
  above: dirs.filter((d) => d.evidence.best !== null && d.evidence.bar !== null && d.evidence.best > d.evidence.bar).length,
  confirmed: dirs.filter((d) => d.hypotheses.some((h) => h.status === 'confirmed')).length,
  active: dirs.filter((d) => d.state === 'active').length, over: dirs.filter((d) => d.state === 'over-budget').length, fresh: dirs.filter((d) => d.state === 'new').length,
}
// One evidence scale for every direction, so best records and noise bars compare across rows.
const evDomain = (() => {
  const [lo, hi] = d3.extent(dirs.flatMap((d) => [d.evidence.best, d.evidence.bar]).filter((v) => v !== null))
  const margin = (hi - lo) * 0.12
  return [lo - margin, hi + margin]
})()
const evTicks = d3.ticks(...evDomain, 4)
const A = fixture.analysis
const rows = A.rows // [x, y, score|null, generation, island, parentIndex, patch, model, cost, cluster, patchType]
const pathSet = new Set(A.path)
const scored = rows.filter((r) => r[2] !== null)
const scoreExtent = d3.extent(scored, (r) => r[2])
const maxGen = d3.max(rows, (r) => r[3])
const best = A.best
const shortModel = (m) => m.replace(/^demo\//, '')

// ======================= glyphs =======================
const GLYPH = {
  supported: '<circle cx="7" cy="7" r="5.5" fill="currentColor"/>',
  confirmed: '<circle cx="7" cy="7" r="5.5" fill="currentColor"/><path d="M4.4 7.2l1.8 1.8 3.4-3.6" stroke="var(--bg)" stroke-width="1.5" fill="none"/>',
  refuted: '<circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M3.6 10.4l6.8-6.8" stroke="currentColor" stroke-width="1.4"/>',
  inconclusive: '<circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M7 2a5 5 0 0 1 0 10z" fill="currentColor"/>',
  running: '<circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" stroke-width="1.4" stroke-dasharray="2.2 1.9"/><circle cx="7" cy="7" r="1.8" fill="currentColor"/>',
  challenged: '<circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M7 4v3.4" stroke="currentColor" stroke-width="1.5"/><circle cx="7" cy="9.6" r=".9" fill="currentColor"/>',
  proposed: '<circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" stroke-width="1.3" stroke-dasharray="1 2.1"/>',
  selected: '<circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" stroke-width="1.3"/>',
  unconfirmed: '<circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M4.8 4.8l4.4 4.4m0-4.4l-4.4 4.4" stroke="currentColor" stroke-width="1.3"/>',
  dropped: '<circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" stroke-width="1.2" stroke-dasharray="3 2"/>',
  error: '<path d="M7 1.6l5.8 10.2H1.2z" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><path d="M7 5.4v3" stroke="currentColor" stroke-width="1.4"/><circle cx="7" cy="10" r=".8" fill="currentColor"/>',
  empty: '<rect x="2" y="2" width="10" height="10" rx="2" fill="none" stroke="currentColor" stroke-width="1.3" stroke-dasharray="2 1.6"/>',
}
const glyph = (key, label, cls = '') => `<svg class="glyph ${cls}" width="14" height="14" viewBox="0 0 14 14" role="img" aria-label="${esc(label)}">${GLYPH[key] ?? GLYPH.selected}</svg>`
const statusGlyph = (c, status) => glyph(status, STATUS[c.lang][status] ?? status)
const STATE_GLYPH = { active: 'running', concluded: 'confirmed', new: 'proposed', 'over-budget': 'challenged' }
const stateGlyph = (c, d) => glyph(STATE_GLYPH[d.state], STATE[c.lang][d.state], d.state === 'over-budget' ? 'is-signal' : '')
const ICON = {
  lock: '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><rect x="2.5" y="6" width="9" height="6.5" rx="1.2" fill="none" stroke="currentColor" stroke-width="1.3"/><path d="M4.5 6V4.3a2.5 2.5 0 0 1 5 0V6" fill="none" stroke="currentColor" stroke-width="1.3"/></svg>',
  unlock: '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><rect x="2.5" y="6" width="9" height="6.5" rx="1.2" fill="none" stroke="currentColor" stroke-width="1.3"/><path d="M4.5 6V4.3a2.5 2.5 0 0 1 4.9-.7" fill="none" stroke="currentColor" stroke-width="1.3"/></svg>',
  search: '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><circle cx="6" cy="6" r="4.2" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M9.2 9.2l3.3 3.3" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>',
  directions: '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M3 13.5V8.5m0 0c0-2.8 2-4 4.8-4M3 8.5c0 2.4 2.2 3.6 4.8 3.6M7.8 4.5h5.2M7.8 12.1H13" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/><circle cx="3" cy="13.5" r="1.3" fill="currentColor"/></svg>',
  runs: '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><rect x="2" y="3" width="12" height="2.6" rx=".6" fill="none" stroke="currentColor" stroke-width="1.3"/><rect x="2" y="6.8" width="12" height="2.6" rx=".6" fill="none" stroke="currentColor" stroke-width="1.3"/><rect x="2" y="10.6" width="12" height="2.6" rx=".6" fill="none" stroke="currentColor" stroke-width="1.3"/></svg>',
  library: '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M2.5 3.2c2-.6 3.8-.4 5.5.8 1.7-1.2 3.5-1.4 5.5-.8v9.6c-2-.6-3.8-.4-5.5.8-1.7-1.2-3.5-1.4-5.5-.8zM8 4v9.6" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/></svg>',
  play: '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 2l7 4-7 4z" fill="currentColor"/></svg>',
  arrow: '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M2 6h7.5M6.5 3l3 3-3 3" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  back: '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M10 3L5 8l5 5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  chevron: '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 4.5l3 3 3-3" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  check: '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M2.5 6.2l2.4 2.4 4.6-5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  cross: '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 3l6 6m0-6l-6 6" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
  download: '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M7 2v7m-3-3l3 3 3-3M2.5 11.5h9" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
}
const MARK = '<svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"><path d="M4 14.5V10m0 0c0-3 2.2-4 5-4m-5 4c0 2.6 2.4 4 5 4M9 6V3.5M9 6c2.8 0 5 1 5 4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><circle cx="9" cy="3.5" r="1.6" fill="currentColor"/><circle cx="14" cy="10" r="1.6" fill="currentColor"/><circle cx="9" cy="14" r="1.6" fill="currentColor"/></svg>'
// Event marks: shape and the event word together, never colour alone.
function eventMark(event) {
  const shapes = {
    'run-failed': '<path d="M3 3l6 6m0-6l-6 6" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>',
    verdict: '<path d="M6 1.5l4.5 4.5L6 10.5 1.5 6z" fill="currentColor"/>',
    skeptic: '<path d="M6 1.8l4.2 4.2L6 10.2 1.8 6z" fill="none" stroke="currentColor" stroke-width="1.3"/>',
    run: '<circle cx="6" cy="6" r="3.2" fill="currentColor"/>',
    scaffold: '<circle cx="6" cy="6" r="3.2" fill="none" stroke="currentColor" stroke-width="1.3"/>',
    probe: '<path d="M6 2.2l4 7H2z" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/>',
  }
  if (event === 'confirm' || event === 'lockbox') return `<span class="evmark">${ICON.unlock}</span>`
  return `<span class="evmark ${event === 'run-failed' ? 'is-signal' : ''}"><svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">${shapes[event] ?? shapes.run}</svg></span>`
}
// Run markers for multi-run comparison: shape, not colour, tells runs apart.
const MARKS = [
  (x, y, s = 5) => `<circle cx="${x}" cy="${y}" r="${s}" fill="var(--text)"/>`,
  (x, y, s = 5) => `<rect x="${x - s + 0.8}" y="${y - s + 0.8}" width="${2 * s - 1.6}" height="${2 * s - 1.6}" fill="var(--bg)" stroke="var(--text)" stroke-width="1.6"/>`,
  (x, y, s = 5) => `<path d="M${x},${y - s - 1}l${s + 1},${2 * s}h${-2 * (s + 1)}z" fill="var(--muted)"/>`,
]
const markIcon = (k, size = 12) => `<svg width="${size}" height="${size}" viewBox="0 0 14 14" aria-hidden="true">${MARKS[k](7, 7, 4.2)}</svg>`

// ======================= shell =======================
const seg = (options, on, label) => `<div class="seg" role="radiogroup" aria-label="${esc(label)}">${options.map((o, k) => `<span role="radio" aria-checked="${k === on}" class="${k === on ? 'is-on' : ''}">${esc(o)}</span>`).join('')}</div>`
function rail(c, active, { narrow = false, direction = null } = {}) {
  const nav = [['directions', dirs.length], ['runs', fixture.runs.length], ['library', fixture.workspace.library?.pages ?? 0]]
  if (narrow) {
    return `<aside class="rail is-narrow"><div class="brand" title="Shinka Research">${MARK}</div>
      <nav aria-label="${c.t('nav')}">${nav.map(([k]) => `<a class="navn ${k === active ? 'is-on' : ''}" ${k === active ? 'aria-current="page"' : ''}>${ICON[k]}<span>${c.t(k + 'Short')}</span></a>`).join('')}</nav>
      <div class="railfoot"><span class="demo" title="${c.t('demoNote')}">DEMO</span></div></aside>`
  }
  return `<aside class="rail"><div class="brand">${MARK}<span>Shinka Research</span></div>
    <div class="ws"><span class="mono">${esc(fixture.workspace.name)}</span><span class="demo">DEMO</span></div>
    <button class="search" type="button">${ICON.search}<span>${c.t('search')}</span><kbd>Ctrl K</kbd></button>
    <nav aria-label="${c.t('nav')}">${nav.map(([k, n]) => `<a class="nav ${k === active ? 'is-on' : ''}" ${k === active ? 'aria-current="page"' : ''}>${ICON[k]}<span>${c.t(k)}</span><span class="count">${int(n)}</span></a>`).join('')}</nav>
    <div class="railhead">${c.t('railDirections')}</div>
    ${dirs.map((d) => `<a class="railitem ${d.id === direction ? 'is-on' : ''}">${stateGlyph(c, d)}${trunc(shortTitle(d))}</a>`).join('')}
    <div class="railfoot"><span class="demo">DEMO</span><span>${c.t('demoNote')}</span></div></aside>`
}
function topbar(c, crumbs, { tabs = null, tab = null, extra = '' } = {}) {
  return `<header class="top"><nav class="crumbs" aria-label="breadcrumb">${crumbs.map((x, i) => (i === crumbs.length - 1 ? `<span class="is-here">${esc(x)}</span>` : trunc(x))).join('<span class="sep" aria-hidden="true">/</span>')}</nav>
    ${tabs ? `<div class="vtabs" role="tablist">${tabs.map(([k, label]) => `<span role="tab" aria-selected="${k === tab}" class="${k === tab ? 'is-on' : ''}">${esc(label)}</span>`).join('')}</div>` : ''}
    <div class="tools">${extra}<span class="updated">${c.t('updated')} <span class="mono">${clock(UPDATED)}</span> UTC</span>
      ${seg([c.t('dark'), c.t('light')], c.theme === 'dark' ? 0 : 1, c.t('theme'))}${seg(['中', 'EN'], c.lang === 'zh' ? 0 : 1, c.t('language'))}</div></header>`
}

// ======================= home =======================
function gauge(c, d, w) {
  const e = d.evidence
  if (e.best === null || e.bar === null) return `<div class="gauge is-empty">${c.t('noTrials')}</div>`
  const x = d3.scaleLinear(evDomain, [5, w - 5])
  const above = e.best > e.bar
  return `<div class="gauge"><svg width="${w}" height="20" viewBox="0 0 ${w} 20" role="img" aria-label="${c.t('best')} ${num(e.best)} · ${c.t('noise')} ${num(e.bar)}">
    <line x1="5" x2="${w - 5}" y1="10" y2="10" stroke="var(--line2)"/>
    ${evTicks.map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="7" y2="13" stroke="var(--stroke)"/>`).join('')}
    <line x1="${x(Math.min(e.bar, e.best))}" x2="${x(Math.max(e.bar, e.best))}" y1="10" y2="10" stroke="${above ? 'var(--signalG)' : 'var(--muted)'}" stroke-width="2.5" ${above ? '' : 'stroke-dasharray="2 2"'}/>
    <rect x="${x(e.bar) - 1}" y="2" width="2" height="16" fill="var(--muted)"/>
    <path d="M${x(e.best)},4l5.5,6-5.5,6-5.5-6z" fill="${above ? 'var(--signalG)' : 'var(--bg)'}" stroke="${above ? 'var(--signalG)' : 'var(--text)'}" stroke-width="1.3"/>
  </svg><div class="gread"><span>${c.t('best')} <b class="mono">${num(e.best)}</b></span><span>${c.t('noise')} <b class="mono">${num(e.bar)}</b></span></div></div>`
}
function gaugeAxis(w) {
  const x = d3.scaleLinear(evDomain, [5, w - 5])
  return `<svg width="${w}" height="14" viewBox="0 0 ${w} 14" aria-hidden="true">${evTicks.map((t) => `<text x="${x(t)}" y="11" text-anchor="middle" class="axis">${num(t)}</text>`).join('')}</svg>`
}
function budget(c, d, w = 72) {
  if (d.budget === null) return `<div class="budget"><span class="mono">${money(d.spend)}</span><span class="faint">${c.t('budgetUnset')}</span></div>`
  const r = d.spend / d.budget
  const over = r > 1
  return `<div class="budget"><span class="mono">${money(d.spend)}<span class="faint"> / ${money(d.budget)}</span></span>
    <span class="bline"><svg width="${w}" height="6" viewBox="0 0 ${w} 6" aria-hidden="true"><rect width="${w}" height="2" y="2" fill="var(--line2)"/><rect width="${(Math.min(1, r) * w).toFixed(1)}" height="2" y="2" fill="${over ? 'var(--signalG)' : 'var(--muted)'}"/>${over ? `<rect x="${w - 3}" y="0" width="3" height="6" fill="var(--signalG)"/>` : ''}</svg><span class="mono ${over ? 'sig' : 'faint'}">${Math.round(r * 100)}%</span></span></div>`
}
const lock = (c, d) => (d.lockbox === 'sealed' ? `<span class="lock">${ICON.lock}<span>${c.t('sealed')}</span></span>` : d.lockbox === 'opened' ? `<span class="lock is-open">${ICON.unlock}<span>${c.t('opened')}</span></span>` : `<span class="lock faint">—</span>`)
const hypDist = (c, d) => `<div class="hyps">${Object.entries(d.evidence.hypothesisStatus).map(([s, n]) => `<span>${statusGlyph(c, s)}<em>${esc(STATUS[c.lang][s])}</em><b class="mono">${n}</b></span>`).join('')}</div>`
function dirTable(c, { gaugeW }) {
  return `<div class="dtable" role="table">
    <div class="drow dhead" role="row"><span>${c.t('direction')}</span><span>${c.t('hypotheses')}</span><span class="evhead">${c.t('evidence')}${gaugeAxis(gaugeW)}</span><span class="num">${c.t('trials')}</span><span>${c.t('budget')}</span><span>${c.t('lockbox')}</span></div>
    ${dirs.map((d) => `<div class="drow ${d.id === 'd1-momentum-regimes' ? 'is-focus' : ''}" role="row" id="${d.id.slice(0, 2)}">
      <div class="dtitle"><b>${esc(shortTitle(d))}</b><span class="dmeta">${stateGlyph(c, d)}<span class="${d.state === 'over-budget' ? 'sig' : ''}">${STATE[c.lang][d.state]}</span><span class="faint">· ${c.t('hypCount')(d.hypotheses.length)}</span></span></div>
      ${hypDist(c, d)}${gauge(c, d, gaugeW)}
      <div class="trials num"><b class="mono">${int(d.trials)}</b>${d.trials ? `<span class="faint">${c.t('failedShort')} <span class="mono">${int(d.evidence.failed)}</span></span>` : ''}</div>
      ${budget(c, d)}${lock(c, d)}</div>`).join('')}</div>`
}
function glyphKey(c) {
  const present = [...new Set(dirs.flatMap((d) => Object.keys(d.evidence.hypothesisStatus)))]
  return `<div class="key"><span class="faint">${c.t('hypothesisKey')}</span>${present.map((s) => `<span>${statusGlyph(c, s)}${esc(STATUS[c.lang][s])}</span>`).join('')}</div>
    <p class="keynote">${c.t('scaleKey')(evTicks.map((t) => num(t)).join(' · '))}</p>`
}
function feed(c, n, cls = '') {
  return `<aside class="feed ${cls}" aria-label="${c.t('activity')}"><header><b>${c.t('activity')}</b><span class="faint">${c.t('allDirections')}</span></header>
    ${fixture.events.slice(0, n).map((e) => `<article class="ev">${eventMark(e.event)}<div class="evbody"><div class="evtitle"><b class="${e.event === 'run-failed' ? 'sig' : ''}">${EVENT[c.lang][e.event] ?? e.event}</b>${trunc(shortTitle(dirById.get(e.direction)), 'faint')}</div>
      <div class="evref mono faint"><span>${esc([e.hypothesis, e.experiment].filter(Boolean).join(' / '))}</span><time>${clock(e.time)}</time></div>${logText(e.note)}</div></article>`).join('')}</aside>`
}
function homeHead(c, { phone = false } = {}) {
  const failRate = totals.trials ? totals.failed / totals.trials : 0
  return `<h1 class="dsent ${phone ? 'is-phone' : ''}">${c.t('sentence')(dirs.length, totals.confirmed, totals.active, totals.over, totals.fresh)}</h1>
    <p class="dsub">${c.t('subline')(int(totals.trials), money(totals.spend), money(totals.budget), totals.above, totals.withTrials)}</p>
    ${phone ? '' : `<div class="kpis">
      <div><span>${c.t('kTrials')}</span><b class="mono">${int(totals.trials)}</b><em>${c.t('kDirs')(dirs.length)}</em></div>
      <div><span>${c.t('kFailed')}</span><b class="mono">${int(totals.failed)}</b><em>${c.t('kFailNote')(pct(failRate, 1))}</em></div>
      <div><span>${c.t('kSpend')}</span><b class="mono">${money(totals.spend)}<i> / ${money(totals.budget)}</i></b><em>${pct(totals.spend / totals.budget)}</em></div>
      <div><span>${c.t('kHyp')}</span><b class="mono">${int(totals.hypotheses)}</b><em>${c.t('kHypNote')(totals.challenged, totals.running)}</em></div>
    </div>`}`
}
function home(c, { narrow = false, stacked = false, gaugeW = 176 } = {}) {
  return `<div class="app">${rail(c, 'directions', { narrow })}<div class="main">${topbar(c, [c.t('directions')])}
    <div class="homebody ${stacked ? 'is-stacked' : ''}"><section class="content">${homeHead(c)}${dirTable(c, { gaugeW })}${glyphKey(c)}${stacked ? feed(c, 4, 'is-below') : ''}</section>${stacked ? '' : feed(c, 9)}</div></div></div>`
}

// ======================= direction report =======================
const rep = fixture.report
const repDir = dirById.get(rep.id)
function bodySection(body, heading) {
  const lines = body.split('\n')
  const start = lines.findIndex((l) => l.replace(/^>\s*/, '').trim().toLowerCase() === `## ${heading}`.toLowerCase())
  if (start === -1) return ''
  const out = []
  for (const l of lines.slice(start + 1)) { if (/^>?\s*#/.test(l)) break; if (l.trim()) out.push(l.replace(/^>\s*/, '').trim()) }
  return out.join(' ')
}
// The chart's right edge is the home gauge: at the last trial a vertical orange segment joins the
// noise line to the best record, so the report and the overview show the same evidence the same way.
function trialChart(c, hid, { w, h }) {
  const pts = rep.points
  const m = { t: 20, r: 132, b: 40, l: 48 }
  const failY = h - m.b - 10
  const plotBottom = failY - 22
  const x = d3.scaleLinear([1, d3.max(pts, (p) => p.n)], [m.l, w - m.r])
  const valid = pts.filter((p) => p.score !== null)
  const [lo, hi] = d3.extent([...valid.map((p) => p.score), ...pts.map((p) => p.bar)])
  const y = d3.scaleLinear([lo - (hi - lo) * 0.04, hi + (hi - lo) * 0.06], [plotBottom, m.t]).nice()
  let running = -Infinity
  const steps = pts.map((p) => { if (p.score !== null) running = Math.max(running, p.score); return [p.n, running, p.bar] }).filter((s) => Number.isFinite(s[1]))
  const stepLine = d3.line().curve(d3.curveStepAfter).x((s) => x(s[0])).y((s) => y(s[1]))
  const barLine = d3.line().curve(d3.curveMonotoneX).x((s) => x(s[0])).y((s) => y(s[2]))
  const last = steps.at(-1)
  const ex = x(last[0])
  const xt = x.ticks(8).filter(Number.isInteger)
  const gap = last[1] - last[2]
  return `<svg class="chart" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${c.t('trialsAndNoise')}">
    ${y.ticks(4).map((t) => `<line x1="${m.l}" x2="${ex}" y1="${y(t)}" y2="${y(t)}" stroke="var(--grid)"/><text x="${m.l - 8}" y="${y(t) + 4}" text-anchor="end" class="axis">${num(t)}</text>`).join('')}
    <text x="${m.l}" y="10" class="axis-title">${c.t('scoreAxis')}</text>
    ${valid.map((p) => p.hypothesis === hid ? `<circle cx="${x(p.n).toFixed(1)}" cy="${y(p.score).toFixed(1)}" r="2" fill="var(--muted)"/>` : `<circle cx="${x(p.n).toFixed(1)}" cy="${y(p.score).toFixed(1)}" r="1.9" fill="none" stroke="var(--stroke)" stroke-width="1"/>`).join('')}
    <line x1="${m.l}" x2="${ex}" y1="${failY - 11}" y2="${failY - 11}" stroke="var(--line)"/>
    <text x="${m.l - 8}" y="${failY + 4}" text-anchor="end" class="axis">${c.t('failBand')}</text>
    ${pts.filter((p) => p.score === null).map((p) => `<path d="M${(x(p.n) - 2.5).toFixed(1)},${failY - 2.5}l5,5m0,-5l-5,5" stroke="var(--signalG)" stroke-width="1.2"/>`).join('')}
    <path d="${barLine(steps)}" fill="none" stroke="var(--muted)" stroke-width="1.4" stroke-dasharray="4 3"/>
    <path d="${stepLine(steps)}" fill="none" stroke="var(--text)" stroke-width="1.6"/>
    <line x1="${ex}" x2="${ex}" y1="${y(last[2])}" y2="${y(last[1])}" stroke="var(--signalG)" stroke-width="2.5"/>
    <rect x="${ex - 5}" y="${y(last[2]) - 1}" width="10" height="2" fill="var(--muted)"/>
    <path d="M${ex},${y(last[1]) - 6}l5.5,6-5.5,6-5.5-6z" fill="var(--signalG)"/>
    <text x="${ex + 12}" y="${y(last[1]) + 4}" class="lbl">${c.t('record')} ${num(last[1])}</text>
    <text x="${ex + 12}" y="${(y(last[1]) + y(last[2])) / 2 + 4}" class="lbl sig">${c.t('clears')} ${signed(gap)}</text>
    <text x="${ex + 12}" y="${y(last[2]) + 4}" class="lbl muted">${c.t('noise')} ${num(last[2])}</text>
    ${xt.map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="${h - m.b + 2}" y2="${h - m.b + 6}" stroke="var(--stroke)"/><text x="${x(t)}" y="${h - m.b + 19}" text-anchor="middle" class="axis">${int(t)}</text>`).join('')}
    <line x1="${m.l}" x2="${ex}" y1="${h - m.b + 2}" y2="${h - m.b + 2}" stroke="var(--stroke)"/>
    <text x="${m.l}" y="${h - 3}" class="axis-title">${c.t('trialAxis')} 1–${int(last[0])}</text>
  </svg>`
}
function ledger(c, ev) {
  const th = ev.thresholds
  const op = (s) => `<span class="op">${s}</span>`
  const rowsL = [
    [c.t('dsr'), num(ev.dsr, 3), `${op('≥')} ${num(th.dsr_supported, 3)}`, ev.dsr >= th.dsr_supported],
    [c.t('positiveSub'), pct(ev.oos_positive_fraction), `${op('≥')} ${pct(th.min_positive_fraction)}`, ev.oos_positive_fraction >= th.min_positive_fraction - 1e-9],
    [c.t('pbo'), num(ev.pbo), `${op('≤')} ${num(th.max_pbo)}`, ev.pbo <= th.max_pbo],
    [c.t('vsBuyHold'), `${num(ev.oos_sharpe)} / ${num(ev.baseline_buy_hold_oos_sharpe)}`, esc(c.t('aboveBaseline')), ev.oos_sharpe > ev.baseline_buy_hold_oos_sharpe],
    [c.t('vsSeed'), `${num(ev.oos_sharpe)} / ${num(ev.baseline_seed_oos_sharpe)}`, '—', null],
    [c.t('trialsInDsr'), int(ev.n_trials), '—', null],
    [c.t('ratio'), num(ev.decay), '—', null],
  ]
  return `<div class="ltable" role="table"><div class="lrow lhead" role="row"><span>${c.t('check')}</span><span class="num">${c.t('result')}</span><span class="num">${c.t('threshold')}</span><span>${c.t('outcome')}</span></div>
    ${rowsL.map(([name, value, threshold, ok]) => `<div class="lrow" role="row"><span>${esc(name)}</span><span class="num mono">${esc(value)}</span><span class="num mono faint">${threshold}</span><span class="${ok === null ? 'faint' : ok ? '' : 'sig'}">${ok === null ? c.t('reference') : `${ok ? ICON.check : ICON.cross}${ok ? c.t('pass') : c.t('fail')}`}</span></div>`).join('')}</div>`
}
function report(c) {
  const h = rep.hypotheses[0]
  const skeptic = h.experiments[0].skeptic
  const confirmNote = fixture.events.find((e) => e.direction === rep.id && e.event === 'confirm' && e.hypothesis === h.id)
  const lockboxBars = rep.lockbox.n_total - rep.lockbox.n_dev
  return `<div class="app">${rail(c, 'directions', { direction: rep.id })}<div class="main">${topbar(c, [c.t('directions'), shortTitle(repDir)])}
    <div class="report">
      <header class="rhead"><h1>${esc(rep.title)}</h1>
        <div class="rmeta"><span class="rstate">${stateGlyph(c, repDir)}${STATE[c.lang][repDir.state]}</span><span class="faint">${c.t('created')} <span class="mono">${clock(rep.created)}</span> UTC</span><span class="faint">${c.t('trialsN')(int(rep.trials))}</span><span class="faint mono">${money(rep.spend)} / ${money(rep.budget)}</span>${lock(c, repDir)}</div>
        <div class="htabs" role="tablist">${rep.hypotheses.map((x, k) => `<span role="tab" aria-selected="${k === 0}" class="${k === 0 ? 'is-on' : ''}"><span class="mono">${x.id}</span>${statusGlyph(c, x.status)}${STATUS[c.lang][x.status]}</span>`).join('')}</div></header>
      <div class="rcols"><div class="rmain">
        <section><h2>${c.t('claim')}</h2><p class="claim">${esc(h.title)}</p>
          <dl class="defs"><dt>${c.t('claim')}</dt><dd>${esc(bodySection(h.body, 'Claim'))}</dd><dt>${c.t('mechanism')}</dt><dd>${esc(bodySection(h.body, 'Mechanism'))}</dd><dt>${c.t('falsification')}</dt><dd>${esc(bodySection(h.body, 'Falsification'))}</dd></dl></section>
        <section><div class="shead"><h2>${c.t('trialsAndNoise')}</h2><div class="legend"><span><i class="ln"></i>${c.t('record')}</span><span><i class="ln is-dash"></i>${c.t('noise')}</span><span><i class="dot"></i>${c.t('thisHyp')}</span><span><i class="dot is-hollow"></i>${c.t('otherHyp')}</span><span>${ICON.cross.replace('currentColor', 'var(--signalG)')}${c.t('failBand')}</span></div></div>
          ${trialChart(c, h.id, { w: 700, h: 262 })}</section>
        <section><h2>${c.t('experiments')}</h2>${h.experiments.map((x) => `<div class="xrow"><b class="mono">${x.id}</b><span class="faint mono">${clock(x.first)} → ${clock(x.last).slice(11)}</span>
          <span><b class="mono">${int(x.evaluated)}</b> ${c.t('evaluated')}</span><span><b class="mono">${int(x.probed)}</b> ${c.t('probed')}</span><span>${c.t('best')} <b class="mono">${num(x.best)}</b></span><span><b class="mono">${money(x.spend)}</b></span><a class="btn">${c.t('openAnalysis')}${ICON.arrow}</a></div>`).join('')}</section>
      </div>
      <aside class="rside">
        <section><h2>${c.t('ledger')}</h2>${ledger(c, skeptic)}</section>
        <section class="verdict"><h2>${c.t('verdict')}</h2><div class="vline">${statusGlyph(c, h.verdict.label)}<b>${STATUS[c.lang][h.verdict.label]}</b><span class="faint">${c.t('log')}</span></div>${logText(h.verdict.reason, 'log is-block')}
          ${confirmNote ? `<div class="vline is-sub">${statusGlyph(c, h.status)}<b>${c.t('lockboxConfirm')} · ${STATUS[c.lang][h.status]}</b><span class="faint mono">${clock(confirmNote.time)} UTC</span></div>${logText(confirmNote.note, 'log is-block')}` : ''}
          <p class="muted small">${c.t('lockboxInfo')(rep.lockbox.start.slice(0, 10), int(lockboxBars), int(rep.lockbox.n_dev))}</p></section>
      </aside></div></div></div></div>`
}

// ======================= all runs and compare =======================
const runLabel = (r) => r.name.replace('/results', '')
const sortedRuns = [...fixture.runs].sort((a, b) => (b.updated ?? 0) - (a.updated ?? 0))
const SELECTED = ['d1-momentum-regimes/experiments/h001/e01/results', 'd1-momentum-regimes/experiments/h002/e01/results', 'd1-momentum-regimes/experiments/h003/e02/results']
function runsPage(c) {
  const counts = { all: fixture.runs.length, ordinary: fixture.runs.filter((r) => r.kind === 'ordinary').length, experiment: fixture.runs.filter((r) => r.kind === 'experiment').length }
  return `<div class="app">${rail(c, 'runs')}<div class="main">${topbar(c, [c.t('runs')])}
    <div class="runs"><div class="toolbar"><label class="input">${ICON.search}<span>${c.t('runSearch')}</span></label>
      ${seg([`${c.t('all')} ${counts.all}`, `${c.t('ordinary')} ${counts.ordinary}`, `${c.t('experiment')} ${counts.experiment}`], 0, c.t('kind'))}
      <span class="select">${c.t('sortBy')}${ICON.chevron}</span>
      <span class="selbar"><span>${c.t('selectedN')(SELECTED.length)}</span><a class="btn is-primary">${c.t('compareSelected')}${ICON.arrow}</a></span></div>
    <div class="rtable" role="table"><div class="rrow rhead2" role="row"><span></span><span>${c.t('run')}</span><span>${c.t('kind')}</span><span class="num">${c.t('programs')}</span><span class="num">${c.t('failed')}</span><span class="num">${c.t('bestScore')}</span><span class="num">${c.t('cost')}</span><span class="num">${c.t('generations')}</span><span class="num">${c.t('islands')}</span><span class="num">${c.t('updatedAt')}</span></div>
      ${sortedRuns.map((r) => { const on = SELECTED.includes(r.name); return `<div class="rrow ${on ? 'is-selected' : ''}" role="row"><span class="cb ${on ? 'is-on' : ''}" role="checkbox" aria-checked="${on}">${on ? ICON.check : ''}</span>
        <span class="rname mono">${esc(runLabel(r))}</span><span class="muted">${r.kind === 'ordinary' ? c.t('ordinary') : c.t('experiment')}</span>
        <span class="num mono">${r.programs ? int(r.programs) : `<span class="empty">${glyph('empty', c.t('emptyRun'))}${c.t('emptyRun')}</span>`}</span><span class="num mono ${r.failed ? '' : 'faint'}">${int(r.failed)}</span><span class="num mono">${num(r.best)}</span><span class="num mono">${money(r.cost)}</span><span class="num mono">${int(r.generations)}</span><span class="num mono">${int(r.islands)}</span><span class="num mono faint">${epochClock(r.updated)}</span></div>` }).join('')}
    </div></div></div></div>`
}
function comparePage(c) {
  const sel = SELECTED.map((name) => fixture.runs.find((r) => r.name === name))
  const metrics = [
    [c.t('bestScore'), (r) => r.best, (v) => num(v), 'max'],
    [c.t('failRate'), (r) => r.failed / r.programs, (v) => pct(v, 1), 'min'],
    [c.t('cost'), (r) => r.cost, (v) => money(v), 'min'],
    [c.t('costPerProgram'), (r) => r.cost / r.programs, (v) => `$${v.toFixed(3)}`, 'min'],
    [c.t('programs'), (r) => r.programs, (v) => int(v), null],
    [c.t('generations'), (r) => r.generations, (v) => int(v), null],
  ]
  const W = 520
  const short = (r) => runLabel(r).replace('d1-momentum-regimes/experiments/', '')
  return `<div class="app">${rail(c, 'runs')}<div class="main">${topbar(c, [c.t('runs'), c.t('compareTitle')(sel.length)])}
    <div class="compare"><header class="chead"><h1>${c.t('compareTitle')(sel.length)}</h1><p class="muted">${esc(shortTitle(dirById.get('d1-momentum-regimes')))} · ${c.t('compareNote')}</p>
      <div class="chips">${sel.map((r, k) => `<span class="chip">${markIcon(k, 14)}<b>${esc(short(r))}</b><span class="faint">${c.t('programsN')(int(r.programs))}</span></span>`).join('')}<span class="chip is-add">+ ${c.t('addRun')}</span></div></header>
    <div class="ctable2" role="table"><div class="crow chd" role="row"><span>${c.t('metric')}</span>${sel.map((r, k) => `<span class="num">${markIcon(k)}<span class="mono">${esc(short(r))}</span></span>`).join('')}<span>${c.t('scaleNote')}</span></div>
    ${metrics.map(([label, get, fmt, better]) => {
      const values = sel.map(get)
      const [lo, hi] = d3.extent(values)
      const pad = (hi - lo) * 0.3 || Math.abs(hi) * 0.1 || 1
      const x = d3.scaleLinear([Math.max(0, lo - pad), hi + pad], [12, W - 12]).nice()
      const lead = better ? values.indexOf(better === 'max' ? hi : lo) : -1
      const ticks = x.ticks(4)
      return `<div class="crow" role="row"><span class="muted">${esc(label)}</span>${values.map((v, k) => `<span class="num mono ${k === lead ? 'is-lead' : ''}">${fmt(v)}</span>`).join('')}
        <svg width="${W}" height="34" viewBox="0 0 ${W} 34" role="img" aria-label="${esc(label)}"><line x1="12" x2="${W - 12}" y1="12" y2="12" stroke="var(--line2)"/>
        ${ticks.map((t, i) => `<line x1="${x(t)}" x2="${x(t)}" y1="9" y2="15" stroke="var(--stroke)"/><text x="${x(t)}" y="31" text-anchor="${i === 0 && x(t) < 30 ? 'start' : i === ticks.length - 1 && x(t) > W - 30 ? 'end' : 'middle'}" class="axis">${fmt(t)}</text>`).join('')}
        ${values.map((v, k) => MARKS[k](x(v), 12)).join('')}</svg></div>`
    }).join('')}</div></div></div></div>`
}

// ======================= analysis =======================
const ramp = (c) => d3.scaleLinear(scoreExtent, [c.th.ramp0, c.th.ramp1]).interpolate(d3.interpolateRgb)
const rampLegend = (c) => { const r = ramp(c); return `<span class="faint">${c.t('scoreRamp')}</span><span class="ramp">${d3.range(5).map((k) => `<i style="background:${r(scoreExtent[0] + (k / 4) * (scoreExtent[1] - scoreExtent[0]))}"></i>`).join('')}</span><span class="mono faint">${num(scoreExtent[0])}–${num(scoreExtent[1])}</span>` }
const VIEWS = (c) => [['lineage', c.t('lineage')], ['progress', c.t('progress')], ['space', c.t('space')], ['contest', c.t('contest')]]
function analysisShell(c, tab, canvas, inspectorTab) {
  const d = dirById.get(A.direction)
  const stats = `<span class="rstats mono faint">${int(A.stats.programs)} ${c.lang === 'zh' ? '程序' : 'programs'} · <span class="sig">${int(A.stats.failed)} ${c.lang === 'zh' ? '失败' : 'failed'}</span> · ${A.stats.generations} ${c.lang === 'zh' ? '代' : 'gens'} · ${A.stats.islands} ${c.lang === 'zh' ? '岛' : 'islands'}</span>`
  return `<div class="app">${rail(c, 'directions', { narrow: true })}<div class="main">${topbar(c, [shortTitle(d), `${A.hypothesis} / ${A.experiment}`], { tabs: VIEWS(c), tab, extra: stats })}
    <div class="analysis"><section class="canvas">${canvas}</section>${inspector(c, inspectorTab)}</div></div></div>`
}
function highlight(line) {
  const commentAt = line.indexOf('#')
  const code = commentAt >= 0 ? line.slice(0, commentAt) : line
  const comment = commentAt >= 0 ? line.slice(commentAt) : ''
  const parts = code.split(/('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")/)
  const KW = /\b(def|return|import|from|for|in|if|elif|else|while|class|lambda|None|True|False|and|or|not|with|as|try|except|raise|pass|yield)\b/g
  const html = parts.map((part, k) => (k % 2 ? `<span class="s">${esc(part)}</span>` : esc(part).replace(KW, '<span class="k">$1</span>'))).join('')
  return html + (comment ? `<span class="cm">${esc(comment)}</span>` : '')
}
function inspector(c, tab, { sheet = false } = {}) {
  const tabs = [['code', c.t('code')], ['diff', c.t('diff')], ['evaluation', c.t('evaluation')], ['llm', c.t('llm')], ['meta', c.t('meta')]]
  const codebar = (file, n) => `<div class="codebar"><span class="mono faint">${file} · ${c.t('lines')(n)}</span><span class="toggle">${c.t('wrap')} <b>${c.t('off')}</b></span></div>`
  let body = ''
  if (tab === 'code') body = `${codebar('main.py', best.code.length)}<pre class="code" tabindex="0" data-scroll-x aria-label="${c.t('codeScroll')}">${best.code.map((l, i) => `<span class="ln">${i + 1}</span>${highlight(l)}`).join('\n')}</pre>`
  if (tab === 'diff') body = `${codebar('main.py', best.diff.length)}<pre class="code is-diff" tabindex="0" data-scroll-x aria-label="${c.t('codeScroll')}">${best.diff.map((l) => `<span class="dl ${l.startsWith('+') && !l.startsWith('+++') ? 'add' : l.startsWith('-') && !l.startsWith('---') ? 'del' : l.startsWith('@@') ? 'hunk' : ''}">${esc(l) || ' '}</span>`).join('')}</pre>`
  if (tab === 'evaluation') body = `<div class="ibody"><div class="metrics">${Object.entries(best.metrics).map(([k, v]) => `<div><span class="mono faint">${esc(k)}</span><b class="mono">${num(v, k === 'is_turnover' ? 3 : 2)}</b></div>`).join('')}</div>
    <h3>${c.t('log')} · stdout</h3>${logText(best.stdout, 'log is-block')}<h3>${c.t('feedback')}</h3>${logText(best.feedback, 'log is-block')}</div>`
  if (tab === 'llm') body = `<div class="ibody"><div class="kv"><span class="faint">${c.t('model')}</span><b class="mono">${esc(best.model)}</b><span class="faint">${c.t('cost')}</span><b class="mono">$${best.cost.toFixed(4)}</b><span class="faint">patch</span><b class="mono">${esc(best.patchType)} · ${esc(best.description)}</b></div>
    <h3>${c.t('reasoning')}</h3>${logText(best.thought, 'log is-block')}</div>`
  if (tab === 'meta') {
    const latest = A.meta.latest
    const plain = (line) => esc(line.replace(/\*\*/g, '').replace(/`/g, ''))
    const block = (lines) => lines.map((line) => (line.startsWith('## ') ? `<h4>${plain(line.slice(3))}</h4>` : line.startsWith('- ') ? `<p class="item">${plain(line.slice(2))}</p>` : /^\d+\.\s/.test(line) ? `<p class="item is-num"><span class="mono faint">${line.match(/^\d+/)[0]}</span>${plain(line.replace(/^\d+\.\s/, ''))}</p>` : line.startsWith('> ') ? `<p class="faint small">${plain(line.slice(2))}</p>` : `<p class="muted small">${plain(line)}</p>`)).join('')
    body = `<div class="ibody is-meta"><div class="metahead"><b>${c.t('metaUpdate')(A.meta.updates.length, latest.processed)}</b><span class="mono faint">${esc(latest.file)}</span></div>
      <h3>${c.t('metaRecs')}</h3><div class="metalist">${block(latest.recommendations)}</div><h3>${c.t('metaInsights')}</h3><div class="metalist">${block(latest.insights)}</div></div>`
  }
  return `<aside class="inspector ${sheet ? 'is-sheet' : ''}" aria-label="${c.t('inspector')}">
    <header class="ihead"><div class="iname"><span class="bestmark">${c.t('best')}</span><h2 class="mono">${esc(best.name)}</h2><span class="faint mono">${best.id.slice(0, 8)}</span></div>
      <div class="iscore"><b class="mono">${num(best.score)}</b><span class="faint">${c.t('parent')} <span class="mono">${num(best.parentScore)}</span> · ${c.t('vsParent')} <span class="mono">${signed(best.score - best.parentScore)}</span></span></div>
      <div class="ichips"><span>${c.t('gen')(best.generation)}</span><span>${c.t('island')} ${best.island}</span><span class="mono">${esc(best.model)}</span><span class="mono">$${best.cost.toFixed(4)}</span></div></header>
    <div class="itabs" role="tablist">${tabs.map(([k, label]) => `<span role="tab" aria-selected="${k === tab}" class="${k === tab ? 'is-on' : ''}">${label}</span>`).join('')}${tab === 'code' ? `<a class="dlbtn" title="${c.t('downloadCode')('main.py')}" aria-label="${c.t('downloadCode')('main.py')}">${ICON.download}</a>` : tab === 'meta' ? `<a class="dlbtn" title="${c.t('download')}" aria-label="${c.t('download')}">${ICON.download}</a>` : ''}</div>
    ${body}</aside>`
}
function lineageView(c, { w, h }) {
  const top = 64
  const scrubH = 70
  const islands = d3.range(A.stats.islands)
  const m = { l: 128, r: 28 }
  const laneH = (h - top - scrubH) / islands.length
  const x = d3.scaleLinear([0, maxGen], [m.l, w - m.r])
  const r = ramp(c)
  const sy = d3.scaleLinear(scoreExtent, [1, 0])
  const yIn = (lane, s) => top + lane * laneH + 34 + sy(s) * (laneH - 70)
  const failY = (lane) => top + (lane + 1) * laneH - 14
  const pos = rows.map((row) => [x(row[3]), row[2] === null ? failY(row[4]) : yIn(row[4], row[2])])
  // Only parent links on the same island are drawn; migrations would cross every lane.
  const edges = rows.map((row, i) => (row[5] === -1 || rows[row[5]][4] !== row[4] ? '' : `M${pos[row[5]][0].toFixed(1)},${pos[row[5]][1].toFixed(1)}L${pos[i][0].toFixed(1)},${pos[i][1].toFixed(1)}`)).join('')
  const pathD = A.path.map((i, k) => `${k ? 'L' : 'M'}${pos[i][0].toFixed(1)},${pos[i][1].toFixed(1)}`).join('')
  const [bx, by] = pos[A.bestIndex]
  const laneTicks = d3.ticks(...scoreExtent, 2)
  // Prompt evolution and meta-analysis updates sit on the same generation axis as the lanes (OD-3).
  const bestPrompt = A.prompts.reduce((a, b) => (b.fitness > a.fitness ? b : a))
  const track = `<text x="20" y="24" class="axis">${c.t('promptTrack')}</text><text x="20" y="50" class="axis">${c.t('metaTrack')}</text>
    <line x1="${m.l}" x2="${w - m.r}" y1="20" y2="20" stroke="var(--line)"/><line x1="${m.l}" x2="${w - m.r}" y1="46" y2="46" stroke="var(--line)"/>
    ${A.prompts.map((p, k) => `<path d="M${x(p.generation)},14l5,8h-10z" fill="${p === bestPrompt ? 'var(--text)' : 'var(--bg)'}" stroke="var(--text)" stroke-width="1.2"/><text x="${x(p.generation) + 8}" y="24" class="axis ${p === bestPrompt ? 'is-strong' : ''}">P${k} ${num(p.fitness)}</text>`).join('')}
    ${A.meta.updates.map((u, k) => `<rect x="${x(u.generation) - 4}" y="42" width="8" height="8" fill="var(--bg)" stroke="var(--text)" stroke-width="1.2"/><text x="${x(u.generation) + 8}" y="50" class="axis">#${k + 1} · ${u.processed}</text>`).join('')}`
  return `<div class="vbar"><div class="legend">${rampLegend(c)}<span><i class="ln is-signal"></i>${c.t('bestPath')}</span><span>${ICON.cross.replace('currentColor', 'var(--signalG)')}${c.t('failedMark')}</span></div><span class="faint small">${c.t('edgesNote')}</span></div>
    <svg class="chart lanes" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${c.t('lineage')}">${track}
    ${islands.map((k) => { const n = rows.filter((row) => row[4] === k).length; const f = rows.filter((row) => row[4] === k && row[2] === null).length; return `<line x1="0" x2="${w}" y1="${top + k * laneH}" y2="${top + k * laneH}" stroke="var(--line)"/>
      <text x="20" y="${top + k * laneH + 26}" class="lane">${c.t('island')} ${k}</text><text x="20" y="${top + k * laneH + 44}" class="axis">${c.t('programsN')(n)}</text><text x="20" y="${top + k * laneH + 60}" class="axis ${f ? 'sig' : ''}">${c.t('failedN')(f)}</text>
      ${laneTicks.map((t) => `<text x="${m.l - 10}" y="${yIn(k, t) + 3.5}" text-anchor="end" class="axis">${num(t, 1)}</text>`).join('')}
      <line x1="${m.l}" x2="${w - m.r}" y1="${failY(k) - 8}" y2="${failY(k) - 8}" stroke="var(--line)" stroke-dasharray="2 3"/>` }).join('')}
    <path d="${edges}" fill="none" stroke="var(--grid)" stroke-width=".8"/>
    ${rows.map((row, i) => (row[2] === null ? `<path d="M${(pos[i][0] - 3).toFixed(1)},${(pos[i][1] - 3).toFixed(1)}l6,6m0,-6l-6,6" stroke="var(--signalG)" stroke-width="1.3"/>` : pathSet.has(i) ? '' : `<circle cx="${pos[i][0].toFixed(1)}" cy="${pos[i][1].toFixed(1)}" r="3.2" fill="${r(row[2])}"/>`)).join('')}
    <path d="${pathD}" fill="none" stroke="var(--signalG)" stroke-width="2" stroke-linejoin="round"/>
    ${A.path.map((i) => `<circle cx="${pos[i][0].toFixed(1)}" cy="${pos[i][1].toFixed(1)}" r="4" fill="var(--bg)" stroke="var(--signalG)" stroke-width="2"/>`).join('')}
    <circle cx="${bx}" cy="${by}" r="5.5" fill="var(--signalG)"/><circle cx="${bx}" cy="${by}" r="9.5" fill="none" stroke="var(--text)" stroke-width="1.5"/>
    <text x="${bx}" y="${by - 16}" text-anchor="middle" class="lbl sig halo">${c.t('best')} ${num(best.score)} · ${c.t('gen')(best.generation)}</text>
    <g transform="translate(0,${h - scrubH})"><line x1="0" x2="${w}" y1="0" y2="0" stroke="var(--line)"/>
      ${x.ticks(8).filter(Number.isInteger).map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="0" y2="5" stroke="var(--stroke)"/><text x="${x(t)}" y="19" text-anchor="middle" class="axis">${t}</text>`).join('')}
      <text x="20" y="19" class="axis">${c.t('generation')}</text>
      <line x1="${m.l}" x2="${w - m.r}" y1="44" y2="44" stroke="var(--stroke)" stroke-width="2"/><line x1="${m.l}" x2="${x(maxGen)}" y1="44" y2="44" stroke="var(--text)" stroke-width="2"/>
      <rect x="${x(maxGen) - 6}" y="37" width="12" height="14" rx="3" fill="var(--text)"/>
      <g transform="translate(20,33)"><rect width="76" height="24" rx="6" fill="none" stroke="var(--line2)"/><g transform="translate(10,6)" style="color:var(--text)">${ICON.play}</g><text x="28" y="16" class="btntext">${c.t('replay')}</text></g>
    </g></svg>
    <p class="scrubnote mono faint">${c.t('scrub')(maxGen, int(rows.length))}</p>`
}
function progressView(c, { w }) {
  const P = A.progress
  const m = { l: 64, r: 28 }
  const x = d3.scaleLinear([0, maxGen], [m.l, w - m.r])
  const bw = Math.max(2, (x(1) - x(0)) * 0.62)
  const panel = (title, legend, height, yDomain, draw, fmt = (v) => num(v)) => {
    const y = d3.scaleLinear(yDomain, [height - 6, 10]).nice()
    return `<div class="pm"><div class="pmhead"><b>${title}</b><span class="legend">${legend}</span></div><svg width="${w}" height="${height}" viewBox="0 0 ${w} ${height}" role="img" aria-label="${esc(title)}">
      ${y.ticks(3).map((t) => `<line x1="${m.l}" x2="${w - m.r}" y1="${y(t)}" y2="${y(t)}" stroke="var(--grid)"/><text x="${m.l - 8}" y="${y(t) + 4}" text-anchor="end" class="axis">${fmt(t)}</text>`).join('')}
      ${draw(y)}</svg></div>`
  }
  const scoreDomain = [d3.min(P, (p) => p.median ?? p.best), d3.max(P, (p) => p.cumulative)]
  const costMax = d3.max(P, (p) => (p.cost ?? 0) + p.metaCost)
  return `<div class="progress">
    ${panel(c.t('cumBest'), `<span><i class="ln"></i>${c.t('cumBest')}</span><span><i class="ln is-dash"></i>${c.t('median')}</span><span><i class="dot"></i>${c.t('perGenBest')}</span>`, 150, scoreDomain, (y) => `
      <path d="${d3.line().x((p) => x(p.generation)).y((p) => y(p.median)).defined((p) => p.median !== null)(P)}" fill="none" stroke="var(--muted)" stroke-width="1.3" stroke-dasharray="4 3"/>
      ${P.filter((p) => p.best !== null).map((p) => `<circle cx="${x(p.generation)}" cy="${y(p.best)}" r="2.2" fill="var(--muted)"/>`).join('')}
      <path d="${d3.line().curve(d3.curveStepAfter).x((p) => x(p.generation)).y((p) => y(p.cumulative))(P)}" fill="none" stroke="var(--text)" stroke-width="1.8"/>
      <circle cx="${x(best.generation)}" cy="${y(best.score)}" r="4.5" fill="var(--signalG)"/><circle cx="${x(best.generation)}" cy="${y(best.score)}" r="8.5" fill="none" stroke="var(--text)" stroke-width="1.5"/>
      <text x="${x(best.generation) - 14}" y="${y(best.score) + 22}" text-anchor="end" class="lbl sig halo">${c.t('best')} ${num(best.score)} · ${c.t('gen')(best.generation)}</text>`)}
    ${panel(c.t('costSplit'), `<span><i class="sw" style="background:var(--muted)"></i>${c.t('modelCost')}</span><span><i class="sw" style="background:var(--text)"></i>${c.t('metaCost')}</span>`, 96, [0, costMax], (y) => P.map((p) => `<rect x="${x(p.generation) - bw / 2}" y="${y(p.cost ?? 0)}" width="${bw}" height="${y(0) - y(p.cost ?? 0)}" fill="var(--muted)"/>${p.metaCost ? `<rect x="${x(p.generation) - bw / 2}" y="${y((p.cost ?? 0) + p.metaCost)}" width="${bw}" height="${y(0) - y(p.metaCost)}" fill="var(--text)"/>` : ''}`).join(''), (v) => `$${v.toFixed(2)}`)}
    ${panel(c.t('failures'), '', 76, [0, d3.max(P, (p) => p.failed) || 1], (y) => P.map((p) => (p.failed ? `<rect x="${x(p.generation) - bw / 2}" y="${y(p.failed)}" width="${bw}" height="${y(0) - y(p.failed)}" fill="var(--signalG)"/>` : '')).join(''), (v) => int(v))}
    ${panel(c.t('throughput'), '', 76, [0, d3.max(P, (p) => p.programs)], (y) => P.map((p) => `<rect x="${x(p.generation) - bw / 2}" y="${y(p.programs)}" width="${bw}" height="${y(0) - y(p.programs)}" fill="var(--stroke)"/>`).join(''), (v) => int(v))}
    <svg width="${w}" height="26" viewBox="0 0 ${w} 26" aria-hidden="true"><line x1="${m.l}" x2="${w - m.r}" y1="2" y2="2" stroke="var(--stroke)"/>${x.ticks(8).filter(Number.isInteger).map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="2" y2="6" stroke="var(--stroke)"/><text x="${x(t)}" y="20" text-anchor="middle" class="axis">${t}</text>`).join('')}<text x="${m.l - 8}" y="20" text-anchor="end" class="axis">${c.t('generation')}</text></svg>
    <section class="models"><h3>${c.t('models')}</h3><div class="mtable" role="table"><div class="mrow mhead" role="row"><span>${c.t('model')}</span><span class="num">${c.t('programs')}</span><span class="num">${c.t('success')}</span><span class="num">${c.t('best')}</span><span class="num">${c.t('mean')}</span><span class="num">${c.t('cost')}</span></div>
      ${A.modelStats.map((s) => `<div class="mrow" role="row"><span class="mono">${esc(s.model)}</span><span class="num mono">${int(s.programs)}</span><span class="num mono">${pct(s.success, 1)}</span><span class="num mono ${s.best === best.score ? 'sig' : ''}">${num(s.best)}</span><span class="num mono">${num(s.mean)}</span><span class="num mono">${money(s.cost)}</span></div>`).join('')}</div>
      <p class="na">${glyph('empty', c.t('posteriorNA'))}<span class="muted small"><b>${c.t('posterior')}</b> · ${c.t('posteriorNA')}</span></p></section>
    <div class="disclosure">${ICON.chevron}<b>${c.t('diagnostics')}</b><span class="faint">${c.t('expand')}</span></div></div>`
}
function spaceView(c, { w, h }) {
  const pts = rows.map((row, i) => ({ x: row[0], y: row[1], s: row[2], k: row[9], i }))
  const pw = w - 320
  const m = { t: 28, r: 24, b: 40, l: 56 }
  // One unit on both axes so distances in the projection are not distorted.
  const [x0, x1] = d3.extent(pts, (p) => p.x), [y0, y1] = d3.extent(pts, (p) => p.y)
  const unit = Math.min((pw - m.l - m.r) / (x1 - x0), (h - m.t - m.b) / (y1 - y0)) * 0.92
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2
  const x = (v) => m.l + (pw - m.l - m.r) / 2 + (v - cx) * unit
  const y = (v) => m.t + (h - m.t - m.b) / 2 - (v - cy) * unit
  const r = ramp(c)
  const contours = d3.contourDensity().x((p) => x(p.x)).y((p) => y(p.y)).size([pw, h]).bandwidth(16).thresholds(5)(pts)
  const geo = d3.geoPath()
  const clusters = d3.rollups(pts, (v) => ({ n: v.length, top: d3.min(v, (p) => y(p.y)), cx: d3.mean(v, (p) => x(p.x)), best: d3.max(v, (p) => p.s), failed: v.filter((p) => p.s === null).length }), (p) => p.k).filter(([k]) => k !== null).sort((a, b) => a[0] - b[0])
  const bp = pts[A.bestIndex]
  const xTicks = d3.ticks(x0, x1, 6), yTicks = d3.ticks(y0, y1, 5)
  return `<div class="vbar"><div class="legend">${rampLegend(c)}<span><i class="ln is-contour"></i>${c.t('density')}</span><span>${ICON.cross.replace('currentColor', 'var(--signalG)')}${c.t('failedMark')}</span></div>${seg([c.t('d2'), c.t('d3')], 0, c.t('space'))}</div>
    <div class="spacewrap"><svg class="chart" width="${pw}" height="${h}" viewBox="0 0 ${pw} ${h}" role="img" aria-label="${c.t('space')}">
      ${contours.map((g) => `<path d="${geo(g)}" fill="none" stroke="var(--stroke)" stroke-width=".8"/>`).join('')}
      ${pts.filter((p) => p.s !== null && p.i !== A.bestIndex).map((p) => `<circle cx="${x(p.x).toFixed(1)}" cy="${y(p.y).toFixed(1)}" r="3" fill="${r(p.s)}"/>`).join('')}
      ${pts.filter((p) => p.s === null).map((p) => `<path d="M${(x(p.x) - 3).toFixed(1)},${(y(p.y) - 3).toFixed(1)}l6,6m0,-6l-6,6" stroke="var(--signalG)" stroke-width="1.3"/>`).join('')}
      ${clusters.map(([k, v]) => `<text x="${v.cx.toFixed(1)}" y="${(v.top - 14).toFixed(1)}" text-anchor="middle" class="clabel halo">${c.t('cluster')(k + 1)}</text>`).join('')}
      <circle cx="${x(bp.x)}" cy="${y(bp.y)}" r="5.5" fill="var(--signalG)"/><circle cx="${x(bp.x)}" cy="${y(bp.y)}" r="9.5" fill="none" stroke="var(--text)" stroke-width="1.5"/>
      <line x1="${x(bp.x) - 10}" y1="${y(bp.y) + 8}" x2="${x(bp.x) - 44}" y2="${y(bp.y) + 52}" stroke="var(--signalG)"/><text x="${x(bp.x) - 48}" y="${y(bp.y) + 66}" text-anchor="end" class="lbl sig">${c.t('best')} ${num(best.score)}</text>
      ${xTicks.map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="${h - m.b + 6}" y2="${h - m.b + 10}" stroke="var(--stroke)"/><text x="${x(t)}" y="${h - m.b + 24}" text-anchor="middle" class="axis">${num(t, 1)}</text>`).join('')}
      ${yTicks.map((t) => `<line x1="${m.l - 10}" x2="${m.l - 6}" y1="${y(t)}" y2="${y(t)}" stroke="var(--stroke)"/><text x="${m.l - 14}" y="${y(t) + 4}" text-anchor="end" class="axis">${num(t, 1)}</text>`).join('')}
      <text x="${pw - m.r}" y="${h - m.b - 4}" text-anchor="end" class="axis-title">PCA 1 →</text><text x="${m.l}" y="14" class="axis-title">↑ PCA 2</text></svg>
    <aside class="sideinfo"><h3>${c.t('clusters')}</h3><div class="mtable" role="table"><div class="mrow chead2" role="row"><span>${c.t('clusters')}</span><span class="num">${c.t('programs')}</span><span class="num">${c.t('best')}</span><span class="num">${c.t('failedShort')}</span></div>
      ${clusters.map(([k, v]) => `<div class="mrow" role="row"><span>${c.t('cluster')(k + 1)}</span><span class="num mono">${int(v.n)}</span><span class="num mono ${v.best === best.score ? 'sig' : ''}">${num(v.best)}</span><span class="num mono">${int(v.failed)}</span></div>`).join('')}</div>
      <h3>${c.t('diversity')}</h3><div class="kv"><span class="faint">${c.t('clusters')}</span><b class="mono">${clusters.length}</b><span class="faint">${c.t('programs')}</span><b class="mono">${int(pts.length)}</b><span class="faint">${c.t('failed')}</span><b class="mono">${int(pts.filter((p) => p.s === null).length)}</b></div></aside></div>`
}
// Battle as in the original view (analysis/viz_tree.html:12048-12064): every valid program of model A
// against every valid program of model B; only a strictly higher score wins, ties count as no win.
function duels() {
  const byModel = new Map(A.modelStats.map((s) => [s.model, rows.filter((row) => row[7] === s.model && row[2] !== null).map((row) => row[2])]))
  const models = A.modelStats.map((s) => s.model)
  return models.map((a) => models.map((b) => {
    if (a === b) return null
    let wins = 0
    for (const sa of byModel.get(a)) for (const sb of byModel.get(b)) wins += sa > sb ? 1 : 0
    const n = byModel.get(a).length * byModel.get(b).length
    return { rate: n ? wins / n : null, n }
  }))
}
const modelScores = () => A.modelStats.map((s) => ({ model: s.model, scores: rows.filter((row) => row[7] === s.model && row[2] !== null).map((row) => row[2]).sort(d3.ascending) }))
function contestView(c, { w }) {
  const ranked = rows.map((row, i) => ({ row, i })).filter((p) => p.row[2] !== null).sort((a, b) => b.row[2] - a.row[2]).slice(0, 8)
  const costed = rows.map((row, i) => ({ s: row[2], cost: row[8], i })).filter((p) => p.s !== null && p.cost > 0)
  const front = []
  for (const p of [...costed].sort((a, b) => a.cost - b.cost || b.s - a.s)) if (!front.length || p.s > front.at(-1).s) front.push(p)
  const frontSet = new Set(front.map((p) => p.i))
  const pw = 292, ph = 252, m = { l: 40, r: 10, t: 10, b: 36 }
  const x = d3.scaleLog(d3.extent(costed, (p) => p.cost), [m.l, pw - m.r])
  const y = d3.scaleLinear(d3.extent(costed, (p) => p.s), [ph - m.b, m.t]).nice()
  const maxCost = d3.max(A.modelStats, (s) => s.cost)
  const D = duels()
  // Decades inside the data range only; the range spans few enough decades that labels keep apart.
  const decades = x.ticks().filter((t) => Math.abs(Math.log10(t) - Math.round(Math.log10(t))) < 1e-9)
  return `<div class="contest"><section><h3>${c.t('leaderboard')}</h3><div class="mtable" role="table"><div class="mrow lbrow mhead" role="row"><span class="num">${c.t('rank')}</span><span>${c.t('program')}</span><span class="num">${c.t('generation')}</span><span class="num">${c.t('island')}</span><span>${c.t('model')}</span><span class="num">${c.t('bestScore')}</span><span class="num">${c.t('vsParent')}</span><span class="num">${c.t('cost')}</span></div>
      ${ranked.map(({ row, i }, k) => { const parent = row[5] === -1 ? null : rows[row[5]][2]; return `<div class="mrow lbrow ${i === A.bestIndex ? 'is-selected' : ''}" role="row"><span class="num mono faint">${k + 1}</span><span class="mono">${trunc(row[6])}</span><span class="num mono">${row[3]}</span><span class="num mono">${row[4]}</span><span class="mono muted">${trunc(shortModel(row[7]))}</span><span class="num mono ${i === A.bestIndex ? 'sig' : ''}">${num(row[2])}</span><span class="num mono">${parent === null ? '—' : signed(row[2] - parent)}</span><span class="num mono">${row[8] === null ? '—' : `$${row[8].toFixed(3)}`}</span></div>` }).join('')}</div></section>
    <div class="cgrid"><section><h3>${c.t('pareto')}</h3><svg class="chart" width="${pw}" height="${ph}" viewBox="0 0 ${pw} ${ph}" role="img" aria-label="${c.t('pareto')}">
      ${y.ticks(4).map((t) => `<line x1="${m.l}" x2="${pw - m.r}" y1="${y(t)}" y2="${y(t)}" stroke="var(--grid)"/><text x="${m.l - 6}" y="${y(t) + 4}" text-anchor="end" class="axis">${num(t)}</text>`).join('')}
      ${decades.map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="${ph - m.b + 2}" y2="${ph - m.b + 6}" stroke="var(--stroke)"/><text x="${x(t)}" y="${ph - m.b + 18}" text-anchor="middle" class="axis">${t >= 0.01 ? `$${t.toFixed(2)}` : `$${t}`}</text>`).join('')}
      ${costed.filter((p) => !frontSet.has(p.i)).map((p) => `<circle cx="${x(p.cost).toFixed(1)}" cy="${y(p.s).toFixed(1)}" r="2.2" fill="none" stroke="var(--stroke)"/>`).join('')}
      <path d="${d3.line().curve(d3.curveStepAfter).x((p) => x(p.cost)).y((p) => y(p.s))(front)}" fill="none" stroke="var(--text)" stroke-width="1.6"/>
      ${front.map((p) => `<circle cx="${x(p.cost).toFixed(1)}" cy="${y(p.s).toFixed(1)}" r="3.2" fill="${p.i === A.bestIndex ? 'var(--signalG)' : 'var(--text)'}"/>`).join('')}
      <line x1="${m.l}" x2="${pw - m.r}" y1="${ph - m.b + 2}" y2="${ph - m.b + 2}" stroke="var(--stroke)"/><text x="${pw - m.r}" y="${ph - 2}" text-anchor="end" class="axis-title">${c.t('paretoAxis')}</text></svg></section>
    <section><h3>${c.t('models')}</h3>${A.modelStats.map((s) => `<div class="mbar"><div class="mbtop"><b class="mono">${esc(shortModel(s.model))}</b><span class="faint">${c.t('programsN')(int(s.programs))}</span></div>
      <div class="mbline"><span class="faint">${c.t('cost')}</span><svg width="120" height="8" viewBox="0 0 120 8" aria-hidden="true"><rect y="3" width="120" height="2" fill="var(--line2)"/><rect y="2" width="${(s.cost / maxCost) * 120}" height="4" fill="var(--muted)"/></svg><b class="mono">${money(s.cost)}</b></div>
      <div class="mbline"><span class="faint">${c.t('success')}</span><svg width="120" height="8" viewBox="0 0 120 8" aria-hidden="true"><rect y="3" width="120" height="2" fill="var(--line2)"/><rect y="2" width="${s.success * 120}" height="4" fill="var(--muted)"/></svg><b class="mono">${pct(s.success, 1)}</b></div>
      <div class="mbline"><span class="faint">${c.t('best')}</span><span></span><b class="mono ${s.best === best.score ? 'sig' : ''}">${num(s.best)}</b></div></div>`).join('')}</section>
    <section><h3>${c.t('duel')}</h3><p class="faint small">${c.t('duelDist')}</p>${(() => {
      const dist = modelScores().map((m) => ({ ...m, q: [d3.quantile(m.scores, 0), d3.quantile(m.scores, 0.25), d3.quantile(m.scores, 0.5), d3.quantile(m.scores, 0.75), d3.quantile(m.scores, 1)] })).sort((a, b) => b.q[2] - a.q[2])
      const bw = 268, bx = d3.scaleLinear(scoreExtent, [92, bw - 8]).nice()
      return `<svg class="chart" width="${bw}" height="${dist.length * 24 + 22}" viewBox="0 0 ${bw} ${dist.length * 24 + 22}" role="img" aria-label="${c.t('duelDist')}">
        ${dist.map((m, k) => { const y = k * 24 + 12; return `<text x="0" y="${y + 4}" class="axis"><tspan class="mono">${'ABC'[A.modelStats.findIndex((s) => s.model === m.model)]}</tspan> ${esc(shortModel(m.model))}</text><line x1="${bx(m.q[0])}" x2="${bx(m.q[4])}" y1="${y}" y2="${y}" stroke="var(--stroke)"/><rect x="${bx(m.q[1])}" y="${y - 6}" width="${bx(m.q[3]) - bx(m.q[1])}" height="12" fill="var(--hover)" stroke="var(--muted)"/><line x1="${bx(m.q[2])}" x2="${bx(m.q[2])}" y1="${y - 6}" y2="${y + 6}" stroke="var(--text)" stroke-width="2"/>` }).join('')}
        ${bx.ticks(4).map((t) => `<text x="${bx(t)}" y="${dist.length * 24 + 18}" text-anchor="middle" class="axis">${num(t, 1)}</text>`).join('')}</svg>`
    })()}
      <div class="duel" role="table"><div class="drow2" role="row"><span></span>${A.modelStats.map((s, k) => `<span class="num mono faint" title="${esc(s.model)}">${'ABC'[k]}</span>`).join('')}</div>
      ${A.modelStats.map((s, a) => `<div class="drow2" role="row"><span class="mono faint dname"><b>${'ABC'[a]}</b>${trunc(shortModel(s.model))}</span>${D[a].map((cell) => (cell === null ? '<span class="num faint">—</span>' : `<span class="num"><b class="mono">${pct(cell.rate)}</b><i class="mono faint">${int(cell.n)}</i></span>`)).join('')}</div>`).join('')}</div>
      <p class="faint small">${c.t('duelNote')}</p>
      <h3 class="gap">${c.t('ensemble')}</h3><p class="na">${glyph('empty', c.t('ensembleNA'))}<span class="muted small">${c.t('ensembleNA')}</span></p></section></div></div>`
}

// ======================= phone =======================
function phoneHome(c) {
  return `<div class="phone"><header class="ptop">${MARK}<b>Shinka Research</b><span class="demo">DEMO</span><span class="psearch" aria-label="${c.t('search')}">${ICON.search}</span></header>
    <div class="pscroll"><div class="psum">${homeHead(c, { phone: true })}<div class="pgauge"><span class="faint">${c.t('evidence')}</span>${gaugeAxis(286)}</div></div>
      ${dirs.map((d) => `<article class="pcard"><div class="dtitle"><b>${esc(shortTitle(d))}</b><span class="dmeta">${stateGlyph(c, d)}<span class="${d.state === 'over-budget' ? 'sig' : ''}">${STATE[c.lang][d.state]}</span><span class="faint">· ${c.t('trialsN')(int(d.trials))}</span></span></div>
        ${hypDist(c, d)}<div class="pev">${gauge(c, d, 286)}</div><div class="prow">${budget(c, d, 120)}${lock(c, d)}</div></article>`).join('')}</div>
    <nav class="ptabs" aria-label="${c.t('nav')}"><a class="is-on" aria-current="page">${ICON.directions}<span>${c.t('directionsShort')}</span></a><a>${ICON.runs}<span>${c.t('runsShort')}</span></a><a>${ICON.library}<span>${c.t('libraryShort')}</span></a></nav></div>`
}
function phoneAnalysis(c) {
  const P = A.progress
  const w = 358, h = 196, m = { l: 36, r: 10, t: 12, b: 24 }
  const x = d3.scaleLinear([0, maxGen], [m.l, w - m.r])
  const y = d3.scaleLinear([d3.min(P, (p) => p.median ?? p.best), d3.max(P, (p) => p.cumulative)], [h - m.b, m.t]).nice()
  return `<div class="phone"><header class="ptop"><span class="pback" aria-label="${c.t('back')}">${ICON.back}</span><b>${A.hypothesis} / ${A.experiment}</b>${trunc(shortTitle(dirById.get(A.direction)), 'faint')}</header>
    <div class="ptabs2" role="tablist">${VIEWS(c).map(([k, label]) => `<span role="tab" aria-selected="${k === 'progress'}" class="${k === 'progress' ? 'is-on' : ''}">${label}</span>`).join('')}</div>
    <div class="pchart"><div class="pmhead"><b>${c.t('cumBest')}</b><span class="legend"><span><i class="ln"></i>${c.t('cumBest')}</span><span><i class="ln is-dash"></i>${c.t('median')}</span></span></div>
      <svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${c.t('cumBest')}">${y.ticks(3).map((t) => `<line x1="${m.l}" x2="${w - m.r}" y1="${y(t)}" y2="${y(t)}" stroke="var(--grid)"/><text x="${m.l - 6}" y="${y(t) + 4}" text-anchor="end" class="axis">${num(t)}</text>`).join('')}
        <path d="${d3.line().x((p) => x(p.generation)).y((p) => y(p.median)).defined((p) => p.median !== null)(P)}" fill="none" stroke="var(--muted)" stroke-width="1.3" stroke-dasharray="4 3"/>
        <path d="${d3.line().curve(d3.curveStepAfter).x((p) => x(p.generation)).y((p) => y(p.cumulative))(P)}" fill="none" stroke="var(--text)" stroke-width="1.8"/>
        <circle cx="${x(best.generation)}" cy="${y(best.score)}" r="5" fill="var(--signalG)"/><circle cx="${x(best.generation)}" cy="${y(best.score)}" r="9" fill="none" stroke="var(--text)" stroke-width="1.5"/>
        ${x.ticks(5).filter(Number.isInteger).map((t) => `<text x="${x(t)}" y="${h - 6}" text-anchor="middle" class="axis">${t}</text>`).join('')}</svg></div>
    <div class="sheet"><span class="grab" aria-hidden="true"></span>${inspector(c, 'diff', { sheet: true })}<p class="pullnote faint">${c.t('pullUp')}</p></div></div>`
}

// ======================= empty and error states =======================
function statesBoard(c) {
  const d3dir = dirById.get('d3-volume-shock')
  const emptyRun = fixture.runs.find((r) => r.programs === 0)
  const failed = A.failedExample
  const hidden = fixture.events.find((e) => e.event === 'skeptic' && hasCjk(e.note))
  const card = (title, body) => `<section class="scard"><h3>${title}</h3><div class="sbody">${body}</div></section>`
  return `<div class="states"><header class="shead2"><h1>${c.t('statesTitle')}</h1><p class="muted">${c.t('statesNote')}</p></header>
    <div class="sgrid">
    ${card(c.t('emptyWorkspace'), `<p class="muted">${c.t('emptyWorkspaceHint')}</p><code class="cmd">shinka_research init &lt;${c.t('cmdDir')}&gt; --direction "&lt;${c.t('cmdQuestion')}&gt;"</code>`)}
    ${card(c.t('unreadable'), `<div class="srow">${glyph('error', c.t('unreadable'), 'is-signal')}<div><b class="mono">d5-broken-notebook</b><span class="sig">${c.t('unreadable')}</span></div></div><p class="muted">${c.t('unreadableHint')}</p>`)}
    ${card(c.t('newDirection'), `<div class="srow">${stateGlyph(c, d3dir)}<div><b>${esc(shortTitle(d3dir))}</b><span class="faint">${STATE[c.lang][d3dir.state]} · ${c.t('hypCount')(d3dir.hypotheses.length)}</span></div></div>${gauge(c, d3dir, 260)}${budget(c, d3dir, 120)}`)}
    ${card(c.t('emptyRunTitle'), `<div class="srow">${glyph('empty', c.t('emptyRunTitle'))}<div><b class="mono">${pathBreaks(runLabel(emptyRun))}</b><span class="faint">${c.t('emptyRun')} · ${c.t('programsN')(0)} · ${money(emptyRun.cost)}</span></div></div><p class="muted">${c.t('emptyRunHint')}</p>`)}
    ${card(c.t('unreadableRun'), `<div class="srow">${glyph('error', c.t('unreadableRun'), 'is-signal')}<div><b class="mono">runs/…/<wbr>programs.sqlite</b><span class="sig">${c.t('unreadableRun')}</span></div></div><p class="muted">${c.t('unreadableRunHint')}</p>`)}
    ${card(c.t('costUnknown'), `<div class="kv"><span class="faint">${c.t('cost')}</span><b class="mono">—</b><span class="faint">${c.t('programs')}</span><b class="mono">${int(220)}</b></div><p class="muted">${c.t('costUnknownHint')}</p>`)}
    ${card(c.t('evalFailed'), `<div class="srow">${glyph('error', c.t('evalFailed'), 'is-signal')}<div><b class="mono">${esc(failed.name)}</b><span class="faint">${c.t('gen')(failed.generation)} · ${c.t('island')} ${failed.island}</span></div></div>${logText(failed.stderr, 'log is-block')}`)}
    ${card(c.t('hiddenValidation'), `<div class="srow">${statusGlyph(c, 'challenged')}<div><b>${esc(shortTitle(dirById.get(hidden.direction)))} · <span class="mono">${hidden.hypothesis}</span></b><span class="faint">${STATUS[c.lang].challenged}</span></div></div><div class="kv"><span class="faint">${c.t('oosSharpe')}</span><b class="faint">${esc(hidden.note)}</b></div><p class="muted">${c.t('hiddenValidationHint')}</p>`)}
    </div></div>`
}

// ======================= CSS =======================
const CSS = `
.f{position:relative;overflow:hidden;background:var(--bg);color:var(--text);font:400 13px/20px InterV,"PingFang SC","Microsoft YaHei","Noto Sans SC",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.f *,.f *::before,.f *::after{box-sizing:border-box} .f a{color:inherit;text-decoration:none} .f h1,.f h2,.f h3,.f p,.f dl,.f dd{margin:0}
.f .mono{font-family:PlexMono,ui-monospace,monospace;font-variant-numeric:tabular-nums} .f .num{text-align:right;justify-self:end}
.f .faint{color:var(--faint)} .f .muted{color:var(--muted)} .f .sig{color:var(--signal)} .f .small{font-size:12px;line-height:18px}
.f .trunc{display:block;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.f .glyph{flex:none;color:var(--text)} .f .glyph.is-signal{color:var(--signal)}
.f .demo{font:500 10px/16px PlexMono,monospace;letter-spacing:.04em;color:var(--muted);border:1px solid var(--line2);border-radius:3px;padding:0 4px}
.f .app{display:flex;height:100%} .f .main{flex:1;min-width:0;display:flex;flex-direction:column;min-height:0}
.f .rail{width:232px;flex:none;background:var(--rail);border-right:1px solid var(--line);padding:12px 10px;display:flex;flex-direction:column;gap:2px;min-height:0}
.f .brand{display:flex;align-items:center;gap:9px;height:32px;padding:0 8px;font-weight:600;font-size:14px}
.f .ws{display:flex;align-items:center;justify-content:space-between;padding:0 8px 0 35px;margin-top:-4px;font-size:12px;color:var(--faint)}
.f .search{display:flex;align-items:center;gap:8px;height:32px;margin:12px 0 10px;padding:0 6px 0 10px;border:1px solid var(--stroke);border-radius:6px;background:var(--bg);color:var(--muted);font:inherit;font-size:12.5px;text-align:left}
.f .search span{flex:1;min-width:0;white-space:nowrap}
.f kbd{font:500 11px/16px PlexMono,monospace;color:var(--muted);border:1px solid var(--line2);border-radius:4px;padding:0 4px;white-space:nowrap}
.f .nav{position:relative;display:flex;align-items:center;gap:10px;height:32px;padding:0 10px;border-radius:6px;color:var(--muted)}
.f .nav .count{margin-left:auto;font:400 12px PlexMono,monospace;color:var(--faint)}
.f .nav.is-on,.f .navn.is-on{background:var(--hover);color:var(--text)} .f .nav.is-on::before{content:"";position:absolute;left:-10px;top:7px;bottom:7px;width:2px;background:var(--text)}
.f .railhead{margin:20px 10px 6px;font-size:12px;color:var(--faint)}
.f .railitem{display:flex;gap:9px;align-items:center;height:30px;padding:0 10px;border-radius:6px;color:var(--muted);min-width:0}
.f .railitem.is-on{background:var(--hover);color:var(--text)}
.f .railfoot{margin-top:auto;display:flex;gap:8px;align-items:center;padding:10px;color:var(--faint);font-size:12px}
.f .rail.is-narrow{width:68px;padding:12px 6px;align-items:stretch} .f .rail.is-narrow .brand{justify-content:center;padding:0;margin-bottom:10px}
.f .navn{position:relative;display:grid;justify-items:center;gap:3px;padding:8px 0 6px;border-radius:6px;color:var(--muted);font-size:11px;line-height:14px}
.f .navn.is-on::before{content:"";position:absolute;left:-6px;top:10px;bottom:10px;width:2px;background:var(--text)} .f .rail.is-narrow .railfoot{justify-content:center;padding:10px 0}
.f .top{height:52px;flex:none;display:flex;align-items:center;gap:24px;padding:0 20px 0 28px;border-bottom:1px solid var(--line)}
.f .crumbs{display:flex;gap:8px;align-items:center;min-width:0;white-space:nowrap;color:var(--muted)} .f .crumbs .trunc{max-width:300px} .f .crumbs .is-here{color:var(--text);font-weight:500} .f .crumbs .sep{color:var(--faint)}
.f .vtabs{display:flex;gap:4px;align-self:stretch} .f .vtabs span{position:relative;display:flex;align-items:center;padding:0 10px;color:var(--muted);white-space:nowrap} .f .vtabs .is-on{color:var(--text);font-weight:500} .f .vtabs .is-on::after{content:"";position:absolute;left:10px;right:10px;bottom:-1px;height:2px;background:var(--text)}
.f .tools{margin-left:auto;display:flex;gap:10px;align-items:center;white-space:nowrap;font-size:12px}
.f .updated{color:var(--faint)} .f .updated .mono{color:var(--muted)}
.f .seg{display:flex;gap:2px;padding:2px;border:1px solid var(--line2);border-radius:7px} .f .seg span{padding:2px 9px;border-radius:5px;border:1px solid transparent;color:var(--muted);font-size:12px;line-height:18px;white-space:nowrap}
.f .seg .is-on{color:var(--text);border-color:var(--stroke);background:var(--raised)}
.f .homebody{flex:1;min-height:0;display:flex} .f .homebody.is-stacked{display:block;overflow:hidden}
.f .content{flex:1;min-width:0;padding:30px 28px 24px;overflow:hidden}
.f .dsent{font-size:28px;line-height:40px;font-weight:600;letter-spacing:-.015em;max-width:880px;text-wrap:balance}
.f .dsent a{white-space:nowrap;text-decoration:underline;text-decoration-color:var(--stroke);text-decoration-thickness:1px;text-underline-offset:7px}
.f .dsent a.is-signal{color:var(--signal);text-decoration-color:currentColor}
.f .dsub{margin-top:8px;color:var(--muted);max-width:820px}
.f .kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));margin:24px 0 26px;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
.f .kpis div{display:grid;gap:2px;padding:12px 16px 14px;border-left:1px solid var(--line)} .f .kpis div:first-child{border-left:0;padding-left:0}
.f .kpis span{color:var(--muted);font-size:12px} .f .kpis b{font-size:22px;line-height:30px;font-weight:500} .f .kpis i{font-style:normal;color:var(--faint);font-size:14px} .f .kpis em{font-style:normal;color:var(--faint);font-size:12px}
.f .dtable{border-top:1px solid var(--line)}
.f .drow{display:grid;grid-template-columns:minmax(0,1fr) var(--hw) var(--gw) 60px 112px 64px;column-gap:16px;align-items:center;padding:10px 8px 10px 12px;border-bottom:1px solid var(--line);min-height:78px;position:relative}
.f .drow.dhead{min-height:0;padding-top:8px;padding-bottom:8px;color:var(--faint);font-size:12px;align-items:end}
.f .drow.is-focus{background:var(--selected)} .f .drow.is-focus::before{content:"";position:absolute;left:0;top:0;bottom:0;width:2px;background:var(--text)}
.f .evhead{display:grid;gap:2px} .f .axis{font:400 10.5px PlexMono,monospace;fill:var(--faint)} .f .axis.sig{fill:var(--signal)} .f .axis-title{font-size:11px;fill:var(--faint)}
.f .dtitle{display:grid;gap:4px;min-width:0} .f .dtitle b{font-weight:500;font-size:13.5px;line-height:20px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.f .dmeta{display:flex;gap:6px;align-items:center;font-size:12px;color:var(--muted);white-space:nowrap}
.f .hyps{display:flex;flex-wrap:wrap;gap:4px 12px} .f .hyps span{display:inline-flex;gap:5px;align-items:center;color:var(--muted);font-size:12px;white-space:nowrap} .f .hyps em{font-style:normal} .f .hyps b{font-weight:500;color:var(--text)}
.f .gauge{display:grid;gap:2px} .f .gauge.is-empty{color:var(--faint);font-size:12px} .f .gread{display:flex;gap:12px;font-size:11.5px;color:var(--faint)} .f .gread b{font-weight:500;color:var(--text);font-size:12px}
.f .trials{display:grid;gap:2px} .f .trials b{font-weight:500;font-size:13px} .f .trials span{font-size:11.5px;white-space:nowrap}
.f .budget{display:grid;gap:3px;font-size:12px} .f .bline{display:flex;align-items:center;gap:8px}
.f .lock{display:inline-flex;gap:6px;align-items:center;color:var(--muted);font-size:12px;white-space:nowrap} .f .lock.is-open{color:var(--text)}
.f .key{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;margin-top:14px;padding-left:12px;font-size:12px;color:var(--muted)} .f .key span{display:inline-flex;gap:5px;align-items:center}
.f .keynote{margin-top:6px;padding-left:12px;font-size:12px;line-height:18px;color:var(--faint);max-width:820px}
.f .feed{width:280px;flex:none;border-left:1px solid var(--line);background:var(--rail);padding:0 16px;overflow:hidden}
.f .feed header{height:48px;display:flex;align-items:center;justify-content:space-between;margin:0 -16px 4px;padding:0 16px;border-bottom:1px solid var(--line);font-size:12.5px} .f .feed header b{font-weight:500}
.f .feed.is-below{width:auto;border:1px solid var(--line);margin-top:24px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));column-gap:24px;background:none}
.f .feed.is-below header{grid-column:1/-1}
.f .ev{display:grid;grid-template-columns:14px minmax(0,1fr);gap:10px;padding:10px 0;border-bottom:1px solid var(--line)}
.f .evmark{display:flex;justify-content:center;padding-top:4px;color:var(--muted)} .f .evmark.is-signal{color:var(--signal)}
.f .evbody{display:grid;gap:2px;min-width:0} .f .evtitle{display:flex;gap:8px;align-items:baseline;min-width:0;white-space:nowrap} .f .evtitle b{font-weight:500;font-size:12.5px;flex:none} .f .evtitle .trunc{font-size:12px}
.f .evref{display:flex;justify-content:space-between;font-size:11px;line-height:16px}
.f code.log{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;margin-top:2px;padding-left:8px;border-left:1px solid var(--line2);font:400 11.5px/17px PlexMono,ui-monospace,monospace;color:var(--muted);overflow-wrap:anywhere}
.f code.log.is-block{display:block;-webkit-line-clamp:unset;padding:8px 10px;border:1px solid var(--line);border-radius:6px;background:var(--bg);white-space:pre-wrap}
.f .report{flex:1;min-height:0;overflow:hidden;padding:24px 32px}
.f .rhead h1{font-size:26px;line-height:34px;font-weight:600;letter-spacing:-.01em}
.f .rmeta{display:flex;gap:16px;align-items:center;margin-top:8px;font-size:12.5px;color:var(--muted);white-space:nowrap} .f .rstate{display:inline-flex;gap:6px;align-items:center;color:var(--text)}
.f .htabs{display:flex;gap:4px;margin-top:16px;border-bottom:1px solid var(--line)} .f .htabs span{position:relative;display:flex;gap:7px;align-items:center;padding:8px 12px 10px;color:var(--muted)} .f .htabs .is-on{color:var(--text)} .f .htabs .is-on::after{content:"";position:absolute;left:12px;right:12px;bottom:-1px;height:2px;background:var(--text)}
.f .rcols{display:grid;grid-template-columns:700px minmax(0,1fr);gap:40px;margin-top:20px} .f .rmain{display:grid;gap:22px;align-content:start} .f .rside{display:grid;gap:22px;align-content:start}
.f .report h2,.f .models h3,.f .sideinfo h3,.f .contest h3{font-size:14px;line-height:22px;font-weight:500;margin-bottom:8px}
.f .claim{font-size:17px;line-height:26px;font-weight:500;margin-bottom:10px}
.f .defs{display:grid;grid-template-columns:72px minmax(0,1fr);gap:4px 16px;color:var(--muted)} .f .defs dt{color:var(--faint);font-size:12px}
.f .shead{display:flex;justify-content:space-between;align-items:baseline;gap:16px}
.f .legend{display:flex;gap:14px;align-items:center;font-size:12px;color:var(--muted);white-space:nowrap} .f .legend span{display:inline-flex;gap:6px;align-items:center}
.f .legend i{display:inline-block} .f .legend .ln{width:16px;border-top:1.6px solid var(--text)} .f .legend .ln.is-dash{border-top:1.4px dashed var(--muted)} .f .legend .ln.is-signal{border-top:2px solid var(--signalG)} .f .legend .ln.is-contour{border-top:1px solid var(--stroke)}
.f .legend .dot{width:6px;height:6px;border-radius:50%;background:var(--muted)} .f .legend .dot.is-hollow{background:none;border:1px solid var(--stroke)} .f .legend .sw{width:10px;height:10px}
.f .ramp{display:inline-flex} .f .ramp i{width:14px;height:10px}
.f .chart{display:block;overflow:visible} .f .lbl{font:500 11.5px PlexMono,monospace;fill:var(--text)} .f .lbl.sig{fill:var(--signal)} .f .lbl.muted{fill:var(--muted)}
.f .halo{paint-order:stroke;stroke:var(--bg);stroke-width:4px;stroke-linejoin:round}
.f .ltable{border-top:1px solid var(--line)} .f .lrow{display:grid;grid-template-columns:minmax(0,1fr) 84px 72px 72px;column-gap:10px;align-items:center;min-height:36px;border-bottom:1px solid var(--line);padding:4px 4px;font-size:12.5px;line-height:18px}
.f .lrow > span:last-child{display:inline-flex;gap:5px;align-items:center} .f .lhead{min-height:30px;color:var(--faint);font-size:12px} .f .op{font-family:"Segoe UI",system-ui,sans-serif}
.f .vline{display:flex;gap:8px;align-items:center;margin-bottom:8px} .f .vline b{font-weight:500;font-size:15px} .f .vline .faint{margin-left:auto;font-size:11.5px} .f .vline.is-sub{margin-top:14px} .f .vline.is-sub b{font-size:13px}
.f .verdict p{margin-top:12px}
.f .xrow{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;padding:12px 14px;border:1px solid var(--line2);border-radius:8px;color:var(--muted);font-size:12.5px} .f .xrow b{color:var(--text);font-weight:500} .f .xrow .btn{margin-left:auto}
.f .btn{display:inline-flex;gap:8px;align-items:center;justify-self:start;height:30px;padding:0 12px;border:1px solid var(--line2);border-radius:6px;font-size:12.5px;font-weight:500;white-space:nowrap;color:var(--text)}
.f .btn.is-primary{background:var(--text);color:var(--bg);border-color:var(--text)}
.f .runs,.f .compare{flex:1;min-height:0;overflow:hidden;padding:22px 28px}
.f .toolbar{display:flex;gap:12px;align-items:center;margin-bottom:16px}
.f .input{display:flex;gap:8px;align-items:center;width:300px;height:32px;padding:0 10px;border:1px solid var(--stroke);border-radius:6px;color:var(--faint)}
.f .select{display:inline-flex;gap:6px;align-items:center;height:32px;padding:0 10px;border:1px solid var(--line2);border-radius:6px;color:var(--muted);white-space:nowrap}
.f .selbar{margin-left:auto;display:flex;gap:12px;align-items:center;color:var(--muted);white-space:nowrap}
.f .rtable{border-top:1px solid var(--line)} .f .rrow{display:grid;grid-template-columns:20px minmax(0,1fr) 64px 76px 72px 56px 76px 40px 44px 128px;column-gap:14px;align-items:center;min-height:42px;padding:0 12px;border-bottom:1px solid var(--line)}
.f .rhead2{min-height:34px;color:var(--faint);font-size:12px} .f .rrow.is-selected{background:var(--selected)}
.f .cb{width:16px;height:16px;border:1px solid var(--stroke);border-radius:4px;display:grid;place-items:center} .f .cb.is-on{background:var(--text);border-color:var(--text);color:var(--bg)}
.f .rname{min-width:0;font-size:12.5px;font-weight:500;overflow-wrap:anywhere}
.f .empty{display:inline-flex;gap:6px;align-items:center;color:var(--faint);font-family:InterV,sans-serif} .f .empty .glyph{color:var(--faint)}
.f .chead h1{font-size:24px;line-height:32px;font-weight:600} .f .chead p{margin-top:4px}
.f .chips{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 20px} .f .chip{display:inline-flex;gap:8px;align-items:center;height:30px;padding:0 12px;border:1px solid var(--line2);border-radius:6px;font-size:12.5px} .f .chip b{font-weight:500;font-family:PlexMono,monospace} .f .chip.is-add{color:var(--muted);border-style:dashed}
.f .ctable2{border-top:1px solid var(--line)} .f .crow{display:grid;grid-template-columns:150px 120px 120px 120px minmax(0,1fr);column-gap:20px;align-items:center;min-height:56px;padding:6px 8px;border-bottom:1px solid var(--line)}
.f .crow.chd{min-height:36px;color:var(--faint);font-size:12px} .f .crow.chd .num{display:inline-flex;gap:6px;align-items:center;color:var(--muted)} .f .crow .is-lead{font-weight:500;box-shadow:inset 0 -1px 0 var(--text)}
.f .analysis{flex:1;min-height:0;display:flex} .f .canvas{flex:1;min-width:0;position:relative;overflow:hidden;display:flex;flex-direction:column}
.f .rstats{font-size:12px}
.f .vbar{height:44px;flex:none;display:flex;align-items:center;justify-content:space-between;gap:16px;padding:0 20px;border-bottom:1px solid var(--line)}
.f .lane{font-size:13px;font-weight:500;fill:var(--text)}
.f .btntext{font-size:12px;fill:var(--text)} .f .scrubnote{position:absolute;right:28px;bottom:18px;font-size:11.5px}
.f .clabel{font-size:12px;font-weight:500;fill:var(--muted)}
.f .inspector{width:440px;flex:none;border-left:1px solid var(--line);background:var(--rail);display:flex;flex-direction:column;min-height:0}
.f .ihead{padding:16px 20px 14px;display:grid;gap:8px;border-bottom:1px solid var(--line)} .f .iname{display:flex;gap:10px;align-items:baseline;min-width:0} .f .iname h2{font-size:15px;line-height:22px;font-weight:500;white-space:nowrap}
.f .bestmark{flex:none;font-size:11px;line-height:18px;padding:0 6px;border-radius:4px;border:1px solid var(--signal);color:var(--signal)}
.f .iscore{display:flex;gap:12px;align-items:baseline} .f .iscore b{font-size:28px;line-height:34px;font-weight:500} .f .iscore span{font-size:12px}
.f .ichips{display:flex;gap:6px;flex-wrap:wrap} .f .ichips span{border:1px solid var(--line2);border-radius:5px;padding:0 7px;color:var(--muted);font-size:12px;line-height:20px}
.f .itabs{display:flex;gap:2px;padding:0 12px 0 14px;height:40px;flex:none;align-items:stretch;border-bottom:1px solid var(--line)} .f .itabs span{position:relative;display:flex;align-items:center;padding:0 8px;color:var(--muted);white-space:nowrap}
.f .itabs .is-on{color:var(--text);font-weight:500} .f .itabs .is-on::after{content:"";position:absolute;left:8px;right:8px;bottom:-1px;height:2px;background:var(--text)} .f .itabs .dlbtn{margin-left:auto;display:flex;align-items:center;color:var(--muted)}
.f .codebar{display:flex;justify-content:space-between;align-items:center;height:34px;flex:none;padding:0 16px 0 20px;border-bottom:1px solid var(--line);font-size:11.5px} .f .toggle{color:var(--muted)} .f .toggle b{font-weight:500;color:var(--text);margin-left:4px;padding:0 6px;border:1px solid var(--stroke);border-radius:4px}
.f .code{margin:0;padding:10px 0 12px;flex:1;min-height:0;overflow:auto;font:400 12px/20px PlexMono,ui-monospace,monospace;color:var(--text);white-space:pre;scrollbar-width:thin}
.f .code .ln{display:inline-block;width:42px;padding-right:14px;text-align:right;color:var(--faint);user-select:none} .f .code .k{font-weight:500} .f .code .s{color:var(--muted)} .f .code .cm{color:var(--faint);font-style:italic}
.f .code.is-diff{padding:10px 0 12px} .f .code .dl{display:block;padding:0 16px 0 20px;min-width:max-content} .f .code .add{box-shadow:inset 2px 0 0 var(--text);background:var(--hover)} .f .code .del{color:var(--muted);text-decoration:line-through;text-decoration-color:var(--stroke)} .f .code .hunk{color:var(--faint)}
.f .ibody{padding:14px 20px;display:grid;gap:10px;align-content:start;overflow:hidden} .f .ibody h3{font-size:12px;font-weight:500;color:var(--muted);margin-top:4px}
.f .ibody.is-meta{overflow:auto;flex:1;min-height:0;gap:8px} .f .metahead{display:flex;justify-content:space-between;align-items:baseline;gap:12px} .f .metahead b{font-weight:500}
.f .metalist{display:grid;gap:6px;font-size:12.5px;line-height:19px} .f .metalist .item{padding-left:12px;position:relative;overflow-wrap:anywhere} .f .metalist .item::before{content:"";position:absolute;left:2px;top:9px;width:4px;height:1px;background:var(--stroke)}
.f .metalist .item.is-num{padding-left:18px} .f .metalist .item.is-num::before{display:none} .f .metalist .item.is-num span{position:absolute;left:0} .f .metalist h4{margin:8px 0 0;font-size:12px;font-weight:500;color:var(--muted)}
.f .axis.is-strong{fill:var(--text);font-weight:500}
.f .metrics{display:grid;grid-template-columns:1fr 1fr;border:1px solid var(--line);border-radius:6px} .f .metrics div{display:grid;gap:2px;padding:8px 12px;border-bottom:1px solid var(--line)} .f .metrics div:nth-child(odd){border-right:1px solid var(--line)} .f .metrics div:nth-last-child(-n+2){border-bottom:0} .f .metrics span{font-size:11px} .f .metrics b{font-weight:500;font-size:15px}
.f .kv{display:grid;grid-template-columns:auto minmax(0,1fr);gap:4px 16px;font-size:12.5px} .f .kv b{font-weight:500;overflow-wrap:anywhere}
.f .progress{padding:4px 0 0;overflow:hidden;flex:1} .f .pm{border-bottom:1px solid var(--line)} .f .pmhead{display:flex;justify-content:space-between;align-items:center;padding:10px 28px 0 20px} .f .pmhead b{font-size:13px;font-weight:500}
.f .models{padding:14px 28px 0 20px} .f .mtable{border-top:1px solid var(--line)} .f .mrow{display:grid;grid-template-columns:minmax(0,1fr) 80px 80px 70px 70px 80px;column-gap:14px;align-items:center;min-height:32px;border-bottom:1px solid var(--line);padding:0 4px;font-size:12.5px}
.f .mhead,.f .chead2{min-height:28px;color:var(--faint);font-size:12px} .f .mrow.is-selected{background:var(--selected)}
.f .disclosure{display:flex;gap:8px;align-items:center;margin:14px 28px 0 20px;padding:10px 12px;border:1px solid var(--line2);border-radius:6px;color:var(--muted)} .f .disclosure b{font-weight:500;color:var(--text)} .f .disclosure .faint{margin-left:auto}
.f .spacewrap{flex:1;display:flex;min-height:0} .f .sideinfo{width:320px;flex:none;border-left:1px solid var(--line);padding:18px 20px;display:grid;gap:10px;align-content:start}
.f .sideinfo .mrow{grid-template-columns:minmax(0,1fr) 56px 52px 44px} .f .sideinfo h3 + .kv{margin-top:-2px}
.f .contest{padding:16px 20px;display:grid;gap:18px;align-content:start;overflow:hidden;flex:1}
.f .lbrow{grid-template-columns:28px minmax(0,1fr) 40px 36px 104px 56px 60px 64px}
.f .cgrid{display:grid;grid-template-columns:292px minmax(0,1fr) 268px;gap:28px}
.f .mbar{display:grid;gap:4px;padding:8px 0;border-bottom:1px solid var(--line)} .f .mbtop{display:flex;justify-content:space-between;align-items:baseline;font-size:12.5px} .f .mbtop b{font-weight:500}
.f .mbline{display:grid;grid-template-columns:48px 120px minmax(0,1fr);gap:10px;align-items:center;font-size:12px} .f .mbline b{font-weight:500;text-align:right}
.f .duel{border-top:1px solid var(--line);margin-bottom:8px} .f .drow2{display:grid;grid-template-columns:104px repeat(3,minmax(0,1fr));column-gap:8px;align-items:center;min-height:34px;border-bottom:1px solid var(--line);font-size:12px}
.f .dname{display:flex;gap:6px;min-width:0} .f .dname b{font-weight:500;color:var(--muted)}
.f .drow2 .num{display:grid;justify-items:end;line-height:15px} .f .drow2 b{font-weight:500} .f .drow2 i{font-style:normal;font-size:10.5px}
.f .contest h3.gap{margin-top:16px} .f .na{display:flex;gap:8px;align-items:flex-start} .f .na .glyph{margin-top:2px;color:var(--faint)} .f .na b{font-weight:500;color:var(--text)} .f .models .na{margin-top:10px}
.f .phone{height:100%;display:flex;flex-direction:column;background:var(--bg);position:relative}
.f .ptop{height:52px;flex:none;display:flex;align-items:center;gap:10px;padding:0 16px;border-bottom:1px solid var(--line);white-space:nowrap;min-width:0} .f .ptop b{font-weight:600} .f .ptop .trunc{font-size:12px}
.f .psearch{margin-left:auto;display:grid;place-items:center;width:36px;height:36px;border:1px solid var(--stroke);border-radius:6px;color:var(--muted)} .f .pback{display:grid;place-items:center;width:32px;height:32px;margin-left:-8px;flex:none}
.f .pscroll{flex:1;min-height:0;overflow-y:auto;scrollbar-width:none}
.f .psum{padding:18px 16px 4px} .f .dsent.is-phone{font-size:21px;line-height:32px;text-underline-offset:5px} .f .psum .dsub{font-size:12.5px;line-height:19px}
.f .pgauge{display:grid;grid-template-columns:1fr 286px;align-items:center;margin-top:14px;padding-top:10px;border-top:1px solid var(--line);font-size:12px}
.f .pcard{margin:0 16px;padding:14px 0;border-bottom:1px solid var(--line);display:grid;gap:10px} .f .pev{display:flex;justify-content:flex-end} .f .prow{display:flex;justify-content:space-between;align-items:flex-end;gap:12px}
.f .ptabs{height:72px;flex:none;display:flex;justify-content:space-around;align-items:flex-start;padding-top:8px;border-top:1px solid var(--line);background:var(--rail)}
.f .ptabs a{display:grid;justify-items:center;gap:2px;min-width:64px;color:var(--muted);font-size:11px} .f .ptabs .is-on{color:var(--text);font-weight:500}
.f .ptabs2{display:flex;gap:2px;padding:0 10px;height:42px;flex:none;border-bottom:1px solid var(--line)} .f .ptabs2 span{position:relative;display:flex;align-items:center;padding:0 10px;color:var(--muted)} .f .ptabs2 .is-on{color:var(--text);font-weight:500} .f .ptabs2 .is-on::after{content:"";position:absolute;left:10px;right:10px;bottom:-1px;height:2px;background:var(--text)}
.f .pchart{padding:10px 16px} .f .pchart .pmhead{padding:0 0 4px}
.f .sheet{position:absolute;left:0;right:0;bottom:0;height:396px;background:var(--rail);border-top:1px solid var(--line2);border-radius:8px 8px 0 0;display:flex;flex-direction:column}
.f .grab{display:block;width:36px;height:4px;border-radius:2px;background:var(--stroke);margin:8px auto 0;flex:none}
.f .inspector.is-sheet{width:auto;border:0;flex:1;min-height:0;background:none} .f .inspector.is-sheet .ihead{padding:8px 16px 12px} .f .inspector.is-sheet .itabs{padding:0 6px}
.f .pullnote{text-align:center;font-size:11px;padding:6px 0 18px;border-top:1px solid var(--line)}
.f .states{padding:28px 32px;height:100%;overflow:hidden} .f .shead2 h1{font-size:24px;line-height:32px;font-weight:600} .f .shead2 p{margin-top:4px}
.f .sgrid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin-top:22px;align-items:stretch}
.f .scard{border:1px solid var(--line2);border-radius:8px;padding:16px;display:grid;gap:12px;align-content:start;background:var(--rail)} .f .scard h3{font-size:13.5px;font-weight:500}
.f .sbody{display:grid;gap:12px;align-content:start} .f .srow{display:grid;grid-template-columns:14px minmax(0,1fr);gap:10px;align-items:start} .f .srow .glyph{margin-top:3px} .f .srow div{display:grid;gap:2px;min-width:0} .f .srow b{font-weight:500}
.f .cmd{display:block;padding:10px;border:1px solid var(--line2);border-radius:6px;background:var(--bg);font:400 11.5px/18px PlexMono,monospace;white-space:pre-wrap}
`

// ======================= frames =======================
const ctx = (theme, lang) => ({ theme, lang, th: THEMES[theme], t: (key) => T[lang][key] })
const FRAMES = [
  { id: 'home-dark-zh', group: 'home', w: 1440, h: 900, theme: 'dark', lang: 'zh', label: '首页 · 深色 · 中文 · 1440×900', render: (c) => home(c) },
  { id: 'home-light-zh', group: 'home', w: 1440, h: 900, theme: 'light', lang: 'zh', label: '首页 · 浅色 · 中文', render: (c) => home(c) },
  { id: 'home-dark-en', group: 'home', w: 1440, h: 900, theme: 'dark', lang: 'en', label: '首页 · 深色 · English', render: (c) => home(c) },
  { id: 'home-1024', group: 'home', w: 1024, h: 768, theme: 'dark', lang: 'zh', label: '首页 · 1024×768 · 带文字窄栏，动态移到表格下方', render: (c) => home(c, { narrow: true, stacked: true, gaugeW: 200 }) },
  { id: 'report', group: 'research', w: 1440, h: 900, theme: 'dark', lang: 'zh', label: '方向研究报告 · 已结论方向 · h001', render: (c) => report(c) },
  { id: 'runs', group: 'runs', w: 1440, h: 900, theme: 'dark', lang: 'zh', label: '全部运行 · 勾选 3 个', render: (c) => runsPage(c) },
  { id: 'compare', group: 'runs', w: 1440, h: 900, theme: 'dark', lang: 'zh', label: '多运行比较 · 运行级指标', render: (c) => comparePage(c) },
  { id: 'lineage', group: 'analysis', w: 1440, h: 900, theme: 'dark', lang: 'zh', label: '实验分析 · 谱系（含提示词与元分析轨道）+ 检查器（元分析）', render: (c) => analysisShell(c, 'lineage', lineageView(c, { w: 932, h: 760 }), 'meta') },
  { id: 'progress', group: 'analysis', w: 1440, h: 900, theme: 'dark', lang: 'zh', label: '实验分析 · 进展 + 检查器（代码）', render: (c) => analysisShell(c, 'progress', progressView(c, { w: 932 }), 'code') },
  { id: 'space', group: 'analysis', w: 1440, h: 900, theme: 'dark', lang: 'zh', label: '实验分析 · 解空间 + 检查器（评估）', render: (c) => analysisShell(c, 'space', spaceView(c, { w: 932, h: 760 }), 'evaluation') },
  { id: 'contest', group: 'analysis', w: 1440, h: 900, theme: 'dark', lang: 'zh', label: '实验分析 · 对比 + 检查器（LLM 记录）', render: (c) => analysisShell(c, 'contest', contestView(c, { w: 932 }), 'llm') },
  { id: 'phone-home', group: 'phone', w: 390, h: 844, theme: 'dark', lang: 'zh', label: '手机首页 · 390×844 · 框内可滚动', render: (c) => phoneHome(c) },
  { id: 'phone-analysis', group: 'phone', w: 390, h: 844, theme: 'dark', lang: 'zh', label: '手机分析 · 检查器抽屉（中间高度，差异）', render: (c) => phoneAnalysis(c) },
  { id: 'states', group: 'states', w: 1440, h: 900, theme: 'dark', lang: 'zh', label: '空与错误状态板', render: (c) => statesBoard(c) },
]
const GROUPS = [['home', '研究方向首页'], ['research', '方向研究报告'], ['runs', '运行与比较'], ['analysis', '实验分析'], ['phone', '手机'], ['states', '空与错误状态']]
const frameHtml = (f) => `<div class="f" data-frame="${f.id}" data-theme="${f.theme}" lang="${f.lang === 'zh' ? 'zh-CN' : 'en'}" style="width:${f.w}px;height:${f.h}px;--gw:${f.id === 'home-1024' ? 200 : 176}px;--hw:${f.lang === 'en' ? 196 : 168}px">${f.render(ctx(f.theme, f.lang))}</div>`
const html = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shinka Research · 高保真样稿（评审用，非最终产品）</title>
<style>
@font-face{font-family:InterV;src:url(data:font/woff2;base64,${INTER}) format("woff2");font-weight:100 900;font-display:block}
@font-face{font-family:PlexMono;src:url(data:font/woff2;base64,${PLEX400}) format("woff2");font-weight:400;font-display:block}
@font-face{font-family:PlexMono;src:url(data:font/woff2;base64,${PLEX500}) format("woff2");font-weight:500;font-display:block}
*{box-sizing:border-box} body{margin:0;background:#d4d4d8;color:#18181b;font:400 14px/1.6 system-ui,"PingFang SC","Microsoft YaHei",sans-serif}
.board-head{padding:28px 32px 4px;max-width:1200px} .board-head h1{margin:0;font-size:22px} .board-head p{margin:6px 0 0;color:#3f3f46}
.group{margin:20px 32px 40px} .group > h2{margin:0 0 12px;font-size:17px}
.slots{display:grid;grid-template-columns:repeat(auto-fill,minmax(560px,1fr));gap:20px;align-items:start} .slots.is-phone{grid-template-columns:repeat(auto-fill,minmax(260px,320px))}
.slot{position:relative;overflow:hidden;background:#000} .slot > .f{position:absolute;left:0;top:0;transform-origin:0 0}
.slot-label{font-size:12.5px;color:#3f3f46;margin:6px 0 0}
body.capture{background:#000} body.capture .board-head,body.capture .group > h2,body.capture .slot-label{display:none} body.capture .group{margin:0} body.capture .slots{display:block}
body.capture .slot{display:none;position:static;overflow:visible} body.capture .slot.is-target{display:block} body.capture .slot > .f{position:relative;transform:none!important} body.capture .group:not(.has-target){display:none}
${themeCss('dark')}
${themeCss('light')}
${CSS}
</style></head><body data-hifi="award-redesign">
<header class="board-head"><h1>Shinka Research 高保真样稿 · 暗场仪器（评审用，非最终产品）</h1>
<p>全部画面使用同一份演示数据（DEMO · 演示数据，非真实研究）。所有者 2026-10-19 选定 A · 暗场仪器为底，吸收 B 的按岛分道谱系、进展小图组与首页数据句标题；默认深色并交付浅色；中英切换。token 与规则见 design.md。</p></header>
${GROUPS.map(([g, title]) => `<section class="group" data-group="${g}"><h2>${title}</h2><div class="slots ${g === 'phone' ? 'is-phone' : ''}">${FRAMES.filter((f) => f.group === g).map((f) => `<div><div class="slot" data-w="${f.w}" data-h="${f.h}">${frameHtml(f)}</div><p class="slot-label">${f.label}</p></div>`).join('')}</div></section>`).join('\n')}
<script>
const params = new URLSearchParams(location.search)
const target = params.get('frame')
// Layout and contrast checks for one frame (?frame=<id>&check=1). The result lands in
// body[data-check] for the capture harness; it is recorded in review-manifest.json.
function hifiCheck(id) {
  const frame = document.querySelector('[data-frame="' + CSS.escape(id) + '"]')
  const fr = frame.getBoundingClientRect()
  const parse = (s) => { const m = s.match(/rgba?\\(([^)]+)\\)/); if (!m) return null; const p = m[1].split(/[ ,/]+/).filter(Boolean).map(Number); return [p[0], p[1], p[2], p[3] === undefined ? 1 : p[3]] }
  const lin = (v) => { v /= 255; return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4) }
  const lum = (c) => 0.2126 * lin(c[0]) + 0.7152 * lin(c[1]) + 0.0722 * lin(c[2])
  const ratio = (a, b) => { const x = Math.max(lum(a), lum(b)), y = Math.min(lum(a), lum(b)); return (x + 0.05) / (y + 0.05) }
  const over = (fg, bg) => [0, 1, 2].map((i) => fg[i] * fg[3] + bg[i] * (1 - fg[3]))
  const bgOf = (el) => { const stack = []; for (let n = el; n; n = n.parentElement) { const c = parse(getComputedStyle(n).backgroundColor); if (c && c[3] > 0) { stack.push(c); if (c[3] >= 1) break } } let bg = [255, 255, 255]; stack.reverse().forEach((c) => { bg = over(c, bg) }); return bg }
  const label = (el) => el.tagName.toLowerCase() + '.' + [...el.classList].join('.') + ': ' + el.textContent.trim().slice(0, 50)
  const out = { overflowX: document.documentElement.scrollWidth > innerWidth, clipped: [], allowedTrunc: 0, badTrunc: [], scrollX: 0, badScroll: [], svgOut: [], overlaps: [], lowContrast: [], minTextContrast: null }
  const seen = (r) => { out.minTextContrast = out.minTextContrast === null ? r : Math.min(out.minTextContrast, r) }
  for (const el of frame.querySelectorAll('*')) {
    if (el.closest('svg')) continue
    if (el.hasAttribute('data-scroll-x')) { out.scrollX++; if (el.getAttribute('tabindex') !== '0' || !el.getAttribute('aria-label')) out.badScroll.push(label(el)); continue }
    if (el.closest('[data-scroll-x]')) continue
    const direct = [...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim())
    const cs = getComputedStyle(el)
    const clipped = (cs.overflow.includes('hidden') || cs.textOverflow === 'ellipsis' || cs.webkitLineClamp !== 'none') && (el.scrollWidth > el.clientWidth + 1 || el.scrollHeight > el.clientHeight + 1)
    if (clipped && el.hasAttribute('data-trunc')) { out.allowedTrunc++; if ((el.getAttribute('title') || '').trim() !== el.textContent.trim()) out.badTrunc.push(label(el)) }
    else if (clipped && direct) out.clipped.push(label(el))
    if (!direct) continue
    const r = el.getBoundingClientRect()
    if (r.width && (r.right > fr.right + 0.5 || r.left < fr.left - 0.5)) out.clipped.push('OUTSIDE ' + label(el))
    const fg = parse(cs.color)
    if (fg) {
      const bg = bgOf(el), size = parseFloat(cs.fontSize), large = size >= 24 || (size >= 18.66 && +cs.fontWeight >= 700)
      const cr = ratio(over(fg, bg), bg); seen(cr)
      if (cr < (large ? 3 : 4.5)) out.lowContrast.push(cr.toFixed(2) + ' ' + label(el))
    }
  }
  for (const svg of frame.querySelectorAll('svg')) {
    const texts = [...svg.querySelectorAll('text')].map((t) => ({ t, r: t.getBoundingClientRect() })).filter((x) => x.r.width)
    const bg = bgOf(svg.parentElement)
    for (const { t, r } of texts) {
      if (r.right > fr.right + 0.5 || r.left < fr.left - 0.5 || r.bottom > fr.bottom + 0.5 || r.top < fr.top - 0.5) out.svgOut.push(t.textContent)
      const fg = parse(getComputedStyle(t).fill)
      if (fg) { const cr = ratio(over(fg, bg), bg); seen(cr); if (cr < 4.5) out.lowContrast.push(cr.toFixed(2) + ' svg: ' + t.textContent.slice(0, 40)) }
    }
    for (let i = 0; i < texts.length; i++) for (let j = i + 1; j < texts.length; j++) {
      const a = texts[i].r, b = texts[j].r
      if (a.left < b.right - 1 && b.left < a.right - 1 && a.top < b.bottom - 1 && b.top < a.bottom - 1) out.overlaps.push(texts[i].t.textContent + ' × ' + texts[j].t.textContent)
    }
  }
  out.minTextContrast = out.minTextContrast === null ? null : Number(out.minTextContrast.toFixed(2))
  return out
}
if (target) {
  document.body.classList.add('capture')
  const node = document.querySelector('[data-frame="' + CSS.escape(target) + '"]')
  node.closest('.slot').classList.add('is-target')
  node.closest('.group').classList.add('has-target')
  if (params.get('check') === '1') document.fonts.ready.then(() => { document.body.dataset.check = JSON.stringify(hifiCheck(target)) })
} else {
  const fit = () => document.querySelectorAll('.slot').forEach((slot) => {
    const w = +slot.dataset.w, h = +slot.dataset.h, s = slot.clientWidth / w
    slot.style.height = h * s + 'px'
    slot.firstElementChild.style.transform = 'scale(' + s + ')'
  })
  addEventListener('resize', fit); fit()
}
</script></body></html>
`
writeFileSync(here('./index.html'), html)
console.log(JSON.stringify({ bytes: html.length, frames: FRAMES.map((f) => f.id) }))
