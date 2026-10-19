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
    leadNote: '加粗为本行更优（最佳取大，失败率与费用取小；显示值相同则并列）', boardRange: (shown, total) => `第 1–${shown} 名 / 共 ${total} 个有效程序`,
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
    cumBest: '累计最佳', median: '中位数', perGenBest: '本代最佳', costSplit: '费用（模型调用、元分析、提示词进化分开）', modelCost: '模型调用', costMeta: '元分析', costPrompt: '提示词进化', failures: '评估失败', throughput: '吞吐（程序 / 代）',
    models: '按模型', model: '模型', success: '成功率', mean: '平均', diagnostics: '诊断 · 11 张分析小图', expand: '展开', feedback: '评估反馈', reasoning: '模型思路',
    density: '密度等高线', clusters: '聚类', cluster: (k) => `簇 ${k}`, d2: '二维', d3: '三维', leaderboard: '排行榜', pareto: '帕累托前沿', paretoAxis: '费用（对数）', frontier: '前沿', rank: '名次', program: '程序',
    duel: '模型对战', duelNote: '每格是行模型的有效程序严格胜过列模型有效程序的比例，平局不算胜；下方小字为比较对数，失败程序不参与。', duelDist: '有效得分分布（按中位数排序）', diversity: '分布',
    posterior: (range) => (range ? `后验（第 ${range} 代）` : '后验'), posteriorAria: (m, a, b) => `${m} 的后验从 ${a} 变为 ${b}`, posteriorNA: '此运行没有记录模型后验（model_posteriors），该列显示 —。',
    posteriorNote: (hi, even) => `后验为每代程序记录的模型后验均值，各行共用 0–${hi} 刻度，虚线为均分 ${even}。`,
    modelsEnsemble: '按模型与集成', posteriorMean: '平均后验', ensembleNote: '平均后验为各代后验均值的平均，竖线为均分；按代轨迹见“进展”。',
    back: '返回', pullUp: '上拉查看完整标签页',
    statesTitle: '空与错误状态', statesNote: '每个状态取自演示数据或服务端真实返回的格式；状态都用形状加文字表达。',
    emptyWorkspace: '还没有研究方向', emptyWorkspaceHint: '在终端创建第一个方向，刷新后出现在这里：', unreadable: '研究记录无法读取（OSError）', unreadableHint: '其余方向照常显示；修复后刷新即可。',
    newDirection: '新方向尚无试验', emptyRunTitle: '运行为空', emptyRunHint: '数据库存在，但还没有任何程序。', unreadableRun: '数据库无法读取', unreadableRunHint: '服务返回 HTTP 500；其他运行不受影响。',
    costUnknown: '费用未知', costUnknownHint: '该运行有 headless 调用没有价目，费用无法得知：运行显示“—”，含它的合计标为下限（≥），不按 0 计算。', evalFailed: '评估失败', hiddenValidation: '判词前隐藏验证段', hiddenValidationHint: '质疑检验已完成但判词未出，验证段数字保持隐藏。', oosSharpe: '验证段 Sharpe',
    cmdDir: '目录', cmdQuestion: '研究问题',
    evaluatedShort: '程序', totalsRow: (n) => `合计 · ${n} 个方向`, lastActivity: '最近', lowerBound: '部分运行的费用未知，合计为下限', spendNote: (n) => `${n} 个运行费用未知`,
    sentenceSub: (trials, evaluated, failed, rate, spend, budget, above, withTrials) => `共 ${trials} 次试验，已评估 ${evaluated} 个程序，其中评估失败 ${failed} 个（${rate}）；花费 ${spend}，预算 ${budget}；有试验的 ${withTrials} 个方向中，${above} 个的最佳记录越过噪声线。`,
    sortBy2: '排序：最近活动', utcNote: '时间为 UTC', stress: '版式压力测试：4 个演示方向各重复 5 次，只检验 20 行时的版式',
    hypTable: '假设', experimentsCol: '实验', bestIs: '最佳（样本内）', oosCol: '验证段 Sharpe', verdictAt: '判词时间', afterVerdict: '出结论后显示',
    hypothesis: '假设', timeline: '时间线', eventsTitle: '事件', programAxis: (n) => `已评估程序（按评估时间）1–${n}`, probesNote: (n) => `另有 ${n} 次试探计入试验数与噪声线，不画点`,
    vsZero: '验证段 Sharpe > 0', isSharpe: '样本内 Sharpe', noiseSharpe: '噪声线 Sharpe（按试验次数的期望最大值）',
    lockboxResults: '封存段确认结果', lbSharpe: '封存段 Sharpe', lbReturn: '年化收益', lbDrawdown: '最大回撤', lbBuyHold: '买入持有 Sharpe', lbPositive: '为正的子区间', lbBars: (n, start) => `${n} 根 K 线，自 ${start}`, lbPassed: '通过', lbFailed: '未通过',
    valid: '有效', stallCol: '停滞（代）', promptsCol: '提示词', runtimeCol: '运行时长', totalsRuns: (n) => `合计 · ${n} 个运行`, scoreCaveat: '各任务的 combined_score 由各自的评估器定义，不能跨任务比较；排序与比较只在同一任务内有意义。', taskGroup: (n) => `${n} 个运行`,
    byGeneration: '累计最佳 · 按代', bySpend: '累计最佳 · 按累计花费', crossBoard: '跨运行排行榜', runCol: '运行', spendAxis: '累计花费', runMetrics: '运行级指标',
    prevGen: '上一代', nextGen: '下一代', genSlider: '代数', globalPanel: '全局结论', promptsPanel: '提示词版本', updateN: (k) => `第 ${k} 次更新`, parentPrompt: '父版本', close: '关闭', fromMarker: '由元分析轨道 #4 打开',
    programSummary: '本程序的元分析总结', summaryFrom: (file, n) => `${file} · 已处理 ${n} 个程序时写入`, patchLabel: '补丁',
    failBandSpace: '失败（按 PCA 1 位置）', failNotCounted: '失败不计入最佳', mapTitle: 'MAP 网格 · 换手 × 最大回撤', diversityTitle: '多样性（各代 PCA 平均两两距离）', contestFails: (n) => `${n} 个评估失败的程序不进入排行榜与前沿`, zip: '导出前 N 个（ZIP）…', idCol: '编号',
    settings: '设置', expandSheet: '展开', hiddenPlaceholder: '（验证段结果，出结论后显示）',
    stale: '连接中断', staleHint: (t) => `界面保留 ${t} UTC 的数据并标注“最后更新”；恢复后自动刷新，不改变当前选择。`, loading: '正在读取', loadingHint: '内容区显示“正在读取”，不放骨架动画，读完一次替换。', compareMin: '比较需要至少 2 个运行', compareMinHint: '只勾选 1 个运行时，“比较所选”按钮不可用并说明原因。',
  },
  en: {
    directions: 'Research', runs: 'All runs', library: 'Library', directionsShort: 'Research', runsShort: 'Runs', libraryShort: 'Library', nav: 'Main navigation',
    search: 'Search', railDirections: 'Directions', demoNote: 'Demo data, not real research', updated: 'Updated', theme: 'Theme', language: 'Language', dark: 'Dark', light: 'Light',
    trials: 'Trials', failed: 'Failed evaluations', failedShort: 'failed', hypotheses: 'Hypotheses', direction: 'Direction', evidence: 'Evidence', budget: 'Budget', lockbox: 'Lockbox',
    best: 'best', noise: 'noise', clears: 'clears by', noTrials: 'No trials yet', sealed: 'Sealed', opened: 'Opened', budgetUnset: 'No budget', activity: 'Activity', allDirections: 'All directions', log: 'Log text',
    hypothesisKey: 'Hypothesis status', scaleKey: (ticks) => `Every row shares one evidence scale (${ticks}): the diamond is the best record, the bar is the noise line at that time, and orange marks how far the record clears it.`,
    sentence: (n, conf, act, over, fresh) => `${n} research directions: <a href="#d2">${conf} confirmed</a>, <a href="#d1">${act} in progress</a>, <a class="is-signal" href="#d4">${over} over budget</a>, <a href="#d3">${fresh} not started</a>.`,
    leadNote: 'Bold marks the better value in each row (higher best, lower failure rate and cost; values tied at the shown precision share it)', boardRange: (shown, total) => `Ranks 1–${shown} of ${total} valid programs`,
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
    cumBest: 'Cumulative best', median: 'Median', perGenBest: 'Generation best', costSplit: 'Cost (model calls, meta-analysis and prompt evolution apart)', modelCost: 'Model calls', costMeta: 'Meta-analysis', costPrompt: 'Prompt evolution', failures: 'Failed evaluations', throughput: 'Throughput (programs / gen)',
    models: 'By model', model: 'Model', success: 'Success', mean: 'Mean', diagnostics: 'Diagnostics · 11 analysis plots', expand: 'Expand', feedback: 'Feedback', reasoning: 'Model reasoning',
    density: 'Density contours', clusters: 'Clusters', cluster: (k) => `C${k}`, d2: '2D', d3: '3D', leaderboard: 'Leaderboard', pareto: 'Pareto front', paretoAxis: 'Cost (log)', frontier: 'Front', rank: '#', program: 'Program',
    duel: 'Model battle', duelNote: 'Each cell is the share of pairs in which a valid program of the row model scores strictly higher than one of the column model; ties are not wins. The small figure is the number of pairs; failed programs take no part.', duelDist: 'Valid score distribution (by median)', diversity: 'Distribution',
    posterior: (range) => (range ? `Posterior (gens ${range})` : 'Posterior'), posteriorAria: (m, a, b) => `${m} posterior moved from ${a} to ${b}`, posteriorNA: 'This run records no model posteriors (model_posteriors); the column shows —.',
    posteriorNote: (hi, even) => `Posterior is the per-generation mean of the model posteriors the programs recorded; rows share a 0–${hi} scale; the dashed line is an even split (${even}).`,
    modelsEnsemble: 'By model and ensemble', posteriorMean: 'Posterior', ensembleNote: 'Posterior is the average of the per-generation means; the tick marks an even split. The per-generation trajectory is in Progress.',
    back: 'Back', pullUp: 'Pull up for all tabs',
    statesTitle: 'Empty and error states', statesNote: 'Each state uses demo data or the server’s real response format; every state has a shape and a word.',
    emptyWorkspace: 'No research directions yet', emptyWorkspaceHint: 'Create the first direction in a terminal; it appears here after a refresh:', unreadable: 'Research record unreadable (OSError)', unreadableHint: 'Other directions still show; refresh after fixing it.',
    newDirection: 'New direction without trials', emptyRunTitle: 'Empty run', emptyRunHint: 'The database exists but holds no programs yet.', unreadableRun: 'Database unreadable', unreadableRunHint: 'The server returned HTTP 500; other runs are unaffected.',
    costUnknown: 'Cost unknown', costUnknownHint: 'The run has headless calls without a price, so its cost is unknown: the run shows “—” and any total that includes it is a lower bound (≥), never zero.', evalFailed: 'Evaluation failed', hiddenValidation: 'Validation hidden before the verdict', hiddenValidationHint: 'The skeptic check ran but no verdict is recorded, so validation numbers stay hidden.', oosSharpe: 'OOS Sharpe',
    cmdDir: 'dir', cmdQuestion: 'research question',
    evaluatedShort: 'programs', totalsRow: (n) => `Total · ${n} directions`, lastActivity: 'Last', lowerBound: 'Some runs have an unknown cost; the total is a lower bound', spendNote: (n) => `${n} run${n === 1 ? '' : 's'} with unknown cost`,
    sentenceSub: (trials, evaluated, failed, rate, spend, budget, above, withTrials) => `${trials} trials and ${evaluated} evaluated programs, ${failed} of them failed (${rate}); ${spend} spent of ${budget} budgeted; ${above} of the ${withTrials} directions with trials have a best record above their noise line.`,
    sortBy2: 'Sort: last activity', utcNote: 'Times in UTC', stress: 'Layout stress test: the 4 demo directions repeated 5 times, only to check the layout at 20 rows',
    hypTable: 'Hypotheses', experimentsCol: 'Experiments', bestIs: 'Best (in-sample)', oosCol: 'OOS Sharpe', verdictAt: 'Verdict at', afterVerdict: 'shown after the verdict',
    hypothesis: 'Hypothesis', timeline: 'Timeline', eventsTitle: 'Events', programAxis: (n) => `Evaluated programs (in evaluation order) 1–${n}`, probesNote: (n) => `${n} probe${n === 1 ? '' : 's'} also count toward trials and the noise line but are not drawn`,
    vsZero: 'OOS Sharpe > 0', isSharpe: 'In-sample Sharpe', noiseSharpe: 'Noise Sharpe (expected maximum for the trial count)',
    lockboxResults: 'Lockbox confirmation', lbSharpe: 'Lockbox Sharpe', lbReturn: 'Annual return', lbDrawdown: 'Max drawdown', lbBuyHold: 'Buy-and-hold Sharpe', lbPositive: 'Positive sub-periods', lbBars: (n, start) => `${n} bars from ${start}`, lbPassed: 'Passed', lbFailed: 'Failed',
    valid: 'Valid', stallCol: 'Stall (gens)', promptsCol: 'Prompts', runtimeCol: 'Runtime', totalsRuns: (n) => `Total · ${n} runs`, scoreCaveat: 'Each task’s evaluator defines its own combined_score, so scores do not compare across tasks; sort and compare within one task.', taskGroup: (n) => `${n} run${n === 1 ? '' : 's'}`,
    byGeneration: 'Cumulative best · by generation', bySpend: 'Cumulative best · by cumulative spend', crossBoard: 'Cross-run leaderboard', runCol: 'Run', spendAxis: 'Cumulative spend', runMetrics: 'Run-level metrics',
    prevGen: 'Previous', nextGen: 'Next', genSlider: 'Generation', globalPanel: 'Global conclusions', promptsPanel: 'Prompt versions', updateN: (k) => `Update ${k}`, parentPrompt: 'Parent', close: 'Close', fromMarker: 'Opened from meta track #4',
    programSummary: 'Meta-analysis summary of this program', summaryFrom: (file, n) => `${file} · written after ${n} programs`, patchLabel: 'Patch',
    failBandSpace: 'Failed (at their PCA 1 position)', failNotCounted: 'failures do not count toward best', mapTitle: 'MAP grid · turnover × max drawdown', diversityTitle: 'Diversity (mean pairwise PCA distance per generation)', contestFails: (n) => `${n} failed programs are left out of the leaderboard and the front`, zip: 'Export top N (ZIP)…', idCol: 'ID',
    settings: 'Settings', expandSheet: 'Expand', hiddenPlaceholder: '(validation result shown after the verdict)',
    stale: 'Connection lost', staleHint: (t) => `The page keeps the data from ${t} UTC under “Updated”; it refreshes on reconnect without changing the selection.`, loading: 'Loading', loadingHint: 'The content area says “Loading” with no skeleton animation and is replaced once.', compareMin: 'Compare needs at least 2 runs', compareMinHint: 'With one run selected, “Compare selected” is disabled and says why.',
  },
}

// ======================= formats (design.md › 字体与数字) =======================
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c])
const int = (n) => (n === null || n === undefined ? '—' : Math.round(n).toLocaleString('en-US'))
// Negative numbers use the minus sign (U+2212), never a hyphen next to a tick.
const num = (n, d = 2) => (n === null || n === undefined || Number.isNaN(n) ? '—' : (Math.abs(n) < 0.5 * 10 ** -d ? 0 : Number(n)).toFixed(d).replace(/^-/, '−'))
const signed = (n, d = 2) => (n === null || n === undefined ? '—' : `${n >= 0 ? '+' : '−'}${num(Math.abs(n), d)}`)
// A difference of two shown scores is computed from the shown (rounded) values so it adds up.
const r2 = (v) => Math.round(v * 100) / 100
const delta = (a, b) => (a === null || b === null || a === undefined || b === undefined ? '—' : signed(r2(a) - r2(b)))
// Money is $0.00; a recorded amount below half a cent shows as <$0.01, never as $0.00.
const money = (n) => (n === null || n === undefined ? '—' : n === 0 ? '$0.00' : Math.abs(n) < 0.005 ? '<$0.01' : `${n < 0 ? '−' : ''}$${Math.abs(n).toFixed(2)}`)
// "At least": the comparison sign comes from a system font, because the bundled Latin subsets lack it.
const GE = '<span class="op">≥</span>'
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
  trials: d3.sum(dirs, (d) => d.trials), evaluated: d3.sum(dirs, (d) => d.evaluated), spend: d3.sum(dirs, (d) => d.spend), budget: d3.sum(dirs, (d) => d.budget ?? 0), failed: d3.sum(dirs, (d) => d.evidence.failed),
  spendKnown: dirs.every((d) => d.spendKnown), hypotheses: d3.sum(dirs, (d) => d.hypotheses.length),
  challenged: d3.sum(dirs, (d) => d.hypotheses.filter((h) => h.status === 'challenged').length),
  running: d3.sum(dirs, (d) => d.hypotheses.filter((h) => h.status === 'running').length),
  withTrials: dirs.filter((d) => d.trials > 0).length,
  above: dirs.filter((d) => d.evidence.best !== null && d.evidence.bar !== null && d.evidence.best > d.evidence.bar).length,
  confirmed: dirs.filter((d) => d.hypotheses.some((h) => h.status === 'confirmed')).length,
  active: dirs.filter((d) => d.state === 'active').length, over: dirs.filter((d) => d.state === 'over-budget').length, fresh: dirs.filter((d) => d.state === 'new').length,
  opened: dirs.filter((d) => d.lockbox === 'opened').length,
}
// One evidence scale for every direction, so best records and noise bars compare across rows; the
// scale snaps to 0.2 steps so the labelled ticks cover every best record and noise bar.
const evDomain = (() => {
  const [lo, hi] = d3.extent(dirs.flatMap((d) => [d.evidence.best, d.evidence.bar]).filter((v) => v !== null))
  return [Math.floor(lo / 0.2) * 0.2, Math.ceil(hi / 0.2) * 0.2]
})()
const evTicks = d3.range(Math.round(evDomain[0] * 5), Math.round(evDomain[1] * 5) + 1).map((k) => k / 5)
const A = fixture.analysis
const rows = A.rows // [x, y, score|null, generation, island, parentIndex, patch, model, cost, cluster, patchType]
const pathSet = new Set(A.path)
const scored = rows.filter((r) => r[2] !== null)
const scoreExtent = d3.extent(scored, (r) => r[2])
const maxGen = d3.max(rows, (r) => r[3])
const best = A.best
// Posteriors sum to 1 over the models of the posterior maps; an even split marks the bandit's neutral point.
const EVEN = A.posteriors.length ? 1 / Object.keys(A.posteriors[0][1]).length : null
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
  // Direction states are squares, hypothesis states circles, so the two never share a shape.
  'dir-active': '<rect x="2" y="2" width="10" height="10" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.4"/><rect x="5" y="5" width="4" height="4" fill="currentColor"/>',
  'dir-concluded': '<rect x="1.5" y="1.5" width="11" height="11" rx="1.5" fill="currentColor"/><path d="M4.3 7.2l1.8 1.8 3.6-3.8" stroke="var(--bg)" stroke-width="1.5" fill="none"/>',
  'dir-new': '<rect x="2" y="2" width="10" height="10" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.3" stroke-dasharray="1.6 1.6"/>',
  'dir-over': '<rect x="2" y="2" width="10" height="10" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M4 10L10 4" stroke="currentColor" stroke-width="1.4"/><path d="M10 4v3M10 4H7" stroke="currentColor" stroke-width="1.4" fill="none"/>',
}
const glyph = (key, label, cls = '') => `<svg class="glyph ${cls}" width="14" height="14" viewBox="0 0 14 14" role="img" aria-label="${esc(label)}">${GLYPH[key] ?? GLYPH.selected}</svg>`
const statusGlyph = (c, status) => glyph(status, STATUS[c.lang][status] ?? status)
const STATE_GLYPH = { active: 'dir-active', concluded: 'dir-concluded', new: 'dir-new', 'over-budget': 'dir-over' }
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
    run: '<rect x="2.5" y="2.5" width="7" height="7" fill="currentColor"/>',
    scaffold: '<path d="M6 2v8M2 6h8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>',
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
function rail(c, active, { narrow = false, direction = null, list = dirs } = {}) {
  const nav = [['directions', list.length], ['runs', fixture.runs.length], ['library', fixture.workspace.library?.pages ?? 0]]
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
    <div class="raillist">${list.map((d) => `<a class="railitem ${d.id === direction ? 'is-on' : ''}">${stateGlyph(c, d)}${trunc(d.title)}</a>`).join('')}</div>
    <div class="railfoot"><span class="demo">DEMO</span><span>${c.t('demoNote')}</span></div></aside>`
}
function topbar(c, crumbs, { tabs = null, tab = null, extra = '' } = {}) {
  return `<header class="top"><nav class="crumbs" aria-label="breadcrumb">${crumbs.map((x, i) => (i === crumbs.length - 1 ? `<span class="is-here">${esc(x)}</span>` : trunc(x))).join('<span class="sep" aria-hidden="true">/</span>')}</nav>
    ${tabs ? `<div class="vtabs" role="tablist">${tabs.map(([k, label]) => `<span role="tab" aria-selected="${k === tab}" class="${k === tab ? 'is-on' : ''}">${esc(label)}</span>`).join('')}</div>` : ''}
    <div class="tools">${extra}<span class="updated">${c.t('updated')} <span class="mono">${clock(UPDATED)}</span> UTC</span>
      ${seg([c.t('dark'), c.t('light')], c.theme === 'dark' ? 0 : 1, c.t('theme'))}${seg(['中', 'EN'], c.lang === 'zh' ? 0 : 1, c.t('language'))}</div></header>`
}

// ======================= home =======================
// Gauges and their header axis share one inset, wide enough for a centred end label.
const EV_PAD = 14
function gauge(c, d, w) {
  const e = d.evidence
  if (e.best === null || e.bar === null) return `<div class="gauge is-empty">${c.t('noTrials')}</div>`
  const x = d3.scaleLinear(evDomain, [EV_PAD, w - EV_PAD])
  const above = e.best > e.bar
  return `<div class="gauge"><svg width="${w}" height="20" viewBox="0 0 ${w} 20" role="img" aria-label="${c.t('best')} ${num(e.best)} · ${c.t('noise')} ${num(e.bar)}">
    <line x1="${EV_PAD}" x2="${w - EV_PAD}" y1="10" y2="10" stroke="var(--line2)"/>
    ${evTicks.map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="7" y2="13" stroke="var(--stroke)"/>`).join('')}
    <line x1="${x(Math.min(e.bar, e.best))}" x2="${x(Math.max(e.bar, e.best))}" y1="10" y2="10" stroke="${above ? 'var(--signalG)' : 'var(--muted)'}" stroke-width="2.5" ${above ? '' : 'stroke-dasharray="2 2"'}/>
    <rect x="${x(e.bar) - 1}" y="2" width="2" height="16" fill="var(--muted)"/>
    <path d="M${x(e.best)},4l5.5,6-5.5,6-5.5-6z" fill="${above ? 'var(--signalG)' : 'var(--bg)'}" stroke="${above ? 'var(--signalG)' : 'var(--text)'}" stroke-width="1.3"/>
  </svg><div class="gread"><span>${c.t('best')} <b class="mono">${num(e.best)}</b></span><span>${c.t('noise')} <b class="mono">${num(e.bar)}</b></span></div></div>`
}
// Every tick is labelled when labels have 44px apart, otherwise every other tick.
function gaugeAxis(w) {
  const x = d3.scaleLinear(evDomain, [EV_PAD, w - EV_PAD])
  const every = (w - 2 * EV_PAD) / (evTicks.length - 1) >= 44 ? 1 : 2
  return `<svg width="${w}" height="14" viewBox="0 0 ${w} 14" aria-hidden="true">${evTicks.filter((_, i) => i % every === 0).map((t) => `<text x="${x(t)}" y="11" text-anchor="middle" class="axis">${num(t)}</text>`).join('')}</svg>`
}
function budget(c, d, w = 72) {
  if (d.budget === null) return `<div class="budget"><span class="mono">${money(d.spend)}</span><span class="faint">${c.t('budgetUnset')}</span></div>`
  const r = d.spend / d.budget
  const over = r > 1
  const atLeast = d.spendKnown === false ? GE : ''
  const missing = d.spendKnown === false ? d.unknownCostRuns : 0
  return `<div class="budget"><span class="mono">${atLeast}${money(d.spend)}<span class="faint"> / ${money(d.budget)}</span></span>
    <span class="bline"><svg width="${w}" height="6" viewBox="0 0 ${w} 6" aria-hidden="true"><rect width="${w}" height="2" y="2" fill="var(--line2)"/><rect width="${(Math.min(1, r) * w).toFixed(1)}" height="2" y="2" fill="${over ? 'var(--signalG)' : 'var(--muted)'}"/>${over ? `<rect x="${w - 3}" y="0" width="3" height="6" fill="var(--signalG)"/>` : ''}</svg><span class="mono ${over ? 'sig' : 'faint'}">${atLeast}${Math.round(r * 100)}%</span></span>
    ${missing ? `<span class="faint small">${c.t('spendNote')(missing)}</span>` : ''}</div>`
}
const lock = (c, d) => (d.lockbox === 'sealed' ? `<span class="lock">${ICON.lock}<span>${c.t('sealed')}</span></span>` : d.lockbox === 'opened' ? `<span class="lock is-open">${ICON.unlock}<span>${c.t('opened')}</span></span>` : `<span class="lock faint">—</span>`)
const hypDist = (c, counts) => `<div class="hyps">${Object.entries(counts).map(([s, n]) => `<span>${statusGlyph(c, s)}<em>${esc(STATUS[c.lang][s])}</em><b class="mono">${n}</b></span>`).join('')}</div>`
const trialsCell = (c, t, evaluated, failed) => `<div class="trials num"><b class="mono">${int(t)}</b>${evaluated ? `<span class="faint">${c.t('evaluatedShort')} <span class="mono">${int(evaluated)}</span></span><span class="faint">${c.t('failedShort')} <span class="mono">${int(failed)}</span></span>` : ''}</div>`
function dirTable(c, { gaugeW, list = dirs, focus = 'd1-momentum-regimes' }) {
  const sum = {}
  for (const d of list) for (const [s, n] of Object.entries(d.evidence.hypothesisStatus)) sum[s] = (sum[s] ?? 0) + n
  const spendKnown = list.every((d) => d.spendKnown)
  const t = { title: c.t('totalsRow')(list.length), trials: d3.sum(list, (d) => d.trials), evaluated: d3.sum(list, (d) => d.evaluated), failed: d3.sum(list, (d) => d.evidence.failed), spend: d3.sum(list, (d) => d.spend), budget: d3.sum(list, (d) => d.budget ?? 0), opened: list.filter((d) => d.lockbox === 'opened').length }
  return `<div class="tablebar"><span class="faint">${c.t('utcNote')}</span><span class="select">${c.t('sortBy2')}${ICON.chevron}</span></div>
    <div class="dtable" role="table">
    <div class="drow dhead" role="row"><span>${c.t('direction')}</span><span>${c.t('hypotheses')}</span><span class="evhead">${c.t('evidence')}${gaugeAxis(gaugeW)}</span><span class="num">${c.t('trials')}</span><span>${c.t('budget')}</span><span>${c.t('lockbox')}</span></div>
    ${list.map((d) => `<div class="drow ${d.id === focus ? 'is-focus' : ''}" role="row" id="${esc(d.id.slice(0, 2))}">
      <div class="dtitle"><b title="${esc(d.title)}">${esc(d.title)}</b><span class="dmeta">${stateGlyph(c, d)}<span class="${d.state === 'over-budget' ? 'sig' : ''}">${STATE[c.lang][d.state]}</span></span>
        <span class="dfolder mono faint">${trunc(d.id)}<span>${c.t('lastActivity')} ${clock(d.evidence.lastActivity)}</span></span></div>
      ${hypDist(c, d.evidence.hypothesisStatus)}${gauge(c, d, gaugeW)}${trialsCell(c, d.trials, d.evaluated, d.evidence.failed)}
      ${budget(c, d)}${lock(c, d)}</div>`).join('')}
    <div class="drow dtotal" role="row"><div class="dtitle"><b>${t.title}</b></div>${hypDist(c, sum)}<span></span>${trialsCell(c, t.trials, t.evaluated, t.failed)}
      ${budget(c, { spend: t.spend, budget: t.budget, spendKnown, id: '' })}<span class="lock is-open">${ICON.unlock}<span>${t.opened} ${c.t('opened')}</span></span></div></div>
    ${spendKnown ? '' : `<p class="keynote">${GE} ${c.t('lowerBound')}</p>`}`
}
function glyphKey(c, list = dirs) {
  const present = [...new Set(list.flatMap((d) => Object.keys(d.evidence.hypothesisStatus)))]
  const states = [...new Set(list.map((d) => d.state))]
  return `<div class="key"><span class="faint">${c.t('direction')}</span>${states.map((s) => `<span>${stateGlyph(c, { state: s })}${esc(STATE[c.lang][s])}</span>`).join('')}</div>
    <div class="key"><span class="faint">${c.t('hypothesisKey')}</span>${present.map((s) => `<span>${statusGlyph(c, s)}${esc(STATUS[c.lang][s])}</span>`).join('')}</div>
    <p class="keynote">${c.t('scaleKey')(evTicks.map((t) => num(t)).join(' · '))}</p>`
}
// Server placeholders are interface text, so they follow the language switch; research notes stay as written.
const noteText = (c, note) => (note === T.zh.hiddenPlaceholder ? c.t('hiddenPlaceholder') : note)
function feed(c, n, cls = '') {
  return `<aside class="feed ${cls}" aria-label="${c.t('activity')}"><header><b>${c.t('activity')}</b><span class="faint">${c.t('utcNote')} · ${c.t('allDirections')}</span></header>
    ${fixture.events.slice(0, n).map((e) => `<article class="ev">${eventMark(e.event)}<div class="evbody"><div class="evtitle"><b class="${e.event === 'run-failed' ? 'sig' : ''}">${EVENT[c.lang][e.event] ?? e.event}</b>${trunc(dirById.get(e.direction).title, 'faint')}</div>
      <div class="evref mono faint"><span>${esc([e.hypothesis, e.experiment].filter(Boolean).join(' / '))}</span><time>${clock(e.time)}</time></div>${logText(noteText(c, e.note))}</div></article>`).join('')}</aside>`
}
function homeHead(c, { phone = false, list = dirs } = {}) {
  const s = { trials: d3.sum(list, (d) => d.trials), evaluated: d3.sum(list, (d) => d.evaluated), failed: d3.sum(list, (d) => d.evidence.failed), spend: d3.sum(list, (d) => d.spend), budget: d3.sum(list, (d) => d.budget ?? 0), known: list.every((d) => d.spendKnown) }
  const count = (state) => list.filter((d) => d.state === state).length
  const confirmed = list.filter((d) => d.hypotheses.some((h) => h.status === 'confirmed')).length
  const withTrials = list.filter((d) => d.trials > 0)
  const above = withTrials.filter((d) => d.evidence.best > d.evidence.bar).length
  return `<h1 class="dsent ${phone ? 'is-phone' : ''}">${c.t('sentence')(list.length, confirmed, count('active'), count('over-budget'), count('new'))}</h1>
    <p class="dsub">${c.t('sentenceSub')(int(s.trials), int(s.evaluated), int(s.failed), pct(s.evaluated ? s.failed / s.evaluated : 0, 1), `${s.known ? '' : GE}${money(s.spend)}`, money(s.budget), above, withTrials.length)}</p>`
}
function home(c, { narrow = false, stacked = false, gaugeW = 176, list = dirs, note = '' } = {}) {
  return `<div class="app">${rail(c, 'directions', { narrow, list })}<div class="main">${topbar(c, [c.t('directions')])}
    <div class="homebody ${stacked ? 'is-stacked' : ''}"><section class="content">${note ? `<p class="stress">${note}</p>` : ''}${homeHead(c, { list })}${dirTable(c, { gaugeW, list })}${glyphKey(c, list)}${stacked ? feed(c, 6, 'is-below') : ''}</section>${stacked ? '' : feed(c, 9)}</div></div></div>`
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
// The chart's right edge is the home gauge: at the last program a vertical orange segment joins the
// noise line to the best record, so the report and the overview show the same evidence the same way.
function trialChart(c, hid, { w, h }) {
  const pts = rep.points
  const m = { t: 20, r: 132, b: 40, l: 48 }
  const failY = h - m.b - 10
  const plotBottom = failY - 22
  const x = d3.scaleLinear([1, d3.max(pts, (p) => p.n)], [m.l, w - m.r])
  const valid = pts.filter((p) => p.score !== null)
  const [lo, hi] = d3.extent([...valid.map((p) => p.score), ...pts.map((p) => p.bar)])
  // The bottom snaps to a labelled tick, so the lowest scores sit above a labelled line; the top
  // stays at the data.
  const top = hi + (hi - lo) * 0.06, bottom = lo - (hi - lo) * 0.04, step = d3.tickStep(bottom, top, 4)
  const y = d3.scaleLinear([Math.floor(bottom / step) * step, top], [plotBottom, m.t])
  let running = -Infinity
  const steps = pts.map((p) => { if (p.score !== null) running = Math.max(running, p.score); return [p.n, running, p.bar] }).filter((s) => Number.isFinite(s[1]))
  const stepLine = d3.line().curve(d3.curveStepAfter).x((s) => x(s[0])).y((s) => y(s[1]))
  const barLine = d3.line().curve(d3.curveMonotoneX).x((s) => x(s[0])).y((s) => y(s[2]))
  const last = steps.at(-1)
  const ex = x(last[0])
  const xt = x.ticks(8).filter(Number.isInteger)
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
    <text x="${ex + 12}" y="${(y(last[1]) + y(last[2])) / 2 + 4}" class="lbl sig">${c.t('clears')} ${delta(last[1], last[2])}</text>
    <text x="${ex + 12}" y="${y(last[2]) + 4}" class="lbl muted">${c.t('noise')} ${num(last[2])}</text>
    ${xt.map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="${h - m.b + 2}" y2="${h - m.b + 6}" stroke="var(--stroke)"/><text x="${x(t)}" y="${h - m.b + 19}" text-anchor="middle" class="axis">${int(t)}</text>`).join('')}
    <line x1="${m.l}" x2="${ex}" y1="${h - m.b + 2}" y2="${h - m.b + 2}" stroke="var(--stroke)"/>
    <text x="${m.l}" y="${h - 3}" class="axis-title">${c.t('programAxis')(int(last[0]))}</text>
  </svg>`
}
// The skeptic's five rules (capability matrix P09-Q4), then the reference figures it reports.
function ledger(c, ev) {
  const th = ev.thresholds
  const op = (s) => `<span class="op">${s}</span>`
  const rowsL = [
    [c.t('vsZero'), num(ev.oos_sharpe), `${op('>')} 0.00`, ev.oos_sharpe > 0],
    [c.t('vsBuyHold'), `${num(ev.oos_sharpe)} / ${num(ev.baseline_buy_hold_oos_sharpe)}`, esc(c.t('aboveBaseline')), ev.oos_sharpe > ev.baseline_buy_hold_oos_sharpe],
    [c.t('dsr'), num(ev.dsr, 3), `${op('≥')} ${num(th.dsr_supported, 3)}`, ev.dsr >= th.dsr_supported],
    [c.t('positiveSub'), pct(ev.oos_positive_fraction), `${op('≥')} ${pct(th.min_positive_fraction)}`, ev.oos_positive_fraction >= th.min_positive_fraction - 1e-9],
    [c.t('pbo'), num(ev.pbo), `${op('≤')} ${num(th.max_pbo)}`, ev.pbo <= th.max_pbo],
    [c.t('isSharpe'), num(ev.is_sharpe), '—', null],
    [c.t('noiseSharpe'), num(ev.expected_max_sharpe_annual), '—', null],
    [c.t('trialsInDsr'), int(ev.n_trials), '—', null],
    [c.t('vsSeed'), `${num(ev.oos_sharpe)} / ${num(ev.baseline_seed_oos_sharpe)}`, '—', null],
    [c.t('ratio'), num(ev.decay), '—', null],
  ]
  return `<div class="ltable" role="table"><div class="lrow lhead" role="row"><span>${c.t('check')}</span><span class="num">${c.t('result')}</span><span class="num">${c.t('threshold')}</span><span>${c.t('outcome')}</span></div>
    ${rowsL.map(([name, value, threshold, ok]) => `<div class="lrow ${ok === null ? 'is-ref' : ''}" role="row"><span>${esc(name)}</span><span class="num mono">${esc(value)}</span><span class="num mono faint">${threshold}</span><span class="${ok === null ? 'faint' : ok ? '' : 'sig'}">${ok === null ? c.t('reference') : `${ok ? ICON.check : ICON.cross}${ok ? c.t('pass') : c.t('fail')}`}</span></div>`).join('')}</div>`
}
// One track per hypothesis from first to last trial (P08), with its checks and verdicts on the track.
function timeline(c, { w }) {
  const evs = fixture.events.filter((e) => e.direction === rep.id)
  const spans = rep.hypotheses.map((h) => ({ h, first: d3.min(h.experiments, (x) => x.first), last: d3.max(h.experiments, (x) => x.last) }))
  const times = [...evs.map((e) => e.time), ...spans.flatMap((s) => [s.first, s.last])].map((t) => new Date(t))
  const day = 864e5
  const x = d3.scaleUtc([new Date(Math.floor(d3.min(times) / day) * day), new Date(Math.ceil(d3.max(times) / day) * day + day)], [176, w - 16])
  const rowH = 40
  const marks = { skeptic: '<path d="M0,-5l5,5-5,5-5-5z" fill="var(--bg)" stroke="var(--text)" stroke-width="1.3"/>', verdict: '<path d="M0,-5l5,5-5,5-5-5z" fill="var(--text)"/>', confirm: '<rect x="-4.5" y="-1" width="9" height="6.5" rx="1" fill="none" stroke="var(--text)" stroke-width="1.3"/><path d="M-2.5,-1v-1.7a2.5,2.5 0 0 1 4.9-.7" fill="none" stroke="var(--text)" stroke-width="1.3"/>' }
  const h = spans.length * rowH + 34
  // The span label sits under its track, clear of the check and verdict marks on the track.
  const spanText = (a, b) => (clock(a).slice(0, 10) === clock(b).slice(0, 10) ? `${clock(a).slice(5)}–${clock(b).slice(11)}` : `${clock(a).slice(5)} – ${clock(b).slice(5)}`)
  return `<svg class="chart" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${c.t('timeline')}">
    ${x.ticks(d3.utcDay.every(1)).map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="0" y2="${h - 24}" stroke="var(--grid)"/><text x="${x(t)}" y="${h - 8}" text-anchor="middle" class="axis">${t.toISOString().slice(5, 10)}</text>`).join('')}
    ${spans.map((s, k) => { const y = k * rowH + 20; return `<text x="0" y="${y + 4}" class="axis"><tspan class="mono">${s.h.id}</tspan></text><g transform="translate(40,${y - 7})">${GLYPH[s.h.status].replace(/currentColor/g, 'var(--text)')}</g><text x="60" y="${y + 4}" class="tl">${esc(STATUS[c.lang][s.h.status])}</text>
      <line x1="${x(new Date(s.first))}" x2="${x(new Date(s.last))}" y1="${y}" y2="${y}" stroke="var(--text)" stroke-width="6" stroke-linecap="round"/>
      <text x="${x(new Date(s.first)) - 3}" y="${y + 18}" class="axis halo">${spanText(s.first, s.last)}</text>
      ${evs.filter((e) => e.hypothesis === s.h.id && marks[e.event]).map((e) => `<g transform="translate(${x(new Date(e.time))},${y})">${marks[e.event]}</g>`).join('')}` }).join('')}
    <text x="0" y="${h - 8}" class="axis-title">${c.lang === 'zh' ? '日期（UTC）' : 'Date (UTC)'}</text></svg>`
}
function eventStream(c) {
  const evs = fixture.events.filter((e) => e.direction === rep.id)
  return `<div class="estream" role="table">${evs.map((e) => `<div class="erow" role="row"><time class="mono faint">${clock(e.time)}</time><span class="mono faint">${esc([e.hypothesis, e.experiment].filter(Boolean).join(' / '))}</span><span class="eword">${eventMark(e.event)}<b class="${e.event === 'run-failed' ? 'sig' : ''}">${EVENT[c.lang][e.event] ?? e.event}</b></span>${logText(noteText(c, e.note), 'log is-line')}</div>`).join('')}</div>`
}
function hypTable(c, selected) {
  return `<div class="htable" role="table"><div class="hrow hhead" role="row"><span>ID</span><span>${c.t('hypTable')}</span><span>${c.t('outcome')}</span><span class="num">${c.t('experimentsCol')}</span><span class="num">${c.t('trials')}</span><span class="num">${c.t('bestIs')}</span><span class="num">${c.t('oosCol')}</span><span class="num">${c.t('verdictAt')}</span></div>
    ${rep.hypotheses.map((h) => { const ev = h.verdict?.evidence; return `<div class="hrow ${h.id === selected ? 'is-selected' : ''}" role="row"><span class="mono">${h.id}</span><span>${esc(h.title)}</span><span class="hstat">${statusGlyph(c, h.status)}${esc(STATUS[c.lang][h.status])}</span>
      <span class="num mono">${h.experiments.length}</span><span class="num mono">${int(d3.sum(h.experiments, (x) => x.evaluated + x.probed))}</span><span class="num mono">${num(d3.max(h.experiments, (x) => x.best))}</span>
      <span class="num mono">${h.verdict ? num(ev.oos_sharpe) : `<span class="faint" title="${c.t('afterVerdict')}">—</span>`}</span><span class="num mono faint">${h.verdict?.created ? clock(h.verdict.created) : '—'}</span></div>` }).join('')}</div>`
}
function lockboxResults(c, h) {
  const k = h.confirmation
  if (!k) return ''
  const rowsK = [[c.t('lbSharpe'), num(k.lockbox_sharpe)], [c.t('lbBuyHold'), num(k.lockbox_buy_hold_sharpe)], [c.t('lbReturn'), pct(k.lockbox_annual_return, 1)], [c.t('lbDrawdown'), pct(k.lockbox_max_drawdown, 1).replace(/^-/, '−')], [c.t('dsr'), num(k.dsr, 3)], [c.t('lbPositive'), pct(k.lockbox_positive_fraction)], [c.t('trialsInDsr'), int(k.n_trials)]]
  return `<section><h2>${c.t('lockboxResults')}</h2><div class="vline">${statusGlyph(c, h.status)}<b>${STATUS[c.lang][h.status]}</b><span class="faint">${k.passed ? c.t('lbPassed') : c.t('lbFailed')}</span></div>
    <div class="kv is-grid">${rowsK.map(([label, value]) => `<span class="faint">${esc(label)}</span><b class="mono">${value}</b>`).join('')}</div>
    <p class="muted small">${c.t('lbBars')(int(k.lockbox_bars), String(k.lockbox_start).slice(0, 10))}</p></section>`
}
function report(c) {
  const h = rep.hypotheses[0]
  const skeptic = h.experiments[0].skeptic
  return `<div class="app">${rail(c, 'directions', { direction: rep.id })}<div class="main">${topbar(c, [c.t('directions'), repDir.title])}
    <div class="report">
      <header class="rhead"><h1>${esc(rep.title)}</h1>
        <div class="rmeta"><span class="rstate">${stateGlyph(c, repDir)}${STATE[c.lang][repDir.state]}</span><span class="faint">${c.t('created')} <span class="mono">${clock(rep.created)}</span> UTC</span><span class="faint">${c.t('trialsN')(int(rep.trials))} · ${c.t('evaluatedShort')} <span class="mono">${int(rep.points.length)}</span></span><span class="faint mono">${money(rep.spend)} / ${money(rep.budget)}</span>${lock(c, repDir)}</div></header>
      <section class="rsec"><h2>${c.t('hypTable')}</h2>${hypTable(c, h.id)}</section>
      <div class="rcols"><div class="rmain">
        <section><h2>${c.t('hypothesis')} <span class="mono">${h.id}</span></h2><p class="claim">${esc(h.title)}</p>
          <dl class="defs"><dt>${c.t('claim')}</dt><dd>${esc(bodySection(h.body, 'Claim'))}</dd><dt>${c.t('mechanism')}</dt><dd>${esc(bodySection(h.body, 'Mechanism'))}</dd><dt>${c.t('falsification')}</dt><dd>${esc(bodySection(h.body, 'Falsification'))}</dd></dl></section>
        <section><div class="shead"><h2>${c.t('trialsAndNoise')}</h2><div class="legend"><span><i class="ln"></i>${c.t('record')}</span><span><i class="ln is-dash"></i>${c.t('noise')}</span><span><i class="dot"></i>${c.t('thisHyp')}</span><span><i class="dot is-hollow"></i>${c.t('otherHyp')}</span><span>${ICON.cross.replace('currentColor', 'var(--signalG)')}${c.t('failBand')}</span></div></div>
          ${trialChart(c, h.id, { w: 744, h: 280 })}<p class="faint small">${c.t('probesNote')(rep.trials - rep.points.length)}</p></section>
        <section><h2>${c.t('ledger')} · <span class="mono">${h.id}</span></h2>${ledger(c, skeptic)}</section>
        <section><h2>${c.t('timeline')}</h2>${timeline(c, { w: 744 })}</section>
        <section><h2>${c.t('eventsTitle')}</h2>${eventStream(c)}</section>
      </div>
      <aside class="rside">
        <section class="verdict"><h2>${c.t('verdict')}</h2><div class="vline">${statusGlyph(c, h.verdict.label)}<b>${STATUS[c.lang][h.verdict.label]}</b><span class="faint mono">${clock(h.verdict.created)} UTC</span></div>${logText(h.verdict.reason, 'log is-block')}</section>
        ${lockboxResults(c, h)}
        <section><h2>${c.t('experiments')}</h2>${h.experiments.map((x) => `<div class="xrow"><b class="mono">${x.id}</b><span class="faint mono">${clock(x.first)} → ${clock(x.last)} UTC</span>
          <span><b class="mono">${int(x.evaluated)}</b> ${c.t('evaluated')}</span><span><b class="mono">${int(x.probed)}</b> ${c.t('probed')}</span><span>${c.t('best')} <b class="mono">${num(x.best)}</b></span><span><b class="mono">${money(x.spend)}</b></span><a class="btn">${c.t('openAnalysis')}${ICON.arrow}</a></div>`).join('')}</section>
      </aside></div></div></div></div>`
}

// ======================= all runs and compare =======================
const runLabel = (r) => r.name.replace('/results', '')
const SELECTED = ['d1-momentum-regimes/experiments/h001/e01/results', 'd1-momentum-regimes/experiments/h002/e01/results', 'd1-momentum-regimes/experiments/h003/e02/results']
const duration = (s) => (s === null || s === undefined ? '—' : s >= 3600 ? `${Math.floor(s / 3600)}h ${String(Math.round((s % 3600) / 60)).padStart(2, '0')}m` : `${Math.round(s / 60)}m`)
// Runs are grouped by evaluator: ordinary runs by task, research experiments by their direction,
// because combined_score only compares within one evaluator (capability matrix P03-Q2).
const runGroup = (r) => (r.kind === 'experiment' ? r.name.split('/')[0] : r.task ?? r.name.split('/')[0])
function runsPage(c) {
  const counts = { all: fixture.runs.length, ordinary: fixture.runs.filter((r) => r.kind === 'ordinary').length, experiment: fixture.runs.filter((r) => r.kind === 'experiment').length }
  const groups = d3.groups(fixture.runs, runGroup).map(([key, list]) => [key, [...list].sort((a, b) => (b.updated ?? 0) - (a.updated ?? 0))]).sort((a, b) => d3.max(b[1], (r) => r.updated ?? 0) - d3.max(a[1], (r) => r.updated ?? 0))
  const known = fixture.runs.every((r) => r.cost !== null)
  const row = (r) => { const on = SELECTED.includes(r.name); return `<div class="rrow ${on ? 'is-selected' : ''}" role="row"><span class="cb ${on ? 'is-on' : ''}" role="checkbox" aria-checked="${on}" aria-label="${esc(runLabel(r))}">${on ? ICON.check : ''}</span>
    <span class="rname mono">${trunc(runLabel(r).replace(`${runGroup(r)}/`, ''))}</span>
    <span class="num mono">${r.programs ? int(r.programs) : `<span class="empty">${glyph('empty', c.t('emptyRun'))}${c.t('emptyRun')}</span>`}</span><span class="num mono">${int(r.valid)}</span><span class="num mono ${r.failed ? '' : 'faint'}">${int(r.failed)}</span><span class="num mono">${num(r.best)}</span>
    <span class="num mono">${r.programs ? int(r.stall) : '—'}</span><span class="num mono">${int(r.prompts)}</span><span class="num mono">${r.cost === null ? `<span class="faint" title="${c.t('costUnknown')}">—</span>` : money(r.cost)}</span><span class="num mono">${duration(r.runtime)}</span><span class="num mono faint">${epochClock(r.updated)}</span></div>` }
  return `<div class="app">${rail(c, 'runs')}<div class="main">${topbar(c, [c.t('runs')])}
    <div class="runs"><div class="toolbar"><label class="input">${ICON.search}<span>${c.t('runSearch')}</span></label>
      ${seg([`${c.t('all')} ${counts.all}`, `${c.t('ordinary')} ${counts.ordinary}`, `${c.t('experiment')} ${counts.experiment}`], 0, c.t('kind'))}
      <span class="select">${c.t('sortBy')}${ICON.chevron}</span>
      <span class="selbar"><span>${c.t('selectedN')(SELECTED.length)}</span><a class="btn is-primary">${c.t('compareSelected')}${ICON.arrow}</a></span></div>
    <p class="caveat">${glyph('challenged', c.t('scoreCaveat'))}<span>${c.t('scoreCaveat')}</span></p>
    <div class="rtable" role="table"><div class="rrow rhead2" role="row"><span></span><span>${c.t('run')}</span><span class="num">${c.t('programs')}</span><span class="num">${c.t('valid')}</span><span class="num">${c.t('failedShort')}</span><span class="num">${c.t('bestScore')}</span><span class="num">${c.t('stallCol')}</span><span class="num">${c.t('promptsCol')}</span><span class="num">${c.t('cost')}</span><span class="num">${c.t('runtimeCol')}</span><span class="num">${c.t('updatedAt')}</span></div>
      ${groups.map(([key, list]) => `<div class="rgroup" role="row"><span>${dirById.has(key) ? `${stateGlyph(c, dirById.get(key))}<b>${esc(dirById.get(key).title)}</b><span class="mono faint">${esc(key)}</span>` : `<b class="mono">${esc(key)}</b>`}</span><span class="faint">${c.lang === 'zh' ? (dirById.has(key) ? '研究实验' : '普通运行') : dirById.has(key) ? 'research experiments' : 'ordinary runs'} · ${c.t('taskGroup')(list.length)}</span></div>${list.map(row).join('')}`).join('')}
      <div class="rrow rtotal" role="row"><span></span><span>${c.t('totalsRuns')(fixture.runs.length)}</span><span class="num mono">${int(d3.sum(fixture.runs, (r) => r.programs))}</span><span class="num mono">${int(d3.sum(fixture.runs, (r) => r.valid))}</span><span class="num mono">${int(d3.sum(fixture.runs, (r) => r.failed))}</span><span class="num faint">—</span><span class="num faint">—</span><span class="num mono">${int(d3.sum(fixture.runs, (r) => r.prompts))}</span><span class="num mono">${known ? '' : GE}${money(d3.sum(fixture.runs, (r) => r.cost ?? 0))}</span><span class="num mono">${duration(d3.sum(fixture.runs, (r) => r.runtime ?? 0))}</span><span></span></div>
    </div>${known ? '' : `<p class="keynote">${GE} ${c.t('lowerBound')} · ${c.t('spendNote')(fixture.runs.filter((r) => r.cost === null).length)}</p>`}</div></div></div>`
}
const DASH = ['', '6 4', '1.5 3']
function compareCurves(c, sel, { w, h, by }) {
  const m = { t: 14, r: 112, b: 36, l: 48 }
  const curves = sel.map((r) => fixture.compare.find((x) => x.name === r.name).curve.filter((p) => p[1] !== null && (by === 'generation' || p[2] !== null)))
  const xv = (p) => (by === 'generation' ? p[0] : p[2])
  const x = d3.scaleLinear([0, d3.max(curves.flat(), xv)], [m.l, w - m.r]).nice()
  const y = d3.scaleLinear(d3.extent(curves.flat(), (p) => p[1]), [h - m.b, m.t]).nice()
  const line = d3.line().curve(d3.curveStepAfter).x((p) => x(xv(p))).y((p) => y(p[1]))
  const ends = curves.map((cv, k) => ({ k, p: cv.at(-1) })).sort((a, b) => y(a.p[1]) - y(b.p[1]))
  let lastY = -Infinity
  for (const e of ends) { e.ly = Math.max(y(e.p[1]) + 4, lastY + 14); lastY = e.ly }
  const short = (r) => runLabel(r).replace('d1-momentum-regimes/experiments/', '')
  return `<svg class="chart" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${by === 'generation' ? c.t('byGeneration') : c.t('bySpend')}">
    ${y.ticks(4).map((t) => `<line x1="${m.l}" x2="${w - m.r}" y1="${y(t)}" y2="${y(t)}" stroke="var(--grid)"/><text x="${m.l - 8}" y="${y(t) + 4}" text-anchor="end" class="axis">${num(t)}</text>`).join('')}
    ${x.ticks(6).map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="${h - m.b + 2}" y2="${h - m.b + 6}" stroke="var(--stroke)"/><text x="${x(t)}" y="${h - m.b + 19}" text-anchor="middle" class="axis">${by === 'generation' ? int(t) : money(t)}</text>`).join('')}
    <line x1="${m.l}" x2="${w - m.r}" y1="${h - m.b + 2}" y2="${h - m.b + 2}" stroke="var(--stroke)"/>
    ${curves.map((cv, k) => `<path d="${line(cv)}" fill="none" stroke="${k === 2 ? 'var(--muted)' : 'var(--text)'}" stroke-width="1.6" ${DASH[k] ? `stroke-dasharray="${DASH[k]}"` : ''}/>${MARKS[k](x(xv(cv.at(-1))), y(cv.at(-1)[1]), 4)}`).join('')}
    ${ends.map((e) => `<text x="${w - m.r + 10}" y="${e.ly}" class="lbl ${e.k === 2 ? 'muted' : ''}">${esc(short(sel[e.k]))} ${num(e.p[1])}</text>`).join('')}
    <text x="${w - m.r}" y="${h - 2}" text-anchor="end" class="axis-title">${by === 'generation' ? c.t('generation') : c.t('spendAxis')}</text></svg>`
}
function comparePage(c) {
  const sel = SELECTED.map((name) => fixture.runs.find((r) => r.name === name))
  const metrics = [
    [c.t('bestScore'), (r) => r.best, (v) => num(v), 'max'],
    [c.t('failRate'), (r) => r.failed / r.programs, (v) => pct(v, 1), 'min'],
    [c.t('cost'), (r) => r.cost, (v) => money(v), 'min'],
    [c.t('costPerProgram'), (r) => r.cost / r.programs, (v) => money(v), 'min'],
    [c.t('stallCol'), (r) => r.stall, (v) => int(v), null],
    [c.t('programs'), (r) => r.programs, (v) => int(v), null],
  ]
  const W = 170
  const short = (r) => runLabel(r).replace('d1-momentum-regimes/experiments/', '')
  const board = fixture.compare.flatMap((x, k) => x.top.map((p) => ({ ...p, k }))).sort((a, b) => b.score - a.score).slice(0, 8)
  return `<div class="app">${rail(c, 'runs')}<div class="main">${topbar(c, [c.t('runs'), c.t('compareTitle')(sel.length)])}
    <div class="compare"><header class="chead"><h1>${c.t('compareTitle')(sel.length)}</h1><p class="muted">${esc(dirById.get('d1-momentum-regimes').title)} · ${c.t('compareNote')}</p>
      <div class="chips">${sel.map((r, k) => `<span class="chip">${markIcon(k, 14)}<svg width="22" height="8" viewBox="0 0 22 8" aria-hidden="true"><line x1="0" x2="22" y1="4" y2="4" stroke="${k === 2 ? 'var(--muted)' : 'var(--text)'}" stroke-width="1.6" ${DASH[k] ? `stroke-dasharray="${DASH[k]}"` : ''}/></svg><b>${esc(short(r))}</b><span class="faint">${c.t('programsN')(int(r.programs))}</span></span>`).join('')}<span class="chip is-add">+ ${c.t('addRun')}</span></div></header>
    <div class="ccharts"><section><h3>${c.t('byGeneration')}</h3>${compareCurves(c, sel, { w: 560, h: 250, by: 'generation' })}</section><section><h3>${c.t('bySpend')}</h3>${compareCurves(c, sel, { w: 560, h: 250, by: 'spend' })}</section></div>
    <div class="cbottom"><section><h3>${c.t('crossBoard')}</h3><div class="mtable" role="table"><div class="mrow cbrow mhead" role="row"><span class="num">${c.t('rank')}</span><span>${c.t('runCol')}</span><span>${c.t('program')}</span><span>${c.t('idCol')}</span><span class="num">${c.t('generation')}</span><span class="num">${c.t('bestScore')}</span><span class="num">${c.t('cost')}</span></div>
      ${board.map((p, i) => `<div class="mrow cbrow" role="row"><span class="num mono faint">${i + 1}</span><span class="runmk">${markIcon(p.k)}<span class="mono">${esc(short(sel[p.k]))}</span></span><span class="mono">${trunc(p.name)}</span><span class="mono faint">${p.id}</span><span class="num mono">${p.generation}</span><span class="num mono">${num(p.score)}</span><span class="num mono">${money(p.cost)}</span></div>`).join('')}</div></section>
    <section><h3>${c.t('runMetrics')}</h3><div class="ctable2" role="table"><div class="crow chd" role="row"><span>${c.t('metric')}</span>${sel.map((r, k) => `<span class="num">${markIcon(k)}</span>`).join('')}<span>${c.t('scaleNote')}</span></div>
    ${metrics.map(([label, get, fmt, better]) => {
      const values = sel.map(get)
      const [lo, hi] = d3.extent(values)
      const pad = (hi - lo) * 0.3 || Math.abs(hi) * 0.1 || 1
      const x = d3.scaleLinear([Math.max(0, lo - pad), hi + pad], [10, W - 10]).nice()
      // The better value is judged at the shown precision: runs tied there lead together, and a row
      // where every run ties has no lead. The triangle is a run's shape, so the lead is typographic.
      const shown = values.map(fmt)
      const target = better ? fmt(better === 'max' ? hi : lo) : null
      const leads = shown.map((s) => target !== null && s === target && shown.some((o) => o !== target))
      const ticks = [x.domain()[0], x.domain()[1]]
      return `<div class="crow" role="row"><span class="muted">${esc(label)}</span>${shown.map((s, k) => `<span class="num mono ${better ? (leads[k] ? 'is-lead' : 'is-trail') : ''}">${s}</span>`).join('')}
        <svg width="${W}" height="30" viewBox="0 0 ${W} 30" role="img" aria-label="${esc(label)}"><line x1="10" x2="${W - 10}" y1="10" y2="10" stroke="var(--line2)"/>
        ${ticks.map((t, i) => `<line x1="${x(t)}" x2="${x(t)}" y1="7" y2="13" stroke="var(--stroke)"/><text x="${x(t)}" y="27" text-anchor="${i ? 'end' : 'start'}" class="axis">${fmt(t)}</text>`).join('')}
        ${values.map((v, k) => MARKS[k](x(v), 10, 4)).join('')}</svg></div>`
    }).join('')}</div><p class="faint small">${c.t('leadNote')}</p></section></div></div></div></div>`
}

// ======================= analysis =======================
// The ramp spans the 5th percentile to the maximum of valid scores, so the bulk of programs is not
// one grey; lower scores take the ramp's low end (still ≥3:1 on bg and rail).
const rampDomain = [d3.quantile(scored.map((r) => r[2]).sort(d3.ascending), 0.05), scoreExtent[1]]
const ramp = (c) => d3.scaleLinear(rampDomain, [c.th.ramp0, c.th.ramp1]).interpolate(d3.interpolateRgb).clamp(true)
const rampLegend = (c) => { const r = ramp(c); return `<span class="faint">${c.t('scoreRamp')}</span><span class="ramp">${d3.range(5).map((k) => `<i style="background:${r(rampDomain[0] + (k / 4) * (rampDomain[1] - rampDomain[0]))}"></i>`).join('')}</span><span class="mono faint"><span class="op">≤</span>${num(rampDomain[0])}–${num(rampDomain[1])}</span>` }
const VIEWS = (c) => [['lineage', c.t('lineage')], ['progress', c.t('progress')], ['space', c.t('space')], ['contest', c.t('contest')]]
const analysisRun = fixture.runs.find((r) => r.name === A.run)
function analysisShell(c, tab, canvas, inspectorTab) {
  const d = dirById.get(A.direction)
  const unit = (zh, en) => (c.lang === 'zh' ? zh : en)
  const strip = `<div class="runbar"><span class="mono">${esc(A.run.replace('/results', ''))}</span><span class="mono faint">${int(A.stats.programs)} ${unit('个程序', 'programs')} · ${int(analysisRun.valid)} ${unit('有效', 'valid')} · <span class="sig">${int(A.stats.failed)} ${unit('失败', 'failed')}</span> · ${A.stats.generations} ${unit('代', 'gens')} · ${A.stats.islands} ${unit('岛', 'islands')} · ${c.t('cost')} ${money(analysisRun.cost)}</span></div>`
  return `<div class="app">${rail(c, 'directions', { narrow: true })}<div class="main">${topbar(c, [d.title, `${A.hypothesis} / ${A.experiment}`], { tabs: VIEWS(c), tab })}${strip}
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
  if (tab === 'llm') body = `<div class="ibody"><div class="kv"><span class="faint">${c.t('model')}</span><b class="mono">${esc(best.model)}</b><span class="faint">${c.t('cost')}</span><b class="mono">${money(best.cost)}</b><span class="faint">${c.t('patchLabel')}</span><b class="mono">${esc(best.patchType)} · ${esc(best.description)}</b></div>
    <h3>${c.t('reasoning')}</h3>${logText(best.thought, 'log is-block')}</div>`
  if (tab === 'meta') {
    // OD-3: the inspector holds the selected program's own summary; global conclusions live in Lineage.
    const summary = best.metaSummary
    const plain = (line) => esc(line.replace(/\*\*/g, '').replace(/`/g, ''))
    const block = (lines) => lines.map((line) => (line.startsWith('**Program Name:') ? `<h4>${plain(line.replace('**Program Name:', '').trim())}</h4>` : line.startsWith('- ') ? `<p class="item">${plain(line.slice(2))}</p>` : `<p class="muted small">${plain(line)}</p>`)).join('')
    body = `<div class="ibody is-meta"><div class="metahead"><b>${c.t('programSummary')}</b><span class="mono faint">${c.t('summaryFrom')(esc(summary.file), summary.processed)}</span></div><div class="metalist">${block(summary.lines)}</div></div>`
  }
  return `<aside class="inspector ${sheet ? 'is-sheet' : ''}" aria-label="${c.t('inspector')}">
    <header class="ihead"><div class="iname"><span class="bestmark">${c.t('best')}</span><h2 class="mono">${esc(best.name)}</h2><span class="faint mono">${best.id.slice(0, 8)}</span></div>
      <div class="iscore"><b class="mono">${num(best.score)}</b><span class="faint">${c.t('parent')} <span class="mono">${num(best.parentScore)}</span> · ${c.t('vsParent')} <span class="mono">${delta(best.score, best.parentScore)}</span></span></div>
      <div class="ichips"><span>${c.t('gen')(best.generation)}</span><span>${c.t('island')} ${best.island}</span><span class="mono">${esc(best.model)}</span><span class="mono">${money(best.cost)}</span></div></header>
    <div class="itabs" role="tablist">${tabs.map(([k, label]) => `<span role="tab" aria-selected="${k === tab}" class="${k === tab ? 'is-on' : ''}">${label}</span>`).join('')}${tab === 'code' ? `<a class="dlbtn" title="${c.t('downloadCode')('main.py')}" aria-label="${c.t('downloadCode')('main.py')}">${ICON.download}</a>` : tab === 'meta' ? `<a class="dlbtn" title="${c.t('download')}" aria-label="${c.t('download')}">${ICON.download}</a>` : ''}</div>
    ${body}</aside>`
}
// The generation scrubber Lineage and Progress share (OD-8): drag, step one generation, or replay
// on request up to the last generation; with reduced motion replay steps once per press.
function scrubber(c, x, w) {
  const ticks = x.ticks(8).filter(Number.isInteger)
  const [x0, x1] = x.range()
  return `<div class="scrub"><div class="scrubctl"><span class="btn is-small" role="button">${ICON.back.replace('width="16" height="16"', 'width="12" height="12"')}${c.t('prevGen')}</span><span class="btn is-small" role="button">${c.t('nextGen')}<svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true"><path d="M6 3l5 5-5 5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg></span><span class="btn is-small" role="button">${ICON.play}${c.t('replay')}</span><span class="mono faint">${c.t('scrub')(maxGen, int(rows.length))}</span></div>
    <svg width="${w}" height="40" viewBox="0 0 ${w} 40" role="slider" aria-label="${c.t('genSlider')}" aria-valuemin="0" aria-valuemax="${maxGen}" aria-valuenow="${maxGen}">
      <text x="20" y="31" class="axis">${c.t('generation')}</text>
      <line x1="${x0}" x2="${x1}" y1="10" y2="10" stroke="var(--stroke)" stroke-width="2"/><line x1="${x0}" x2="${x(maxGen)}" y1="10" y2="10" stroke="var(--text)" stroke-width="2"/>
      <rect x="${x(maxGen) - 6}" y="3" width="12" height="14" rx="3" fill="var(--text)"/>
      ${ticks.map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="16" y2="20" stroke="var(--stroke)"/><text x="${x(t)}" y="33" text-anchor="middle" class="axis">${t}</text>`).join('')}</svg></div>`
}
function globalPanel(c) {
  const latest = A.meta.latest
  const plain = (line) => esc(line.replace(/\*\*/g, '').replace(/`/g, ''))
  const recs = latest.recommendations.filter((l) => /^\d+\.\s/.test(l))
  return `<aside class="gpanel" aria-label="${c.t('globalPanel')}"><header><b>${c.t('globalPanel')} · ${c.t('updateN')(A.meta.updates.length)}</b><span class="faint mono">${esc(latest.file)}</span><span class="btn is-small" role="button">${c.t('close')}</span></header>
    <p class="faint small">${c.t('fromMarker')} · ${c.t('metaUpdate')(A.meta.updates.length, latest.processed)}</p>
    <h3>${c.t('metaRecs')}</h3><div class="metalist">${recs.map((l) => `<p class="item is-num"><span class="mono faint">${l.match(/^\d+/)[0]}</span>${plain(l.replace(/^\d+\.\s/, ''))}</p>`).join('')}</div>
    <h3>${c.t('promptsPanel')}</h3><div class="mtable" role="table"><div class="mrow prow2 mhead" role="row"><span>P</span><span class="num">${c.t('generation')}</span><span>${c.t('parentPrompt')}</span><span class="num">${c.t('programs')}</span><span class="num">${c.lang === 'zh' ? '适应度' : 'Fitness'}</span><span>${c.lang === 'zh' ? '说明' : 'Description'}</span></div>
      ${A.prompts.map((p, k) => `<div class="mrow prow2" role="row"><span class="mono">P${k}</span><span class="num mono">${p.generation}</span><span class="mono faint">${p.parent ? `P${A.prompts.findIndex((q) => q.name === p.parent)}` : '—'}</span><span class="num mono">${int(p.programs)}</span><span class="num mono">${num(p.fitness)}</span>${trunc(p.description, 'muted')}</div>`).join('')}</div></aside>`
}
function lineageView(c, { w, h }) {
  const top = 64
  const islands = d3.range(A.stats.islands)
  const m = { l: 128, r: 28 }
  const laneH = (h - top) / islands.length
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
    ${A.meta.updates.map((u, k) => `<rect x="${x(u.generation) - 4}" y="42" width="8" height="8" fill="${k === A.meta.updates.length - 1 ? 'var(--text)' : 'var(--bg)'}" stroke="var(--text)" stroke-width="1.2"/><text x="${x(u.generation) + 8}" y="50" class="axis ${k === A.meta.updates.length - 1 ? 'is-strong' : ''}">#${k + 1} · ${u.processed}</text>`).join('')}`
  return `<div class="vbar"><div class="legend">${rampLegend(c)}<span><i class="ln is-signal"></i>${c.t('bestPath')}</span><span>${ICON.cross.replace('currentColor', 'var(--signalG)')}${c.t('failedMark')}</span><span><i class="ln is-grid"></i>${c.lang === 'zh' ? '同岛父子' : 'Same-island parent'}</span></div></div>
    <svg class="chart lanes" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${c.t('lineage')}">${track}
    ${islands.map((k) => { const n = rows.filter((row) => row[4] === k).length; const f = rows.filter((row) => row[4] === k && row[2] === null).length; return `<line x1="0" x2="${w}" y1="${top + k * laneH}" y2="${top + k * laneH}" stroke="var(--line)"/>
      <text x="20" y="${top + k * laneH + 26}" class="lane">${c.t('island')} ${k}</text><text x="20" y="${top + k * laneH + 44}" class="axis">${c.t('programsN')(n)}</text><text x="20" y="${top + k * laneH + 60}" class="axis ${f ? 'sig' : ''}">${c.t('failedN')(f)}</text>
      ${laneTicks.map((t) => `<text x="${m.l - 10}" y="${yIn(k, t) + 3.5}" text-anchor="end" class="axis">${num(t)}</text>`).join('')}
      <line x1="${m.l}" x2="${w - m.r}" y1="${failY(k) - 8}" y2="${failY(k) - 8}" stroke="var(--line)" stroke-dasharray="2 3"/>` }).join('')}
    <path d="${edges}" fill="none" stroke="var(--grid)" stroke-width=".8"/>
    ${rows.map((row, i) => (row[2] === null ? `<path d="M${(pos[i][0] - 3).toFixed(1)},${(pos[i][1] - 3).toFixed(1)}l6,6m0,-6l-6,6" stroke="var(--signalG)" stroke-width="1.3"/>` : pathSet.has(i) ? '' : `<circle cx="${pos[i][0].toFixed(1)}" cy="${pos[i][1].toFixed(1)}" r="3.2" fill="${r(row[2])}"/>`)).join('')}
    <path d="${pathD}" fill="none" stroke="var(--signalG)" stroke-width="2" stroke-linejoin="round"/>
    ${A.path.map((i) => `<circle cx="${pos[i][0].toFixed(1)}" cy="${pos[i][1].toFixed(1)}" r="4" fill="var(--bg)" stroke="var(--signalG)" stroke-width="2"/>`).join('')}
    <circle cx="${bx}" cy="${by}" r="5.5" fill="var(--signalG)"/><circle cx="${bx}" cy="${by}" r="9.5" fill="none" stroke="var(--text)" stroke-width="1.5"/>
    <text x="${bx}" y="${by - 16}" text-anchor="middle" class="lbl sig halo">${c.t('best')} ${num(best.score)} · ${c.t('gen')(best.generation)}</text></svg>
    ${scrubber(c, x, w)}${globalPanel(c)}`
}
function progressView(c, { w }) {
  const P = A.progress
  const m = { l: 64, r: 128 }
  const x = d3.scaleLinear([0, maxGen], [m.l, w - m.r])
  const bw = Math.max(2, (x(1) - x(0)) * 0.62)
  const panel = (title, legend, height, yDomain, draw, fmt = (v) => num(v)) => {
    const y = d3.scaleLinear(yDomain, [height - 6, 10]).nice()
    return `<div class="pm"><div class="pmhead"><b>${title}</b><span class="legend">${legend}</span></div><svg width="${w}" height="${height}" viewBox="0 0 ${w} ${height}" role="img" aria-label="${esc(title)}">
      ${y.ticks(3).map((t) => `<line x1="${m.l}" x2="${w - m.r}" y1="${y(t)}" y2="${y(t)}" stroke="var(--grid)"/><text x="${m.l - 8}" y="${y(t) + 4}" text-anchor="end" class="axis">${fmt(t)}</text>`).join('')}
      ${draw(y)}</svg></div>`
  }
  // The direction's noise line over this run's generations, ending in the same orange bracket as home and the report.
  const noise = A.noise
  const lastNoise = noise.at(-1)[1]
  const lastBest = P.at(-1).cumulative
  const scoreDomain = [d3.min(P, (p) => p.median ?? p.best), d3.max([...P.map((p) => p.cumulative), ...noise.map((n) => n[1])])]
  const costMax = d3.max(P, (p) => (p.modelCost ?? 0) + (p.metaCost ?? 0) + (p.promptCost ?? 0))
  const stack = (y, p) => {
    let base = 0
    return [['modelCost', 'var(--muted)'], ['metaCost', 'var(--stroke)'], ['promptCost', 'var(--text)']].map(([key, fill]) => {
      const v = p[key] ?? 0
      if (!v) return ''
      const top = base + v
      const rect = `<rect x="${x(p.generation) - bw / 2}" y="${y(top)}" width="${bw}" height="${Math.max(1, y(base) - y(top) - 1)}" fill="${fill}"/>`
      base = top
      return rect
    }).join('')
  }
  // Model posteriors per generation on one shared scale (L08-Q1/Q2, OD-4); the dashed line is an even split.
  const Q = A.posteriors
  const px = d3.scaleLinear(Q.length ? [Q[0][0], Q.at(-1)[0]] : [0, 1], [2, 146])
  const py = d3.scaleLinear([0, d3.max(Q, (q) => d3.max(Object.values(q[1]))) ?? 1], [22, 2]).nice()
  const has = (q, model) => q[1][model] !== undefined && q[1][model] !== null
  const spark = (s) => {
    const own = Q.filter((q) => has(q, s.model))
    if (!own.length) return '<span class="faint">—</span>'
    const [g, v] = [own.at(-1)[0], own.at(-1)[1][s.model]]
    return `<span class="spark"><svg width="148" height="24" viewBox="0 0 148 24" role="img" aria-label="${esc(c.t('posteriorAria')(s.model, num(own[0][1][s.model]), num(v)))}"><line x1="0" x2="148" y1="${py(EVEN)}" y2="${py(EVEN)}" stroke="var(--stroke)" stroke-dasharray="3 3"/>
      <path d="${d3.line().defined((q) => has(q, s.model)).x((q) => px(q[0])).y((q) => py(q[1][s.model]))(Q)}" fill="none" stroke="var(--text)" stroke-width="1.3"/><circle cx="${px(g)}" cy="${py(v)}" r="2.4" fill="var(--text)"/></svg><b class="mono">${num(v)}</b></span>`
  }
  return `<div class="progress">
    ${panel(c.t('cumBest'), `<span><i class="ln"></i>${c.t('cumBest')}</span><span><i class="ln is-dash"></i>${c.t('noise')}</span><span><i class="ln is-thin"></i>${c.t('median')}</span><span><i class="dot"></i>${c.t('perGenBest')}</span>`, 136, scoreDomain, (y) => `
      <path d="${d3.line().x((p) => x(p.generation)).y((p) => y(p.median)).defined((p) => p.median !== null)(P)}" fill="none" stroke="var(--stroke)" stroke-width="1"/>
      <path d="${d3.line().curve(d3.curveMonotoneX).x((n) => x(n[0])).y((n) => y(n[1]))(noise)}" fill="none" stroke="var(--muted)" stroke-width="1.4" stroke-dasharray="4 3"/>
      ${P.filter((p) => p.best !== null).map((p) => `<circle cx="${x(p.generation)}" cy="${y(p.best)}" r="2.2" fill="var(--muted)"/>`).join('')}
      <path d="${d3.line().curve(d3.curveStepAfter).x((p) => x(p.generation)).y((p) => y(p.cumulative))(P)}" fill="none" stroke="var(--text)" stroke-width="1.8"/>
      <line x1="${x(maxGen)}" x2="${x(maxGen)}" y1="${y(lastNoise)}" y2="${y(lastBest)}" stroke="var(--signalG)" stroke-width="2.5"/><rect x="${x(maxGen) - 5}" y="${y(lastNoise) - 1}" width="10" height="2" fill="var(--muted)"/>
      <path d="M${x(maxGen)},${y(lastBest) - 6}l5.5,6-5.5,6-5.5-6z" fill="var(--signalG)"/>
      <text x="${x(maxGen) + 12}" y="${y(lastBest) + 4}" class="lbl">${c.t('record')} ${num(lastBest)}</text><text x="${x(maxGen) + 12}" y="${(y(lastBest) + y(lastNoise)) / 2 + 4}" class="lbl sig">${c.t('clears')} ${delta(lastBest, lastNoise)}</text><text x="${x(maxGen) + 12}" y="${y(lastNoise) + 4}" class="lbl muted">${c.t('noise')} ${num(lastNoise)}</text>
      <circle cx="${x(best.generation)}" cy="${y(best.score)}" r="8.5" fill="none" stroke="var(--text)" stroke-width="1.5"/>
      <text x="${x(best.generation) - 14}" y="${y(best.score) - 10}" text-anchor="end" class="lbl halo">${c.t('best')} ${num(best.score)} · ${c.t('gen')(best.generation)}</text>`)}
    ${panel(c.t('costSplit'), `<span><i class="sw" style="background:var(--muted)"></i>${c.t('modelCost')}</span><span><i class="sw" style="background:var(--stroke)"></i>${c.t('costMeta')}</span><span><i class="sw" style="background:var(--text)"></i>${c.t('costPrompt')}</span>`, 72, [0, costMax], (y) => P.map((p) => stack(y, p)).join(''), (v) => money(v))}
    ${panel(c.t('failures'), `<span>${ICON.cross.replace('currentColor', 'var(--signalG)')}${c.lang === 'zh' ? '每个 × 一个失败程序' : 'one × per failed program'}</span>`, 56, [0, Math.max(3, d3.max(P, (p) => p.failed))], (y) => P.map((p) => d3.range(p.failed).map((k) => `<path d="M${x(p.generation) - 3},${y(k + 0.5) - 3}l6,6m0,-6l-6,6" stroke="var(--signalG)" stroke-width="1.3"/>`).join('')).join(''), (v) => int(v))}
    ${panel(c.t('throughput'), '', 50, [0, d3.max(P, (p) => p.programs)], (y) => P.map((p) => `<rect x="${x(p.generation) - bw / 2}" y="${y(p.programs)}" width="${bw}" height="${y(0) - y(p.programs)}" fill="var(--stroke)"/>`).join(''), (v) => int(v))}
    ${scrubber(c, x, w)}
    <section class="models"><h3>${c.t('models')}</h3><div class="mtable" role="table">
      <div class="mrow prow3 mhead" role="row"><span>${c.t('model')}</span><span class="num">${c.t('programs')}</span><span class="num">${c.t('success')}</span><span class="num">${c.t('best')}</span><span class="num">${c.t('mean')}</span><span class="num">${c.t('cost')}</span><span>${c.t('posterior')(Q.length ? `${Q[0][0]}–${Q.at(-1)[0]}` : '')}</span></div>
      ${A.modelStats.map((s) => `<div class="mrow prow3" role="row"><span class="mono">${esc(s.model)}</span><span class="num mono">${int(s.programs)}</span><span class="num mono">${pct(s.success, 1)}</span><span class="num mono">${s.best === best.score ? '<i class="leadmark" aria-hidden="true">◆</i>' : ''}${num(s.best)}</span><span class="num mono">${num(s.mean)}</span><span class="num mono">${money(s.cost)}</span>${spark(s)}</div>`).join('')}</div>
      ${Q.length ? `<p class="faint small mnote">${c.t('posteriorNote')(num(py.domain()[1]), num(EVEN))}</p>` : `<p class="na mnote">${glyph('empty', c.t('posteriorNA'))}<span class="muted small">${c.t('posteriorNA')}</span></p>`}</section>
    <div class="disclosure">${ICON.chevron}<b>${c.t('diagnostics')}</b><span class="faint">${c.t('expand')}</span></div></div>`
}
function spaceView(c, { w, h }) {
  const pts = rows.map((row, i) => ({ x: row[0], y: row[1], s: row[2], k: row[9], g: row[3], i }))
  const pw = w - 320
  const band = 34
  const m = { t: 28, r: 24, b: 40 + band, l: 56 }
  // One unit on both axes so distances in the projection are not distorted.
  const [x0, x1] = d3.extent(pts, (p) => p.x), [y0, y1] = d3.extent(pts, (p) => p.y)
  const unit = Math.min((pw - m.l - m.r) / (x1 - x0), (h - m.t - m.b) / (y1 - y0)) * 0.92
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2
  const x = (v) => m.l + (pw - m.l - m.r) / 2 + (v - cx) * unit
  const y = (v) => m.t + (h - m.t - m.b) / 2 - (v - cy) * unit
  const r = ramp(c)
  const valid = pts.filter((p) => p.s !== null)
  const failedPts = pts.filter((p) => p.s === null)
  const contours = d3.contourDensity().x((p) => x(p.x)).y((p) => y(p.y)).size([pw, h]).bandwidth(16).thresholds(5)(valid)
  const geo = d3.geoPath()
  const clusters = d3.rollups(pts, (v) => ({ n: v.length, top: d3.min(v, (p) => y(p.y)), cx: d3.mean(v, (p) => x(p.x)), best: d3.max(v, (p) => p.s), failed: v.filter((p) => p.s === null).length }), (p) => p.k).filter(([k]) => k !== null).sort((a, b) => a[0] - b[0])
  const bp = pts[A.bestIndex]
  const xTicks = d3.ticks(x0, x1, 6), yTicks = d3.ticks(y0, y1, 5)
  const bandY = h - band - 18
  // MAP-Elites style grid (L11): two public metrics as behaviour axes, best valid score per cell.
  const mk = (key) => A.metricKeys.indexOf(key)
  const cells = 6, gw = 276, gh = 168
  const mp = A.metrics.map((v, i) => (v ? { tx: v[mk('is_turnover')], dy: v[mk('is_max_drawdown')], s: rows[i][2], i } : null)).filter(Boolean)
  const bx = d3.scaleLinear(d3.extent(mp, (p) => p.tx), [0, cells - 1e-9]), by = d3.scaleLinear(d3.extent(mp, (p) => p.dy), [0, cells - 1e-9])
  const grid = new Map()
  for (const p of mp) { const key = `${Math.floor(bx(p.tx))},${Math.floor(by(p.dy))}`; const cur = grid.get(key); if (!cur || p.s > cur.s) grid.set(key, p) }
  const bestCell = [...grid].find(([, p]) => p.i === A.bestIndex)?.[0]
  // Diversity (L13): mean pairwise distance in the 2-D projection among each generation's programs.
  const div = d3.range(maxGen + 1).map((g) => { const q = pts.filter((p) => p.g === g); let sum = 0, n = 0; for (let a = 0; a < q.length; a++) for (let b = a + 1; b < q.length; b++) { sum += Math.hypot(q[a].x - q[b].x, q[a].y - q[b].y); n++ } return [g, n ? sum / n : null] })
  const dx = d3.scaleLinear([0, maxGen], [34, gw - 6]), dy = d3.scaleLinear([0, d3.max(div, (d) => d[1])], [70, 8]).nice()
  const lastDiv = div.filter((d) => d[1] !== null).at(-1)
  return `<div class="vbar"><div class="legend">${rampLegend(c)}<span><i class="ln is-contour"></i>${c.t('density')}</span><span>${ICON.cross.replace('currentColor', 'var(--signalG)')}${c.t('failedMark')} · ${c.t('failNotCounted')}</span></div>${seg([c.t('d2'), c.t('d3')], 0, c.t('space'))}</div>
    <div class="spacewrap"><svg class="chart" width="${pw}" height="${h}" viewBox="0 0 ${pw} ${h}" role="img" aria-label="${c.t('space')}">
      ${contours.map((g) => `<path d="${geo(g)}" fill="none" stroke="var(--stroke)" stroke-width=".8"/>`).join('')}
      ${valid.filter((p) => p.i !== A.bestIndex).map((p) => `<circle cx="${x(p.x).toFixed(1)}" cy="${y(p.y).toFixed(1)}" r="3" fill="${r(p.s)}"/>`).join('')}
      ${clusters.map(([k, v]) => `<text x="${v.cx.toFixed(1)}" y="${(v.top - 14).toFixed(1)}" text-anchor="middle" class="clabel halo">${c.t('cluster')(k + 1)}</text>`).join('')}
      <circle cx="${x(bp.x)}" cy="${y(bp.y)}" r="5.5" fill="var(--signalG)"/><circle cx="${x(bp.x)}" cy="${y(bp.y)}" r="9.5" fill="none" stroke="var(--text)" stroke-width="1.5"/>
      <line x1="${x(bp.x) + 10}" y1="${y(bp.y) - 6}" x2="${x(bp.x) + 44}" y2="${y(bp.y) - 30}" stroke="var(--signalG)"/><text x="${x(bp.x) + 48}" y="${y(bp.y) - 32}" class="lbl sig halo">${c.t('best')} ${num(best.score)}</text>
      ${yTicks.map((t) => `<text x="${m.l - 12}" y="${y(t) + 4}" text-anchor="end" class="axis">${num(t, 1)}</text>`).join('')}
      <line x1="${m.l}" x2="${pw - m.r}" y1="${bandY - 16}" y2="${bandY - 16}" stroke="var(--line)"/>
      <text x="${m.l - 12}" y="${bandY + 4}" text-anchor="end" class="axis">${c.t('failBand')}</text>
      ${failedPts.map((p) => `<path d="M${(x(p.x) - 3).toFixed(1)},${bandY - 3}l6,6m0,-6l-6,6" stroke="var(--signalG)" stroke-width="1.3"/>`).join('')}
      <text x="${pw - m.r}" y="${bandY - 22}" text-anchor="end" class="axis-title">${c.t('failBandSpace')} · ${int(failedPts.length)}</text>
      ${xTicks.map((t) => `<text x="${x(t)}" y="${h - 4}" text-anchor="middle" class="axis">${num(t, 1)}</text>`).join('')}
      <text x="${m.l}" y="14" class="axis-title">↑ PCA 2</text><text x="${pw - m.r}" y="${bandY + 22}" text-anchor="end" class="axis-title">PCA 1 →</text></svg>
    <aside class="sideinfo"><h3>${c.t('clusters')}</h3><div class="mtable" role="table"><div class="mrow chead2" role="row"><span>${c.t('clusters')}</span><span class="num">${c.t('programs')}</span><span class="num">${c.t('best')}</span><span class="num">${c.t('failedShort')}</span></div>
      ${clusters.map(([k, v]) => `<div class="mrow" role="row"><span>${c.t('cluster')(k + 1)}</span><span class="num mono">${int(v.n)}</span><span class="num mono">${v.best === best.score ? '<i class="leadmark" aria-hidden="true">◆</i>' : ''}${num(v.best)}</span><span class="num mono">${int(v.failed)}</span></div>`).join('')}</div>
      <h3>${c.t('mapTitle')}</h3><svg class="chart" width="${gw}" height="${gh + 34}" viewBox="0 0 ${gw} ${gh + 34}" role="img" aria-label="${c.t('mapTitle')}">
        ${d3.range(cells).flatMap((i) => d3.range(cells).map((j) => { const p = grid.get(`${i},${j}`); const cw = (gw - 40) / cells, ch = gh / cells; return `<rect x="${40 + i * cw + 1}" y="${gh - (j + 1) * ch + 1}" width="${cw - 2}" height="${ch - 2}" fill="${p ? r(p.s) : 'none'}" stroke="${`${i},${j}` === bestCell ? 'var(--signalG)' : 'var(--line2)'}" stroke-width="${`${i},${j}` === bestCell ? 2 : 1}"/>` })).join('')}
        <text x="40" y="${gh + 14}" class="axis">${num(bx.domain()[0])}</text><text x="${gw}" y="${gh + 14}" text-anchor="end" class="axis">${num(bx.domain()[1])}</text><text x="${(gw + 40) / 2}" y="${gh + 30}" text-anchor="middle" class="axis-title">${c.lang === 'zh' ? '换手 →' : 'turnover →'}</text>
        <text x="34" y="${gh}" text-anchor="end" class="axis">${num(by.domain()[0])}</text><text x="34" y="10" text-anchor="end" class="axis">${num(by.domain()[1])}</text></svg>
      <p class="faint small">${c.lang === 'zh' ? `格内为最佳得分，纵轴为最大回撤；已占 ${grid.size} / ${cells * cells} 格，橙框为最佳程序所在格` : `Each cell holds its best score; vertical: max drawdown; ${grid.size} of ${cells * cells} cells occupied, the orange outline holds the best program`}</p>
      <h3>${c.t('diversityTitle')}</h3><svg class="chart" width="${gw}" height="84" viewBox="0 0 ${gw} 84" role="img" aria-label="${c.t('diversityTitle')}">
        ${dy.ticks(2).map((t) => `<line x1="34" x2="${gw - 6}" y1="${dy(t)}" y2="${dy(t)}" stroke="var(--grid)"/><text x="28" y="${dy(t) + 4}" text-anchor="end" class="axis">${num(t, 1)}</text>`).join('')}
        <path d="${d3.line().defined((d) => d[1] !== null).x((d) => dx(d[0])).y((d) => dy(d[1]))(div)}" fill="none" stroke="var(--text)" stroke-width="1.4"/>
        <text x="34" y="82" class="axis">0</text><text x="${gw - 6}" y="82" text-anchor="end" class="axis">${maxGen} ${c.lang === 'zh' ? '代' : 'gens'}</text>
        <circle cx="${dx(lastDiv[0])}" cy="${dy(lastDiv[1])}" r="2.5" fill="var(--text)"/></svg></aside></div>`
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
  const valid = rows.map((row, i) => ({ row, i })).filter((p) => p.row[2] !== null).sort((a, b) => b.row[2] - a.row[2])
  const ranked = valid.slice(0, 8)
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
  return `<div class="contest"><section><div class="sechead"><h3>${c.t('leaderboard')}</h3><span class="faint small">${ICON.cross.replace('currentColor', 'var(--signalG)')} ${c.t('contestFails')(A.stats.failed)}</span><span class="btn is-small" role="button">${ICON.download}${c.t('zip')}</span></div><div class="mtable" role="table"><div class="mrow lbrow mhead" role="row"><span class="num">${c.t('rank')}</span><span>${c.t('program')}</span><span>${c.t('idCol')}</span><span class="num">${c.t('generation')}</span><span class="num">${c.t('island')}</span><span>${c.t('model')}</span><span class="num">${c.t('bestScore')}</span><span class="num">${c.t('vsParent')}</span><span class="num">${c.t('cost')}</span></div>
      ${ranked.map(({ row, i }, k) => { const parent = row[5] === -1 ? null : rows[row[5]][2]; return `<div class="mrow lbrow ${i === A.bestIndex ? 'is-selected' : ''}" role="row"><span class="num mono faint">${k + 1}</span><span class="mono">${trunc(row[6])}</span><span class="mono faint">${esc(A.ids[i])}</span><span class="num mono">${row[3]}</span><span class="num mono">${row[4]}</span><span class="mono muted">${trunc(shortModel(row[7]))}</span><span class="num mono">${i === A.bestIndex ? '<i class="leadmark" aria-hidden="true">◆</i>' : ''}${num(row[2])}</span><span class="num mono">${delta(row[2], parent)}</span><span class="num mono">${money(row[8])}</span></div>` }).join('')}</div><p class="faint small lbfoot">${c.t('boardRange')(ranked.length, int(valid.length))}</p></section>
    <div class="cgrid"><section><h3>${c.t('pareto')}</h3><svg class="chart" width="${pw}" height="${ph}" viewBox="0 0 ${pw} ${ph}" role="img" aria-label="${c.t('pareto')}">
      ${y.ticks(4).map((t) => `<line x1="${m.l}" x2="${pw - m.r}" y1="${y(t)}" y2="${y(t)}" stroke="var(--grid)"/><text x="${m.l - 6}" y="${y(t) + 4}" text-anchor="end" class="axis">${num(t)}</text>`).join('')}
      ${decades.map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="${ph - m.b + 2}" y2="${ph - m.b + 6}" stroke="var(--stroke)"/><text x="${x(t)}" y="${ph - m.b + 18}" text-anchor="middle" class="axis">${t >= 0.01 ? money(t) : `$${t}`}</text>`).join('')}
      ${costed.filter((p) => !frontSet.has(p.i)).map((p) => `<circle cx="${x(p.cost).toFixed(1)}" cy="${y(p.s).toFixed(1)}" r="2.2" fill="none" stroke="var(--stroke)"/>`).join('')}
      <path d="${d3.line().curve(d3.curveStepAfter).x((p) => x(p.cost)).y((p) => y(p.s))(front)}" fill="none" stroke="var(--text)" stroke-width="1.6"/>
      ${front.map((p) => `<circle cx="${x(p.cost).toFixed(1)}" cy="${y(p.s).toFixed(1)}" r="3.2" fill="${p.i === A.bestIndex ? 'var(--signalG)' : 'var(--text)'}"/>`).join('')}
      <line x1="${m.l}" x2="${pw - m.r}" y1="${ph - m.b + 2}" y2="${ph - m.b + 2}" stroke="var(--stroke)"/><text x="${pw - m.r}" y="${ph - 2}" text-anchor="end" class="axis-title">${c.t('paretoAxis')}</text></svg></section>
    <section><h3>${c.t('modelsEnsemble')}</h3>${A.modelStats.map((s) => `<div class="mbar"><div class="mbtop"><b class="mono">${esc(shortModel(s.model))}</b><span class="faint">${c.t('programsN')(int(s.programs))}</span></div>
      <div class="mbline"><span class="faint">${c.t('cost')}</span>${s.cost === null ? '<span></span>' : `<svg width="120" height="8" viewBox="0 0 120 8" aria-hidden="true"><rect y="3" width="120" height="2" fill="var(--line2)"/><rect y="2" width="${(s.cost / maxCost) * 120}" height="4" fill="var(--muted)"/></svg>`}<b class="mono">${money(s.cost)}</b></div>
      <div class="mbline"><span class="faint">${c.t('success')}</span>${s.success === null ? '<span></span>' : `<svg width="120" height="8" viewBox="0 0 120 8" aria-hidden="true"><rect y="3" width="120" height="2" fill="var(--line2)"/><rect y="2" width="${s.success * 120}" height="4" fill="var(--muted)"/></svg>`}<b class="mono">${pct(s.success, 1)}</b></div>
      <div class="mbline"><span class="faint">${c.t('posteriorMean')}</span>${s.posteriorMean === null ? '<span></span>' : `<svg width="120" height="8" viewBox="0 0 120 8" aria-hidden="true"><rect y="3" width="120" height="2" fill="var(--line2)"/><rect y="2" width="${s.posteriorMean * 120}" height="4" fill="var(--muted)"/><rect x="${120 * EVEN}" width="1" height="8" fill="var(--faint)"/></svg>`}<b class="mono">${num(s.posteriorMean)}</b></div></div>`).join('')}
      <p class="faint small mnote">${c.t('ensembleNote')}</p></section>
    <section><h3>${c.t('duel')}</h3><p class="faint small">${c.t('duelDist')}</p>${(() => {
      const dist = modelScores().map((m) => ({ ...m, q: [d3.quantile(m.scores, 0), d3.quantile(m.scores, 0.25), d3.quantile(m.scores, 0.5), d3.quantile(m.scores, 0.75), d3.quantile(m.scores, 1)] })).sort((a, b) => b.q[2] - a.q[2])
      const bw = 268, bx = d3.scaleLinear(scoreExtent, [92, bw - 8]).nice()
      return `<svg class="chart" width="${bw}" height="${dist.length * 24 + 22}" viewBox="0 0 ${bw} ${dist.length * 24 + 22}" role="img" aria-label="${c.t('duelDist')}">
        ${dist.map((m, k) => { const y = k * 24 + 12; return `<text x="0" y="${y + 4}" class="axis"><tspan class="mono">${'ABC'[A.modelStats.findIndex((s) => s.model === m.model)]}</tspan> ${esc(shortModel(m.model))}</text><line x1="${bx(m.q[0])}" x2="${bx(m.q[4])}" y1="${y}" y2="${y}" stroke="var(--stroke)"/><rect x="${bx(m.q[1])}" y="${y - 6}" width="${bx(m.q[3]) - bx(m.q[1])}" height="12" fill="var(--hover)" stroke="var(--muted)"/><line x1="${bx(m.q[2])}" x2="${bx(m.q[2])}" y1="${y - 6}" y2="${y + 6}" stroke="var(--text)" stroke-width="2"/>` }).join('')}
        ${bx.ticks(2).map((t) => `<text x="${bx(t)}" y="${dist.length * 24 + 18}" text-anchor="middle" class="axis">${num(t)}</text>`).join('')}</svg>`
    })()}
      <div class="duel" role="table"><div class="drow2" role="row"><span></span>${A.modelStats.map((s, k) => `<span class="num mono faint" title="${esc(s.model)}">${'ABC'[k]}</span>`).join('')}</div>
      ${A.modelStats.map((s, a) => `<div class="drow2" role="row"><span class="mono faint dname"><b>${'ABC'[a]}</b>${trunc(shortModel(s.model))}</span>${D[a].map((cell) => (cell === null ? '<span class="num faint">—</span>' : `<span class="num"><b class="mono">${pct(cell.rate)}</b><i class="mono faint">${int(cell.n)}</i></span>`)).join('')}</div>`).join('')}</div>
      <p class="faint small">${c.t('duelNote')}</p></section></div></div>`
}

// ======================= phone =======================
function phoneHome(c) {
  return `<div class="phone"><header class="ptop">${MARK}<b>Shinka Research</b><span class="demo">DEMO</span><span class="psearch" role="button" aria-label="${c.t('search')}">${ICON.search}</span><span class="psettings" role="button">${c.t('settings')}</span></header>
    <div class="pscroll"><div class="psum">${homeHead(c, { phone: true })}<p class="pupdated faint">${c.t('updated')} <span class="mono">${clock(UPDATED)}</span> UTC</p><div class="pgauge"><span class="faint">${c.t('evidence')}</span>${gaugeAxis(286)}</div></div>
      ${dirs.map((d) => `<article class="pcard"><div class="dtitle"><b>${esc(d.title)}</b><span class="dmeta">${stateGlyph(c, d)}<span class="${d.state === 'over-budget' ? 'sig' : ''}">${STATE[c.lang][d.state]}</span><span class="faint">· ${c.t('trialsN')(int(d.trials))} · ${c.t('evaluatedShort')} ${int(d.evaluated)}</span></span></div>
        ${hypDist(c, d.evidence.hypothesisStatus)}<div class="pev">${gauge(c, d, 286)}</div><div class="prow">${budget(c, d, 120)}${lock(c, d)}</div></article>`).join('')}</div>
    <nav class="ptabs" aria-label="${c.t('nav')}"><a class="is-on" aria-current="page">${ICON.directions}<span>${c.t('directionsShort')}</span></a><a>${ICON.runs}<span>${c.t('runsShort')}</span></a><a>${ICON.library}<span>${c.t('libraryShort')}</span></a></nav></div>`
}
function phoneAnalysis(c) {
  const P = A.progress
  const w = 358, h = 196, m = { l: 36, r: 10, t: 12, b: 24 }
  const x = d3.scaleLinear([0, maxGen], [m.l, w - m.r])
  const y = d3.scaleLinear([d3.min(P, (p) => p.median ?? p.best), d3.max([...P.map((p) => p.cumulative), ...A.noise.map((n) => n[1])])], [h - m.b, m.t]).nice()
  return `<div class="phone"><header class="ptop"><span class="pback" role="button" aria-label="${c.t('back')}">${ICON.back}</span><b>${A.hypothesis} / ${A.experiment}</b>${trunc(dirById.get(A.direction).title, 'faint')}<span class="demo">DEMO</span></header>
    <div class="ptabs2" role="tablist">${VIEWS(c).map(([k, label]) => `<span role="tab" aria-selected="${k === 'progress'}" class="${k === 'progress' ? 'is-on' : ''}">${label}</span>`).join('')}</div>
    <div class="pchart"><div class="pmhead"><b>${c.t('cumBest')}</b><span class="legend"><span><i class="ln"></i>${c.t('cumBest')}</span><span><i class="ln is-dash"></i>${c.t('noise')}</span></span></div>
      <svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${c.t('cumBest')}">${y.ticks(3).map((t) => `<line x1="${m.l}" x2="${w - m.r}" y1="${y(t)}" y2="${y(t)}" stroke="var(--grid)"/><text x="${m.l - 6}" y="${y(t) + 4}" text-anchor="end" class="axis">${num(t)}</text>`).join('')}
        <path d="${d3.line().curve(d3.curveMonotoneX).x((n) => x(n[0])).y((n) => y(n[1]))(A.noise)}" fill="none" stroke="var(--muted)" stroke-width="1.3" stroke-dasharray="4 3"/>
        <path d="${d3.line().curve(d3.curveStepAfter).x((p) => x(p.generation)).y((p) => y(p.cumulative))(P)}" fill="none" stroke="var(--text)" stroke-width="1.8"/>
        <circle cx="${x(best.generation)}" cy="${y(best.score)}" r="5" fill="var(--signalG)"/><circle cx="${x(best.generation)}" cy="${y(best.score)}" r="9" fill="none" stroke="var(--text)" stroke-width="1.5"/>
        ${x.ticks(5).filter(Number.isInteger).map((t) => `<text x="${x(t)}" y="${h - 6}" text-anchor="middle" class="axis">${t}</text>`).join('')}</svg></div>
    <div class="sheet"><div class="sheetbar"><span class="grab" aria-hidden="true"></span><span class="btn is-small" role="button">${c.t('expandSheet')}</span></div>${inspector(c, 'diff', { sheet: true })}<p class="pullnote faint">${c.t('pullUp')}</p></div></div>`
}

// ======================= empty and error states =======================
function statesBoard(c) {
  const d3dir = dirById.get('d3-volume-shock')
  const emptyRun = fixture.runs.find((r) => r.programs === 0)
  const unknownRun = fixture.runs.find((r) => r.cost === null && r.kind === 'ordinary')
  const failed = A.failedExample
  const hidden = fixture.events.find((e) => e.event === 'skeptic' && hasCjk(e.note))
  const card = (title, body) => `<section class="scard"><h3>${title}</h3><div class="sbody">${body}</div></section>`
  return `<div class="states"><header class="shead2"><h1>${c.t('statesTitle')}</h1><p class="muted">${c.t('statesNote')}</p></header>
    <div class="sgrid">
    ${card(c.t('emptyWorkspace'), `<p class="muted">${c.t('emptyWorkspaceHint')}</p><code class="cmd">shinka_research init &lt;${c.t('cmdDir')}&gt; --direction "&lt;${c.t('cmdQuestion')}&gt;"</code>`)}
    ${card(c.t('unreadable'), `<div class="srow">${glyph('error', c.t('unreadable'), 'is-signal')}<div><b class="mono">d5-broken-notebook</b><span class="sig">${c.t('unreadable')}</span></div></div><p class="muted">${c.t('unreadableHint')}</p>`)}
    ${card(c.t('newDirection'), `<div class="srow">${stateGlyph(c, d3dir)}<div><b>${esc(d3dir.title)}</b><span class="faint">${STATE[c.lang][d3dir.state]} · ${c.t('hypCount')(d3dir.hypotheses.length)}</span></div></div>${gauge(c, d3dir, 260)}${budget(c, d3dir, 120)}`)}
    ${card(c.t('emptyRunTitle'), `<div class="srow">${glyph('empty', c.t('emptyRunTitle'))}<div><b class="mono">${pathBreaks(runLabel(emptyRun))}</b><span class="faint">${c.t('emptyRun')} · ${c.t('programsN')(0)} · ${money(emptyRun.cost)}</span></div></div><p class="muted">${c.t('emptyRunHint')}</p>`)}
    ${card(c.t('unreadableRun'), `<div class="srow">${glyph('error', c.t('unreadableRun'), 'is-signal')}<div><b class="mono">runs/…/<wbr>programs.sqlite</b><span class="sig">${c.t('unreadableRun')}</span></div></div><p class="muted">${c.t('unreadableRunHint')}</p>`)}
    ${card(c.t('costUnknown'), `<div class="srow">${glyph('empty', c.t('costUnknown'))}<div><b class="mono">${pathBreaks(runLabel(unknownRun))}</b><span class="faint">${c.t('programsN')(int(unknownRun.programs))} · ${c.t('cost')} —</span></div></div><p class="muted">${c.t('costUnknownHint')}</p>`)}
    ${card(c.t('evalFailed'), `<div class="srow">${glyph('error', c.t('evalFailed'), 'is-signal')}<div><b class="mono">${esc(failed.name)}</b><span class="faint">${c.t('gen')(failed.generation)} · ${c.t('island')} ${failed.island}</span></div></div>${logText(failed.stderr, 'log is-block')}`)}
    ${card(c.t('hiddenValidation'), `<div class="srow">${statusGlyph(c, 'challenged')}<div><b>${esc(dirById.get(hidden.direction).title)} · <span class="mono">${hidden.hypothesis}</span></b><span class="faint">${STATUS[c.lang].challenged}</span></div></div><div class="kv"><span class="faint">${c.t('oosSharpe')}</span><b class="faint">${esc(noteText(c, hidden.note))}</b></div><p class="muted">${c.t('hiddenValidationHint')}</p>`)}
    ${card(c.t('stale'), `<div class="srow">${glyph('error', c.t('stale'), 'is-signal')}<div><b>${c.t('stale')}</b><span class="faint">${c.t('updated')} <span class="mono">${clock(UPDATED)}</span> UTC</span></div></div><p class="muted">${c.t('staleHint')(clock(UPDATED))}</p>`)}
    ${card(c.t('loading'), `<div class="srow">${glyph('proposed', c.t('loading'))}<div><b>${c.t('loading')}…</b><span class="faint mono">/api/workspace</span></div></div><p class="muted">${c.t('loadingHint')}</p>`)}
    ${card(c.t('compareMin'), `<div class="srow">${glyph('selected', c.t('compareMin'))}<div><b>${c.t('selectedN')(1)}</b><span class="faint">${c.t('compareSelected')} · ${c.lang === 'zh' ? '不可用' : 'disabled'}</span></div></div><p class="muted">${c.t('compareMinHint')}</p>`)}
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
.f .tablebar{display:flex;justify-content:space-between;align-items:center;margin:22px 0 10px;font-size:12px} .f .tablebar .select{height:28px}
.f .stress{margin-bottom:14px;padding:8px 12px;border:1px dashed var(--stroke);color:var(--muted);font-size:12px}
.f .dtable{border-top:1px solid var(--line)}
.f .drow{display:grid;grid-template-columns:minmax(0,1fr) var(--hw) var(--gw) 84px 120px 72px;column-gap:16px;align-items:center;padding:10px 8px 10px 12px;border-bottom:1px solid var(--line);min-height:84px;position:relative}
.f .drow.dtotal{min-height:64px;background:var(--rail)} .f .drow.dtotal .dtitle b{font-size:12.5px;color:var(--muted)}
.f .dfolder{display:grid;gap:0;min-width:0;font-size:11px;line-height:16px} .f .dfolder span:last-child{white-space:nowrap}
.f .drow.dhead{min-height:0;padding-top:8px;padding-bottom:8px;color:var(--faint);font-size:12px;align-items:end}
.f .drow.is-focus{background:var(--selected)} .f .drow.is-focus::before{content:"";position:absolute;left:0;top:0;bottom:0;width:2px;background:var(--text)}
.f .evhead{display:grid;gap:2px} .f .axis{font:400 10.5px PlexMono,monospace;fill:var(--faint)} .f .axis.sig{fill:var(--signal)} .f .axis-title{font-size:11px;fill:var(--faint)}
.f .dtitle{display:grid;gap:4px;min-width:0} .f .dtitle b{font-weight:500;font-size:13.5px;line-height:20px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.f .dmeta{display:flex;flex-wrap:wrap;gap:2px 6px;align-items:center;font-size:12px;color:var(--muted)} .f .dmeta > *{white-space:nowrap}
.f .hyps{display:flex;flex-wrap:wrap;gap:4px 12px} .f .hyps span{display:inline-flex;gap:5px;align-items:center;color:var(--muted);font-size:12px;white-space:nowrap} .f .hyps em{font-style:normal} .f .hyps b{font-weight:500;color:var(--text)}
.f .gauge{display:grid;gap:2px} .f .gauge.is-empty{color:var(--faint);font-size:12px} .f .gread{display:flex;gap:12px;font-size:11.5px;color:var(--faint)} .f .gread b{font-weight:500;color:var(--text);font-size:12px}
.f .trials{display:grid;gap:2px} .f .trials b{font-weight:500;font-size:13px} .f .trials span{font-size:11.5px;white-space:nowrap}
.f .budget{display:grid;gap:3px;font-size:12px} .f .bline{display:flex;align-items:center;gap:8px}
.f .lock{display:inline-flex;gap:6px;align-items:center;color:var(--muted);font-size:12px;white-space:nowrap} .f .lock.is-open{color:var(--text)} .f .lock svg{flex:none}
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
.f code.log{display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;margin-top:2px;padding-left:8px;border-left:1px solid var(--line2);font:400 11.5px/17px PlexMono,ui-monospace,monospace;color:var(--muted);overflow-wrap:anywhere}
.f code.log.is-line{display:block;-webkit-line-clamp:unset;margin:0;padding:0;border:0;white-space:normal}
.f code.log.is-block{display:block;-webkit-line-clamp:unset;padding:8px 10px;border:1px solid var(--line);background:var(--bg);white-space:pre-wrap}
.f .report{flex:1;min-height:0;overflow:hidden;padding:24px 32px}
.f .rhead h1{font-size:26px;line-height:34px;font-weight:600;letter-spacing:-.01em}
.f .rmeta{display:flex;gap:16px;align-items:center;margin-top:8px;font-size:12.5px;color:var(--muted);white-space:nowrap} .f .rstate{display:inline-flex;gap:6px;align-items:center;color:var(--text)}
.f .rsec{margin-top:20px} .f .htable{border-top:1px solid var(--line)} .f .hrow{display:grid;grid-template-columns:52px minmax(0,1fr) 120px 64px 64px 112px 112px 140px;column-gap:16px;align-items:center;min-height:40px;padding:4px 8px;border-bottom:1px solid var(--line)}
.f .hhead{min-height:30px;color:var(--faint);font-size:12px} .f .hrow.is-selected{background:var(--selected);box-shadow:inset 2px 0 0 var(--text)} .f .hstat{display:inline-flex;gap:6px;align-items:center}
.f .rcols{display:grid;grid-template-columns:744px minmax(0,1fr);gap:36px;margin-top:24px} .f .rmain{display:grid;gap:24px;align-content:start} .f .rside{display:grid;gap:24px;align-content:start}
.f .tl{font-size:12px;fill:var(--text)} .f .estream{border-top:1px solid var(--line)} .f .erow{display:grid;grid-template-columns:128px 72px 120px minmax(0,1fr);column-gap:14px;align-items:baseline;padding:8px 4px;border-bottom:1px solid var(--line);font-size:12px}
.f .eword{display:inline-flex;gap:6px;align-items:center} .f .eword b{font-weight:500} .f .eword .evmark{padding-top:0}
.f .kv.is-grid{grid-template-columns:minmax(0,1fr) auto;gap:6px 14px;margin-bottom:8px}
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
.f .xrow{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;padding:12px 14px;border:1px solid var(--line2);color:var(--muted);font-size:12.5px} .f .xrow b{color:var(--text);font-weight:500} .f .xrow .btn{margin-left:auto}
.f .btn{display:inline-flex;gap:8px;align-items:center;justify-self:start;height:30px;padding:0 12px;border:1px solid var(--line2);border-radius:6px;font-size:12.5px;font-weight:500;white-space:nowrap;color:var(--text)}
.f .btn.is-primary{background:var(--text);color:var(--bg);border-color:var(--text)}
.f .runs,.f .compare{flex:1;min-height:0;overflow:hidden;padding:22px 28px}
.f .toolbar{display:flex;gap:12px;align-items:center;margin-bottom:16px}
.f .input{display:flex;gap:8px;align-items:center;width:300px;height:32px;padding:0 10px;border:1px solid var(--stroke);border-radius:6px;color:var(--faint)}
.f .select{display:inline-flex;gap:6px;align-items:center;height:32px;padding:0 10px;border:1px solid var(--line2);border-radius:6px;color:var(--muted);white-space:nowrap}
.f .selbar{margin-left:auto;display:flex;gap:12px;align-items:center;color:var(--muted);white-space:nowrap}
.f .rtable{border-top:1px solid var(--line)} .f .rrow{display:grid;grid-template-columns:20px minmax(0,1fr) 64px 56px 48px 56px 72px 56px 76px 72px 128px;column-gap:14px;align-items:center;min-height:38px;padding:0 12px;border-bottom:1px solid var(--line)}
.f .rgroup{display:flex;justify-content:space-between;align-items:center;gap:16px;min-height:36px;padding:10px 12px 4px;font-size:12.5px} .f .rgroup > span:first-child{display:inline-flex;gap:8px;align-items:center;min-width:0} .f .rgroup b{font-weight:500}
.f .rtotal{background:var(--rail);color:var(--muted);font-size:12.5px} .f .caveat{display:flex;gap:8px;align-items:center;margin:-4px 0 14px;color:var(--muted);font-size:12px}
.f .rhead2{min-height:34px;color:var(--faint);font-size:12px} .f .rrow.is-selected{background:var(--selected)}
.f .cb{width:16px;height:16px;border:1px solid var(--stroke);border-radius:4px;display:grid;place-items:center} .f .cb.is-on{background:var(--text);border-color:var(--text);color:var(--bg)}
.f .rname{min-width:0;font-size:12.5px;font-weight:500;overflow-wrap:anywhere}
.f .empty{display:inline-flex;gap:6px;align-items:center;color:var(--faint);font-family:InterV,sans-serif} .f .empty .glyph{color:var(--faint)}
.f .chead h1{font-size:24px;line-height:32px;font-weight:600} .f .chead p{margin-top:4px}
.f .chips{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 20px} .f .chip{display:inline-flex;gap:8px;align-items:center;height:30px;padding:0 12px;border:1px solid var(--line2);border-radius:6px;font-size:12.5px} .f .chip b{font-weight:500;font-family:PlexMono,monospace} .f .chip.is-add{color:var(--muted);border-style:dashed}
.f .ctable2{border-top:1px solid var(--line)} .f .crow{display:grid;grid-template-columns:120px 72px 72px 72px minmax(0,1fr);column-gap:14px;align-items:center;min-height:48px;padding:4px 6px;border-bottom:1px solid var(--line)}
.f .crow.chd{min-height:32px;color:var(--faint);font-size:12px} .f .crow.chd .num{display:inline-flex;gap:6px;align-items:center;color:var(--muted)} .f .crow .is-lead{font-weight:500;color:var(--text)} .f .crow .is-trail{color:var(--muted)} .f .lbfoot{margin-top:6px}
.f .leadmark{font-style:normal;font-size:9px;margin-right:4px;color:var(--text)} .f .ccharts{display:grid;grid-template-columns:repeat(2,560px);gap:28px;margin-bottom:22px} .f .cbottom{display:grid;grid-template-columns:minmax(0,1fr) 560px;gap:28px}
.f .mrow.cbrow{grid-template-columns:28px 92px minmax(0,1fr) 72px 40px 56px 56px} .f .runmk{display:inline-flex;gap:6px;align-items:center} .f .compare h3{font-size:14px;line-height:22px;font-weight:500;margin-bottom:8px}
.f .analysis{flex:1;min-height:0;display:flex} .f .canvas{flex:1;min-width:0;position:relative;overflow:hidden;display:flex;flex-direction:column}
.f .runbar{height:32px;flex:none;display:flex;gap:16px;align-items:center;padding:0 20px 0 28px;border-bottom:1px solid var(--line);font-size:12px;white-space:nowrap}
.f .vbar{height:44px;flex:none;display:flex;align-items:center;justify-content:space-between;gap:16px;padding:0 20px;border-bottom:1px solid var(--line)}
.f .lane{font-size:13px;font-weight:500;fill:var(--text)}
.f .scrub{flex:none;border-top:1px solid var(--line);padding-top:8px} .f .scrubctl{display:flex;gap:8px;align-items:center;padding:0 20px 4px;font-size:11.5px} .f .scrubctl .mono{margin-left:auto}
.f .btn.is-small{height:26px;padding:0 9px;gap:6px;font-size:12px} .f .legend .ln.is-grid{border-top:1px solid var(--grid)} .f .legend .ln.is-thin{border-top:1px solid var(--stroke)}
.f .gpanel{position:absolute;right:16px;top:112px;width:404px;max-height:560px;overflow:hidden;padding:14px 16px;background:var(--raised);border:1px solid var(--line2);border-radius:8px;display:grid;gap:8px;align-content:start}
.f .gpanel header{display:flex;gap:10px;align-items:baseline} .f .gpanel header b{font-weight:500} .f .gpanel header .btn{margin-left:auto} .f .gpanel h3{font-size:12px;font-weight:500;color:var(--muted);margin-top:4px}
.f .mrow.prow2{grid-template-columns:28px 32px 44px 52px 48px minmax(0,1fr);font-size:12px;min-height:26px}
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
.f .ibody.is-meta{overflow:auto;flex:1;min-height:0;gap:8px} .f .metahead{display:grid;gap:2px} .f .metahead b{font-weight:500} .f .metahead span{font-size:11.5px}
.f .metalist{display:grid;gap:6px;font-size:12.5px;line-height:19px} .f .metalist .item{padding-left:12px;position:relative;overflow-wrap:anywhere} .f .metalist .item::before{content:"";position:absolute;left:2px;top:9px;width:4px;height:1px;background:var(--stroke)}
.f .metalist .item.is-num{padding-left:18px} .f .metalist .item.is-num::before{display:none} .f .metalist .item.is-num span{position:absolute;left:0} .f .metalist h4{margin:8px 0 0;font-size:12px;font-weight:500;color:var(--muted)}
.f .axis.is-strong{fill:var(--text);font-weight:500}
.f .metrics{display:grid;grid-template-columns:1fr 1fr;border:1px solid var(--line)} .f .metrics div{display:grid;gap:2px;padding:8px 12px;border-bottom:1px solid var(--line)} .f .metrics div:nth-child(odd){border-right:1px solid var(--line)} .f .metrics div:nth-last-child(-n+2){border-bottom:0} .f .metrics span{font-size:11px} .f .metrics b{font-weight:500;font-size:15px}
.f .kv{display:grid;grid-template-columns:auto minmax(0,1fr);gap:4px 16px;font-size:12.5px} .f .kv b{font-weight:500;overflow-wrap:anywhere}
.f .progress{padding:4px 0 0;overflow:hidden;flex:1} .f .pm{border-bottom:1px solid var(--line)} .f .pmhead{display:flex;justify-content:space-between;align-items:center;padding:10px 28px 0 20px} .f .pmhead b{font-size:13px;font-weight:500}
.f .models{padding:14px 28px 0 20px} .f .mtable{border-top:1px solid var(--line)} .f .mrow{display:grid;grid-template-columns:minmax(0,1fr) 80px 80px 70px 70px 80px;column-gap:14px;align-items:center;min-height:32px;border-bottom:1px solid var(--line);padding:0 4px;font-size:12.5px}
.f .mrow.prow3{grid-template-columns:minmax(0,1fr) 80px 80px 70px 70px 80px 196px} .f .spark{display:flex;gap:8px;align-items:center} .f .spark svg{flex:none} .f .spark b{font-weight:500}
.f .mhead,.f .chead2{min-height:28px;color:var(--faint);font-size:12px} .f .mrow.is-selected{background:var(--selected)}
.f .disclosure{display:flex;gap:8px;align-items:center;margin:14px 28px 0 20px;padding:10px 12px;border:1px solid var(--line2);border-radius:6px;color:var(--muted)} .f .disclosure b{font-weight:500;color:var(--text)} .f .disclosure .faint{margin-left:auto}
.f .spacewrap{flex:1;display:flex;min-height:0} .f .sideinfo{width:320px;flex:none;border-left:1px solid var(--line);padding:18px 20px;display:grid;gap:10px;align-content:start}
.f .sideinfo .mrow{grid-template-columns:minmax(0,1fr) 56px 52px 44px} .f .sideinfo h3 + .kv{margin-top:-2px}
.f .contest{padding:16px 20px;display:grid;gap:18px;align-content:start;overflow:hidden;flex:1} .f .sechead{display:flex;gap:14px;align-items:baseline} .f .sechead .btn{margin-left:auto} .f .sechead .small{display:inline-flex;gap:6px;align-items:center}
.f .mrow.lbrow{grid-template-columns:28px minmax(0,1fr) 72px 40px 36px 104px 56px 60px 56px}
.f .cgrid{display:grid;grid-template-columns:292px minmax(0,1fr) 268px;gap:28px}
.f .mbar{display:grid;gap:4px;padding:8px 0;border-bottom:1px solid var(--line)} .f .mbtop{display:flex;justify-content:space-between;align-items:baseline;font-size:12.5px} .f .mbtop b{font-weight:500}
.f .mbline{display:grid;grid-template-columns:56px 120px minmax(0,1fr);gap:10px;align-items:center;font-size:12px} .f .mbline b{font-weight:500;text-align:right}
.f .duel{border-top:1px solid var(--line);margin-bottom:8px} .f .drow2{display:grid;grid-template-columns:104px repeat(3,minmax(0,1fr));column-gap:8px;align-items:center;min-height:34px;border-bottom:1px solid var(--line);font-size:12px}
.f .dname{display:flex;gap:6px;min-width:0} .f .dname b{font-weight:500;color:var(--muted)}
.f .drow2 .num{display:grid;justify-items:end;line-height:15px} .f .drow2 b{font-weight:500} .f .drow2 i{font-style:normal;font-size:10.5px}
.f .na{display:flex;gap:8px;align-items:flex-start} .f .na .glyph{margin-top:2px;color:var(--faint)} .f .mnote{margin-top:10px}
.f .phone{height:100%;display:flex;flex-direction:column;background:var(--bg);position:relative}
.f .ptop{height:52px;flex:none;display:flex;align-items:center;gap:10px;padding:0 16px;border-bottom:1px solid var(--line);white-space:nowrap;min-width:0} .f .ptop b{font-weight:600} .f .ptop .trunc{font-size:12px}
.f .psearch,.f .psettings{display:grid;place-items:center;min-width:40px;height:40px;padding:0 8px;border:1px solid var(--stroke);border-radius:6px;color:var(--muted);font-size:12.5px} .f .psearch{margin-left:auto} .f .pback{display:grid;place-items:center;width:40px;height:40px;margin-left:-10px;flex:none}
.f .pupdated{margin-top:6px;font-size:12px} .f .ptop .demo{flex:none}
.f .pscroll{flex:1;min-height:0;overflow-y:auto;scrollbar-width:none}
.f .psum{padding:18px 16px 4px} .f .dsent.is-phone{font-size:21px;line-height:32px;text-underline-offset:5px} .f .psum .dsub{font-size:12.5px;line-height:19px}
.f .pgauge{display:grid;grid-template-columns:1fr 286px;align-items:center;margin-top:14px;padding-top:10px;border-top:1px solid var(--line);font-size:12px}
.f .pcard{margin:0 16px;padding:14px 0;border-bottom:1px solid var(--line);display:grid;gap:10px} .f .pev{display:flex;justify-content:flex-end} .f .prow{display:flex;justify-content:space-between;align-items:flex-end;gap:12px}
.f .ptabs{height:72px;flex:none;display:flex;justify-content:space-around;align-items:flex-start;padding-top:8px;border-top:1px solid var(--line);background:var(--rail)}
.f .ptabs a{display:grid;justify-items:center;gap:2px;min-width:64px;color:var(--muted);font-size:11px} .f .ptabs .is-on{color:var(--text);font-weight:500}
.f .ptabs2{display:flex;gap:2px;padding:0 10px;height:42px;flex:none;border-bottom:1px solid var(--line)} .f .ptabs2 span{position:relative;display:flex;align-items:center;padding:0 10px;color:var(--muted)} .f .ptabs2 .is-on{color:var(--text);font-weight:500} .f .ptabs2 .is-on::after{content:"";position:absolute;left:10px;right:10px;bottom:-1px;height:2px;background:var(--text)}
.f .pchart{padding:10px 16px} .f .pchart .pmhead{padding:0 0 4px}
.f .sheet{position:absolute;left:0;right:0;bottom:0;height:396px;background:var(--rail);border-top:1px solid var(--line2);border-radius:8px 8px 0 0;display:flex;flex-direction:column}
.f .sheetbar{position:relative;display:flex;justify-content:center;align-items:center;height:30px;flex:none} .f .sheetbar .btn{position:absolute;right:12px;top:4px}
.f .grab{display:block;width:36px;height:4px;border-radius:2px;background:var(--stroke);flex:none}
.f .inspector.is-sheet{width:auto;border:0;flex:1;min-height:0;background:none} .f .inspector.is-sheet .ihead{padding:8px 16px 12px} .f .inspector.is-sheet .itabs{padding:0 6px}
.f .pullnote{text-align:center;font-size:11px;padding:6px 0 18px;border-top:1px solid var(--line)}
.f .states{padding:28px 32px;height:100%;overflow:hidden} .f .shead2 h1{font-size:24px;line-height:32px;font-weight:600} .f .shead2 p{margin-top:4px}
.f .sgrid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0;margin-top:22px;align-items:stretch;border-top:1px solid var(--line2);border-left:1px solid var(--line2)}
.f .scard{border-right:1px solid var(--line2);border-bottom:1px solid var(--line2);padding:16px;display:grid;gap:12px;align-content:start;background:var(--rail)} .f .scard h3{font-size:13.5px;font-weight:500}
.f .sbody{display:grid;gap:12px;align-content:start} .f .srow{display:grid;grid-template-columns:14px minmax(0,1fr);gap:10px;align-items:start} .f .srow .glyph{margin-top:3px} .f .srow div{display:grid;gap:2px;min-width:0} .f .srow b{font-weight:500}
.f .cmd{display:block;padding:10px;border:1px solid var(--line2);background:var(--bg);font:400 11.5px/18px PlexMono,monospace;white-space:pre-wrap}
`

// ======================= frames =======================
const ctx = (theme, lang) => ({ theme, lang, th: THEMES[theme], t: (key) => T[lang][key] })
// Layout stress test only (owner-decisions.md › 交给高保真阶段的已知缺口): the 4 demo directions × 5.
const dirs20 = d3.range(5).flatMap((k) => dirs.map((d) => ({ ...d, id: k ? `${d.id}-r${k + 1}` : d.id })))
const FRAMES = [
  { id: 'home-dark-zh', group: 'home', w: 1440, h: 900, theme: 'dark', lang: 'zh', label: '首页 · 深色 · 中文 · 1440×900', render: (c) => home(c) },
  { id: 'home-light-zh', group: 'home', w: 1440, h: 900, theme: 'light', lang: 'zh', label: '首页 · 浅色 · 中文', render: (c) => home(c) },
  { id: 'home-dark-en', group: 'home', w: 1440, h: 900, theme: 'dark', lang: 'en', label: '首页 · 深色 · English', render: (c) => home(c) },
  { id: 'home-1024', group: 'home', w: 1024, h: 1400, theme: 'dark', lang: 'zh', label: '首页 · 1024 宽整页 · 带文字窄栏，动态移到表格下方', render: (c) => home(c, { narrow: true, stacked: true, gaugeW: 200 }) },
  { id: 'home-20', group: 'home', w: 1440, h: 900, theme: 'dark', lang: 'zh', label: '首页 · 20 个方向的版式压力测试（演示方向重复 5 次）', render: (c) => home(c, { list: dirs20, note: c.t('stress') }) },
  { id: 'report', group: 'research', w: 1440, h: 1560, theme: 'dark', lang: 'zh', label: '方向研究报告 · 已结论方向 · h001 · 整页', render: (c) => report(c) },
  { id: 'runs', group: 'runs', w: 1440, h: 1060, theme: 'dark', lang: 'zh', label: '全部运行 · 按评估器分组 · 勾选 3 个 · 整页', render: (c) => runsPage(c) },
  { id: 'compare', group: 'runs', w: 1440, h: 960, theme: 'dark', lang: 'zh', label: '多运行比较 · 累计最佳曲线、跨运行排行与运行级指标', render: (c) => comparePage(c) },
  { id: 'lineage', group: 'analysis', w: 1440, h: 900, theme: 'dark', lang: 'zh', label: '实验分析 · 谱系（提示词与元分析轨道、全局结论面板）+ 检查器（本程序元分析）', render: (c) => analysisShell(c, 'lineage', lineageView(c, { w: 932, h: 676 }), 'meta') },
  { id: 'progress', group: 'analysis', w: 1440, h: 900, theme: 'dark', lang: 'zh', label: '实验分析 · 进展 + 检查器（代码）', render: (c) => analysisShell(c, 'progress', progressView(c, { w: 932 }), 'code') },
  { id: 'space', group: 'analysis', w: 1440, h: 900, theme: 'dark', lang: 'zh', label: '实验分析 · 解空间（失败带、MAP 与多样性）+ 检查器（评估）', render: (c) => analysisShell(c, 'space', spaceView(c, { w: 932, h: 760 }), 'evaluation') },
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
