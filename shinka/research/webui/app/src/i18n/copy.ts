// Every interface string in Chinese and English (award-redesign task 2.2).
// {name} is filled by t(key, { name }); <tag>text</tag> marks a span that rich() hands to a renderer,
// so a sentence keeps its own word order in each language. Research content (titles, notes, Markdown)
// is data, not copy: it renders as written inside translate="no".
export const COPY = {
  // Shell
  'nav.label': { zh: '工作区导航', en: 'Workspace navigation' },
  'nav.research': { zh: '研究方向', en: 'Directions' },
  'nav.runs': { zh: '全部运行', en: 'All runs' },
  'nav.library': { zh: '资料库', en: 'Library' },
  'nav.pending': { zh: '{n} 份资料待整理', en: '{n} sources awaiting triage' },
  'top.trials': { zh: '试验数', en: 'Trials' },
  'top.budget': { zh: '预算', en: 'Budget' },
  'top.lockbox': { zh: '封存段', en: 'Lockbox' },
  'theme.label': { zh: '主题', en: 'Theme' },
  'theme.dark': { zh: '深色', en: 'Dark' },
  'theme.light': { zh: '浅色', en: 'Light' },
  'crumb.label': { zh: '位置', en: 'Breadcrumb' },
  'icon.error': { zh: '错误', en: 'Error' },
  'icon.warning': { zh: '注意', en: 'Warning' },
  'icon.info': { zh: '说明', en: 'Information' },
  'table.sortAsc': { zh: '按“{header}”升序排列', en: 'Sort rows by {header} in ascending order' },
  'table.sortDesc': { zh: '按“{header}”降序排列', en: 'Sort rows by {header} in descending order' },
  'table.sortNone': { zh: '取消按“{header}”排序', en: 'Stop sorting by {header}' },
  'lang.label': { zh: '语言', en: 'Language' },
  'action.refresh': { zh: '刷新', en: 'Refresh' },
  'state.loading': { zh: '载入中…', en: 'Loading…' },
  'lockbox.unset': { zh: '未设', en: 'Not set' },
  'lockbox.opened': { zh: '已开封', en: 'Opened' },
  'lockbox.sealed': { zh: '封存', en: 'Sealed' },
  'event.withheld': { zh: '（验证段结果，出结论后显示）', en: '(validation result, shown after the verdict)' },

  // Page heads
  'crumb.workspace': { zh: '工作区', en: 'Workspace' },
  'crumb.back': { zh: '返回研究 · {where}', en: 'Back to research · {where}' },
  'crumb.direction': { zh: '研究方向', en: 'Direction' },
  'crumb.analysis': { zh: '实验分析', en: 'Experiment analysis' },
  'head.runs': { zh: '运行与分析', en: 'Runs and analysis' },
  'head.library': { zh: '共享资料库', en: 'Shared library' },
  'head.direction': { zh: '研究方向', en: 'Direction' },
  'head.workspace': { zh: '工作区', en: 'Workspace' },
  'head.workspaceNamed': { zh: '工作区 · <name>{name}</name>', en: 'Workspace · <name>{name}</name>' },
  'lede.runs': {
    zh: '浏览运行、比较结果并检查完整实验分析。研究由对话主导；网页只读，私有指标不公开。',
    en: 'Browse runs, compare results and inspect the full experiment analysis. Research is led from the conversation; this page is read-only and private metrics stay hidden.',
  },
  'lede.library': { zh: '所有研究方向共享同一份资料库。', en: 'Every direction shares one library.' },
  'lede.workspace': {
    zh: '选择研究方向查看假设与结论。<trials>试验数</trials>按方向累计；<sealed>封存段</sealed>状态按方向展示。同一工作区共用一个资料库，研究操作仍通过 CLI 执行。',
    en: 'Choose a direction to see its hypotheses and verdicts. <trials>Trials</trials> add up per direction, and each direction shows its own <sealed>lockbox</sealed> state. The workspace shares one library; research actions still go through the CLI.',
  },
  'lede.direction': {
    zh: '每个假设在<insample>样本内</insample>进化，再由质疑者在<validation>验证段</validation>检验；<validation2>验证段</validation2>结果只在出结论后显示，<sealed>封存段</sealed>只在最终确认时打开一次。',
    en: 'Each hypothesis evolves <insample>in-sample</insample>, then the skeptic tests it on the <validation>validation segment</validation>. <validation2>Validation</validation2> results appear only after a verdict, and the <sealed>lockbox</sealed> opens once, at final confirmation.',
  },
  'error.read': { zh: '读取失败：', en: 'Could not read:' },
  'error.readHint': {
    zh: '{error}。确认 shinka_research ui 还在运行，然后点刷新。',
    en: '{error}. Check that shinka_research ui is still running, then refresh.',
  },
  'empty.hypotheses': { zh: '还没有假设：', en: 'No hypotheses yet:' },
  'empty.hypothesesHint': { zh: '用 shinka_research hypothesis add 添加第一个假设。', en: 'Add the first one with shinka_research hypothesis add.' },

  // Workspace table
  'ws.empty': { zh: '还没有研究方向——用 <cmd>…</cmd> 初始化。', en: 'No directions yet. Create one with <cmd>…</cmd>.' },
  'ws.initCommand': {
    zh: 'shinka_research init <workspace>/<direction> --direction "研究方向"',
    en: 'shinka_research init <workspace>/<direction> --direction "Research question"',
  },
  'ws.table': { zh: '研究方向列表', en: 'Directions' },
  'ws.colDirection': { zh: '研究方向', en: 'Direction' },
  'ws.colHypotheses': { zh: '假设数', en: 'Hypotheses' },
  'ws.colTrials': { zh: '试验数', en: 'Trials' },
  'ws.colSpend': { zh: '花费 / 预算', en: 'Spend / budget' },
  'ws.colLockbox': { zh: '封存段', en: 'Lockbox' },
  'ws.linkLabel': { zh: '{direction}（目录：{id}）', en: '{direction} (folder: {id})' },
  'ws.readError': { zh: '研究记录无法读取（{error}）', en: 'Research notebook unreadable ({error})' },

  // Direction page
  'dir.hypotheses': { zh: '假设', en: 'Hypotheses' },
  'dir.trials': { zh: '试验', en: 'Trials' },
  'dir.timeline': { zh: '实验时间线', en: 'Experiment timeline' },
  'dir.events': { zh: '事件', en: 'Events' },
  'dir.noProgramsProbes': {
    zh: '还没有进化出的程序（只有 {n} 次试探，试探不画点）——用 <cmd>shinka_research run</cmd> 跑实验。',
    en: 'No evolved programs yet ({n} probes only; probes are not plotted). Run an experiment with <cmd>shinka_research run</cmd>.',
  },
  'dir.noTrials': {
    zh: '还没有试验——用 <cmd>shinka_research run</cmd> 跑实验。',
    en: 'No trials yet. Run an experiment with <cmd>shinka_research run</cmd>.',
  },
  'dir.noTimeline': {
    zh: '还没有试验，画不出时间线——用 <cmd>shinka_research run</cmd> 跑实验。',
    en: 'No trials yet, so there is no timeline. Run an experiment with <cmd>shinka_research run</cmd>.',
  },
  'dir.noEvents': { zh: '还没有事件。', en: 'No events yet.' },

  // Hypothesis table
  'hyp.table': { zh: '假设列表', en: 'Hypotheses' },
  'hyp.colId': { zh: '编号', en: 'ID' },
  'hyp.colTitle': { zh: '假设', en: 'Hypothesis' },
  'hyp.colStatus': { zh: '状态', en: 'Status' },
  'hyp.colExperiments': { zh: '实验数', en: 'Experiments' },
  'hyp.colTrials': { zh: '<trials>试验数</trials>', en: '<trials>Trials</trials>' },
  'hyp.colBest': { zh: '最好<insample>样本内</insample>分数', en: 'Best <insample>in-sample</insample> score' },
  'hyp.colOos': { zh: '<validation>验证段</validation> <sharpe>Sharpe</sharpe>', en: '<validation>Validation</validation> <sharpe>Sharpe</sharpe>' },

  // Hypothesis status words (design.md › 状态)
  'status.proposed': { zh: '待跑', en: 'queued' },
  'status.selected': { zh: '已选', en: 'selected' },
  'status.running': { zh: '运行中', en: 'running' },
  'status.challenged': { zh: '待判定', en: 'challenged' },
  'status.supported': { zh: '成立', en: 'supported' },
  'status.refuted': { zh: '否定', en: 'refuted' },
  'status.inconclusive': { zh: '未定', en: 'inconclusive' },
  'status.confirmed': { zh: '已确认', en: 'confirmed' },
  'status.unconfirmed': { zh: '未确认', en: 'unconfirmed' },
  'status.dropped': { zh: '放弃', en: 'dropped' },

  // Trial chart
  'chart.selected': { zh: '所选假设', en: 'Selected hypothesis' },
  'chart.other': { zh: '其他假设', en: 'Other hypotheses' },
  'chart.record': { zh: '新纪录', en: 'New record' },
  'chart.frontier': { zh: '前沿', en: 'Frontier' },
  'chart.noise': { zh: '噪声线', en: 'Noise line' },
  'chart.axisTrial': { zh: '试验序号', en: 'Trial' },
  'chart.axisScore': { zh: '样本内分数', en: 'In-sample score' },
  'chart.note': {
    zh: '{n} 次试验，其中 {failed} 次程序失败（未画出）。',
    en: '{n} trials, {failed} of them failed programs (not plotted). ',
  },
  'chart.noteProbes': {
    zh: '另有 {probes} 次试探没有分数记录，不画点，但已计入噪声线。',
    en: '{probes} more probes have no recorded score: they are not plotted but count toward the noise line. ',
  },
  'chart.noteAxes': {
    zh: '横轴是整个方向的试验顺序，纵轴是<insample>样本内</insample>分数；虚线是<noise>噪声线</noise>，线下的分数和碰运气分不开。',
    en: 'The x-axis is the order of trials across the direction and the y-axis the <insample>in-sample</insample> score; the dashed line is the <noise>noise line</noise>, and scores under it cannot be told apart from luck.',
  },

  // Spec card
  'spec.id': { zh: '编号', en: 'ID' },
  'spec.status': { zh: '状态', en: 'Status' },
  'spec.created': { zh: '创建', en: 'Created' },
  'spec.experiments': { zh: '实验', en: 'Experiments' },
  'spec.trials': { zh: '<trials>试验数</trials>', en: '<trials>Trials</trials>' },
  'spec.spend': { zh: '花费', en: 'Spend' },
  'spec.analysis': { zh: '实验分析', en: 'Experiment analysis' },
  'spec.openAnalysis': { zh: '查看实验分析', en: 'Open the analysis' },
  'spec.noVerdict': { zh: '还没出结论：', en: 'No verdict yet:' },
  'spec.noVerdictHint': { zh: '验证段结果出结论后显示。', en: 'Validation results appear after the verdict.' },
  'spec.next': { zh: '下一步', en: 'Next step' },
  'spec.nextScaffold': {
    zh: '这个假设还没有实验。用 <cmd>shinka_research scaffold</cmd> 搭好实验，再用 <cmd2>shinka_research run</cmd2> 开始进化。',
    en: 'This hypothesis has no experiment yet. Set one up with <cmd>shinka_research scaffold</cmd>, then start evolving with <cmd2>shinka_research run</cmd2>.',
  },
  'spec.nextRun': {
    zh: '实验已搭好，还没有试验。用 <cmd>shinka_research run</cmd> 开始进化。',
    en: 'The experiment is set up but has no trials. Start evolving with <cmd>shinka_research run</cmd>.',
  },
  'spec.progress': { zh: '进度', en: 'Progress' },
  'spec.trialsFailed': { zh: '{n}（失败 {failed}）', en: '{n} ({failed} failed)' },
  'spec.best': { zh: '最好<insample>样本内</insample>分数', en: 'Best <insample>in-sample</insample> score' },
  'spec.noiseNow': { zh: '此刻<noise>噪声线</noise>', en: '<noise>Noise line</noise> now' },
  'spec.readingFailed': { zh: '这个假设的试验都失败了，还没有分数可读。', en: 'Every trial of this hypothesis failed; there is no score to read yet.' },
  'spec.readingAbove': {
    zh: '最好分数高于噪声线：有信号，但这只是样本内，还没经过验证段检验。',
    en: 'The best score is above the noise line: a signal, but in-sample only, not yet tested on the validation segment.',
  },
  'spec.readingBelow': { zh: '最好分数低于噪声线：目前和碰运气分不开。', en: 'The best score is below the noise line: so far it cannot be told apart from luck.' },
  'spec.ledger': { zh: '检查台账', en: 'Check ledger' },
  'spec.checkOos': { zh: '<validation>验证段</validation> <sharpe>Sharpe</sharpe> > 0', en: '<validation>Validation</validation> <sharpe>Sharpe</sharpe> > 0' },
  'spec.checkHold': { zh: '跑赢<buyhold>买入持有</buyhold>', en: 'Beats <buyhold>buy and hold</buyhold>' },
  'spec.checkDsr': { zh: '<dsr>DSR</dsr> ≥ {v}', en: '<dsr>DSR</dsr> ≥ {v}' },
  'spec.checkSub': { zh: '<subperiod>子区间</subperiod>为正 ≥ {v}', en: 'Positive <subperiod>sub-periods</subperiod> ≥ {v}' },
  'spec.checkPbo': { zh: '<pbo>PBO</pbo> ≤ {v}', en: '<pbo>PBO</pbo> ≤ {v}' },
  'spec.pboNone': { zh: '无法计算，视为通过', en: 'not computable, counted as a pass' },
  'spec.isSharpe': { zh: '<insample>样本内</insample> <sharpe>Sharpe</sharpe>', en: '<insample>In-sample</insample> <sharpe>Sharpe</sharpe>' },
  'spec.noise': { zh: '<noise>噪声线</noise>', en: '<noise>Noise line</noise>' },
  'spec.lockboxResult': { zh: '<sealed>封存段</sealed>结果', en: '<sealed>Lockbox</sealed> result' },
  'confirm.oos_sharpe': { zh: '封存段 Sharpe', en: 'Lockbox Sharpe' },
  'confirm.is_sharpe': { zh: '样本内 Sharpe', en: 'In-sample Sharpe' },
  'confirm.dsr': { zh: 'DSR', en: 'DSR' },
  'confirm.pbo': { zh: 'PBO', en: 'PBO' },
  'confirm.oos_positive_fraction': { zh: '子区间为正比例', en: 'Positive sub-period share' },
  'confirm.baseline_buy_hold_oos_sharpe': { zh: '买入持有 Sharpe', en: 'Buy-and-hold Sharpe' },
  'confirm.baseline_seed_oos_sharpe': { zh: '初始程序 Sharpe', en: 'Seed program Sharpe' },
  'confirm.n_trials': { zh: '试验数', en: 'Trials' },
  'confirm.oos_bars': { zh: '封存段 bar 数', en: 'Lockbox bars' },
  'confirm.program_id': { zh: '程序', en: 'Program' },
  'confirm.experiment': { zh: '实验', en: 'Experiment' },
  'confirm.lockbox_start': { zh: '封存段起点', en: 'Lockbox start' },
  'confirm.lockbox_bars': { zh: '封存段 bar 数', en: 'Lockbox bars' },
  'confirm.lockbox_sharpe': { zh: '封存段 Sharpe', en: 'Lockbox Sharpe' },
  'confirm.lockbox_annual_return': { zh: '封存段年化收益', en: 'Lockbox annual return' },
  'confirm.lockbox_max_drawdown': { zh: '封存段最大回撤', en: 'Lockbox maximum drawdown' },
  'confirm.lockbox_buy_hold_sharpe': { zh: '封存段买入持有 Sharpe', en: 'Lockbox buy-and-hold Sharpe' },
  'confirm.lockbox_positive_fraction': { zh: '封存段子区间为正比例', en: 'Lockbox positive sub-period share' },
  'confirm.passed': { zh: '通过', en: 'Passed' },
  'confirm.yes': { zh: '是', en: 'yes' },
  'confirm.no': { zh: '否', en: 'no' },

  // Library
  'lib.empty': { zh: '还没有资料库——用 <cmd>shinka_research library add</cmd> 加资料。', en: 'No library yet. Add sources with <cmd>shinka_research library add</cmd>.' },
  'lib.pending': { zh: '有 {n} 份资料待整理：', en: '{n} sources awaiting triage:' },
  'lib.pendingMore': { zh: '{list} 等', en: '{list} and more' },
  'lib.listSeparator': { zh: '、', en: ', ' },
  'lib.pages': { zh: '资料库页面', en: 'Library pages' },
  'lib.index': { zh: '目录', en: 'Index' },
  'lib.log': { zh: '日志', en: 'Log' },
  'lib.pageGroup': { zh: '页面（{n}）', en: 'Pages ({n})' },
  'lib.noPages': { zh: 'wiki 里还没有页面，研究员整理资料后会写到这里。', en: 'The wiki has no pages yet; the researcher writes them here after triaging sources.' },
  'lib.emptyPage': { zh: '这一页是空的。', en: 'This page is empty.' },

  // Analysis host
  'an.section': { zh: '运行浏览与实验分析', en: 'Runs and experiment analysis' },
  'an.context': {
    zh: '本页显示各运行的原始 <code>combined_score</code>，只供诊断；不同评估器的得分不可直接比较。研究结论和全方向试验的噪声基准以对应研究方向页为准，本页不另行推算。',
    en: 'This page shows each run’s raw <code>combined_score</code> for diagnosis only; scores from different evaluators are not comparable. Research verdicts and the direction-wide noise benchmark live on the direction page; this page does not recompute them.',
  },
  'an.readError': { zh: '分析读取失败：', en: 'Analysis failed to load:' },
  'an.compareSelection': {
    zh: '比较链接缺少运行选择。请从“全部运行”选择至少两个运行。',
    en: 'The compare link has no run selection. Pick at least two runs in All runs.',
  },
  'an.badLink': { zh: '分析链接无效。请从研究实验或“全部运行”重新进入。', en: 'This analysis link is not valid. Open it again from an experiment or from All runs.' },
  'an.notLoaded': { zh: '分析资源未加载。请检查前端构建和静态资源后刷新页面。', en: 'The analysis assets did not load. Check the frontend build and static files, then reload.' },
  'an.leftWorkspace': { zh: '分析页面离开了工作区。请重新进入“全部运行”。', en: 'The analysis page left the workspace. Open All runs again.' },

  // Terms (plain-language definitions shown on hover and focus)
  'term.sharpe': {
    zh: 'Sharpe（夏普比率）：每承担一份波动换来多少收益，越大越好，0 以下说明不如不做。',
    en: 'Sharpe ratio: return per unit of volatility. Higher is better; below 0, doing nothing would have been better.',
  },
  'term.insample': {
    zh: '样本内：进化过程中程序能看到、拿来打分的那段历史数据。',
    en: 'In-sample: the stretch of history the programs see and are scored on during evolution.',
  },
  'term.validation': {
    zh: '验证段：进化时看不到的一段数据，质疑者用它检验结果；只在出结论后才显示。',
    en: 'Validation segment: data hidden during evolution that the skeptic uses to test the result; shown only after a verdict.',
  },
  'term.sealed': {
    zh: '封存段：数据最末尾单独封起来的一段，整个方向只在最终确认时打开一次。',
    en: 'Lockbox: the final stretch of data, sealed apart; the direction opens it once, at final confirmation.',
  },
  'term.trials': {
    zh: '试验数：整个方向里一共试过多少个程序（含探测），试得越多，碰巧撞上好结果的机会越大。',
    en: 'Trials: how many programs the whole direction has tried, probes included. The more you try, the likelier a good result is luck.',
  },
  'term.noise': {
    zh: '噪声线：试了这么多次，纯靠运气也能拿到的最高 Sharpe；真实结果要明显高于它才可信。',
    en: 'Noise line: the best Sharpe luck alone would reach after this many trials; a real result has to clear it clearly.',
  },
  'term.dsr': {
    zh: 'DSR（紧缩夏普比率）：扣掉「试得多总会撞上好运」的部分后，Sharpe 真的大于 0 的概率。',
    en: 'DSR (deflated Sharpe ratio): the probability that the Sharpe is truly above 0 once the luck of many trials is taken out.',
  },
  'term.pbo': {
    zh: 'PBO（回测过拟合概率）：样本内最好的那个程序，换一段数据后排名掉到中位数以下的可能性，越小越好。',
    en: 'PBO (probability of backtest overfitting): how likely the best in-sample program drops below the median on other data. Lower is better.',
  },
  'term.subperiod': {
    zh: '子区间：把验证段切成几等份，看每一份是否都赚钱，避免靠一两段行情撑起整体结果。',
    en: 'Sub-periods: the validation segment cut into equal parts, each checked for profit, so one or two lucky stretches cannot carry the result.',
  },
  'term.buyhold': {
    zh: '买入持有：一直满仓拿着不动的基准做法，策略至少要比它强才有意义。',
    en: 'Buy and hold: the baseline of staying fully invested; a strategy has to beat it to matter.',
  },
  'term.tbd': { zh: '出结论后显示', en: 'Shown after the verdict' },
} as const satisfies Record<string, { zh: string; en: string }>

export type CopyKey = keyof typeof COPY
