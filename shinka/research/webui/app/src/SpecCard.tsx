import { InlineNotification, Link, Tile } from '@carbon/react'
import { Markdown } from './Markdown'
import { STATUS, checksOf, fmt, num, pct, spendOf, trialsOf, usd, when } from './format'
import { Term } from './Term'
import type { Evidence, Hypothesis, TrialPoint } from './types'

function Spec({ rows }: { rows: [React.ReactNode, React.ReactNode][] }) {
  return (
    <dl className="spec">
      {rows.map(([k, v], i) => (
        <div key={i}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  )
}

const CONFIRM_LABELS: Record<string, string> = {
  oos_sharpe: '封存段 Sharpe',
  is_sharpe: '样本内 Sharpe',
  dsr: 'DSR',
  pbo: 'PBO',
  oos_positive_fraction: '子区间为正比例',
  baseline_buy_hold_oos_sharpe: '买入持有 Sharpe',
  baseline_seed_oos_sharpe: '初始程序 Sharpe',
  n_trials: '试验数',
  oos_bars: '封存段 bar 数',
  program_id: '程序',
}

const show = (v: unknown): string =>
  typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toFixed(3)) : typeof v === 'string' ? v : JSON.stringify(v)

function Confirmation({ data }: { data: Evidence }) {
  const keys = Object.keys(data).filter((k) => data[k] !== undefined && typeof data[k] !== 'object')
  return (
    <section>
      <h3>
        <Term k="sealed">封存段</Term>结果
      </h3>
      <Spec rows={keys.map((k) => [CONFIRM_LABELS[k] ?? k, <span className="mono">{show(data[k])}</span>])} />
    </section>
  )
}

function Ledger({ h }: { h: Hypothesis }) {
  const v = h.verdict!
  const ev = v.evidence
  const th = ev.thresholds
  const noise = num(ev.expected_max_sharpe_annual)
  return (
    <section>
      <h3>检查台账</h3>
      <ul className="ledger">
        {checksOf(ev, th).map((c) => (
          <li key={c.key}>
            <rux-status status={c.pass === null ? 'off' : c.pass ? 'normal' : 'critical'} />
            <span className="ledger-label">
              {c.key === 'oos' && (<><Term k="validation">验证段</Term> <Term k="sharpe">Sharpe</Term> {'> 0'}</>)}
              {c.key === 'hold' && (<>跑赢<Term k="buyhold">买入持有</Term></>)}
              {c.key === 'dsr' && (<><Term k="dsr">DSR</Term> ≥ {th.dsr_supported}</>)}
              {c.key === 'sub' && (<><Term k="subperiod">子区间</Term>为正 ≥ {pct(th.min_positive_fraction)}</>)}
              {c.key === 'pbo' && (<><Term k="pbo">PBO</Term> ≤ {th.max_pbo}</>)}
            </span>
            <span className="mono ledger-value">{c.value}</span>
          </li>
        ))}
      </ul>
      <Spec
        rows={[
          [<><Term k="insample">样本内</Term> <Term k="sharpe">Sharpe</Term></>, <span className="mono">{fmt(ev.is_sharpe)}</span>],
          [<Term k="noise">噪声线</Term>, <span className="mono">{fmt(noise)}</span>],
          [<Term k="trials">试验数</Term>, <span className="mono">{fmt(ev.n_trials, 0)}</span>],
        ]}
      />
      <p className="reason">{v.reason}</p>
    </section>
  )
}

const Code = ({ children }: { children: string }) => <code className="mono">{children}</code>

// 还没有试验的两种情形：没搭实验，或搭了没跑。
function NextStep({ h }: { h: Hypothesis }) {
  return (
    <section>
      <h3>下一步</h3>
      <p className="reason">
        {h.experiments.length === 0 ? (
          <>这个假设还没有实验。用 <Code>shinka_research scaffold</Code> 搭好实验，再用 <Code>shinka_research run</Code> 开始进化。</>
        ) : (
          <>实验已搭好，还没有试验。用 <Code>shinka_research run</Code> 开始进化。</>
        )}
      </p>
    </section>
  )
}

// 有试验、没结论：只看这个假设自己的样本内进度，对着此刻的噪声线读。
function Progress({ pts, bar }: { pts: TrialPoint[]; bar: number | null }) {
  const scores = pts.flatMap((p) => (p.score === null ? [] : [p.score]))
  const best = scores.length ? Math.max(...scores) : null
  let reading = '这个假设的试验都失败了，还没有分数可读。'
  if (best !== null && bar !== null)
    reading =
      best > bar
        ? '最好分数高于噪声线：有信号，但这只是样本内，还没经过验证段检验。'
        : '最好分数低于噪声线：目前和碰运气分不开。'
  return (
    <section>
      <h3>进度</h3>
      <Spec
        rows={[
          [<Term k="trials">试验数</Term>, <span className="mono">{pts.length}（失败 {pts.length - scores.length}）</span>],
          [<>最好<Term k="insample">样本内</Term>分数</>, <span className="mono">{fmt(best)}</span>],
          [<>此刻<Term k="noise">噪声线</Term></>, <span className="mono">{fmt(bar)}</span>],
        ]}
      />
      <p className="reason">{reading}</p>
    </section>
  )
}

function Trees({ h, treeUrl }: { h: Hypothesis; treeUrl: string }) {
  const exps = h.experiments.filter((e) => e.tree_db)
  if (exps.length === 0) return null
  return (
    <section>
      <h3>进化树</h3>
      <ul className="trees">
        {exps.map((e) => (
          <li key={e.id}>
            <span className="mono">{e.id}</span>
            <Link href={`${treeUrl}?db_path=${encodeURIComponent(e.tree_db!)}`} target="_blank" rel="noopener noreferrer">
              看进化树 ↗
            </Link>
          </li>
        ))}
      </ul>
    </section>
  )
}

type Props = { h: Hypothesis; points: TrialPoint[]; treeUrl: string | null }

export function SpecCard({ h, points, treeUrl }: Props) {
  const st = STATUS[h.status]
  const mine = points.filter((p) => p.hypothesis === h.id)
  const bar = points.length ? points[points.length - 1].bar : null
  return (
    <Tile className="spec-card">
      <h2 className="spec-title">{h.title}</h2>
      <Spec
        rows={[
          ['编号', <span className="mono">{h.id}</span>],
          [
            '状态',
            <span className="status-cell">
              <rux-status status={st.rux} />
              {st.label}
            </span>,
          ],
          ['创建', <span className="mono">{when(h.created)}</span>],
          ['实验', <span className="mono">{h.experiments.length}</span>],
          [<Term k="trials">试验数</Term>, <span className="mono">{trialsOf(h)}</span>],
          ['花费', <span className="mono">{usd(spendOf(h))}</span>],
        ]}
      />
      {h.body.trim() && <Markdown text={h.body} />}
      {treeUrl && <Trees h={h} treeUrl={treeUrl} />}
      {h.verdict ? (
        <Ledger h={h} />
      ) : mine.length === 0 ? (
        <NextStep h={h} />
      ) : (
        <>
          <Progress pts={mine} bar={bar} />
          <InlineNotification
            kind="info"
            lowContrast
            hideCloseButton
            title="还没出结论："
            subtitle="验证段结果出结论后显示。"
          />
        </>
      )}
      {h.confirmation && <Confirmation data={h.confirmation} />}
    </Tile>
  )
}
