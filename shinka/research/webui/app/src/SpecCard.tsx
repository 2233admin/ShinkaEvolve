import { InlineNotification, Tile } from '@carbon/react'
import { STATUS, checksOf, fmt, num, spendOf, trialsOf, usd, when } from './format'
import { Term } from './Term'
import type { Evidence, Hypothesis } from './types'

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
  const noise = num(ev.expected_max_sharpe_annual)
  return (
    <section>
      <h3>检查台账</h3>
      <ul className="ledger">
        {checksOf(ev).map((c) => (
          <li key={c.key}>
            <rux-status status={c.pass === null ? 'off' : c.pass ? 'normal' : 'critical'} />
            <span className="ledger-label">
              {c.key === 'oos' && (<><Term k="validation">验证段</Term> <Term k="sharpe">Sharpe</Term> {'> 0'}</>)}
              {c.key === 'hold' && (<>跑赢<Term k="buyhold">买入持有</Term></>)}
              {c.key === 'dsr' && (<><Term k="dsr">DSR</Term> ≥ 0.95</>)}
              {c.key === 'sub' && (<><Term k="subperiod">子区间</Term>为正 ≥ 2/3</>)}
              {c.key === 'pbo' && (<><Term k="pbo">PBO</Term> ≤ 0.5</>)}
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

export function SpecCard({ h }: { h: Hypothesis }) {
  const st = STATUS[h.status]
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
      {h.body.trim() && <pre className="body">{h.body}</pre>}
      {h.verdict ? (
        <Ledger h={h} />
      ) : (
        <InlineNotification
          kind="info"
          lowContrast
          hideCloseButton
          title="还没出结论："
          subtitle="验证段结果出结论后显示。"
        />
      )}
      {h.confirmation && <Confirmation data={h.confirmation} />}
    </Tile>
  )
}
