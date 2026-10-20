import { useEffect, useMemo, useRef } from 'react'
import { ComboChart } from '@carbon/charts-react'
import '@carbon/charts-react/styles.css'
import { Rich, useCopy } from './i18n/lang'
import { PARTS } from './Term'
import { mix, useTokens } from './tokens'
import type { TrialPoint } from './types'

type Props = { points: TrialPoint[]; probes: number; selected: string; light: boolean }

export function TrialChart({ points, probes, selected, light }: Props) {
  const { t } = useCopy()
  // Carbon Charts names each dataset by its label, so the legend follows the language.
  const G = useMemo(() => ({
    selected: t('chart.selected'),
    other: t('chart.other'),
    record: t('chart.record'),
    frontier: t('chart.frontier'),
    noise: t('chart.noise'),
  }), [t])
  const tk = useTokens(light)
  const box = useRef<HTMLDivElement>(null)
  const failed = points.filter((p) => p.score === null).length

  const data = useMemo(() => {
    const rows: Record<string, unknown>[] = []
    let best: number | null = null
    for (const p of points) {
      rows.push({ group: G.noise, n: p.n, score: p.bar })
      if (p.score === null) continue
      if (p.record) {
        rows.push({ group: G.record, n: p.n, score: p.score })
        rows.push({ group: G.frontier, n: p.n, score: p.score })
        best = p.score
      } else {
        rows.push({ group: p.hypothesis === selected ? G.selected : G.other, n: p.n, score: p.score })
      }
    }
    // 前沿是阶梯线：最后一个纪录一直延伸到最后一次试验。
    const last = points[points.length - 1]
    if (best !== null && last) rows.push({ group: G.frontier, n: last.n, score: best })
    return rows
  }, [points, selected, G])

  const options = useMemo(() => {
    if (!tk) return null
    return {
      theme: light ? 'white' : 'g100',
      height: '320px',
      toolbar: { enabled: false },
      legend: { position: 'bottom', clickable: false },
      curve: 'curveStepAfter',
      points: { radius: 3, fillOpacity: 1, enabled: true },
      grid: { x: { enabled: false } },
      axes: {
        bottom: { mapsTo: 'n', scaleType: 'linear', title: t('chart.axisTrial') },
        left: { mapsTo: 'score', scaleType: 'linear', title: t('chart.axisScore') },
      },
      color: {
        scale: {
          [G.selected]: tk.lineStrong,
          [G.other]: mix(tk.lineStrong, tk.base, 0.62),
          [G.record]: tk.ink,
          [G.frontier]: tk.ink,
          [G.noise]: tk.accent,
        },
      },
      comboChartTypes: [
        { type: 'scatter', correspondingDatasets: [G.selected, G.other, G.record] },
        { type: 'line', options: { points: { enabled: false } }, correspondingDatasets: [G.frontier, G.noise] },
      ],
    }
  }, [tk, light, G, t])

  // Carbon 的线没有虚线选项：认出噪声线（唯一用强调色描边的线）后在 DOM 上加 dasharray。
  useEffect(() => {
    const el = box.current
    if (!el || !tk) return
    const probe = document.createElement('i')
    probe.style.color = tk.accent
    document.body.appendChild(probe)
    const rgb = getComputedStyle(probe).color
    probe.remove()
    const dash = () =>
      el.querySelectorAll<SVGPathElement>('path.line').forEach((p) => {
        p.style.strokeDasharray = getComputedStyle(p).stroke === rgb ? '5 4' : ''
      })
    dash()
    const mo = new MutationObserver(dash)
    mo.observe(el, { childList: true, subtree: true, attributes: true, attributeFilter: ['d', 'style', 'class'] })
    return () => mo.disconnect()
  }, [tk, data])

  return (
    <div className="trial-chart" ref={box}>
      <p className="chart-note">
        {t('chart.note', { n: points.length, failed })}
        {probes > 0 && t('chart.noteProbes', { probes })}
        <Rich k="chart.noteAxes" parts={PARTS} />
      </p>
      {options && <ComboChart data={data as never} options={options as never} />}
    </div>
  )
}
