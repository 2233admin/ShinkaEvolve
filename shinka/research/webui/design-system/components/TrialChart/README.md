# TrialChart

The trial chart plots every in-sample score by trial order with the best-so-far frontier and the noise line, so no score is ever read without the chance level beside it.

## What the consumer provides

- `points`: the direction's trial points (`n`, `hypothesis`, `score` or `null` for a failed program, `record`, `bar`), `probes` (trials without a score), the selected hypothesis id, and the theme flag.
- Colours are read at run time from the CSS custom properties (`ink`, `muted`, `base`, `line`, `line-strong`, `accent`) because the chart draws on a canvas-like SVG that cannot use `var()`; re-read on theme change.

## Built from

- Carbon Charts `ComboChart` (`@carbon/charts-react` 1.27) with `comboChartTypes`: `scatter` for 所选假设, 其他假设, 新纪录 and `line` for 前沿, 噪声线. Options: `theme` `g100` or `white`, `height: '320px'`, `toolbar` off, `legend` at the bottom and not clickable, `curve: 'curveStepAfter'` (the frontier is a step line), `points.radius: 3`, x grid off, axes `n` (试验序号) and `score` (样本内分数), both linear.
- Colour scale: 所选假设 `line-strong`; 其他假设 `line-strong` mixed 62% toward `base`; 新纪录 and 前沿 `ink`; 噪声线 `accent`, dashed (`stroke-dasharray: 5 4` applied on the DOM, because Carbon has no dash option).
- bundle.css strips Carbon's grey backdrop, sets grid lines to `line` and axis text to `muted`.
- Above the chart: `p.chart-note` (`text-note`, muted) with 样本内 and 噪声线 as `Term`s and the counts of failed programs and probes.
- The preview draws a static SVG with the same tokens; it is not the Carbon component.

## States

- Points present: caption, chart, legend.
- Failures: counted in the caption (`N 次程序失败（未画出）`), never plotted as 0.
- Probes: when `probes > 0` the caption adds 另有 N 次试探没有分数记录，不画点，但已计入噪声线.
- No points: no chart, a muted sentence: 还没有试验——用 `shinka_research run` 跑实验。 or, when only probes exist, 还没有进化出的程序（只有 N 次试探，试探不画点）——用 `shinka_research run` 跑实验。

## Don'ts

- Never draw a frontier without the noise line, and never drop the noise line from the legend.
- Never plot a failed program as 0 or hide the failure count.
- Never plot validation or lockbox scores here; this chart is in-sample only.
- Orange is the noise line only: no other series, no highlight.
- Never use Carbon's default categorical colours; every series is mapped to a token.
