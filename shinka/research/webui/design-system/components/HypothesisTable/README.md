# HypothesisTable

A compact, sortable table of every hypothesis in the direction; selecting a row drives the spec card, trial chart highlight and timeline.

## What the consumer provides

- `hyps`: the snapshot's hypotheses (id, title, status, experiments, verdict).
- `selected`: the current hypothesis id, and `onSelect(id)`; the table never owns selection.

## Built from

- Carbon `DataTable` (`size="lg"`) with `Table`, `TableHead`, `TableHeader`, `TableBody`, `TableRow`, `TableCell`; the markup is `cds--data-table cds--data-table--lg` inside `.table-wrap` (horizontal scroll, `min-width: 720px`).
- Columns: 编号 · 假设 · 状态 · 实验数 (sortable: `isSortable`), 试验数 · 最好样本内分数 · 验证段 Sharpe (not sortable, headers carry `Term`s). A custom `sortRow` sorts numbers numerically and strings with `localeCompare(…, 'zh')`; empty values are `-Infinity` and sort last either way.
- The status cell is `.status-cell`: Astro `rux-status` and the Chinese label side by side. Ids and numbers use `.mono`; numbers also `.num` (tabular).
- The selected row is `tr.hrow.is-selected`: `raised` ground and a 2px `accent` inset bar on the first cell; `aria-selected`; the whole row is keyboard-operable (`tabIndex=0`, Enter or Space). Focus is a 2px `accent` outline inset.
- Header underline is `line-strong`; row rules come from Carbon, mapped to `line`.

## Status map

| Label | `rux-status` |
| --- | --- |
| 待跑, 放弃 | `off` |
| 已选, 运行中 | `standby` |
| 待判定, 未定 | `caution` |
| 成立, 已确认 | `normal` |
| 否定, 未确认 | `critical` |

## States

- Default: all rows, first row's hypothesis selected when nothing was chosen.
- Selected: one row marked; there is always exactly one.
- No verdict: the 验证段 Sharpe cell is `—` in a `DefinitionTooltip` that says 出结论后显示. Never a number, a dash with no explanation, or a blurred figure.
- No trials: 试验数 `0`, 最好样本内分数 `—`.
- Empty table: not rendered; the page shows the info Notice 还没有假设.
- Narrow screens: the table scrolls inside `.table-wrap`; the page does not scroll sideways.

## Don'ts

- Never show a validation figure for a hypothesis without a verdict.
- Never use filled status pills or coloured row backgrounds; state is the `rux-status` shape plus the word.
- Never use orange for anything but the selected-row bar and focus.
- Never add row actions or inline editing; the table is read-only.
