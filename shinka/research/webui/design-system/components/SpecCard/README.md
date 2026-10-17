# SpecCard

The spec card is the right-hand sheet for the selected hypothesis: its title, a key–value spec sheet, its notes, evolution-tree links and then whatever the stage allows: next step, progress or the check ledger.

## What the consumer provides

- The selected `Hypothesis` (title, id, status, `created`, experiments, `body` Markdown, `verdict`, `confirmation`), the direction's trial points (for progress and the current noise line) and the evolution-tree base URL (`tree_url`, may be `null`).
- It sits in the sticky side column (`--side-width`); on narrow screens it follows the table in normal flow.

## Built from

- Carbon `Tile` with class `spec-card` (a 1px `line` border, no fill step, `space-5` padding, square), `h2.spec-title` (`title-card`).
- The spec sheet: `dl.spec`, each row `<div><dt>key</dt><dd>value</dd></div>` on a `--spec-key-width` key column. Keys are `label-spec` (mono, muted); values `text-field`, figures and ids in `.mono`. Keys that are terms use `Term`. Rows: 编号 · 状态 (`.status-cell`: Astro `rux-status` + word) · 创建 (`YYYY-MM-DD HH:MM`) · 实验 · 试验数 · 花费 (`$0.00`).
- `.md` for the hypothesis notes (rendered Markdown, headings shrunk to 13px inside the card); `ul.trees` of experiment ids with a Carbon `Link` "看进化树 ↗" opening in a new tab; section labels are `h3` in `label-section`.
- Carbon `InlineNotification` (`kind="info"`, `lowContrast`, `hideCloseButton`) for the no-verdict notice. See CheckLedger for the verdict section.

## States, by stage

| Stage | Below the spec sheet and notes |
| --- | --- |
| No experiments | 下一步: 这个假设还没有实验。用 `shinka_research scaffold` 搭好实验，再用 `shinka_research run` 开始进化。 |
| Experiments, no trials | 下一步: 实验已搭好，还没有试验。用 `shinka_research run` 开始进化。 |
| Running (trials, no verdict) | 进度 (试验数 with failures, 最好样本内分数, 此刻噪声线) · one reading sentence: higher than the noise line says *有信号，但这只是样本内，还没经过验证段检验*, lower says *目前和碰运气分不开*, no score says *这个假设的试验都失败了，还没有分数可读* · the info notice 还没出结论：验证段结果出结论后显示。 |
| Verdict | CheckLedger. No notice. |
| Confirmed or unconfirmed (`confirmation` present) | CheckLedger, then 封存段结果: a spec sheet of the confirmation figures (封存段 Sharpe, 样本内 Sharpe, DSR, PBO, 子区间为正比例, 买入持有 Sharpe, 初始程序 Sharpe, 试验数, 封存段 bar 数, 程序). |
| No notes / no tree | The `.md` block and the 进化树 section are simply absent. |

## Don'ts

- Never show a validation figure (Sharpe, DSR, PBO, sub-period share) before a verdict exists, and never show lockbox figures before `confirmation`. This is the product's core rule.
- Never show a progress number without the noise line beside it.
- Never fill the card with a raised background or add a shadow; it is a ruled sheet.
- Never use orange here except for the link and focus.
- Never add an edit, confirm or lockbox control.
