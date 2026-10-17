---
version: alpha
name: Shinka Research 研究记录本
description: A warm-paper lab-notebook console for Shinka Research. Read-only pages (workspace, direction, experiment, library) record a quant research direction the way a careful researcher keeps a notebook — margin-numbered entries, status stamps in coloured ink, a ✓/✗ check ledger behind every verdict, and a dashed noise bar on every trial chart that rises with the number of trials. Serif for the notebook's hand (titles, stamps, long reading), system sans for everything that must read on any OS (prose, labels, numbers). Chinese interface copy; statistical terms keep English abbreviations.

colors:
  primary: "#1f1c17"
  ink: "#1f1c17"
  ink-secondary: "#4a443a"
  ink-muted: "#665d50"
  on-ink: "#f5f0e6"
  paper: "#f5f0e6"
  paper-deep: "#efe7d8"
  paper-raised: "#faf6ee"
  rule: "#d3c8b5"
  rule-strong: "#b9ad99"
  supported: "#2c5a37"
  refuted: "#8a2b1f"
  inconclusive: "#6b5a2a"
  running: "#1f4a7a"
  link: "#1f4a7a"
  focus: "#1f4a7a"
  noise-bar: "#8a2b1f"
  frontier: "#1f1c17"
  trial-dot: "#b9ad99"
  decider-mark: "#8a2b1f"
  diff-add: "#e3ecd9"
  diff-del: "#f3dcd5"
  diff-hunk: "#e6ddcb"

typography:
  display-lg:
    fontFamily: "Noto Serif CJK SC, Source Han Serif SC, Songti SC, STSong, serif"
    fontSize: 30px
    fontWeight: 600
    lineHeight: 1.35
    letterSpacing: 0
  display-md:
    fontFamily: "Noto Serif CJK SC, Source Han Serif SC, Songti SC, STSong, serif"
    fontSize: 24px
    fontWeight: 600
    lineHeight: 1.4
    letterSpacing: 0
  entry-title:
    fontFamily: "Noto Serif CJK SC, Source Han Serif SC, Songti SC, STSong, serif"
    fontSize: 18px
    fontWeight: 600
    lineHeight: 1.45
    letterSpacing: 0
  entry-id:
    fontFamily: "Noto Serif CJK SC, Source Han Serif SC, Songti SC, STSong, serif"
    fontSize: 15px
    fontWeight: 600
    lineHeight: 1.4
    letterSpacing: 0
  stamp:
    fontFamily: "Noto Serif CJK SC, Source Han Serif SC, Songti SC, STSong, serif"
    fontSize: 12px
    fontWeight: 600
    lineHeight: 1.4
    letterSpacing: 0.1em
  reading:
    fontFamily: "Noto Serif CJK SC, Source Han Serif SC, Songti SC, Microsoft YaHei, sans-serif"
    fontSize: 16px
    fontWeight: 400
    lineHeight: 1.8
    letterSpacing: 0
  reading-h2:
    fontFamily: "Noto Serif CJK SC, Source Han Serif SC, Songti SC, STSong, serif"
    fontSize: 20px
    fontWeight: 600
    lineHeight: 1.45
    letterSpacing: 0
  body-md:
    fontFamily: "system-ui, -apple-system, Segoe UI, PingFang SC, Hiragino Sans GB, Microsoft YaHei, Noto Sans CJK SC, sans-serif"
    fontSize: 15px
    fontWeight: 400
    lineHeight: 1.75
    letterSpacing: 0
  body-sm:
    fontFamily: "system-ui, -apple-system, Segoe UI, PingFang SC, Hiragino Sans GB, Microsoft YaHei, Noto Sans CJK SC, sans-serif"
    fontSize: 13px
    fontWeight: 400
    lineHeight: 1.6
    letterSpacing: 0
  label:
    fontFamily: "system-ui, -apple-system, Segoe UI, PingFang SC, Hiragino Sans GB, Microsoft YaHei, Noto Sans CJK SC, sans-serif"
    fontSize: 12px
    fontWeight: 500
    lineHeight: 1.5
    letterSpacing: 0.08em
  data-md:
    fontFamily: "system-ui, -apple-system, Segoe UI, PingFang SC, Hiragino Sans GB, Microsoft YaHei, Noto Sans CJK SC, sans-serif"
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.6
    letterSpacing: 0
    fontFeature: '"tnum" 1'
  data-lg:
    fontFamily: "system-ui, -apple-system, Segoe UI, PingFang SC, Hiragino Sans GB, Microsoft YaHei, Noto Sans CJK SC, sans-serif"
    fontSize: 20px
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: 0
    fontFeature: '"tnum" 1'
  chart:
    fontFamily: "system-ui, -apple-system, Segoe UI, PingFang SC, Hiragino Sans GB, Microsoft YaHei, Noto Sans CJK SC, sans-serif"
    fontSize: 11px
    fontWeight: 400
    lineHeight: 1.2
    letterSpacing: 0
    fontFeature: '"tnum" 1'
  code:
    fontFamily: "ui-monospace, SF Mono, Menlo, Consolas, DejaVu Sans Mono, Noto Sans Mono CJK SC, monospace"
    fontSize: 13px
    fontWeight: 400
    lineHeight: 1.55
    letterSpacing: 0

rounded:
  none: 0px
  seal: 9999px

spacing:
  xxs: 2px
  xs: 4px
  sm: 8px
  md: 12px
  lg: 20px
  xl: 32px
  xxl: 48px
  margin-col: 150px
  gutter: 40px
  page: 1040px
  measure: 40em
  reading-measure: 36em
  top-bar: 48px

components:
  page:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.body-md}"
    width: 1040px
  top-bar:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink-muted}"
    typography: "{typography.label}"
    height: 48px
  top-bar-link:
    textColor: "{colors.ink-secondary}"
    typography: "{typography.body-sm}"
  top-bar-link-current:
    textColor: "{colors.ink}"
    typography: "{typography.body-sm}"
  breadcrumb:
    textColor: "{colors.ink-muted}"
    typography: "{typography.body-sm}"
  kicker:
    textColor: "{colors.ink-muted}"
    typography: "{typography.label}"
  page-title:
    textColor: "{colors.ink}"
    typography: "{typography.display-lg}"
  page-lede:
    textColor: "{colors.ink-secondary}"
    typography: "{typography.body-md}"
    width: 40em
  margin-note:
    textColor: "{colors.ink-muted}"
    typography: "{typography.body-sm}"
    width: 150px
  facts-row:
    textColor: "{colors.ink}"
    typography: "{typography.data-md}"
    padding: 10px 0
  facts-rule-top:
    backgroundColor: "{colors.ink}"
    height: 1.5px
  facts-rule-bottom:
    backgroundColor: "{colors.rule-strong}"
    height: 1px
  fact-label:
    textColor: "{colors.ink-muted}"
    typography: "{typography.label}"
  fact-value:
    textColor: "{colors.ink}"
    typography: "{typography.data-lg}"
  fact-value-unset:
    textColor: "{colors.ink-muted}"
    typography: "{typography.body-sm}"
  budget-meter:
    backgroundColor: "{colors.rule}"
    height: 3px
  budget-meter-fill:
    backgroundColor: "{colors.ink}"
    height: 3px
  budget-meter-fill-exhausted:
    backgroundColor: "{colors.refuted}"
    height: 3px
  lockbox-seal-sealed:
    textColor: "{colors.refuted}"
    typography: "{typography.stamp}"
    rounded: "{rounded.seal}"
    size: 26px
  lockbox-seal-opened:
    textColor: "{colors.ink-muted}"
    typography: "{typography.stamp}"
    rounded: "{rounded.seal}"
    size: 26px
  section-heading:
    textColor: "{colors.ink-muted}"
    typography: "{typography.label}"
  group-heading:
    textColor: "{colors.ink-muted}"
    typography: "{typography.label}"
  entry-rule:
    backgroundColor: "{colors.rule}"
    height: 1px
  entry-id:
    textColor: "{colors.ink}"
    typography: "{typography.entry-id}"
  entry-title:
    textColor: "{colors.ink}"
    typography: "{typography.entry-title}"
  entry-evidence:
    textColor: "{colors.ink-secondary}"
    typography: "{typography.body-md}"
    width: 40em
  entry-collapsed:
    textColor: "{colors.ink-secondary}"
    typography: "{typography.body-sm}"
    padding: 8px 0
  status-stamp:
    textColor: "{colors.ink-muted}"
    typography: "{typography.stamp}"
    rounded: "{rounded.none}"
    padding: 0 6px
  status-stamp-supported:
    textColor: "{colors.supported}"
    typography: "{typography.stamp}"
  status-stamp-confirmed:
    textColor: "{colors.supported}"
    typography: "{typography.stamp}"
  status-stamp-refuted:
    textColor: "{colors.refuted}"
    typography: "{typography.stamp}"
  status-stamp-unconfirmed:
    textColor: "{colors.refuted}"
    typography: "{typography.stamp}"
  status-stamp-inconclusive:
    textColor: "{colors.inconclusive}"
    typography: "{typography.stamp}"
  status-stamp-running:
    textColor: "{colors.running}"
    typography: "{typography.stamp}"
  status-stamp-challenged:
    textColor: "{colors.running}"
    typography: "{typography.stamp}"
  status-stamp-proposed:
    textColor: "{colors.ink-muted}"
    typography: "{typography.stamp}"
  status-stamp-selected:
    textColor: "{colors.ink-secondary}"
    typography: "{typography.stamp}"
  status-stamp-dropped:
    textColor: "{colors.ink-muted}"
    typography: "{typography.stamp}"
  check-pass:
    textColor: "{colors.supported}"
    typography: "{typography.body-sm}"
  check-fail:
    textColor: "{colors.refuted}"
    typography: "{typography.body-sm}"
  check-decider:
    textColor: "{colors.decider-mark}"
    typography: "{typography.body-sm}"
  check-not-applicable:
    textColor: "{colors.ink-muted}"
    typography: "{typography.body-sm}"
  toggle:
    textColor: "{colors.link}"
    typography: "{typography.body-sm}"
    height: 28px
  toggle-expanded:
    textColor: "{colors.ink}"
    typography: "{typography.body-sm}"
  text-link:
    textColor: "{colors.link}"
    typography: "{typography.body-md}"
  text-link-hover:
    textColor: "{colors.ink}"
  focus-ring:
    textColor: "{colors.focus}"
  term:
    textColor: "{colors.ink}"
  term-tooltip:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.on-ink}"
    typography: "{typography.body-sm}"
    rounded: "{rounded.none}"
    padding: 8px 10px
    width: 280px
  trial-chart:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink-muted}"
    typography: "{typography.chart}"
  trial-chart-band:
    backgroundColor: "{colors.paper-deep}"
  trial-chart-gridline:
    backgroundColor: "{colors.rule}"
  trial-chart-dot:
    backgroundColor: "{colors.trial-dot}"
    size: 3.6px
  trial-chart-kept:
    backgroundColor: "{colors.frontier}"
    size: 5.6px
  trial-chart-frontier:
    backgroundColor: "{colors.frontier}"
    width: 1.6px
  trial-chart-noise-bar:
    backgroundColor: "{colors.noise-bar}"
    width: 1.6px
  trial-chart-tooltip:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.on-ink}"
    typography: "{typography.chart}"
    padding: 6px 8px
  event-row:
    textColor: "{colors.ink-secondary}"
    typography: "{typography.body-sm}"
    padding: 6px 0
  direction-row:
    textColor: "{colors.ink}"
    typography: "{typography.entry-title}"
    padding: 12px 0
  hash-chip:
    textColor: "{colors.ink-secondary}"
    typography: "{typography.code}"
  ledger-entry:
    textColor: "{colors.ink-secondary}"
    typography: "{typography.body-md}"
    padding: 12px 0
  code-block:
    backgroundColor: "{colors.paper-raised}"
    textColor: "{colors.ink}"
    typography: "{typography.code}"
    rounded: "{rounded.none}"
    padding: 12px 16px
  diff-line-add:
    backgroundColor: "{colors.diff-add}"
    textColor: "{colors.ink}"
    typography: "{typography.code}"
  diff-line-del:
    backgroundColor: "{colors.diff-del}"
    textColor: "{colors.ink}"
    typography: "{typography.code}"
  diff-line-hunk:
    backgroundColor: "{colors.diff-hunk}"
    textColor: "{colors.ink-muted}"
    typography: "{typography.code}"
  wiki-reader:
    textColor: "{colors.ink}"
    typography: "{typography.reading}"
    width: 36em
  wiki-heading:
    textColor: "{colors.ink}"
    typography: "{typography.reading-h2}"
  wikilink:
    textColor: "{colors.link}"
    typography: "{typography.reading}"
  wikilink-missing:
    textColor: "{colors.ink-muted}"
    typography: "{typography.reading}"
  wiki-quote:
    textColor: "{colors.ink-secondary}"
    typography: "{typography.reading}"
    padding: 0 0 0 16px
  empty-state:
    textColor: "{colors.ink-secondary}"
    typography: "{typography.body-md}"
  error-state:
    textColor: "{colors.refuted}"
    typography: "{typography.body-md}"
  page-footer:
    textColor: "{colors.ink-muted}"
    typography: "{typography.body-sm}"
    padding: 32px 0
---

## Overview

Shinka Research's console reads like a **lab notebook kept by a careful researcher**. The base
atmosphere is warm paper (`{colors.paper}` — #f5f0e6) with dark warm ink (`{colors.ink}` —
#1f1c17). Titles, entry headings and status stamps are set in a **CJK serif** — the notebook's hand
— while prose, labels and every number run in the **system sans**, so the page reads the same on
macOS, Windows, Linux and phones. The combination feels like a bound research record, not a
monitoring dashboard.

Every page is a record of research state, never a control surface. The viewer — the owner, or
someone the owner shared a link with, who may not know Shinka — should understand within one screen
which hypotheses hold, how many variants were tried, how high that pushed the bar for "supported",
how much budget is left, and whether the lockbox is still sealed.

The system has four pages, all on the same paper:
1. **工作区 Workspace** — the table of contents: one row per research direction.
2. **方向 Direction** — the core page: question, facts row, trial chart, hypothesis entries, event log.
3. **实验 Experiment** — one experiment of one hypothesis: settings and hashes, per-generation chart,
   the evolution ledger with diffs, the best program; links out to the upstream Shinka tree view.
4. **资料库 Library** — the shared wiki: index, page reader, uncompiled sources, library log.

**Key Characteristics:**
- Warm paper canvas (`{colors.paper}`) and warm ink (`{colors.ink}`); no white, no cool grey.
- The **ledger**: margin-numbered entries in a 150px margin column, a status stamp in coloured ink,
  a sentence of evidence, and a ✓/✗ check ledger with the deciding rule underlined.
- The **noise bar**: a dashed `{colors.noise-bar}` rule on every trial chart showing the best
  score N pure-noise trials would reach. It rises with N — the cost of trying more, made visible.
- The **lockbox seal**: a 26px ring with 封 (sealed) or 启 (opened); the only round shape.
- Four status inks are the only hues: supported green-ink, refuted oxblood, inconclusive ochre,
  running blue-black. Status is always also written as text and a glyph.
- Flat paper: no shadows, no cards, no gradients. Hierarchy comes from rules, whitespace and type.
- Read-only and static: no buttons that change research state, no animation.

## Colors

### Ink
- **Ink** (`{colors.ink}` — #1f1c17): Titles, figures, entry ids, frontier line. 15.0:1 on paper.
  Also `{colors.primary}`: the system's one dominant colour is ink, not an accent.
- **Ink Secondary** (`{colors.ink-secondary}` — #4a443a): Evidence sentences, lede, event notes,
  empty states. 8.5:1 on paper.
- **Ink Muted** (`{colors.ink-muted}` — #665d50): Margin notes, labels, breadcrumbs, chart axes,
  proposed/dropped stamps. 5.7:1 on paper — the lowest text contrast in the system.
- **On Ink** (`{colors.on-ink}` — #f5f0e6): Text on the ink tooltips.

### Paper
- **Paper** (`{colors.paper}` — #f5f0e6): Every page floor.
- **Paper Deep** (`{colors.paper-deep}` — #efe7d8): Alternating hypothesis bands in trial charts only.
- **Paper Raised** (`{colors.paper-raised}` — #faf6ee): Code blocks and diffs — one step lighter, like a
  pasted-in printout.
- **Rule** (`{colors.rule}` — #d3c8b5): Hairlines between entries, chart gridlines, meter track.
- **Rule Strong** (`{colors.rule-strong}` — #b9ad99): The rule under the facts row; trial dots.

### Status inks
- **Supported** (`{colors.supported}` — #2c5a37): supported and confirmed stamps, passing checks. 7.0:1.
- **Refuted** (`{colors.refuted}` — #8a2b1f): refuted and unconfirmed stamps, failing checks, the lockbox
  seal, exhausted budget. 7.6:1.
- **Inconclusive** (`{colors.inconclusive}` — #6b5a2a): inconclusive stamps. 5.9:1.
- **Running** (`{colors.running}` — #1f4a7a): running and challenged stamps. 8.0:1.

### Functional
- **Link / Focus** (`{colors.link}`, `{colors.focus}` — #1f4a7a): Links, toggles, the 2px focus ring.
  Shares the running ink: blue-black means "in motion / go somewhere".
- **Noise Bar / Decider Mark** (`{colors.noise-bar}`, `{colors.decider-mark}` — #8a2b1f): Oxblood marks
  every limit the research may not cross — the noise bar, the deciding failed rule, the seal.
- **Frontier** (`{colors.frontier}` — #1f1c17) and **Trial Dot** (`{colors.trial-dot}` — #b9ad99):
  kept trials in ink, discarded trials faint.
- **Diff Add / Del / Hunk** (`{colors.diff-add}` #e3ecd9, `{colors.diff-del}` #f3dcd5, `{colors.diff-hunk}`
  #e6ddcb): Line grounds in diffs, always paired with a leading `+` / `−` / `@@` so grey-scale prints read.

Colour never carries a result alone: every status has its Chinese word and a glyph, every check
has ✓ or ✗, every diff line has its sign.

## Typography

### Font Family
Two voices, no webfonts:
- **CJK serif** — `Noto Serif CJK SC, Source Han Serif SC, Songti SC, STSong, serif` — the notebook's
  hand: page titles, entry ids and titles, stamps, wiki reading.
- **System sans** — `system-ui, -apple-system, Segoe UI, PingFang SC, Hiragino Sans GB, Microsoft
  YaHei, Noto Sans CJK SC, sans-serif` — prose under entries, labels, tables, facts, chart text.
- **System mono** — `ui-monospace, SF Mono, Menlo, Consolas, DejaVu Sans Mono, Noto Sans Mono CJK SC,
  monospace` — code, diffs, hashes.

Windows ships no good CJK serif (only SimSun). Serif is therefore limited to short display text,
where a SimSun fallback is tolerable. The wiki reading stack (`{typography.reading}`) lists
Microsoft YaHei before any generic family, so on a Windows machine without Noto/Source Han Serif,
long reading switches to the sans instead of SimSun.

### Hierarchy

| Token | Size | Weight | Line Height | Tracking | Use |
|---|---|---|---|---|---|
| `{typography.display-lg}` | 30px | 600 | 1.35 | 0 | Direction question, experiment title — serif |
| `{typography.display-md}` | 24px | 600 | 1.4 | 0 | Workspace title, wiki page title — serif |
| `{typography.entry-title}` | 18px | 600 | 1.45 | 0 | Hypothesis title, direction row title — serif |
| `{typography.entry-id}` | 15px | 600 | 1.4 | 0 | `h003` / generation number in the margin — serif |
| `{typography.stamp}` | 12px | 600 | 1.4 | 0.1em | Status stamps, seal glyph — serif |
| `{typography.reading}` | 16px | 400 | 1.8 | 0 | Wiki body — serif |
| `{typography.reading-h2}` | 20px | 600 | 1.45 | 0 | Wiki section headings — serif |
| `{typography.body-md}` | 15px | 400 | 1.75 | 0 | Evidence sentences, lede, ledger text — sans |
| `{typography.body-sm}` | 13px | 400 | 1.6 | 0 | Margin notes, checks, events, toggles — sans |
| `{typography.label}` | 12px | 500 | 1.5 | 0.08em | Kickers, fact labels, group headings — sans |
| `{typography.data-md}` | 14px | 400 | 1.6 | 0 | Figures in rows and tables — sans, tabular |
| `{typography.data-lg}` | 20px | 600 | 1.3 | 0 | Facts-row values — sans, tabular |
| `{typography.chart}` | 11px | 400 | 1.2 | 0 | Axis ticks, band labels, chart tooltip — sans, tabular |
| `{typography.code}` | 13px | 400 | 1.55 | 0 | Code, diffs, hashes — mono |

### Principles
- Weights 400 and 600 only; no synthesized italics or bold for CJK.
- Never negative tracking on CJK. Positive tracking only on `label` and `stamp`.
- Numbers: ASCII digits, tabular figures, right-aligned in columns, minus as `−` (U+2212), two
  decimals for Sharpe/DSR/PBO, `$` with one decimal for money.
- Chinese prose uses full-width punctuation（“”、，。：；（））. One space between Han and Latin
  or numerals; none before `%` or full-width punctuation. `line-break: strict`.
- Emphasis is weight, never underline (underline means link or term).
- Desktop prose measure 30–40 Han characters (`{spacing.measure}`), wiki 36em
  (`{spacing.reading-measure}`); 18–25 on mobile.

## Layout

### Spacing System
- **Base unit:** 4px (with a 2px half-step for stamp padding).
- **Tokens:** `{spacing.xxs}` 2 · `{spacing.xs}` 4 · `{spacing.sm}` 8 · `{spacing.md}` 12 ·
  `{spacing.lg}` 20 · `{spacing.xl}` 32 · `{spacing.xxl}` 48.
- **Between entries:** `{spacing.md}` above and below the entry rule.
- **Between sections** (facts → chart → groups → events): `{spacing.xl}`.
- **Page top:** `{spacing.top-bar}` bar, then `{spacing.xxl}` to the kicker.

### Grid & Container
- **Page:** centered `{spacing.page}` (1040px) with a `{spacing.margin-col}` (150px) right-aligned
  margin column and a `{spacing.gutter}` (40px) gutter; the main column is 850px.
- **Margin column** holds what a researcher writes beside an entry: ids, trial counts, generations,
  timestamps, chart captions, wiki page metadata.
- **Main column:** prose capped at `{spacing.measure}`; facts row, charts, diffs and code may use the
  full 850px.

### Page structures
- **Direction:** top bar → kicker 研究方向 → question → lede → facts row → trial chart → groups 在研 /
  已有结论 / 已关闭 → event log → footer.
- **Experiment:** top bar → kicker `h003 · e01` → hypothesis title + stamp → facts row (settings,
  hashes, trials, cost) → per-generation chart → evolution ledger → best program (collapsed) → skeptic
  report (after verdict) → link to the upstream tree → footer.
- **Workspace:** top bar → title → direction rows (question, facts summary, counts per status) →
  library summary → footer.
- **Library:** top bar → title → uncompiled sources → index (by type: concept / paper / finding) →
  library log; a wiki page uses the reader layout with metadata in the margin.

### Grouping and density
Hypotheses are grouped so a long direction stays scannable:
- **在研** — proposed, selected, running, challenged — full entries.
- **已有结论** — supported, inconclusive, confirmed, unconfirmed — full entries.
- **已关闭** — refuted, dropped — one-line collapsed entries that expand in place.
Within a group, order is by hypothesis id.

### Whitespace Philosophy
Generous between sections, tight inside an entry. The margin column is the page's breathing space
and its index at once: scanning down the margin lists every id with its trial count.

## Elevation & Depth

| Level | Treatment | Use |
|---|---|---|
| Flat | No border, no shadow | Page, entries, facts row, charts |
| Hairline | 1px `{colors.rule}` | Between entries, event rows, gridlines |
| Strong rule | 1.5px `{colors.ink}` above, 1px `{colors.rule-strong}` below | Facts row |
| Raised paper | `{colors.paper-raised}` ground, no border | Code blocks, diffs |
| Ink overlay | `{colors.ink}` ground, `{colors.on-ink}` text | Term and chart tooltips — the only layer above the page |

No shadows, gradients, blur or glass anywhere. Depth is rules and grounds only.

## Shapes

### Border Radius Scale

| Token | Value | Use |
|---|---|---|
| `{rounded.none}` | 0px | Everything: stamps, code blocks, tooltips, meters |
| `{rounded.seal}` | 9999px | The lockbox seal ring only |

### Glyphs and marks
- Status glyphs: ✓ supported/confirmed · ✗ refuted/unconfirmed · ? inconclusive · ▸ running ·
  ◇ challenged · ○ proposed · ● selected · – dropped.
- Check glyphs: ✓ pass · ✗ fail · — not applicable.
- No icon set, no logo, no illustrations, no photography. The seal is drawn in CSS (ring + one
  character).

## Components

### Navigation

**`top-bar`** — 48px strip on `{colors.paper}` with no rule. Left: `{component.breadcrumb}`
（工作区 / btc-research / 方向）. Right: `{component.top-bar-link}` 工作区 · 资料库, with
`{component.top-bar-link-current}` for the current page. Collapses to breadcrumb only below 760px.

**`breadcrumb`** — `{typography.body-sm}` in `{colors.ink-muted}`; each crumb is a
`{component.text-link}` except the last.

**`page-footer`** — One line in `{colors.ink-muted}`: 只读 · 数据来自 `<workspace path>` · 读取于
`<time>`. Reminds shared viewers the page is a snapshot.

### Page head

**`kicker`** — `{typography.label}`: 研究方向 / `h003 · e01` / 资料库.

**`page-title`** — `{typography.display-lg}` serif, max two lines, full-width punctuation.

**`page-lede`** — One sentence in `{typography.body-md}` `{colors.ink-secondary}` explaining how to
read the page, with `{component.term}` for N, DSR, PBO and lockbox.

### Facts row

**`facts-row`** — Between `{component.facts-rule-top}` and `{component.facts-rule-bottom}`; items
laid out with `{spacing.xl}` gaps. Each item is `{component.fact-label}` over `{component.fact-value}`:
- 开发数据 — date range.
- 预算 — `$spent` / `$limit` with `{component.budget-meter}`; `{component.budget-meter-fill-exhausted}`
  and the word 已用完 when spent ≥ limit; `{component.fact-value-unset}` 未设置 without a budget.
- 试验数 N — N plus 噪声门槛 value (the current noise bar).
- Lockbox — `{component.lockbox-seal-sealed}` 封 + 未启封 · 自 `<start>`, or
  `{component.lockbox-seal-opened}` 启 + 已启封 `<date>`; 未切分 before the first scaffold.
- 资料库 — pages and 待编译 count (direction page only).

### Hypothesis entries

**`entry`** — Margin: `{component.entry-id}` and 96 次试验 in `{component.margin-note}`. Main:
`{component.entry-title}` followed by its `{component.status-stamp-*}`; then
`{component.entry-evidence}`; then the check ledger; then 查看实验 → link.
- **Verdict present:** “验证段 Sharpe 1.21，DSR 0.97（N = 412），PBO 0.22，子区间为正 3/3；买入持有 0.78。”
- **No verdict yet (running/challenged):** in-sample only — “进化中，第 23/40 代，样本内最优 Sharpe 0.93，
  118 次试验。验证段证据出结论后才显示。”
- **Proposed/selected:** the hypothesis claim's first sentence and 等待人选择 / 已选，待 scaffold.
- **Confirmed/unconfirmed:** the verdict sentence plus a second sentence with lockbox figures.

**`entry-collapsed`** — One line: id · title · stamp · shortest evidence (“验证段 Sharpe −0.12”).
The whole line is a `{component.toggle}` (`aria-expanded`); expanding shows the full entry in place.

**`status-stamp`** — 1.5px border in the status ink, `{typography.stamp}`, padding 0 × 6px, glyph +
word. Variants: `-proposed` 待选 · `-selected` 已选 · `-running` 运行中 · `-challenged` 质疑中 ·
`-supported` 支持 · `-refuted` 证伪 · `-inconclusive` 不确定 · `-confirmed` 已确认 ·
`-unconfirmed` 未确认 · `-dropped` 已放弃. Never filled; the stamp is ink on paper.

**Check ledger** — Five inline items in `{typography.body-sm}`: 验证段 Sharpe > 0 · 跑赢买入持有 ·
DSR ≥ 0.95 · 子区间为正 ≥ 2/3 · PBO ≤ 0.5 (thresholds from the direction's settings).
`{component.check-pass}` ✓ / `{component.check-fail}` ✗; the first failing rule in the decision order
is `{component.check-decider}` — underlined 1px in `{colors.decider-mark}` — and its tooltip says
这一条决定了结论. `{component.check-not-applicable}` — when PBO could not be computed (NaN).

### Trial chart

**`trial-chart`** — Inline SVG, full main-column width, 230px tall (direction) or 200px (experiment).
- X: trial number across the direction (direction page) or generation (experiment page).
- `{component.trial-chart-band}` alternates per hypothesis, labelled `h001`… under the axis.
- `{component.trial-chart-dot}` for each discarded trial; `{component.trial-chart-kept}` for each new
  best; `{component.trial-chart-frontier}` as a step line per hypothesis.
- `{component.trial-chart-noise-bar}` dashed 5/4, labelled 噪声门槛 `1.36` at the right end.
- `{component.trial-chart-gridline}` every 0.5 Sharpe; ticks in `{typography.chart}`.
- Margin caption: 虚线：N 个纯噪声试验的期望最高分。实线：每个假设的最优前沿。
- Hover or keyboard focus on a kept point shows `{component.trial-chart-tooltip}`: 第 212 次 · h002 ·
  第 14 代 · 样本内 Sharpe 1.48. Failed trials are not plotted; their count appears in the caption
  (失败 7 次).
- Empty: fewer than 2 trials → `{component.empty-state}` 还没有可画的试验.

### Event log

**`event-row`** — results.tsv rows, newest first, 10 by default with 显示全部 toggle. Margin: date and
time `10-16 22:40`; main: `h002` · event word (scaffold 建立实验 · run 进化 · probe 试探 · skeptic 质疑 ·
verdict 结论 · confirm 启封) · note.

### Workspace

**`direction-row`** — Margin: direction folder name. Main: question in `{typography.entry-title}`, one
`{typography.data-md}` line (N · 预算 · lockbox state) and a status tally written as stamps with counts
(支持 1 · 证伪 1 · 不确定 1 · 运行中 1 · 待选 1). The row is a link to the direction page.

### Experiment page

**`hash-chip`** — First 12 hex characters of the evaluator and data SHA-256 in `{typography.code}`,
full hash in the tooltip; labelled 评估器 / 数据. A tampered experiment (hash mismatch) shows
`{component.error-state}` 评估器已被改动，结果不可信.

**`ledger-entry`** — One per improvement on the best path, oldest first. Margin: 第 14 代 and the
in-sample score. Main: the patch name in `{typography.entry-title}`, its description in
`{typography.body-md}`, and 展开改动 `{component.toggle}` revealing the diff.

**`code-block`** — `{colors.paper-raised}` ground, `{typography.code}`, horizontal scroll inside the
block, line numbers in `{colors.ink-muted}`. The best program sits collapsed under 最优程序.

**`diff-line-add` / `diff-line-del` / `diff-line-hunk`** — Unified diff, one line per row, leading
`+` / `−` / `@@`, no syntax colouring.

Skeptic report (`skeptic.md`) renders as wiki-style reading text only after the hypothesis has a
verdict; before that the section reads 质疑报告在出结论后显示.

The upstream link is a plain `{component.text-link}`: 完整进化树、岛屿与嵌入（Shinka 界面）→.

### Library

**`wiki-reader`** — `{typography.reading}` at `{spacing.reading-measure}`; headings
`{component.wiki-heading}`; lists, tables (hairline rules), `{component.code-block}`,
`{component.wiki-quote}` (2px `{colors.rule-strong}` left rule). Margin: type (概念 / 论文 / 发现),
sources (raw/… links), updated date, backlinks.

**`wikilink`** / **`wikilink-missing`** — `[[page]]` becomes a link; a missing page is
`{colors.ink-muted}` with a dotted underline and the tooltip 这个页面还不存在.

Uncompiled sources list each raw path with 待编译; the index groups pages by type with one-line
summaries from `index.md`.

### Shared

**`text-link`** — `{colors.link}`, underline on hover (`{component.text-link-hover}` turns ink).

**`toggle`** — Text button in `{colors.link}` with ▸ / ▾ prefix; 28px tall on desktop, 44px touch
target on mobile; `{component.toggle-expanded}` once open.

**`term`** — Dotted underline; on hover or focus shows `{component.term-tooltip}` with one sentence:
- N：这个方向里评估过的所有变体（含 probe）。试得越多，门槛越高。
- DSR：考虑试验次数 N 后，真实 Sharpe 超过噪声门槛的概率。
- PBO：样本内最优的策略，在样本外排到中位数以下的比例。
- 噪声门槛：N 个纯噪声试验的期望最高分。
- 验证段：agent 看不到的样本外数据，用来出结论。
- Lockbox：研究期间谁都看不到的数据尾段；收尾时 confirm 一次性打开。

**`focus-ring`** — 2px solid `{colors.focus}` outline, 2px offset, on every link, toggle, term and
chart point.

**`empty-state`** — A notebook sentence naming what is missing and the CLI that creates it:
还没有假设——用 `shinka_research hypothesis add` 添加。 / 还没有切分数据——第一次 `scaffold` 时切出 lockbox。
/ 资料库是空的——把资料放进 `library/raw/`。

**`error-state`** — `{colors.refuted}` sentence naming the unreadable file and the reason:
无法读取 experiments/h002/e01/results/programs.sqlite：文件不存在。

## Do's and Don'ts

### Do
- State every verdict and its evidence as a sentence; numbers support the sentence.
- Show N and the noise bar wherever a best score is shown.
- Hide validation evidence until a verdict and lockbox figures until `confirm`.
- Keep status readable without colour: word + glyph, ✓/✗, diff signs.
- Put hypothesis ids, generations and timestamps in the margin column.
- Print well: grey-scale and A4 must still read.

### Don't
- Don't add any control that writes research state; never a confirm or lockbox button.
- Don't show a best score without N.
- Don't use white or cool grey grounds, cards, shadows, gradients, rounded corners (except the seal).
- Don't fill stamps; don't use emoji or icon fonts.
- Don't show a crashed trial as 0.0; it is 失败 and is counted, not plotted.
- Don't load webfonts or CDN scripts; the console must work offline.
- Don't use serif for numbers, tables or controls.

## Responsive Behavior

### Breakpoints

| Name | Width | Key Changes |
|---|---|---|
| Mobile | < 760px | Margin column folds into the entry header (id · trials before the title); facts row 2-up; top bar shows breadcrumb only; charts full width at 4:3; check ledger wraps one item per line; 16px side gutter |
| Tablet | 760–1100px | Margin column 110px, gutter 24px; facts row wraps to 3-up; charts keep aspect |
| Desktop | 1100–1440px | Full layout: 1040px page, 150px margin, 40px gutter |
| Wide | > 1440px | Same as desktop, centred; no extra columns |

### Touch Targets
- `{component.toggle}` and collapsed entries: ≥ 44px tall on mobile.
- Links in the top bar and breadcrumbs: ≥ 44 × 44px hit area on mobile via padding.
- Chart points: 12px invisible hit radius; on touch, tapping a band shows its hypothesis summary.

### Collapsing Strategy
- The margin becomes an inline prefix, never a hidden drawer.
- Code blocks and diffs scroll horizontally inside their block; code never wraps.
- Charts simplify on mobile: band labels drop to every other band, tick count halves.
- No horizontal page scroll at any width.

## Iteration Guide

1. Change one component at a time and reference its key (`{component.entry}`, `{component.trial-chart}`).
2. New states are separate `components:` entries (`-expanded`, `-exhausted`, `-opened`).
3. Use `{token.refs}`; never inline hex in pages.
4. Serif for the notebook's hand (titles, ids, stamps, wiki reading); sans for everything else. The
   split is unbreakable.
5. Paper + ink + four status inks is the whole palette. Don't add a fifth hue.
6. When something needs emphasis, give it a margin note or a sentence before giving it colour.
7. Every new number on screen must say what N it was chosen from.

## Known Gaps

- CJK serif availability differs by OS; Windows shows display text in SimSun and wiki reading in
  Microsoft YaHei. Accepted; verify with real copy during QA.
- Print stylesheet (A4, page breaks between entry groups) is specified in intent only.
- Very large directions (thousands of trials) may need chart point thinning; not yet specified.
- The upstream Shinka tree view keeps its own styling; the link out is the only bridge.
- No dark theme by design; a request for one would be a new direction, not a variant.

## Product Context

Shinka Research (fork 2233admin/ShinkaEvolve) turns a research direction into falsifiable hypotheses,
tests each by evolutionary code search against a frozen evaluator, challenges results with skeptic
statistics (DSR, PBO, sub-periods, baselines) and closes a direction with a one-time lockbox. Users are
the owner and people the owner shares results with; most of them are not Shinka experts. Read-only and
visibility rules: `docs/research/design.md` §7.3. Owner decisions:
`design/changes/research-ui/decisions/grill.md`.

## Source Decisions

Change: `design/changes/research-ui` — direction preview `direction-preview.json`, candidate
`notebook` selected by the owner on 2026-10-17 with the instruction to combine it with the existing
Shinka WebUI features; document structure and completeness modelled on the Claude DESIGN.md in
VoltAgent/awesome-design-md (`design-md/claude/DESIGN.md`), inspiration only.

Adopted:
- Claude DESIGN.md (awesome-design-md): document structure and completeness — full token set, every
  state as its own component entry, typography/elevation/radius tables, responsive breakpoints, touch
  targets, collapsing strategy, iteration guide, known gaps; warm-ground + serif-display editorial
  register as a precedent.
- Karpathy autoresearch progress plot — faint discarded trials, highlighted kept trials, step frontier —
  as the trial chart; extended with hypothesis bands and the noise bar.
- WorldQuant BRAIN's explicit pass/fail tests — as the check ledger.
- Optuna Dashboard's small fixed set of views reading files directly — four pages, no panel builder.
- Shinka WebUI's best-path and diff views — re-expressed as the evolution ledger; tree, islands and
  embeddings stay upstream, linked.
- Preview candidate B: N paired with its noise bar in the facts row. Candidate C: marking the deciding rule.

Rejected:
- Claude's coral accent, dark product surfaces, rounded cards, licensed typefaces and marketing
  components (hero, pricing, CTA bands) — Shinka Research is not a marketing site and has its own ink palette.
- W&B auto-generated panel workspaces, MLflow evidence-free stage labels, AI-Scientist-v2's per-run
  static tree as the only view.
- Card grids, dark terminal styling, serif body text on all platforms.
- Any control that writes research state; a confirm/lockbox button.
