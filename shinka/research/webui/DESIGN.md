---
version: alpha
name: Shinka Research 研究记录本
description: Read-only research console for Shinka Research, styled as a lab notebook — workspace, direction, experiment and library pages.
colors:
  primary: "#1f1c17"
  paper: "#f5f0e6"
  paper-deep: "#efe7d8"
  rule: "#d3c8b5"
  rule-strong: "#b9ad99"
  ink: "#1f1c17"
  ink-secondary: "#4a443a"
  ink-muted: "#665d50"
  supported: "#2c5a37"
  refuted: "#8a2b1f"
  inconclusive: "#6b5a2a"
  running: "#1f4a7a"
  link: "#1f4a7a"
  diff-add: "#e3ecd9"
  diff-del: "#f3dcd5"
typography:
  display:
    fontFamily: "Noto Serif CJK SC, Source Han Serif SC, Songti SC, STSong, serif"
    fontSize: 30px
    fontWeight: 600
    lineHeight: 1.35
  entry-title:
    fontFamily: "Noto Serif CJK SC, Source Han Serif SC, Songti SC, STSong, serif"
    fontSize: 18px
    fontWeight: 600
    lineHeight: 1.45
  stamp:
    fontFamily: "Noto Serif CJK SC, Source Han Serif SC, Songti SC, STSong, serif"
    fontSize: 12px
    fontWeight: 600
    lineHeight: 1.4
    letterSpacing: 0.1em
  body:
    fontFamily: "system-ui, -apple-system, Segoe UI, PingFang SC, Hiragino Sans GB, Microsoft YaHei, Noto Sans CJK SC, sans-serif"
    fontSize: 15px
    fontWeight: 400
    lineHeight: 1.75
  reading:
    fontFamily: "Noto Serif CJK SC, Source Han Serif SC, Songti SC, STSong, serif"
    fontSize: 16px
    fontWeight: 400
    lineHeight: 1.8
  label:
    fontFamily: "system-ui, -apple-system, Segoe UI, PingFang SC, Hiragino Sans GB, Microsoft YaHei, Noto Sans CJK SC, sans-serif"
    fontSize: 12px
    fontWeight: 500
    lineHeight: 1.5
    letterSpacing: 0.08em
  data:
    fontFamily: "system-ui, -apple-system, Segoe UI, PingFang SC, Hiragino Sans GB, Microsoft YaHei, Noto Sans CJK SC, sans-serif"
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.6
    fontFeature: '"tnum" 1'
  code:
    fontFamily: "ui-monospace, SF Mono, Menlo, Consolas, DejaVu Sans Mono, Noto Sans Mono CJK SC, monospace"
    fontSize: 13px
    fontWeight: 400
    lineHeight: 1.55
rounded:
  none: 0px
  seal: 999px
spacing:
  xs: 4px
  sm: 8px
  md: 12px
  lg: 20px
  xl: 32px
  margin-col: 150px
  gutter: 40px
  page: 1040px
  measure: 40em
components:
  page:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
  status-stamp:
    textColor: "{colors.ink-muted}"
    typography: "{typography.stamp}"
    rounded: "{rounded.none}"
    padding: 0 6px
  status-stamp-supported:
    textColor: "{colors.supported}"
  status-stamp-refuted:
    textColor: "{colors.refuted}"
  status-stamp-inconclusive:
    textColor: "{colors.inconclusive}"
  status-stamp-running:
    textColor: "{colors.running}"
  lockbox-seal:
    textColor: "{colors.refuted}"
    rounded: "{rounded.seal}"
    size: 26px
  facts-row:
    textColor: "{colors.ink}"
    typography: "{typography.data}"
    padding: 10px 0
  entry-margin:
    textColor: "{colors.ink-muted}"
    typography: "{typography.label}"
    width: 150px
  diff-line-add:
    backgroundColor: "{colors.diff-add}"
    typography: "{typography.code}"
  diff-line-del:
    backgroundColor: "{colors.diff-del}"
    typography: "{typography.code}"
  chart-band:
    backgroundColor: "{colors.paper-deep}"
  entry-prose:
    textColor: "{colors.ink-secondary}"
    typography: "{typography.body}"
  entry-title:
    textColor: "{colors.ink}"
    typography: "{typography.entry-title}"
  page-title:
    textColor: "{colors.ink}"
    typography: "{typography.display}"
  text-link:
    textColor: "{colors.link}"
  entry-rule:
    backgroundColor: "{colors.rule}"
    height: 1px
  facts-rule:
    backgroundColor: "{colors.rule-strong}"
    height: 1px
  wiki-reader:
    textColor: "{colors.ink}"
    typography: "{typography.reading}"
    width: 40em
---

# Shinka Research 研究记录本

## Overview

The research console reads like a lab notebook kept by a careful researcher. Every page is a
record: a direction states its question, each hypothesis is a numbered entry whose verdict is
written out as a sentence with its evidence, and each experiment keeps a ledger of the
improvements evolution actually made. The viewer — the owner, or someone the owner shared the
page with — should understand within one screen where the research stands: which hypotheses hold,
how many variants were tried, how high that pushed the bar for "supported", how much budget is
left, and whether the lockbox is still sealed.

The product signature is **the ledger**: margin-numbered entries, a stamp for the status, and a
line of ✓/✗ checks that names the exact rule behind every verdict. The second signature is **the
noise bar**: a dashed ink rule on every trial chart showing the best score N pure-noise trials
would be expected to reach. It rises as N grows; that is the cost of trying more, made visible.

The console is read-only. Nothing on it changes research state, and the lockbox is never opened
from it. Interface copy is Chinese; statistical terms keep their English abbreviations (Sharpe,
DSR, PBO, N) with a hover explanation.

## Colors

Warm paper and ink. `paper` is the ground everywhere; `paper-deep` only marks alternating
hypothesis bands in charts. `ink` is for titles and figures, `ink-secondary` for prose,
`ink-muted` for margin notes and labels (5.7:1 on paper). Rules are drawn in `rule` (hairlines
between entries) and `rule-strong`/`ink` (the 1.5px rule above the facts row).

Status inks are the only hues: `supported` green-ink, `refuted` oxblood, `inconclusive` ochre,
`running` blue-black. Proposed, selected and dropped use `ink-muted`. Every status color passes
4.5:1 on `paper`, and every status is also spelled out in its stamp text and carries a glyph
(✓ ✗ ? ▸ ○ –); color never carries a result alone. `refuted` also draws the lockbox seal and the
noise bar, because both mark a limit the research may not cross.

Diffs use pale `diff-add`/`diff-del` grounds with ink text, plus a leading `+`/`−`, so a
grey-scale print still reads. The console has no dark theme: the paper metaphor is the product,
and shared screenshots and prints must look the same as the screen.

## Typography

Two voices. **Serif** (Noto Serif CJK SC / Source Han Serif SC / Songti SC) is the notebook's
hand: page titles, entry titles, stamps, and long-form wiki reading. **System sans** carries
everything that must stay legible on any OS — prose under entries, labels, tables, facts and
numbers — because Windows has no reliable CJK serif and falls back badly.

- Display 30px/1.35; entry titles 18px/1.45; wiki reading 16px/1.8 with a 36–40 Han-character
  measure; body 15px/1.75; labels 12px with 0.08em tracking; data 14px tabular.
- Numbers always use tabular figures and ASCII digits; a minus sign is `−` (U+2212). Columns of
  numbers are right-aligned.
- Chinese prose uses full-width punctuation (“”、，。：；（）). Latin terms and numerals sit
  next to Han characters with a single space; no space before `%` or full-width punctuation.
- Code and diffs use the system monospace stack at 13px.
- No webfonts are downloaded. Weights used: 400 and 600 only; no synthesized italics.

## Layout

A centered 1040px page with a 150px **margin column** and a 40px gutter. The margin holds what a
researcher writes beside an entry: the hypothesis id, trial count, timestamps, chart captions.
Content sits in the main column at a 40em measure for prose; tables and charts may use the full
main column.

Page order puts the state first: kicker → title → one-sentence lede → facts row → trial chart
→ entries. The chart is above the fold on a 1440×900 screen.

Entries are grouped so a long direction stays scannable: **在研**（proposed, selected, running,
challenged）, **已有结论**（supported, inconclusive, confirmed, unconfirmed）, and **已关闭**
（refuted, dropped）. Closed entries collapse to one line — id, title, stamp, one-sentence
evidence — and expand on click. With 20+ hypotheses the page is still one screen per group.

Below 760px the margin column folds into the entry header (id and trial count inline before the
title), the facts row wraps to two columns, and charts keep their aspect ratio at full width.
There is no horizontal page scroll; wide tables and diffs scroll inside their own box.

Spacing uses a 4px base: 4, 8, 12, 20, 32.

## Elevation & Depth

Flat paper. No shadows, cards, glass or gradients. Hierarchy comes from rules (hairline between
entries, a heavier rule above the facts row), whitespace and type. Expanded content (a diff, the
best program, a collapsed entry) opens inline on the same paper, indented to the main column.

## Shapes

Square corners everywhere. The only round shape is the lockbox seal: a 26px ring with the
character 封 (sealed) or 启 (opened). Stamps are rectangles drawn with a 1.5px border in the status
ink.

## Components

- **Facts row** — label over value, separated by whitespace: development data range, budget
  (spent / limit, plus a hairline meter), trials N with the current noise bar, lockbox seal with
  its start date, library pages and uncompiled sources. Missing values read “未设置”.
- **Hypothesis entry** — margin: id and trial count. Main: serif title + status stamp; one
  sentence of evidence; the check ledger (✓/✗ per rule, rule failing first is underlined). Before a
  verdict the sentence reports in-sample figures only and says validation evidence appears after
  the verdict.
- **Status stamp** — text label in status ink with glyph; states: proposed 待选, selected 已选,
  running 运行中, challenged 质疑中, supported 支持, refuted 证伪, inconclusive 不确定, confirmed
  已确认, unconfirmed 未确认, dropped 已放弃.
- **Check ledger** — five inline items: 验证段 Sharpe > 0, 跑赢买入持有, DSR ≥ threshold, 子区间为正
  ≥ threshold, PBO ≤ threshold; thresholds come from the direction's settings.
- **Trial chart** — every trial a small `rule-strong` dot; per-hypothesis frontier as an ink step
  line with kept points as solid dots; alternating `paper-deep` bands labelled by hypothesis; the
  dashed `refuted` noise bar labelled 噪声门槛 with its value. Hover on a point shows trial number,
  hypothesis, generation and in-sample score. On the experiment page the same chart plots one
  experiment by generation.
- **Evolution ledger** (experiment page) — one entry per improvement on the best path: margin
  shows generation and score; main shows the patch name and its description; a 展开改动 toggle
  reveals the diff. Above it: the frozen evaluator and data hashes (short form), settings, trials,
  and cost. Below it: the best program, collapsed. Heavy views — the full tree, islands,
  embeddings — are a plain text link 完整进化树（Shinka 界面）.
- **Event log** — results.tsv rows as margin timestamp + hypothesis + event + note.
- **Wiki reader** — serif reading measure, `[[page]]` rendered as links (missing pages shown in
  `ink-muted` with a dotted underline), page metadata (type, sources, updated) in the margin.
- **Term** — dotted underline; hover or focus shows a one-sentence Chinese explanation.
- **Empty and error states** — written as a notebook line in `ink-secondary`, saying what is
  missing and the CLI command that creates it (e.g. 还没有假设——用 `shinka_research hypothesis
  add` 添加). Unreadable files are named with the reason.
- **Focus** — 2px `running` outline offset 2px on every interactive element.

## Do's and Don'ts

### Do

- State the verdict and its evidence in words; numbers support the sentence.
- Show the trial count and the noise bar wherever a best score is shown.
- Keep validation evidence hidden until a verdict and lockbox figures hidden until `confirm`.
- Print well: the page must read in grey-scale and on paper.

### Don't

- Don't add buttons that change research state; the console is read-only.
- Don't show a best score without N.
- Don't use color alone, emoji, icons-for-decoration, gradients, shadows or card grids.
- Don't use magic values (a crashed trial is shown as 失败, never 0.0).
- Don't load webfonts or CDN scripts; the console must work offline.

## Product Context

Shinka Research (fork 2233admin/ShinkaEvolve) turns a research direction into hypotheses tested
by evolutionary code search against a frozen evaluator, with skeptic statistics (DSR, PBO,
sub-periods, baselines) and a one-time lockbox. Users are the owner and people the owner shares
results with; most of them are not Shinka experts. Design rationale and the read-only decision:
`docs/research/design.md` §7.3. Owner decisions: `design/changes/research-ui/decisions/grill.md`.

## Source Decisions

Change: `design/changes/research-ui` (direction preview `direction-preview.json`, selected
`notebook` by the owner on 2026-10-17, with the instruction to combine it with the existing Shinka
WebUI features).

Adopted:
- Karpathy autoresearch progress plot — faint discarded trials, highlighted kept trials, step
  frontier — as the trial chart; extended with per-hypothesis bands and the noise bar.
- WorldQuant BRAIN's explicit pass/fail tests — as the check ledger.
- Optuna Dashboard's small, fixed set of views reading files directly — four pages, no panel
  builder.
- Shinka WebUI's best-path and diff views — re-expressed as the evolution ledger; the full tree,
  islands and embeddings stay in the upstream interface, linked.
- Direction preview candidate B's pairing of N with its noise bar in the facts row.
- Direction preview candidate C's idea of marking the deciding rule — as the underlined first
  failing check.

Rejected:
- W&B auto-generated panel workspaces (clutter), MLflow stage labels without evidence,
  AI-Scientist-v2's per-run static tree as the only view.
- Card grids, dark terminal styling and serif body text on all platforms (Windows fallback).
- Any control that writes research state, and a confirm/lockbox button.
