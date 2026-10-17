---
version: alpha
name: Shinka Research 研究记录本
description: Read-only research console for Shinka Research, kept like a lab notebook. Tokens below are generated from design-system/tokens.json; usage rules, guidelines and live previews live in design-system/.
colors:
  paper: "#f5f0e6"
  paper-deep: "#efe7d8"
  paper-raised: "#faf6ee"
  rule: "#d3c8b5"
  rule-strong: "#b9ad99"
  ink: "#1f1c17"
  primary: "#1f1c17"
  ink-secondary: "#4a443a"
  ink-muted: "#665d50"
  on-ink: "#f5f0e6"
  supported: "#2c5a37"
  refuted: "#8a2b1f"
  inconclusive: "#6b5a2a"
  running: "#1f4a7a"
  link: "#1f4a7a"
  focus: "#1f4a7a"
  noise-bar: "#8a2b1f"
  decider: "#8a2b1f"
  frontier: "#1f1c17"
  trial-dot: "#8a7d68"
  diff-add: "#e3ecd9"
  diff-del: "#f3dcd5"
  diff-hunk: "#e6ddcb"
typography:
  display-lg:
    fontFamily: "\"Noto Serif CJK SC\", \"Source Han Serif SC\", \"Songti SC\", STSong, serif"
    fontSize: 30px
    fontWeight: 600
    lineHeight: 1.35
  display-md:
    fontFamily: "\"Noto Serif CJK SC\", \"Source Han Serif SC\", \"Songti SC\", STSong, serif"
    fontSize: 24px
    fontWeight: 600
    lineHeight: 1.4
  entry-title:
    fontFamily: "\"Noto Serif CJK SC\", \"Source Han Serif SC\", \"Songti SC\", STSong, serif"
    fontSize: 18px
    fontWeight: 600
    lineHeight: 1.45
  entry-id:
    fontFamily: "\"Noto Serif CJK SC\", \"Source Han Serif SC\", \"Songti SC\", STSong, serif"
    fontSize: 15px
    fontWeight: 600
    lineHeight: 1.4
  stamp:
    fontFamily: "\"Noto Serif CJK SC\", \"Source Han Serif SC\", \"Songti SC\", STSong, serif"
    fontSize: 12px
    fontWeight: 600
    lineHeight: 1.4
    letterSpacing: 0.1em
  reading:
    fontFamily: "\"Noto Serif CJK SC\", \"Source Han Serif SC\", \"Songti SC\", \"Microsoft YaHei\", sans-serif"
    fontSize: 16px
    fontWeight: 400
    lineHeight: 1.8
  reading-h2:
    fontFamily: "\"Noto Serif CJK SC\", \"Source Han Serif SC\", \"Songti SC\", STSong, serif"
    fontSize: 20px
    fontWeight: 600
    lineHeight: 1.45
  body-md:
    fontFamily: "system-ui, -apple-system, \"Segoe UI\", \"PingFang SC\", \"Hiragino Sans GB\", \"Microsoft YaHei\", \"Noto Sans CJK SC\", sans-serif"
    fontSize: 15px
    fontWeight: 400
    lineHeight: 1.75
  body-sm:
    fontFamily: "system-ui, -apple-system, \"Segoe UI\", \"PingFang SC\", \"Hiragino Sans GB\", \"Microsoft YaHei\", \"Noto Sans CJK SC\", sans-serif"
    fontSize: 13px
    fontWeight: 400
    lineHeight: 1.6
  label:
    fontFamily: "system-ui, -apple-system, \"Segoe UI\", \"PingFang SC\", \"Hiragino Sans GB\", \"Microsoft YaHei\", \"Noto Sans CJK SC\", sans-serif"
    fontSize: 12px
    fontWeight: 500
    lineHeight: 1.5
    letterSpacing: 0.08em
  data-md:
    fontFamily: "system-ui, -apple-system, \"Segoe UI\", \"PingFang SC\", \"Hiragino Sans GB\", \"Microsoft YaHei\", \"Noto Sans CJK SC\", sans-serif"
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.6
  data-lg:
    fontFamily: "system-ui, -apple-system, \"Segoe UI\", \"PingFang SC\", \"Hiragino Sans GB\", \"Microsoft YaHei\", \"Noto Sans CJK SC\", sans-serif"
    fontSize: 20px
    fontWeight: 600
    lineHeight: 1.3
  chart:
    fontFamily: "system-ui, -apple-system, \"Segoe UI\", \"PingFang SC\", \"Hiragino Sans GB\", \"Microsoft YaHei\", \"Noto Sans CJK SC\", sans-serif"
    fontSize: 11px
    fontWeight: 400
    lineHeight: 1.2
  code:
    fontFamily: "ui-monospace, \"SF Mono\", Menlo, Consolas, \"DejaVu Sans Mono\", \"Noto Sans Mono CJK SC\", monospace"
    fontSize: 13px
    fontWeight: 400
    lineHeight: 1.55
rounded:
  radius-none: 0px
  radius-seal: 9999px
spacing:
  space-1: 4px
  space-2: 8px
  space-3: 12px
  space-5: 20px
  space-8: 32px
  space-12: 48px
  margin-col: 150px
  gutter: 40px
  page: 1040px
  measure: 40em
  reading-measure: 36em
---

# Shinka Research 研究记录本

## Overview

A warm-paper lab notebook: margin-numbered hypothesis entries, status stamps in coloured ink, a
✓/✗ check ledger behind every verdict, and a dashed noise bar on every trial chart. The usage rules
are `design-system/README.md`; read it before building any page.

## Colors

Paper and ink plus four status inks. Every token's use and its contrast on each ground it sits on
are in `design-system/tokens.json` (`usage`, `contrast`).

## Typography

Serif for the notebook's hand (titles, ids, stamps, wiki reading), system sans for prose, labels and
numbers, mono for code. Styles and their uses: `design-system/tokens.json` `type`.

## Layout

A 1040px page with a 150px margin column and a 40px gutter; state first (facts row, trial chart),
then grouped entries. Rules and breakpoints: `design-system/README.md` › Visual foundations.

## Components

Nineteen components, each with guidelines and a live preview under `design-system/components/`;
the shared stylesheet is `design-system/components/bundle.css`.

## Do's and Don'ts

See `design-system/README.md` › Rules that protect the research: read-only, N beside every best
score, validation evidence only after a verdict, lockbox figures only after `confirm`, nothing
loaded from the network.

## Product Context

Shinka Research (fork 2233admin/ShinkaEvolve) turns a research direction into hypotheses tested by
evolution against a frozen evaluator, with skeptic statistics and a one-time lockbox. Readers are the
owner and people the owner shares results with. See `docs/research/design.md` §7.3 and
`design/changes/research-ui/decisions/grill.md`.

## Source Decisions

Change: `design/changes/research-ui`; direction `notebook` selected by the owner (2026-10-17).

Adopted: Claude's Design System format for the package (usage-noted tokens, usage-rule README,
per-component guidelines with live previews, cover); autoresearch's frontier plot; BRAIN's explicit
pass/fail tests; Optuna Dashboard's small fixed set of views; Shinka WebUI's best path and diffs as
the evolution ledger.

Rejected: W&B auto-generated panels, MLflow evidence-free stage labels, card grids, dark terminal
styling, serif body text on every platform, any control that writes research state.
