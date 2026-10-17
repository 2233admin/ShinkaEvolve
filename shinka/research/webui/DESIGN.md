---
version: alpha
name: Shinka Research 运维规格台
description: Read-only research console for Shinka Research, set as an ink-black ops-spec sheet built from Astro UXDS and Carbon components. Tokens below are generated from design-system/tokens.json (dark theme shown; the light theme Technical Paper is in the same file); usage rules, guidelines and live previews live in design-system/.
colors:
  primary: "#efeee8"
  ink: "#efeee8"
  muted: "#a9a8a0"
  base: "#0b0b0a"
  raised: "#141413"
  hover: "#1c1c1a"
  selected: "#232320"
  line: "#2c2c29"
  line-strong: "#6e6e68"
  accent: "#f45d22"
  accent-hover: "#ff7a3e"
  accent-text: "#f45d22"
  on-accent: "#0b0b0a"
  ok: "#9cd1ac"
  warn: "#e5c06f"
  err: "#ffa095"
typography:
  title-page:
    fontFamily: "'Inter Variable', 'PingFang SC', 'Noto Sans SC', sans-serif"
    fontSize: 28px
    fontWeight: 600
    lineHeight: 1.3
  title-card:
    fontFamily: "'Inter Variable', 'PingFang SC', 'Noto Sans SC', sans-serif"
    fontSize: 18px
    fontWeight: 600
    lineHeight: 1.4
  heading-reading:
    fontFamily: "'Inter Variable', 'PingFang SC', 'Noto Sans SC', sans-serif"
    fontSize: 16px
    fontWeight: 600
    lineHeight: 1.4
  text-reading:
    fontFamily: "'Inter Variable', 'PingFang SC', 'Noto Sans SC', sans-serif"
    fontSize: 13px
    fontWeight: 400
    lineHeight: 1.75
  text-field:
    fontFamily: "'Inter Variable', 'PingFang SC', 'Noto Sans SC', sans-serif"
    fontSize: 13px
    fontWeight: 400
    lineHeight: 1.5
  text-note:
    fontFamily: "'Inter Variable', 'PingFang SC', 'Noto Sans SC', sans-serif"
    fontSize: 12px
    fontWeight: 400
    lineHeight: 1.7
  label-section:
    fontFamily: "'IBM Plex Mono', 'PingFang SC', monospace"
    fontSize: 11px
    fontWeight: 500
    lineHeight: 1
    letterSpacing: 0.08em
  label-spec:
    fontFamily: "'IBM Plex Mono', 'PingFang SC', monospace"
    fontSize: 11px
    fontWeight: 500
    lineHeight: 1.4
    letterSpacing: 0.08em
  data-cell:
    fontFamily: "'IBM Plex Mono', 'PingFang SC', monospace"
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.2857
  data-sm:
    fontFamily: "'IBM Plex Mono', 'PingFang SC', monospace"
    fontSize: 12px
    fontWeight: 400
    lineHeight: 1.5
rounded:
  radius-none: 0px
spacing:
  space-1: 4px
  space-2: 8px
  space-3: 12px
  space-4: 16px
  space-5: 20px
  space-6: 24px
  space-8: 32px
  space-16: 64px
  page-max: 1480px
  side-width: 420px
  spec-key-width: 110px
  library-nav-width: 240px
  stripe-height: 8px
---

# Shinka Research 运维规格台

## Overview

An ops-spec sheet for a research workspace and each direction: spec-sheet key–value blocks, rules instead of filled panels,
orange index stripes under the top bar, and orange used only for limits and signals (the sealed
lockbox, the noise line, the selected-row mark, actions). Default theme is neutral ink-black; the
switchable light theme is Technical Paper. The signal layer (top bar, status symbols, timeline, event
log) is Astro UXDS 8.0; the content layer (tables, tabs, tooltips, notices, tree, charts) is Carbon
1.118. The usage rules are `design-system/README.md`; read it before building any page.

## Colors

Ink on ink-black in the dark theme, ink on paper in the light theme, three status colours (`ok`,
`warn`, `err`) and one orange (`accent`, with `accent-text` for text and `accent-hover`). The values
above are the dark theme; the light theme and every token's use and its contrast on each ground it sits
on are in `design-system/tokens.json` (`color`, `usage`, `contrast`).

## Typography

Inter for text and IBM Plex Mono for ids, figures, keys and section labels; CJK falls back to PingFang
SC and Noto Sans SC. Styles and their uses: `design-system/tokens.json` `type`.

## Layout

A page of at most `page-max` with a grid of table, trial chart, timeline and events beside a sticky
`side-width` spec card; stacked below 900px. All corners are square. Rules and breakpoints:
`design-system/README.md` › Visual foundations.

The workspace uses a Carbon table of real direction summaries, not a card grid. A direction link opens
a new document; its breadcrumb returns to the workspace. Both views reuse the same library reader.
Directory identity, empty/error states and read-only navigation follow
[`docs/research/workspace.md`](../../../docs/research/workspace.md).

## Components

Ten components, each with guidelines and a live preview under `design-system/components/`: TopBar,
HypothesisTable, SpecCard, CheckLedger, TrialChart, Timeline, EventStream, LibraryReader, Term and
Notice. The shared stylesheet, which also maps the tokens onto Astro and Carbon, is
`design-system/components/bundle.css`.

## Do's and Don'ts

See `design-system/README.md` › Rules that protect the research: read-only, a noise line beside every
best score, validation evidence only after a verdict, lockbox figures only after confirmation. Orange is
for limits and signals only; take ready-made Astro and Carbon pieces instead of drawing components.

## Product Context

Shinka Research (fork 2233admin/ShinkaEvolve) turns a research direction into hypotheses tested by
evolution against a frozen evaluator, with skeptic statistics and a one-time lockbox. Readers are the
owner and people the owner shares results with. See `docs/research/design.md` §7.3 and
`design/changes/research-ui/decisions/grill.md`.

## Source Decisions

Change: `design/changes/research-ui`. The owner first picked the notebook direction (2026-10-17), then
chose direction B, ops-spec (运维规格台), and asked for ready-made component libraries rather than
hand-built components.

Adopted: Neo-Industrial structure for the sheet (spec-sheet key–value blocks, rules, orange index
stripes, orange only for limits and signals); Astro UXDS 8.0 for the signal layer and Carbon 1.118 for
content, split as in `design/changes/research-ui/component-selection/README.md` (owner-approved
2026-10-18); a neutral ink-black dark theme instead of the Operations Dark green-grey, per the owner
2026-10-18; the light theme Technical Paper; Inter and IBM Plex Mono; Claude's Design System format for
the package (usage-noted tokens, usage-rule README, per-component guidelines with live previews,
cover); autoresearch's frontier plot; BRAIN's explicit pass/fail tests.

Rejected: the paper-notebook direction (serif, paper colours, hand-drawn components) and its package;
the other previewed directions (night ledger, tape archive); a shared component library with
KatanaConsole, which stays a separate project with its own libraries; Astro tabs, accordion and
notification, and `rux-classification-marking`; W&B auto-generated panels, MLflow evidence-free stage
labels, card grids, any control that writes research state.
