---
version: alpha
name: Shinka Research 暗场仪器
description: Read-only research workbench for Shinka Research, drawn as a dark-field instrument with the app's own components. The values below are the dark theme as defined in app/src/design/tokens.css, the only source of values; the light theme is in the same file. Usage rules and the live preview are in design-system/.
colors:
  primary: "#ededef"
  bg: "#0a0a0b"
  rail: "#0e0e10"
  raised: "#16161a"
  hover: "#1c1c21"
  selected: "#232329"
  text: "#ededef"
  muted: "#a1a1aa"
  faint: "#8a8a93"
  signal: "#f2622e"
  signal-graphic: "#f2622e"
  stroke: "#6d6d76"
  grid: "#34343b"
  notice-error-bg: "#262626"
  ramp-0: "#606066"
  ramp-1: "#f1f1f3"
typography:
  headline:
    fontFamily: "'Inter Variable', 'PingFang SC', 'Microsoft YaHei', 'Noto Sans SC', system-ui, sans-serif"
    fontSize: 28px
    fontWeight: 600
    lineHeight: 40px
  title:
    fontFamily: "'Inter Variable', 'PingFang SC', 'Microsoft YaHei', 'Noto Sans SC', system-ui, sans-serif"
    fontSize: 26px
    fontWeight: 590
    lineHeight: 34px
  section:
    fontFamily: "'Inter Variable', 'PingFang SC', 'Microsoft YaHei', 'Noto Sans SC', system-ui, sans-serif"
    fontSize: 15px
    fontWeight: 590
    lineHeight: 22px
  body:
    fontFamily: "'Inter Variable', 'PingFang SC', 'Microsoft YaHei', 'Noto Sans SC', system-ui, sans-serif"
    fontSize: 13px
    fontWeight: 400
    lineHeight: 20px
  small:
    fontFamily: "'Inter Variable', 'PingFang SC', 'Microsoft YaHei', 'Noto Sans SC', system-ui, sans-serif"
    fontSize: 12px
    fontWeight: 400
    lineHeight: 18px
  data:
    fontFamily: "'IBM Plex Mono', 'PingFang SC', 'Microsoft YaHei', monospace"
    fontSize: 12px
    fontWeight: 400
    lineHeight: 18px
rounded:
  radius-panel: 0px
  radius-control: 6px
  radius-float: 8px
spacing:
  space-1: 4px
  space-2: 8px
  space-3: 12px
  space-4: 16px
  space-5: 20px
  space-6: 24px
  space-8: 32px
  space-10: 40px
---

# Shinka Research 暗场仪器

## Overview

A dark-field instrument for reading research evidence. Near-black grounds, hairline rules and one signal orange. Every page answers a question with data:
- the directions overview opens with a data-sentence title;
- a direction report runs claim → trials against the noise line → check ledger → verdict;
- a run's analysis has four views (Lineage, Progress, Solution space, Contest) and one resident program inspector.

Dark is the default and a light theme ships beside it. The whole interface switches between English (the default) and Chinese. There is no dedicated phone design or acceptance; the existing narrow layout remains, and the accepted widths are 1920, 1440 and 1024.

The page is read-only. Every component is the project's own DOM, SVG or D3. react-aria-components supplies only interaction and accessibility behaviour; no component library styles are loaded. Usage rules are in `design-system/README.md`; the live preview is `design-system/preview.html` (`pnpm design` in `app/`).

## Colors

All values live in `app/src/design/tokens.css`; the front matter repeats the dark theme.
- **Grounds:** `bg`, `rail`, `raised`, `hover` and `selected`.
- **Text:** `text`, `muted` and `faint`.
- **Rules:** `line` and `line-strong` (white or black at 8% and 16%). They separate regions and never carry information alone.
- **Graphics:** `stroke` for axes, input and checkbox borders and the selected segment of a switch; `grid` for auxiliary grid lines. The score ramp `ramp-0` → `ramp-1` shades lineage marks.

**Contrast requirements**, worked out with the WCAG 2.x relative-luminance formula on every ground:
- body and small text reach at least 4.5:1;
- axes, borders, selection marks and data marks reach at least 3:1.

| Token | Dark | Light |
|---|---|---|
| `text` | 13.36–16.93 | 14.36–17.72 |
| `faint` | from 4.57 | from 4.61 |
| `signal` | 4.88–6.18 | text `#b83e0b` 4.56–5.63, graphics `#e3550c` 3.05–3.77 |

The preview recomputes these figures from the cascade.

**Signal orange** marks evidence that needs attention, and nothing else: the part of a best record above the noise line, the best program's ancestry, an overspent budget, a failed evaluation. There is no second accent, no decorative gradient and no glass.

**Restored-page aliases.** The names `ink`, `base`, `accent`, `accent-text`, `ok`, `warn` and `err` exist only for the restored analysis page, which keeps serving Solution space and Contest (owner decision A); the native views never use them.

## Typography

- Inter Variable for the interface; IBM Plex Mono for numbers, IDs and code. Both fonts are bundled locally.
- Chinese text falls back to PingFang SC, Microsoft YaHei and Noto Sans SC.
- Roles are the `--type-*` tokens: headline 28/40 (600) for the data-sentence title, title 26/34, section 15/22, body 13/20 and small 12/18. The title and section roles use weight 590.
- Numbers use tabular figures and are right-aligned.
- Scores and Sharpe ratios have two decimals and DSR has three; money is `$0.00`.
- Time is `YYYY-MM-DD HH:MM` marked UTC.
- A missing value is `—`; a recorded zero shows as zero.
- A total that includes runs of unknown cost starts with `≥`, because it is a lower bound.
- The minus sign is U+2212.

## Layout

The shell is a left rail with three parts: the directions overview, all runs, and the library with the list of directions. Above the content sits a top bar holding:
- the breadcrumb and view tabs;
- the "last updated" indicator and the auto-refresh pause;
- the theme and language switches;
- the Ctrl K command palette.

Routing is by query parameters (`?view=directions|direction|runs|compare|run|library`). Every selection is written to the URL, so direct links, refresh and Back restore the same state.

| Width | Layout |
|---|---|
| ≥1920 | Rail expanded; content up to 1600px; inspector 440px |
| 1440 | Same; content follows the width |
| 1024 | Rail narrows to a labelled strip; inspector 380px; the home side column moves below the content. Where the run page's top bar cannot hold the four view tabs, its tools move to a second row and the run name does not wrap |

On spacing and shape:
- spacing follows a 4px baseline (4, 8, 12, 16, 20, 24, 32, 40);
- controls have 6px corners and floating layers 8px;
- tables, panels, cards and code or log blocks stay square, separated by hairlines rather than shadows.

## Elevation & Depth

Flat. Regions are separated by `line` hairlines and whitespace, never by shadows. A selected row rises to `selected` with a 2px `text` bar on its left edge; a pressed toggle gets `raised` with a 1px `stroke` inset ring. Floating layers sit above the page on `raised` with a `line-strong` border and an 8px radius:
- chart read-outs;
- the lineage panel;
- dialogs;
- the command palette.

Dialogs and the palette dim the page behind them to 50% black. The palette alone also carries a soft drop shadow (`0 16px 48px`, 40% black), so it reads as the topmost layer. Definition popovers are inverted instead: `bg` text on a `text` ground with 2px corners, the metrics of the Carbon tooltip the app used before.

## Shapes

Status is a shape plus a word, never colour alone:
- **Hypothesis states** use circles. The words are fixed: 待跑, 已选, 运行中, 待判定, 成立, 否定, 未定, 已确认, 未确认, 放弃. In English they are queued, selected, running, challenged, supported, refuted, inconclusive, confirmed, unconfirmed, dropped.
- **Direction states** use squares:
  - in progress: a square with a filled inner square;
  - concluded: a filled square with a tick;
  - new: a dashed square;
  - over budget: a square with an arrow, in signal orange.
- **Events:**
  - ＋ set up;
  - ■ run finished;
  - ✕ run failed (signal orange);
  - ◇ skeptic check;
  - ◆ verdict;
  - △ probe;
  - an open lock for the lockbox confirmation.
- **Lineage marks** show the patch type: init ◆, full ●, diff ■, cross ✚, fix ▲.
- **Runs** on one chart are told apart by shape plus line style: ● solid, □ dashed, ▲ dotted.

## Components

All components live in the app, and the preview renders them on demo data:
- **Shell** (`app/src/shell/`): rail, top bar, view tabs, palette and modal.
- **Directions overview** (`overview/`): data-sentence title and a sortable direction table with a shared evidence scale.
- **Report** (`report/`): `HypothesisTable`, `TrialChart`, `Ledger`, `Verdict`/`LockboxResults`/`Experiments`, `Timeline` and `Events`.
- **Runs** (`runs/`) and **Compare** (`compare/`).
- **Run analysis** (`run/`): `Lineage`, `Progress`, `Scrubber`, `Inspector` and `LineagePanel`.
- **Shared** (`Notice`, `Definition`/`Term`, `Markdown`), with styles in `design/base.css`.
- **Restored analysis page** (`AnalysisWorkspace`, `legacyFrame.ts`): hosted for Solution space and Contest.

Charts follow one grammar:
- the noise line is drawn by the chart itself;
- charts that are compared share a scale and label its ticks;
- trial and generation axes have integer ticks only;
- failed evaluations sit in a separate band below the score axis, are counted, and never enter best or frontier;
- anything shown on hover can also be read in the inspector by click or focus.

## Do's and Don'ts

- Do keep the page read-only. Controls only navigate, switch theme or language, refresh, or pause the automatic refresh.
- Do show validation figures only after a verdict, and lockbox figures only after confirmation. Before that the cell is `—` with the note 出结论后显示.
- Do put the direction's noise line beside every research best score. Generic analysis shows `combined_score` for diagnosis only and never invents a null model.
- Do mark state with shape plus word, and use the signal orange only for the four signals above.
- Don't introduce a colour outside `tokens.css` (the black scrim and the palette's shadow are the only literals), add a decorative shadow or gradient, animate an entrance, or load a component library's styles.
- Don't add a control that writes research state, confirms a hypothesis or opens the lockbox.

## Product Context

Shinka Research (fork 2233admin/ShinkaEvolve) turns a research direction into hypotheses. Evolution tests them against a frozen evaluator, with skeptic statistics and a one-time lockbox. The readers are the owner and the people the owner shares results with. The research contract is `docs/research/design.md` §7.3; the page contract is `docs/research/workspace.md`.

## Source Decisions

Change: `openspec/changes/award-redesign`.

On 2026-10-19 the owner chose direction A, 暗场仪器 (dark-field instrument), on `direction-previews/index.html`. A absorbed three things from B:
- island-laned lineage;
- progress small multiples;
- the data-sentence home title.

The owner also set four product rules: dark default with a light theme, a full Chinese and English switch, and separate pages for directions and all runs. The high-fidelity mock-ups were approved on 2026-10-20 (`hifi/index.html`), together with the rule that no existing capability is removed. On 2026-10-22 the owner made three more decisions:
- react-aria-components replaced Carbon and Astro UXDS as behaviour-only dependencies (component route A);
- dedicated phone design and acceptance were dropped, keeping the existing narrow layout;
- Solution space and Contest stay on the restored analysis page (decision A).

All of these are recorded in `owner-decisions.md`.

Adopted: direction A's dark grounds, hairline rules and single signal orange; B's island-laned lineage, progress small multiples and data-sentence title; Inter Variable and IBM Plex Mono; react-aria-components for behaviour only; shape-plus-word status; the noise line drawn by every research chart.

Rejected: the previous ops-spec sheet (运维规格台) and its Astro UXDS and Carbon components; card grids; a second accent colour; animated chart entrances; any control that writes research state.
