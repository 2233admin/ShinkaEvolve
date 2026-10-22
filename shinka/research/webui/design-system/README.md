Shinka Research's read-only workbench is a dark-field instrument: near-black grounds, hairline rules, one signal orange for evidence that needs attention, and the project's own components with react-aria-components for behaviour only.

This package holds the usage rules (this file) and the live preview (`preview.html`). Values live in one place, `app/src/design/tokens.css`. `DESIGN.md` and `MOTION.md` beside this folder summarise the system for other agents.

## The preview

`pnpm design` in `app/` serves `preview.html` with `app/vite.design.config.ts`. The page is never built into the product. It renders the app's own components with the app's stylesheets, in the same order as `app/src/main.tsx`, on a trimmed fixture of the deterministic demo workspace (`app/src/preview/fixture.json`). Regenerate the fixture with `node src/preview/fixture.mjs <ui-url>` against `shinka_research ui` on that workspace.

The page shows, from top to bottom:
1. **Tokens.** Both themes side by side, read back from the cascade:
   - every colour token with its value;
   - text-on-ground contrast for `text`, `muted`, `faint` and `signal` on `bg`, `raised` and `hover`;
   - the five type roles;
   - the spacing scale, both radii and the focus ring.
2. **Component states.**
   - The view tabs with each tab current: Lineage, Progress, Solution space and Contest.
   - An info and an error notice, plus an empty-state sentence.
   - Two terms with their definitions.
3. **Report**, on a real direction snapshot:
   - the hypothesis table with sorting and selection;
   - the claim;
   - the trial chart with the noise line and failure band;
   - the check ledger;
   - the verdict, lockbox confirmation and experiments;
   - the timeline and the events.
4. **Lineage and Progress**, on one run's first ten generations:
   - the island lanes and tracks;
   - the scrubber and replay;
   - the small multiples;
   - the per-model table and the disclosures.

The theme and language switches at the top act on the whole page, so each chart family can be read in dark or light and in Chinese or English. Because the preview imports the components themselves, `tsc -b` in `pnpm build` fails when a component's props change and the preview no longer matches.

## Content fundamentals

- **Language.**
  - Every interface string is in both Chinese and English (`app/src/i18n/copy.ts`); research content stays as written.
  - Commands, IDs, file names and Sharpe, DSR and PBO stay in Latin in both languages.
  - Chinese copy puts a space between Chinese and a Latin word or a number where the copy table does (`有 2 份资料待整理`).
- **Voice.**
  - Plain statements of fact: no first person, no exclamation marks, no apology, no blame.
  - Say what is missing and the command that fixes it: 还没有试验——用 `shinka_research run` 跑实验。
  - An empty state is one muted sentence, never an illustration.
- **Notice titles** end with a colon (full-width in Chinese) and the subtitle follows: 读取失败：, 还没有假设：.
- **Terms.**
  - Explain each statistical term once, in place, with `Term` (`app/src/Term.tsx`). Its definitions are the `term.*` keys of the copy table.
  - The sentence must still read without the definition.
  - The names are fixed: 样本内, 验证段, 封存段, 试验数, 噪声线, Sharpe, DSR, PBO, 子区间, 买入持有.
- **Status words** are fixed and always paired with their shape:
  - Chinese: 待跑, 已选, 运行中, 待判定, 成立, 否定, 未定, 已确认, 未确认, 放弃.
  - English: queued, selected, running, challenged, supported, refuted, inconclusive, confirmed, unconfirmed, dropped.
- **Numbers.**
  - IDs and figures are set in IBM Plex Mono with tabular figures, right-aligned.
  - Sharpe and scores have two decimals, DSR three, PBO two; money is `$0.00`; time is `YYYY-MM-DD HH:MM` marked UTC.
  - A missing figure is `—`, never `0`, `N/A` or blank; a recorded zero is shown as zero.
  - A total that includes runs of unknown cost starts with `≥`; the minus sign is U+2212.
- **Withheld results.** Validation figures appear only after a verdict, and lockbox figures only after confirmation. Before that the cell says `—` with the note 出结论后显示.

## Visual foundations

- **Colour.**
  - Two themes come from one token set: `[data-theme]` on `<html>` (`app/src/design/theme.ts`) switches every token, and dark is the default.
  - Grounds are `bg`, `rail` (rail, side columns, inspector), `raised` (floating layers) and `hover`/`selected`. Text is `text`, `muted` or `faint`. Rules are `line` and `line-strong`, which never carry information alone.
  - Axes, input and checkbox borders and the selected segment of a switch use `stroke`; auxiliary grid lines use `grid`. Lineage marks are shaded along the score ramp `ramp-0` → `ramp-1`.
  - Use token names only; the black scrim behind dialogs and the palette's shadow are the only literals.
- **Signal orange** (`signal` for text, `signal-graphic` for marks) is evidence that needs attention, and only that:
  - the part of a best record above the noise line;
  - the best program's ancestry and its badge;
  - an overspent budget;
  - a failed evaluation.

  Nothing decorative is orange, and there is no second accent.
- **Contrast.**
  - Body and small text reach at least 4.5:1, and axes, borders, selection marks and data marks at least 3:1, on every ground they sit on.
  - The preview recomputes the text figures. The design's table is `openspec/changes/award-redesign/design.md` › 颜色.
- **Type.**
  - Inter Variable for the interface, IBM Plex Mono for figures, IDs and code, both bundled.
  - Roles are `--type-headline` 28/40 (the home data sentence), `--type-title` 26/34, `--type-section` 15/22, `--type-body` 13/20 and `--type-small` 12/18.
- **Spacing and shape.**
  - Spacing follows a 4px baseline (`--space-1` … `--space-10`).
  - Controls have 6px corners and floating layers 8px.
  - Tables, panels, cards and code or log blocks are square and separated by hairlines.
- **Depth.**
  - Flat; no shadows except the palette's.
  - Floating layers (read-outs, the lineage panel, dialogs, the palette) are `raised` with a `line-strong` border.
- **States.**
  - Selected: `selected` ground plus a 2px `text` bar. Navigation, view tabs and table rows never rely on the ground alone.
  - Pressed toggle: `raised` with a 1px `stroke` inset.
  - Hover: `hover`.
  - Focus: the 2px `text` outline `--focus-ring`, at least 13.36:1 on the selected ground.
  - Disabled: `faint` text, no hover.
- **Charts.**
  - Native charts are SVG and D3 coloured from tokens.
  - The noise line is part of the chart; charts that are compared share a scale and label its ticks; trial and generation axes have integer ticks.
  - Failed evaluations sit in a band below the score axis and never enter best or frontier.
  - Any hover read-out can also be read in the inspector.
  - Solution space and Contest are the restored page's Plotly and Chart.js charts, which keep their own colours (owner decision A).
- **Motion.** Static; see `MOTION.md`.
- **Layout.** See `DESIGN.md` › Layout for the shell and the 1920/1440/1024 widths. Narrower widths keep their existing layout but are not designed for.

## Components

All components are in `app/src/`. The preview renders the report, run and shared ones directly.

| Group | Files | Notes |
|---|---|---|
| Shell | `shell/` (`Rail`, `TopBar`, `Palette`, `Modal`, `route.ts`) | Rail with section counts and the direction list; top bar with breadcrumb, view tabs, "last updated", the refresh pause, and the theme and language switches; Ctrl K jumps to any page, direction, run, library page or program of the open run |
| Notice | `Notice.tsx`, styles in `design/base.css` | A kind (`info` or `error`), a title ending in a colon, then a subtitle. Keeps the metrics of Carbon's low-contrast notice: 3px left edge, 1px frame at 40%, at least 48px tall. Error uses `notice-error-bg` with a `signal` edge. No close button, no action, never for success |
| Term | `Term.tsx`, `Definition.tsx`, styles in `design/base.css` | A dotted `stroke` underline; react-aria opens the definition on hover and focus, closes it on Escape, and opens it on tap. The popover is inverted (`bg` on `text`) |
| Overview | `overview/` | Data-sentence title, sortable direction table with a shared evidence scale, totals row, recent activity |
| Report | `report/` | `HypothesisTable`, `TrialChart`, `Ledger`, `Side` (verdict, lockbox, experiments), `Timeline`, `Events` |
| Runs and Compare | `runs/`, `compare/` | Grouped run table with totals; step charts by generation and spend, leaderboard and per-run metrics |
| Run analysis | `run/` | `Lineage`, `Progress` with `ProgressSections`, `Scrubber`, `Inspector`, `LineagePanel`, `PromptTree`, `Offspring` |
| Markdown | `Markdown.tsx`, `.md` in `design/base.css` | Library pages and hypothesis bodies, sanitized; rules and type only; links keep the text colour with a `stroke` underline |
| Restored page host | `AnalysisWorkspace.tsx`, `legacyFrame.ts`, `analysis/` | Hosts `/viz_tree.html` for Solution space, Contest and the run bar's "original workbench". `analysis/workbench.css` maps the restored page onto the tokens through the aliases in `tokens.css` |

## Rules that protect the research

- Read-only. No control writes research state, and there is never a confirm or lockbox button. Controls only navigate, switch theme or language, refresh, or pause the automatic refresh.
- Validation evidence appears only after a verdict, and lockbox figures only after confirmation. The server filters them out (`ui.snapshot`), so the browser never receives them.
- Every research best score has its direction's noise line beside it. The generic run analysis shows the original `combined_score` for diagnosis only; it never invents a Sharpe null model and never maps trial-order noise onto generation or cost axes.
- Failed evaluations stay in counts and costs but never become a valid zero-score frontier.
- The evaluator, the data split and the out-of-sample metrics are never reachable from the page.
