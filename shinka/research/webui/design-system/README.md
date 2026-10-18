Shinka Research's read-only console is an ink-black ops-spec sheet: rules and key–value blocks instead of filled panels, orange reserved for limits and signals, and ready-made Astro UXDS and Carbon components instead of hand-drawn ones. `tokens.json` holds every value, `components/bundle.css` maps them onto the two libraries and styles the shared pieces, and the app and the previews load the same files.

## Content fundamentals

- **Language.** Interface copy is Chinese. Commands, ids, file names and `SHINKA RESEARCH` stay in Latin. Put a space between Chinese and a Latin word or number only where the source copy does (`有 2 份资料待整理`, `Sharpe`).
- **Voice.** Plain statements of fact, no first person, no exclamation marks, no apology, no blame. Say what is missing and the command that fixes it: 还没有试验——用 `shinka_research run` 跑实验。 Empty states are one muted sentence, never an illustration.
- **Notice titles** end with a full-width colon and the subtitle follows: 读取失败：, 还没有假设：, 有 2 份资料待整理：, 还没出结论：.
- **Terms.** Explain every statistical term once, in place, with `Term` (a Carbon definition tooltip) and the definitions in `terms.ts`; the sentence must still read without the tooltip. The names are fixed: 样本内, 验证段, 封存段, 试验数, 噪声线, Sharpe, DSR, PBO, 子区间, 买入持有.
- **Status words** are fixed and always paired with their symbol: 待跑, 已选, 运行中, 待判定, 成立, 否定, 未定, 已确认, 未确认, 放弃.
- **Numbers.** Set ids and figures in `data-cell` or `data-sm` (mono, tabular). Sharpe and scores two decimals, DSR three, PBO two, fractions as whole percents, money `$0.00`, time `YYYY-MM-DD HH:MM` (UTC). A missing figure is `—`, never `0`, `N/A` or blank.
- **Casing.** Section labels are set in `label-section` and `label-spec` (mono, `letter-spacing: .08em`); write them in plain case, the style does the rest. Commands, ids and file names are always in `<code class="mono">`.
- **Withheld results.** Validation figures appear only after a verdict and lockbox figures only after confirmation; before that the cell says `—` with the tooltip 出结论后显示.

## Visual foundations

- **Colour.** Two themes from one token set: `dark` (the default, neutral ink-black: `base` as the ground, `raised` one step up) and `light` (Technical Paper). Text is `ink`; secondary text, labels and keys are `muted`; rules are `line`, and anything that must be seen as a mark is `line-strong`. The three status colours are `ok`, `warn`, `err` and are marks, always paired with a shape or a word. Use only token names; never write a hex value in a component or a page.
- **Orange is a limit or a signal.** `accent` appears on: the index stripes under the top bar, the sealed-lockbox icon, the selected-row mark, the noise line in the trial chart, action borders, focus rings. `accent-text` is the orange for text (links, tertiary actions); `accent-hover` is the hover. Nothing decorative is orange.
- **Structure.** Rules instead of filled panels: sections are separated by `line` hairlines and whitespace; a region is lifted with `raised` only for the selected table row and the spec card's border. The spec sheet (`dl.spec`) is the signature shape: mono key on the left (`--spec-key-width`), value on the right, one hairline between rows.
- **Type.** Inter for text, IBM Plex Mono for ids, figures, keys and section labels (`--font-sans`, `--font-mono`; CJK falls back to PingFang SC / Noto Sans SC). Page title `title-page` (28px, 22px under 900px), spec card title `title-card`, reading text `text-reading`, spec values `text-field`, captions `text-note`, section labels `label-section`, spec keys `label-spec`, table figures `data-cell`, ledger values and code `data-sm`, library headings `heading-reading`. Weights are 400, 500 (mono labels) and 600 (titles).
- **Spacing and layout.** 4px base: `space-1`…`space-16`. The page is at most `page-max` wide with `space-8` side padding (`space-4` below 900px); the research view is a grid of the table, trial chart, timeline and events beside a sticky `side-width` spec card; below 900px everything stacks and the spec card follows the table. The library is a `library-nav-width` tree beside a reading column of at most `78ch`. Tables scroll inside their wrapper; the page never scrolls sideways.
- **Borders and radii.** Hairline 1px `line`; `line-strong` for header underlines and quote rules. Every corner is square (`radius-none`); Astro's and Carbon's radii are mapped to it.
- **Depth.** None. No shadows, gradients (except the stripes), blur or overlays; the only layer above the page is Carbon's popover for a definition.
- **States.** Hover: Carbon's own step to `hover`. Selected: `selected` for a tree node or time region, `raised` plus a 2px `accent` inset bar for a table row. Focus: a 2px `accent` outline (3:1 on `base`, `raised` and `hover`). Disabled: Carbon's default, set in `muted`. Withheld: `—` with a tooltip.
- **Motion.** None of our own; only the libraries' micro-transitions. See MOTION.md.
- **The two libraries.** Astro UXDS 8.0 owns the signal layer: `rux-global-status-bar`, `rux-monitoring-icon`, `rux-status`, `rux-timeline`, `rux-log`. Carbon 1.118 owns the content: `DataTable`, `Tabs`, `DefinitionTooltip`, `InlineNotification`, `Toggle`, `TreeView`, `StructuredList` and `Tile` for blocks, `Tag`, Carbon Charts for plots. Astro's tabs, accordion and notification are not used (their accent use is too loud); neither is `rux-classification-marking`. Do not hand-draw a component a library already provides; map its variables in `bundle.css` instead.
- **Theme switch.** The app sets `data-theme` on `<html>` and the Carbon `Theme` (`g100` for dark, `white` for light); `body.light-theme` is set for Astro. Both come from the same `tokens.json`.

## Iconography

- No custom icon set and no logo: the product name is `SHINKA` in the bold weight and `RESEARCH` in regular, set by the Astro status bar.
- Icons come from the libraries: Astro monitoring icons (`assessment`, `attach-money`, `lock`, `lock-open`) in the top bar, Astro `rux-status` symbols for state (circle, square, triangle, ring and dot shapes), Carbon's own sort, tree and notification icons. Do not add another icon font or SVG set.
- Status is shape plus word, never colour alone; the sealed lockbox is the only orange icon.
- `↗` follows a link that leaves the page; `—` marks a withheld or missing value.

## Rules that protect the research

- Read-only: no control writes research state, and never a confirm or lockbox button. Controls only change the theme, refresh the displayed files or navigate between the workspace, a direction and the shared library.
- Validation evidence only after a verdict; lockbox figures only after confirmation.
- Every research evidence best score has its direction-wide noise benchmark beside it. The unified generic analysis shows original `combined_score` for diagnosis, explicitly refers research interpretation back to the direction view, and never invents a Sharpe null model or maps trial-order noise onto generation/cost axes. Failed evaluations stay in counts/resources but never become a valid zero-score frontier.
- The evaluator, data split and out-of-sample metrics are never reachable from this page.

## Using the package

- The app reads `tokens.json` through the Vite plugin in `app/vite.config.ts` (`virtual:tokens.css`) and imports `components/bundle.css`; nothing is copied. Change a value in `tokens.json`, regenerate the `DESIGN.md` token groups (`design-system design-md-tokens`) and run the checks.
- Previews load the pinned library builds (Astro Web Components 8.0.0, `@carbon/styles` 1.117.0) from a CDN inside a CSS layer, so `bundle.css` always wins over library defaults; offline they stay readable through the `:not(:defined)` fallbacks at the end of `bundle.css`.
