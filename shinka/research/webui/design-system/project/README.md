Shinka Research is a research notebook, not a dashboard. Every page records where a research direction stands — which hypotheses hold, how many variants were tried, how high that pushed the bar for “supported”, how much budget is left, and whether the lockbox is still sealed — so that the owner, or someone the owner shared a link with, understands it within one screen. The console only reads files; nothing on it changes research state.

## Content fundamentals

- Write Chinese interface copy. Keep statistical terms in their English abbreviations — Sharpe, DSR, PBO, N — and explain each with `Term` the first time it appears on a page.
- State a verdict as a sentence with its evidence, numbers inside the sentence: “验证段 Sharpe 1.21，DSR 0.97（N = 412），PBO 0.22，子区间为正 3/3；买入持有 0.78。” Never a bare number without what it measures.
- Name what a figure was chosen from. A best score always travels with its N and the noise bar (门槛 1.36).
- Say what is hidden and when it appears, instead of hiding silently: “验证段证据出结论后才显示。”
- Empty and error states name what is missing and the command that creates it: “还没有假设——用 `shinka_research hypothesis add` 添加。”
- Tone: plain, exact, unhurried. No exclamation marks, no emoji, no praise (“表现优异”), no hedging adverbs. Prefer the researcher's words: 假设、证伪、不确定、试验、门槛、启封.
- Punctuation: full-width Chinese punctuation（，。：；（）“”）in Chinese sentences; one space between Han characters and Latin words or numerals, none before % or full-width punctuation. Numbers are ASCII digits; minus is − (U+2212); Sharpe, DSR and PBO take two decimals; money takes `$` and one decimal.
- Status words are fixed: 待选 · 已选 · 运行中 · 质疑中 · 支持 · 证伪 · 不确定 · 已确认 · 未确认 · 已放弃. Event words are fixed: 建立实验 · 进化 · 试探 · 质疑 · 结论 · 启封.

## Visual foundations

**Colour.** Paper and ink. Set every page on `paper`; text in `ink` (titles, figures), `ink-secondary` (sentences) or `ink-muted` (margin notes, labels — the lowest contrast allowed, 5.7:1). Draw hairlines between entries in `rule` and the rule under the facts row in `rule-strong`, with a 1.5px `ink` rule above it. Use `paper-deep` only for alternating hypothesis bands in charts and `paper-raised` only under code and diffs. The four status inks — `supported`, `refuted`, `inconclusive`, `running` — are the only hues; spend them on stamps, checks, the seal and the noise bar, never on grounds or headings. `refuted` marks every limit the research may not cross: refuted verdicts, the deciding failed rule (`decider`), the noise bar (`noise-bar`), the sealed lockbox, an exhausted budget. Links, toggles and the focus ring use `link` / `focus` (the running blue-black). There is one theme, Paper: prints and shared screenshots must look like the screen.

**Colour never carries meaning alone.** Every status is a word plus a glyph, every check is ✓ or ✗, every diff line starts with + or −. A grey-scale print must still read.

**Type.** Two voices. The serif (`--font-serif`) is the notebook's hand: `display-lg` for the direction question and experiment title, `display-md` for workspace, library and wiki titles, `entry-title` for hypotheses and ledger entries, `entry-id` for ids in the margin, `stamp` for stamps. The sans (`--font-sans`) carries everything that must read on any OS: `body-md` for sentences, `body-sm` for margin notes, checks and events, `label` for kickers and fact labels, `data-md` and `data-lg` for figures (tabular numerals, right-aligned in columns), `chart` inside charts. Wiki pages read in `reading` (serif, falling back to Microsoft YaHei rather than SimSun on Windows) with `reading-h2` headings. Code, diffs and hashes use `code`. Weights are 400 and 600 only; never negative tracking on Chinese; emphasis is weight, never underline.

**Layout.** A centred `page` (1040px) with a right-aligned `margin-col` (150px) and a `gutter` (40px). The margin column holds what a researcher writes beside an entry — ids, trial counts, generations, timestamps, chart captions, wiki metadata — so scanning down it indexes the page. Prose stops at `measure` (40em); wiki text at `reading-measure` (36em). Space with `space-3` around entry rules, `space-8` between sections, `space-12` from the top bar to the kicker. Put the state first: kicker → title → lede → `FactsRow` → `TrialChart` → entries → `EventLog`. Group hypotheses under 在研, 已有结论 and 已关闭; closed entries collapse to one line.

**Depth, borders, radii.** Flat paper: no shadows, no cards, no gradients, no blur. Hierarchy is rules, whitespace and type. Every corner is square (`radius-none`) except the lockbox seal (`radius-seal`). The only layer above the page is the `ink` tooltip with `on-ink` text.

**States.** Hover underlines links and turns them `ink`. Keyboard focus is a solid 2px `focus` outline offset 2px, 8.0:1 on `paper`. Expanded toggles turn `ink` with ▾. Nothing animates (MOTION.md: static).

**Responsive.** Below 1100px the margin narrows to 110px; below 760px it folds into a line above each entry, the facts row goes two-up, checks stack, toggles grow to 44px and the page takes a 16px gutter. Code and diffs scroll inside their block; the page never scrolls sideways.

## Iconography

There is no icon set and no logo. Glyphs carry state: ✓ ✗ ? ▸ ◇ ○ ● – in stamps (set in the sans via `<span class="g">`), ✓ ✗ — in check ledgers, ▸ ▾ on toggles, → after links that leave the page, + − @@ in diffs. The lockbox seal is a CSS ring around one serif character, 封 or 启. The product name is set in plain type.

## Rules that protect the research

- Never add a control that writes research state, and never a confirm or lockbox button.
- Show validation evidence only after a hypothesis has a verdict; show lockbox figures only after `confirm`.
- Count failed trials and say so; never plot them as 0.
- Load nothing from the network: no webfonts, no CDN scripts.
