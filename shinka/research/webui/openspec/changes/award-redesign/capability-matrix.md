# Shinka Research — capability matrix for the analysis regroup (for owner approval)

Status: read-only inventory, 2026-10-19, branch `research/unified-analysis`. Nothing was executed and no code was changed. Every `VT:`, `IX:`, `CP:` and other line cite refers to commit `351c660`; `viz_tree.html` and `compare.html` changed afterwards (task 3.0 cost rule), so read cited lines with `git show 351c660:<path>`.

Use in `award-redesign`: this table is the acceptance spine for the owner's 2026-10-20 rule that no existing feature is removed (`owner-decisions.md` › 高保真评审). Each 6.x task checks its group's rows in the new UI; task 7.1 deletes an old implementation only after all of its rows pass, and keeps it when any row cannot be reproduced. The owner's answers to §9 are in `owner-decisions.md`.

Scope: 13 left tabs plus 10 right tabs of the analysis workbench, the run list, run compare, every React research page, and the read API.
Purpose: every question the UI answers today gets a destination in the new linked surfaces, so no capability is lost when tabs are merged.

## 0. Reading guide

### Path abbreviations

| Abbrev | Path |
|---|---|
| VT | `shinka/research/webui/analysis/viz_tree.html` |
| IX | `shinka/research/webui/analysis/index.html` (run list) |
| CP | `shinka/research/webui/analysis/compare.html` |
| AN | `shinka/research/analysis.py` |
| UI | `shinka/research/ui.py` |
| VIS | `shinka/webui/visualization.py` (upstream handler, reused through `AnalysisHandler`) |
| DB | `shinka/database/dbase.py` |
| VA | `shinka/research/webui/app/vite.analysis.config.ts` |
| RT | `shinka/research/webui/app/src/analysis/runtime.js` |
| AX | `shinka/research/webui/app/src/analysis/accessibility.js` |
| AC | `shinka/research/webui/app/src/analysis/actions.js` |
| WB | `shinka/research/webui/app/src/analysis/workbench.css` |
| SRC/ | `shinka/research/webui/app/src/` |

### Destinations

| Code | Destination |
|---|---|
| **RPT** | Report (direction/hypothesis research report) |
| **LIN** | Lineage (tree, path, islands, prompt evolution) |
| **PRG** | Progress (metrics, cost, throughput) |
| **SOL** | Solution space (embeddings, clusters incl. 3D, MAPs, diversity) |
| **CON** | Contest (battle, Pareto, multi-run compare) |
| **INS** | Persistent Program inspector |
| **RUN** | Runs list/compare |
| **LIB** | Library |

Some items fit none of the eight destinations. These rows say `no fitting destination — OD-n`, which points to an owner decision in §6.

### Row kinds

| Kind | Meaning |
|---|---|
| Q | A question the view answers. The wording comes from on-screen labels, titles and axes; `[INFERENCE]` marks a question I derived myself. |
| C | A control or interaction. |
| S | URL or stored state. |
| E | An empty, loading, error or failed-evaluation state. |
| X | An export. |
| I | Internationalisation (i18n). |
| SEC | Rendering of untrusted text. |
| D | A defect visible in the code. |

For D and SEC rows the destination column says `carry-over: <surface>`. These rows are constraints the rebuild must honour, not affordances to place.

### Global facts that apply to every VT/IX/CP view

| Fact | Cite |
|---|---|
| At build time every CDN `<script>`/`<link>` is stripped, and every `innerHTML` / `outerHTML` / `insertAdjacentHTML` is rewritten to `ShinkaHTML()`, a DOMPurify wrapper. Only `onclick/onchange/oninput` handlers whose function is declared in the document survive. | VA:9-37, VA:45-54 |
| Local libraries replace the CDN: Chart.js, d3, markdown-it (`html:false`, exposed as the `marked` shim), highlight.js with **python only**, diff, JSZip, Plotly (full bundle, so 3D works) and DOMPurify. html2pdf and jsPDF are dropped. **No VT code references html2pdf or jsPDF beyond the script tags, so nothing is lost by dropping them.** | RT:1-22; VT:2671-2680; grep of VT finds only those tags |
| DOMPurify drops `on*` attributes unless the action passes the finite grammar. It forbids `style`, `iframe`, `object`, `embed` and `form`, `url()` inside `style`, and remote `src`/`href` on anything other than `a`. d3 `.html()` is routed through the same sanitizer. `open()` may only target same-origin `/plot_file/`. | RT:35-81; AC:3-39 |
| **Plotly hover text (`text:` + `hoverinfo:'text'`) and Chart.js labels do not pass through ShinkaHTML.** Plotly renders its own pseudo-HTML subset. | RT:106-112 (only the theme is wrapped) |
| Private metrics are never sent (`show_private=False`). The UI shows "私有指标已隐藏". | AN:78; VIS:1722-1723; VT:5855-5857 |
| Lockbox, library and raw directories, and `<direction>/data`, are excluded from program-DB discovery. | AN:99-115 |
| Theme comes from the parent document or `localStorage['shinka-theme']`. Chart and Plotly colours are re-tokenised. | RT:83-125 |
| Keyboard: `role=tab` with Arrow/Home/End on `.left-tab`, `.tab` and `.meta-sub-tab`. Tree, prompt, ensembling-card and path nodes get `role=button` with Enter/Space and a Chinese `aria-label` (failed nodes are read as "评估失败"). | RT:127-159; AX:17-51 |
| Responsive: below 900px the tree and details panels stack at a fixed 600px each, the iframe is 1400px tall, and `prefers-reduced-motion` kills CSS animation. VT has no `@media` rules of its own and **no touch handlers** (grep found none). | WB:114-135; SRC/app.css:20-41 |

---

## 1. Read API and route consumers

| Route | Returns | Consumers (cite) | Destination of the data |
|---|---|---|---|
| `/list_databases` | `{path, actual_path, name, task}` sorted by date in the path (AN:213-254) | VT:3857 (hidden selectors); IX:524 | RUN |
| `/get_programs_summary` | All programs without code/embeddings, including PCA and cluster id (DB:1799-1852), plus failed proposal nodes (VIS:568-571); retries, then 503 when the DB is busy (VIS:579-596) | VT:4386, 4126 (silent refresh); CP:445 | LIN, PRG, SOL, CON, RUN |
| `/get_programs` | Full programs including `embedding` | VT:9333 (Embed heatmap only) | SOL |
| `/get_program_count` | count + max_timestamp | VT:4081 (auto-refresh every 3s) | Shell (OD-17) |
| `/get_program_details` | One program including code | VT:6181 (code/diff), VT:6580 (initial program for "diff vs initial"), RT:23-30 (hydrate Path and ZIP) | INS, LIN (path), Shell export |
| `/get_meta_files`, `/get_meta_content` | `meta/meta_N.txt` (N must be decimal, AN:143-152) | VT:3144, 3258, 3631 (XHR) | OD-3 (Scratchpad) |
| `/download_meta_pdf` | **501 disabled** (AN:311-315) | **orphan**; button disabled at VT:2495-2497 | OD-12 |
| `/get_plots` | `gen_N/results/plots/*.png|gif|jpg` (AN:323-342) | VT:6349 | INS |
| `/plot_file/<ns path>` | Image; pattern-checked (AN:168-190) | VT:6376, 6386; `open()` allow-list RT:75-77 | INS |
| `/get_system_prompts` | prompts.db rows, `[]` if absent, `{error}` on read failure (AN:264-287) | VT:4390, 4127, 4201, 17585 | LIN (prompt evolution), PRG (prompt cost) |
| `/get_database_stats` | Counts, best, cost; unknown pricing gives `null` (AN:19-34, 289-321) | IX:687 | RUN |
| `/api/workspace` | `{name, directions[], library}` (UI:254-298) | SRC/App.tsx:49-60 | RPT (index), LIB |
| `/api/snapshot?direction=` | Direction report payload (UI:176-215); 400/404/500 (UI:327-350) | SRC/App.tsx:49-60 | RPT, LIB |
| static `/analysis/i18n_{en,zh}.json` | i18n dictionaries | VT:18586 | Shell (OD-13) |
| pages `/analysis/index.html`, `/viz_tree.html`, `/compare.html` | Workbench documents (AN:38-41; VA:43) | Iframe in SRC/AnalysisWorkspace.tsx:4-8, 102-105 | Replaced by native surfaces |

Payload facts that matter for the mapping:

- `trial_points` carry `n, time, hypothesis, experiment, generation, score, record, bar` and **no program id** (UI:129-143). A trial dot on the Report cannot open the Program inspector today.
- Skeptic and verdict notes are hidden until a verdict exists (UI:35, 88-109). The experiment's `skeptic` field is likewise gated (UI:78-79).
- `analysis_url` is how an experiment deep-links into run analysis: `/?view=runs&direction&hypothesis&analysis=/viz_tree.html?db_path=workspace/...` (UI:63-76).

---

## 2. Shell, header and cross-cutting state of the analysis workbench (VT)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| S-01 | C | "← Dashboard" link to `/`. Inside the iframe it is intercepted to stay in the workspace. | VT:2144-2146; SRC/AnalysisWorkspace.tsx:81-90 | RUN |
| S-02 | C/I | Language toggle EN/中 stored in `localStorage['shinka-lang']`, default zh. `applyLang` rewrites only `[data-i18n]` and runtime-translated labels; rendered charts are not re-rendered. | VT:2149, 18579-18623; AX:4, 54-58 | no fitting destination — OD-13 |
| S-03 | C/E | Reload button reloads the current run. With no run selected it shows `alert()`. Status text says "Reloading data…". | VT:2150, 2774-2796 | no fitting destination — OD-17 |
| S-04 | C | Auto-refresh toggle. It polls `/get_program_count` every 3s and silently reloads summaries and prompts. It starts automatically after every load (VT:4411, 4436). | VT:2151, 2802-2819, 4030-4117, 4120-4281 | no fitting destination — OD-17 |
| S-05 | E | Scan status line: "正在读取运行…", "已载入 N 个程序 · time", 503 "数据库正忙；可以使用刷新重试。" or "读取失败：…" (`role=alert`). | VT:2152, 4343, 4421-4423, 4438-4448 | no fitting destination — OD-17 |
| S-06 | X | "导出程序": ZIP of the top-N valid programs (N from `window.prompt`, 1-100). Contains README.txt (TSV: rank/gen/score/name/type/lang/cost/complexity/model/ids/description/filename) and one file per program with a comment header. | VT:2153, 13592-13726, 17552-17557 | CON (export of ranked programs) |
| S-07 | S | URL keys `db_path`, `left_tab`, `right_tab`, `meta_section`, `selected_node`, written with `replaceState` on every tab change or selection. | VT:9094-9122, 2895, 2928, 5816 | All surfaces (deep links must survive) |
| S-08 | S | `localStorage['shinka_preferences']` holds embedding sort, colour scale, cell size, heat min/max and tree panel width. | VT:9124-9189 | SOL (embed prefs); Shell (panel width) |
| S-09 | S | Restore order: tab clicks, then wait for `shinka-data-ready`, then select the node, then the right tab and meta section. | VT:9191-9244 | All surfaces |
| S-10 | S | The iframe URL is projected into the parent URL (`analysis=`, `compare=`) so Back/Forward work. | SRC/AnalysisWorkspace.tsx:57-80 | RUN |
| S-11 | C | Draggable divider between the left and right panels; width persisted. | VT:2446, 3668-3771 | Shell (layout) — OD-11 |
| S-12 | E | Empty run: "该运行尚无程序记录。…" in the left pane and "还没有可分析的程序指标。" in the right pane. | VT:4456-4486, 4489-4493 | All surfaces |
| S-13 | C | Hidden task/result selects are kept only for compatibility with old JS; run selection effectively comes from the URL. | VT:2157-2162, 2739-2764, 9196-9216 | RUN |
| S-14 | Q/C | Persistent node summary above the right tabs: name (Gen), ID, Parent ID, score, error. | VT:2448, 5822-5828 | INS (header) |
| S-15 | C | `selectNodeById` defaults to **switching the left pane to Tree and the right pane to Code**. Pareto, the Meta charts, Metrics, migration links and the prompt→program table all call it. | VT:8348-8370; callers 6899, 6912, 7635, 10594, 12938, 13290, 18319 | INS (see OD-11) |
| D-S1 | D | `processData` errors are logged to the console only; the user sees no error. | VT:4567-4569 | carry-over: all |
| D-S2 | D | `auto-refresh-toggle` initial `aria-pressed="false"` is never updated by VT JS; the button only toggles a class. | VT:2151, 4039-4042, 4060-4063 | carry-over: Shell |
| D-S3 | D | `[INFERENCE]` `lastDataTimestamp = Date.now()` (milliseconds) is compared with server `max_timestamp` (seconds) for change detection. | VT:4430, 4093-4094 | carry-over: Shell |
| D-S4 | D | URL `left_tab` is interpolated into a CSS selector; a quote throws, and the error is caught silently. | VT:9221 | carry-over: Shell |
| D-S5 | D | Containers for Diversity and Cost are not cleared on reload, unlike the others. They re-render on activation. | VT:4540-4548 | carry-over: SOL/PRG |
| SEC-S1 | SEC | Node summary: `agentName` (LLM patch_name) and `data.error` are interpolated unescaped. Safe only because ShinkaHTML is applied at build. | VT:5823, 5827 | carry-over: INS (text nodes) |

---

## 3. Left tabs (13)

### L01 Tree — `tree-view` (VT:2164, 2180-2192)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| L01-Q1 | Q | How did programs descend from each other? (parent→child tree, gen 0 merged under a "unified root") | VT:4641-4720 | LIN |
| L01-Q2 | Q | Which program is best and what is its ancestry? Best = correct and finite max score. **Failed evaluations stay visible but never get the winner marker.** A single best path is drawn orange. | VT:4759-4767, 4805-4899, 5096-5125 | LIN (+INS for best) |
| L01-Q3 | Q | What mutation type and island produced each node? Shape encodes patch type (init ◆ / full ● / diff ■ / cross ✚ / fix ▲); island colour badge `I{n}`. | VT:4916-4942, 5163-5188 | LIN |
| L01-Q4 | Q | Which nodes failed? Red fill. Score colour uses viridis over correct programs. | VT:5086-5095, 4908-4914 | LIN |
| L01-C1 | C | Zoom and pan (d3.zoom 0.1-8). | VT:4636 | LIN |
| L01-C2 | C | Transpose to left→right layout (checkbox). | VT:2183, 3007-3014, 4712-4752 | LIN |
| L01-C3 | C | Animate: reveal by depth, looping. | VT:2187, 3017-3048, 5191-5283 | LIN — OD-8 |
| L01-C4 | C | Node click highlights ancestors, opens details and forces the Code tab. | VT:5022-5027, 4583-4606 | INS |
| L01-C5 | C | Hover tooltip: patch name, score, type, island, model. **Mouse only.** | VT:5028-5074 | INS (preview) — OD-16 (phone/touch) |
| L01-C6 | C | Legend: Best, Error, patch shapes, performance heatbar min/max, collapsible islands list. | VT:2408-2443, 5291-5392, 8064-8100 | LIN |
| L01-C7 | C | Keyboard: nodes get Enter/Space and an aria-label. | AX:17-51 | LIN |
| L01-E1 | E | No valid best node: tree renders without a best path. Render skipped when the data hash is unchanged. | VT:4769-4776, 4613-4618 | LIN |
| L01-I1 | I | Labels are hardcoded EN ("Transpose Tree", "Animate Tree", legend text); runtime dictionary translation applies to headings only. | VT:2184, 2188, 2411-2438; AX:10-27 | carry-over: LIN |
| SEC-L01 | SEC | Unified-root tooltip `patchName` and model name are unescaped. | VT:5049, 5060 | carry-over: LIN |
| D-L01a | D | The unified root copies the **first** gen-0 program's score and code, so it shows a fabricated score. | VT:4646-4668 | carry-over: LIN |
| D-L01b | D | Colour domain and fill use truthy `d.correct`. A string `'false'` would be treated as correct (the best-node logic is strict). | VT:4909, 5089 vs 4761 | carry-over: LIN |
| D-L01c | D | Ties for best are silently truncated to the first ("FORCE SINGLE PATH"). | VT:4889-4899 | carry-over: LIN |

### L02 PromptEvo — `meta-prompt-view` (VT:2165, 2363-2406)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| L02-Q1 | Q | How did system prompts evolve (prompt tree) and which prompt is best? Best = highest fitness with ≥1 program. | VT:17568-17641, 17643-17898 | LIN |
| L02-Q2 | Q | For one prompt: patch type, prompt generation, program generation, archive flag, live fitness percentile vs stored, programs (correct), total improvement, improved/correct, LLM metadata, created, parent. | VT:18216-18270, 18180-18212 | LIN (prompt detail) — OD-3b |
| L02-Q3 | Q | What did this prompt change vs its parent? Full text plus a line diff in a modal `<dialog>`. | VT:18275-18279, 18407-18466, 18469-18560 | LIN |
| L02-Q4 | Q | Which programs did this prompt generate and how did they do? Table sortable by Gen / Score / Δ / Beat%; a row opens the program. | VT:18099-18104, 18323-18405, 18318-18320 | LIN → INS |
| L02-C1 | C | Zoom and pan. **Two zoom behaviours** are attached (0.3-4 and 0.3-3). | VT:17734-17736, 17885-17894 | LIN |
| L02-C2 | C | Prompt node click; resizable split between tree and detail; "Show more" on descriptions. | VT:17776-17779, 3773-3841, 18301-18316 | LIN |
| L02-C3 | C | Legend: patch shapes, average improvement heatbar, Best marker. | VT:2372-2392, 17900-17960 | LIN |
| L02-S1 | S | Selected prompt kept across auto-refresh. **Not in the URL.** | VT:4252-4275 | LIN (add to URL) |
| L02-E1 | E | "Please select a database first."; "Loading prompts..." (`role=status`); "No meta-prompt evolution data available… `evolve_prompts: true`"; 503 busy vs read failure (`role=alert`). | VT:17575-17606, 17629-17640 | LIN |
| L02-I1 | I | Mixed: status line in ZH ("已载入 N 个提示"), everything else EN. | VT:17624, 18221-18276 | carry-over: LIN |
| SEC-L02 | SEC | LLM-generated `prompt.name` and `prompt.description` are interpolated **unescaped**. Prompt text itself is escaped. | VT:18228, 18232 vs 18279, 18447 | carry-over: LIN |
| D-L02 | D | `promptData.patch_type.toUpperCase()` throws when null. | VT:18219 | carry-over: LIN |

### L03 Programs — `table-view` (VT:2166, 2193-2222)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| L03-Q1 | Q | Ranking of programs by score. Columns: Rank, Gen, Archive, Patch name, Type, Island, Score, Cost, Complexity, Novelty/Resample/Patch attempts, Model. | VT:2204-2216, 8248-8262 | CON — OD-2 |
| L03-C1 | C | Sort on 9 columns (▲/▼ indicator; invalid values last). | VT:2204-2215, 8137-8204, 12538-12556 | CON |
| L03-C2 | C | "Show incorrect programs" (off by default). Incorrect rows are tinted red and rank "—". | VT:2196, 2997-3004, 8110-8131, 8227-8231 | CON |
| L03-C3 | C | Pagination Prev/Next with "Showing a-b of N". | VT:8291-8346 | CON |
| L03-C4 | C | Row click selects the program and opens Code; selection is synced to the table row. | VT:8269-8288, 5796-5803 | INS |
| L03-S1 | S | Sort state is not persisted (in-memory `sortState`). | VT:2703 | CON |
| L03-I1 | I | Headers hardcoded EN; "查看程序" aria-label in ZH. | VT:2204-2216, 8252 | carry-over: CON |
| SEC-L03 | SEC | `title="${rawModel}"` unescaped. | VT:8261 | carry-over: CON |
| D-L03a | D | **Correct programs with a null score disappear from the table** (neither the "correct" nor the "incorrect" bucket takes them). | VT:8120-8121 | carry-over: CON |
| D-L03b | D | API cost `0` and complexity `0` render as "N/A" (truthy checks). | VT:8233-8234 | carry-over: CON |
| L03-Q2 | Q | No text search or filter beyond the incorrect toggle. Gap noted, not a lost capability. | VT:2193-2222 | CON |

### L04 Metrics — `metrics-view` (VT:2167, 2357-2362)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| L04-Q1 | Q | How does each metric evolve over generations? One plot per metric: `combined_score`, numeric `public_metrics.*`, and `private_metrics.*` (private is always hidden server-side, so in practice public only). | VT:13049-13068 | PRG |
| L04-Q2 | Q | Per metric: individual programs (scatter), per-generation best, and cumulative min (toggle to cumulative max). | VT:13153-13215, 13255-13283 | PRG |
| L04-C1 | C | Per-plot toggle "Switch to Cumulative Max". | VT:13105-13115 | PRG (this is the de-facto objective-direction control — OD-7) |
| L04-C2 | C | Point click selects the program and switches to Code. | VT:13285-13293 | INS |
| L04-E1 | E | Correct programs with a score only; "No valid programs with metrics found." | VT:13018-13022 | PRG |
| L04-I1 | I | Hardcoded EN titles ("Generation", "Value"). | VT:13229-13237 | carry-over: PRG |
| SEC-L04 | SEC | Plotly hover `text` contains `patch_name` **unescaped** and does not pass through ShinkaHTML. | VT:13164 | carry-over: PRG |
| D-L04 | D | Rows of two plots with min-width 400px each: horizontal overflow on phones. | VT:13070-13094 | carry-over: PRG |

### L05 Embed — `embeddings-view` (VT:2168, 2223-2264)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| L05-Q1 | Q | How similar are programs' code embeddings? Cosine similarity matrix of all programs, plus per-island sub-matrices when there is more than one island. | VT:8373-8403, 8574-8668 | SOL |
| L05-Q2 | Q | Do similar programs cluster (hierarchical ordering) or cluster by performance? | VT:8737-8756, 8404-8449 | SOL |
| L05-C1 | C | Sort: chronological, hierarchical clustering or performance. | VT:2230-2234 | SOL |
| L05-C2 | C | Colour scale (viridis/plasma/inferno/magma), cell size 3-15 (sub-plots clamped to 4-8), heat min/max (auto-set per DB). | VT:2238-2259, 8705-8735, 8766-8767 | SOL |
| L05-C3 | C | Cell hover tooltip and row-label hover. Cell click opens the column program in details. | VT:8836-8934, 9080-9092 | INS |
| L05-S1 | S | Preferences in `localStorage` (S-08). | VT:9124-9189 | SOL |
| L05-E1 | E | "Waiting for data to load...", "Loading embedding data..." (fetches full `/get_programs` once per DB), "Failed to load embedding data.", "No embedding data available…". Duplicate gen-0 copies are deduplicated. | VT:9284-9358, 8541-8548, 8451-8481 | SOL |
| D-L05a | D | **Hangs forever on "Loading full embedding data..."** if any program lacks an embedding (for example a failed evaluation). The completeness check requires every program to have one. | VT:8483-8496, 8528-8535 | carry-over: SOL |
| D-L05b | D | The clicked cell index comes from `indexOf(value)`, so it is wrong when a row contains equal similarity values. | VT:8838, 8870 | carry-over: SOL |
| L05-I1 | I | Hardcoded EN. | VT:2226-2259, 8579 | carry-over: SOL |

### L06 Clusters — `clusters-view` (VT:2169, 2345-2356)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| L06-Q1 | Q | Where do programs sit in 2D PCA space, coloured by cluster and sized by score, with the best marked? | VT:12392-12426, 12379-12387 | SOL |
| L06-Q2 | Q | The same in **3D PCA** (scatter3d, colour bar "C-ID", rotate/zoom via Plotly). | VT:12437-12489 | SOL — OD-1 |
| L06-C1 | C | Point click in 2D or 3D selects the program; the left tab does not switch, the right tab switches to Code. | VT:12428-12435, 12482-12489 | INS |
| L06-C2 | C | Size legend (score min/mid/max). | VT:12491-12535 | SOL |
| L06-E1 | E | Correct programs with PCA 2D, PCA 3D, cluster id and score only. Otherwise "No programs with PCA embeddings and cluster IDs found." **Failed programs are invisible here.** | VT:12352-12363 | SOL — OD-10 |
| L06-L | C | Layout is fixed side by side 50/50, with no 2D/3D toggle. | VT:2348-2353 | SOL — OD-1 |
| D-L06 | D | The 2D plot uses the 3D symbol array (diamond); the star array is unused. | VT:12384-12405 | carry-over: SOL |
| L06-I1 | I | Hardcoded EN ("2D PCA of Program Embeddings", "PC1"). | VT:12414-12415, 12469 | carry-over: SOL |

### L07 Islands — `islands-view` (VT:2170, 2265-2281)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| L07-Q1 | Q | How does each island's best score progress? ("Performance Progression by Island", Best Score vs Generation) | VT:9360-9676, 10014 | LIN (islands) — dup PRG (D-1) |
| L07-Q2 | Q | How large is each island's population over time? ("Population Size by Island", cumulative programs) | VT:9678-9933, 10017 | LIN |
| L07-Q3 | Q | Per-island tree plots. | VT:10019-10560 | LIN |
| L07-Q4 | Q | Which programs migrated between islands, and when? Migration Summary table with Gen, linked program, score, from → to. | VT:10562-10600 | LIN |
| L07-C1 | C | Y-axis min/max overrides. Their listeners live inside `initializeScoreTypeListeners`. | VT:2265-2281, 7939-7962, 10008-10014 | LIN |
| L07-C2 | C | Collapsible legends; per-island tree node click; migration link → `selectNodeById`. | VT:9664-9673, 9923-9932, 10353, 10594 | LIN → INS |
| L07-E1 | E | "No data to visualize."; **correct programs only** ("No correct programs with island assignments and scores found."); "No migration events have been recorded." | VT:9948-9960, 10563 | LIN — OD-10 |
| L07-I1 | I | Hardcoded EN. | VT:2268, 9403, 9721 | carry-over: LIN |

### L08 Ensembling (model posteriors) — `model-posteriors-view` (VT:2171, 2334-2343)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| L08-Q1 | Q | Which LLM is the ensemble favouring, and how did model posteriors shift over generations? (lines or stacked) | VT:11299-11373, 11440-11446, 11501-11680 | PRG (per-generation trajectory) / CON (average posterior) — OD-4 |
| L08-Q2 | Q | Per model: programs generated, success rate, latest posterior (model overview cards). | VT:11436-11499 | CON |
| L08-Q3 | Q | Model comparison table: programs, success rate, average posterior, total cost. | VT:11453, 11683-11740 | CON |
| L08-Q4 | Q | `[INFERENCE]` Cumulative programs per model (computed for the cards). | VT:11375-11396 | CON |
| L08-C1 | C | Card click highlights a model; lines/stacked toggle; sortable table with `aria-sort`; hover tooltips (mouse only). | VT:11469-11483, 11671-11680, 11709-11736, 11586-11597, 11649-11668 | CON |
| L08-E1 | E | `t('ens_no_data')` when no posteriors exist in `metadata.llm_result.model_posteriors` or `metadata.model_posteriors`. | VT:11320-11341 | CON |
| L08-I1 | I | Uses `t()` (EN/ZH). | VT:11436-11498 | carry-over: CON |

### L09 Path — `best-path-view` (VT:2172, 2282-2285)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| L09-Q1 | Q | What chain of edits led from the root to the best program? Flowchart with gen/name/score/type per step and Δscore between steps. | VT:10630-10710 | LIN (path) |
| L09-Q2 | Q | How did the score climb along that path? (score curve) | VT:10712-10767 | LIN |
| L09-Q3 | Q | What exactly changed at each step? (diff chain; middle steps collapsed) | VT:10769-10830 | LIN (+INS diff) |
| L09-C1 | C | Clicking a flow node, curve dot or diff dot syncs all three and selects the program without switching tabs. Toggles show/hide each diff. | VT:10659-10674, 10707, 10766, 10785, 10821-10828 | LIN → INS |
| L09-E1 | E | `t('path_no_data')`, `t('path_no_best')`, `t('path_empty')`; hydration failure "路径详情读取失败" (`role=alert`); `t('path_no_diff')`. | VT:10625-10654, 10814 | LIN |
| L09-I1 | I | Uses `t()`. | VT:10681, 10715, 10772 | carry-over: LIN |
| D-L09 | D | Null scores are treated as 0 in Δ and in the curve (a failed ancestor plots at 0). | VT:10690, 10720 | carry-over: LIN |

### L10 Cost — `cost-view` (VT:2173, 2287-2289)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| L10-Q1 | Q | What did the run cost: total API, embedding, overall, average per program? | VT:11774-11798 | PRG |
| L10-Q2 | Q | How did the best valid score rise against cumulative cost? (line chart with tooltip) | VT:11806-11937 | PRG |
| L10-Q3 | Q | Which model buys improvement most cheaply? ($/Δscore bars) | VT:11940-11987 | CON (model contest) / PRG — OD-4 |
| L10-Q4 | Q | Zero-cost mode: when recorded model cost is $0, everything switches to compute time (s/Δscore). | VT:11778-11790, 11806-11807, 11841-11843 | PRG |
| L10-E1 | E | `t('cost_no_data')`; incomplete records give "费用或计算时间记录不完整；不绘制缺失的资源数值。" and nothing is plotted (honest unknown cost). "至少需要同模型的两个不同有效分数…" | VT:11769-11771, 11817-11821, 11966-11967 | PRG |
| L10-I1 | I | Mixed: `t()` plus hardcoded ZH (zero-cost labels) and EN ("lower is more efficient"). | VT:11786-11807, 11980-11982 | carry-over: PRG |
| D-L10a | D | Cost here **excludes prompt-evolution cost**, while the Meta tab includes it (see D-2). | VT:11763-11766 vs 5602-5603 | carry-over: PRG |
| D-L10b | D | Tie-break sorts by subtracting string ids (NaN). | VT:11829 | carry-over: PRG |

### L11 MAPs (MAP-Elites) — `maps-view` (VT:2174, 2291-2324)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| L11-Q1 | Q | Which behaviour cells (any two numeric descriptors) are occupied, and by which elite? The best score is kept per cell. | VT:11054-11097, 11238 | SOL |
| L11-Q2 | Q | How did coverage grow up to generation g? (generation slider "≤g"/"All") | VT:11004-11037 | SOL |
| L11-C1 | C | X and Y axis selects, populated from nested metric discovery up to depth 3. Defaults: `lines_of_code` × `compute_time`. | VT:2295-2300, 10854-10927, 10956-10987 | SOL |
| L11-C2 | C | Bins slider (3-20), fitness vmin/vmax, generation slider. | VT:2303-2318, 10992-11011 | SOL |
| L11-C3 | C | Cell hover tooltip; cell click selects the elite without switching tabs. | VT:11171-11194 | INS |
| L11-E1 | E | "No data to visualize.", "Please select axes.", "No correct programs found.", "No programs with valid values for selected axes." **Correct programs only.** | VT:10942-10945, 11039-11068 | SOL — OD-10 |
| L11-I1 | I | Tab label not translatable (no `data-i18n`); hardcoded EN. | VT:2174, 11238, 11284 | carry-over: SOL |
| D-L11a | D | Bins label shows "10" while the slider value is 20. | VT:2304-2305 | carry-over: SOL |
| D-L11b | D | Axis options are populated once and **never reset when the run changes** (Pareto does reset), so a newly loaded run keeps stale axes. | VT:10851, 10948-11013 vs 4371-4375 | carry-over: SOL |

### L12 Battle — `battle-view` (VT:2175, 2326-2328)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| L12-Q1 | Q | How are valid scores distributed per **LLM mutation model**? (box plot sorted by median) | VT:12006-12046, 12093-12183 | CON |
| L12-Q2 | Q | Head-to-head: how often does model A's program beat model B's? (pairwise win-rate matrix) | VT:12048-12066, 12254-12262 | CON |
| L12-Q3 | Q | Model ranking by median, best and program count. | VT:12313-12318 | CON |
| L12-C1 | C | Hover tooltips (mouse only). Computed client-side only; there is no fetch in VT:11991-12340. | VT:12084-12091 | CON |
| L12-E1 | E | `t('battle_no_data')` when there are no correct, finite-score programs with a model. | VT:12015-12017 | CON |
| L12-I1 | I | `t()` plus EN literal "Win rate". | VT:12071-12078, 12257 | carry-over: CON |
| L12-N | Q | Note: "Battle" is model vs model, **not** program vs program. Program vs program lives in Pareto (R02) and Programs (L03). | VT:12006-12013 | CON |

### L13 Diversity — `diversity-view` (VT:2176, 2330-2332)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| L13-Q1 | Q | Is the population collapsing or exploring? Per-generation score standard deviation vs average pairwise PCA-2D distance (dual-axis line). | VT:13333-13458 | SOL |
| L13-Q2 | Q | Where does generation g's population sit in PCA space? Scatter with a generation slider; **includes failed programs, labelled "—（评估失败）"**. | VT:13460-13555 | SOL |
| L13-Q3 | Q | Per-generation statistics table: N, embedding distance, score std, score range, cluster count. | VT:13558-13588 | SOL |
| L13-C1 | C | Generation slider (starts at the first generation); hover tooltip; **no click to select**. | VT:13468-13470, 13537-13555 | SOL (add INS link) |
| L13-E1 | E | `t('div_no_data')`; scatter only when PCA data exists. | VT:13315-13317, 13321, 13461, 13487-13489 | SOL |
| L13-I1 | I | `t()`; column headers "N" and "clusters" are EN. | VT:13563-13565 | carry-over: SOL |
| D-L13 | D | Minimum width 500px forces horizontal scroll on phones. | VT:13387 | carry-over: SOL |

---

## 4. Right tabs (10)

### R01 Meta — `agent-info` (VT:2450, 2462-2464; template VT:5419-5522)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| R01-Q1 | Q | Run overview: total generations, correct % (n/N), total cost, average cost per program. | VT:5426-5430, 5728-5738 | PRG (headline) / RUN row |
| R01-Q2 | Q | Best solution: score, name, generation, island. | VT:5434-5438, 5740-5747 | PRG headline → INS |
| R01-Q3 | Q | Cost breakdown: API / Embed / Novelty / Meta / PromptEvo. **Unknown cost shows "—" rather than $0.** | VT:5442-5447, 5523-5528, 5597-5603, 5732-5738 | PRG |
| R01-Q4 | Q | "Performance Score Across Program Evaluations": cumulative max, best path, individual scores, mean. | VT:5456-5470, 6839-6922 | PRG — dup D-1 |
| R01-Q5 | Q | "Performance Score Across API Budget". | VT:5474-5488, 7072 | PRG |
| R01-Q6 | Q | "Performance Score Across ShinkaEvolve Runtime". | VT:5492-5506, 7348 | PRG |
| R01-Q7 | Q | "Cumulative API Cost Breakdown": stacked bars per generation plus a cumulative line. | VT:5509-5512, 7491-7640 | PRG |
| R01-Q8 | Q | "Cumulative Runtime Across Program Evaluations" / "Runtime per Generation". | VT:5515-5518, 7734-7800 | PRG |
| R01-C1 | C | Six Y min/max inputs (debounced redraw). | VT:5459-5500, 6650-6654 | PRG |
| R01-C2 | C | Click a score point to select that program; click a cumulative-max or cost bar to select the best program of that generation (switches to Tree + Code). | VT:6890-6921, 7620-7638 | INS |
| R01-C3 | C | When a node is selected, the charts are redrawn with its generation (`createCharts(data.generation)`). `[INFERENCE]` The generation is highlighted. | VT:5832-5835, 6656 | PRG |
| R01-E1 | E | "正在读取程序指标。" placeholder; renders only while the tab is active. | VT:5422, 6659 | PRG |
| R01-I1 | I | Mixed: EN labels, ZH placeholder and aria-labels. | VT:5421-5518 | carry-over: PRG |
| D-R01 | D | Best agent uses truthy `correct`, while the Tree uses strict `correct`, so the two views can disagree. | VT:5566, 5578 vs 4760-4762 | carry-over: PRG |
| DEAD-R01 | D | Score-type radio (`input[name=score-type]`, min vs max objective) is wired in JS, but **the markup does not exist**, so the objective switch is unreachable. | VT:2880, 7877-8010; no markup match | OD-7 |

### R02 Pareto — `pareto-front` (VT:2451, 2465-2485)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| R02-Q1 | Q | Which programs are Pareto-optimal on two chosen objectives (for example score vs compute time or complexity)? | VT:12609-12637, 12961-12969 | CON |
| R02-Q2 | Q | Dominated vs incorrect (red ×) vs Pareto-optimal points, boundary line, dominated region. | VT:12665-12894 | CON |
| R02-C1 | C | X/Y metric selects (`combined_score`, `complexity`, `metadata.compute_time`, `public_metrics.*`) and per-axis "Higher/Lower is better" selects (min is the default for cost, time, complexity and latency). | VT:2466-2483, 12569-12607, 12971-12988 | CON |
| R02-C2 | C | Point click → `selectNodeById` (switches to Tree + Code). | VT:12936-12939 | INS |
| R02-S1 | S | Axis and objective choices are not persisted; reset on run load. | VT:4370-4375 | CON |
| R02-E1 | E | No empty-state message. | VT:12945-13004 | CON |
| R02-I1 | I | Hardcoded EN. | VT:12909-12916 | carry-over: CON |
| SEC-R02 | SEC | Metric names (from metric keys) go into Plotly hover and axis titles unescaped. `patch_name` is escaped. | VT:12640-12657, 12911-12916 | carry-over: CON |
| D-R02 | D | `{...find(), objective}` is never falsy, so the empty-metric guard never fires and the plot is built from undefined paths. Also recomputed on every load even when hidden. | VT:12997-13000, 4520 | carry-over: CON |

### R03 Scratchpad — `meta-analysis` (VT:2452, 2486-2509)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| R03-Q1 | Q | What did the meta-LLM conclude at meta update N? Three sections: Individual Programs, Global Insights, Meta Recommendations. | VT:2500-2503, 3501-3542, 3467-3499 | no fitting destination — OD-3 |
| R03-C1 | C | "Meta Update" slider snaps to available `processed_count`; defaults to latest. | VT:2489-2491, 3077-3096, 3159-3168, 3188-3215 | OD-3 |
| R03-C2 | C | Refresh button; sub-tab click (saved as URL `meta_section`). | VT:2494, 3100-3129 | OD-3 |
| R03-X1 | X | PDF export button **disabled** with explanatory note; server returns 501. | VT:2495-2497; AN:311-315 | OD-12 |
| R03-E1 | E | "Select a database to view meta analysis."; "No meta analysis files found for this database."; "Failed to load meta files(.list)"; "Failed to render meta analysis content." | VT:2507, 3171-3179, 3195, 3579-3582 | OD-3 |
| R03-I1 | I | Hardcoded EN. | VT:2489-2503, 3565 | carry-over |
| SEC-R03 | SEC | Individual Programs uses a regex mini-markdown (`processSimpleMarkdown`) that concatenates **raw LLM text into HTML without escaping**; safe only through ShinkaHTML. The other sections use markdown-it with `html:false`. | VT:3341-3418, 3525-3528; RT:18-21 | carry-over |
| DEAD-R03 | D | Diff highlighting between meta updates (`generateSimpleDiffHTML`, `convertStructuredMetaToHTML`, `formatStructuredEntries`) has no callers. The previous update is prefetched (3278) but never diffed. | VT:3295, 3420, 3586, 3621 | OD-19 |

### R04 Node — `node-details` (VT:2453, 2510-2512)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| R04-Q1 | Q | Everything recorded about one program: General (language, island, timestamp), Combined Score, Public Metrics, Private Metrics & Feedback (hidden notice plus `text_feedback`), Metadata (JSON; meta_* and novelty_explanation are collapsed), Code Analysis, Patch Information, Inspiration, Migration History, Thought Process (markdown). | VT:5838-6151 | INS |
| R04-C1 | C | Expandable `<details>` for long metadata. | VT:5875-5883 | INS |
| R04-E1 | E | "Select a node to view its details."; "No public metrics available."; "私有指标已隐藏…"/"没有私有指标记录。"; "No text feedback available." | VT:2511, 5846, 5855-5857, 5964 | INS |
| R04-I1 | I | Mixed EN/ZH. | VT:5932-5971, 5856 | carry-over: INS |
| SEC-R04 | SEC | Metric **keys** are unescaped (5843, 5852); `thought` is rendered with markdown-it `html:false` (6138); values are escaped. | VT:5843, 5852, 5869, 6138 | carry-over: INS |

### R05 Code — `agent-code` (VT:2454, 2513-2521)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| R05-Q1 | Q | What is this program's source? Line-number gutter plus syntax highlighting. | VT:5753-5784 | INS |
| R05-X1 | X | Copy to clipboard ("Copied" feedback); download as `<patch_name>_gen<N>.<ext>`. | VT:2515-2516, 2822-2876 | INS |
| R05-E1 | E | "Loading code..."; "Failed to load code: …"; "No code available for this node."; controls hidden when there is no code. Code is lazy-loaded from `/get_program_details`. | VT:6174-6226, 5754-5757 | INS |
| R05-I1 | I | Buttons use `data-i18n`. | VT:2515-2516 | carry-over: INS |
| D-R05 | D | Only Python highlighting is registered, so cpp/cuda/js render plain. | RT:4-5, 17; VT:5769 | carry-over: INS |

### R06 Diff — `code-diff` (VT:2455, 2522-2534)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| R06-Q1 | Q | What changed relative to the parent (stored `code_diff`)? | VT:6530-6538, 6461-6474 | INS |
| R06-Q2 | Q | What changed relative to the initial program? Computed client-side after fetching the initial program. | VT:6503-6527, 6549-6603, 6606-6623, 6580 | INS |
| R06-C1 | C | Parent/Initial toggle; "Show Changes Only"/"Show All Lines". | VT:2526-2529, 6477-6490 | INS |
| R06-E1 | E | "Loading diff..."; "No parent/initial diff available…"; "Failed to load diff." | VT:6178, 6210, 6218, 6537, 6543 | INS |
| R06-I1 | I | Hardcoded EN. | VT:2524-2529 | carry-over: INS |

### R07 Evaluation — `log-output` (VT:2456, 2535-2537)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| R07-Q1 | Q | What did the evaluator print (STDOUT/STDERR)? Escaped `<pre>`. | VT:6228-6248 | INS |
| R07-Q2 | Q | Which plots or animations did the evaluator emit for this program? Images are laid out by aspect ratio; clicking opens the file via the `/plot_file/` allow-list. | VT:6339-6459, 6247-6253; RT:75-77 | INS |
| R07-E1 | E | "No log output available for this node."; plots fail silently on non-OK responses (console only). | VT:6242-6244, 6352-6355 | INS |
| R07-I1 | I | EN, plus emoji headings. | VT:6364, 6388 | carry-over: INS |

### R08 LLM — `llm-result` (VT:2457, 2538-2540)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| R08-Q1 | Q | What did the LLM return for this mutation? Key/value table; `content`, `msg`, `system_msg`, `new_msg_history` and `thought` are collapsed; nested values shown as JSON. | VT:6255-6325 | INS |
| R08-E1 | E | "No LLM result available for this node."; fallback to pretty-printed JSON. | VT:6260-6269 | INS |
| R08-SEC | SEC | All values are escaped. | VT:6295-6304 | carry-over: INS |
| R08-I1 | I | Tab label "LLM" has no `data-i18n`. | VT:2457 | carry-over: INS |

### R09 Analysis — `analysis` (VT:2458, 2541-2624; driver VT:17490-17547)

Every plot below is computed client-side from summaries. Each has its own "No … data available." state. **None of them has click-through to a program**; selection is only passed in for highlighting (no `plotly_click` or `selectNodeById` in VT:14783-17488). The whole tab recomputes on every node selection (VT:6277-6280). Hardcoded EN.

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| R09-Q1 | Q | Program Correctness Rate: rolling %, overall and per patch type. | VT:2544-2545, 14304-14377, 15980-16043 | PRG |
| R09-Q2 | Q | Program Improvement Rate [P vs. C]: rolling %, per patch type. | VT:2548-2549, 14379-14475, 16045-16108 | PRG |
| R09-Q3 | Q | Lines of Code across Generations (code bloat). | VT:2552-2553, 14209-14248, 15774-15858 | PRG |
| R09-Q4 | Q | Mutation Size vs Improvement. | VT:2556-2557, 14250-14302, 15860-15978 | LIN; also in the PRG diagnostics block — OD-6 |
| R09-Q5 | Q | Novelty Success Rates: success rate per LLM, success vs failures per patch type. | VT:2559-2570, 13871-14006, 15156, 15409 | PRG (diagnostics) — OD-6 |
| R09-Q6 | Q | Resample Success Rates (same pair). | VT:2572-2583, 14008-14144, 15527, 15653 | PRG (diagnostics) — OD-6 |
| R09-Q7 | Q | Patch Success Rate per LLM and per patch type. | VT:2585-2596, 13731-13869, 14783, 15045 | PRG (diagnostics) — OD-6 |
| R09-Q8 | Q | Patch Improvement Rate per LLM and per patch type. | VT:2598-2609, 14914, 15100 | PRG (diagnostics) — OD-6 |
| R09-Q9 | Q | Cost by Model vs Patch Type (heatmap). | VT:2611-2613, 14490-14718, 16219-16330 | PRG (cost) — OD-6 |
| R09-Q10 | Q | Average Cost Across Generations. | VT:2615-2617, 16331 | PRG |
| R09-Q11 | Q | Parent Lineage: Offspring Count & Improvements (most fertile parents: total vs improving children). | VT:2619-2621, 14146-14207, 15711-15772 | LIN; also in the PRG diagnostics block — OD-6 |
| R09-E1 | E | "No data available. Please load a database." (only in the code-bloat slot); error banner "Error computing analysis statistics". | VT:17496-17502, 17536-17545 | PRG/LIN |
| SEC-R09 | SEC | Lineage-fertility hover interpolates `p.name`, `patchType` and `modelName` **unescaped** into Plotly hover. | VT:15734-15737 | carry-over: LIN, PRG (the diagnostics block reuses the Lineage component) |
| D-R09 | D | Two sub-plots are both titled "Improvement Rate" under "Patch Improvement Rate". | VT:2602, 2606 | carry-over: PRG |
| DEAD-R09 | D | Renderers with no caller and no DOM target: novelty LLM improvement (15282), novelty improvement by patch type (15467), cost efficiency per generation (16110), search dynamics by cluster (17426 plus `computeSearchDynamicsStats` 14720). | as cited | OD-19 |

### R10 Throughput 吞吐 — `throughput` (VT:2459, 2625-2665; driver VT:17389-17424)

All eight items need per-program `metadata.{pipeline,sampling,evaluation,postprocess}_{started,finished}_at`, `timeline_lane_mode='pool_slots'`, `*_worker_id`, `*_worker_capacity` and `source_job_id` (VT:16457-16534, 16633-16636). Hardcoded EN, but the tab label is hardcoded ZH "吞吐".

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| R10-Q1 | Q | Throughput Summary. | VT:2627-2629, 17054-17096 | PRG |
| R10-Q2 | Q | Generation Runtime Timeline (lanes per worker slot), with a summary line; the selected program is highlighted. | VT:2631-2634, 16749, 17417 | PRG |
| R10-Q3 | Q | Worker Occupancy Over Time. | VT:2636-2638, 17098-17173 | PRG |
| R10-Q4 | Q | Normalized Worker Occupancy Over Time (%). | VT:2640-2642, 17175-17249 | PRG |
| R10-Q5 | Q | Evaluation Occupancy Distribution. | VT:2647-2648, 17251-17280 | PRG |
| R10-Q6 | Q | Completed Evaluations per Minute. | VT:2651-2652, 17282-17311 | PRG |
| R10-Q7 | Q | Stage Duration Summary (table). | VT:2657-2658, 17313-17349 | PRG |
| R10-Q8 | Q | Resource Utilization Summary (table). | VT:2661-2662, 17351-17387 | PRG |
| R10-E1 | E | "No throughput data available. Please load a database."; "No pool-based runtime data available." in every slot. | VT:17395-17413 | PRG |
| R10-N | Q | Island copies are deduplicated by `source_job_id`. | VT:16515-16534 | PRG |

---

## 5. Run list, compare, and React research pages

### P01 Run list — `analysis/index.html` (iframe default at `/?view=runs`)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| P01-Q1 | Q | Which runs exist, grouped by task? Columns: task, run directory, programs, valid programs, max generation, generations since improvement (stall, coloured at >10 and >50), best valid score, prompt-evo count (cost in title), cost (coloured at >$1 and >$5), updated (UTC). | IX:485-500, 636-651, 759-793 | RUN |
| P01-Q2 | Q | Totals across runs: run count, peak score (direction-agnostic max), total cost (only when every run is loaded and every cost is known), total programs. | IX:432-449, 731-742 | RUN |
| P01-C1 | C | Search over task and run name; `/` focuses search. | IX:452-456, 590-596, 829-832, 860-865 | RUN |
| P01-C2 | C | Sort by name, best score, last update or cost. | IX:457-461, 565-571 | RUN |
| P01-C3 | C | Multi-select checkboxes → "比较所选运行" (enabled at ≥2) → compare page. | IX:464-468, 637-639, 795-821 | RUN → CON (OD-5) |
| P01-C4 | C | Collapse task groups (persisted); row click or run link opens `/viz_tree.html?db_path=`. | IX:603-623, 641, 653, 744-752 | RUN |
| P01-C5 | C | Auto-refresh every 10s (toggle with `aria-pressed`); GitHub link (external). | IX:420-426, 834-858 | RUN — OD-17 |
| P01-S1 | S | URL `selected` (JSON of `{path,label}`), `q` and `sort`; `localStorage['collapsed-task-*']`. | IX:511-520, 806-815, 606, 746 | RUN |
| P01-E1 | E | Loading "正在读取运行记录…"; empty "没有可显示的运行"; discovery error "无法读取运行目录"; no match "没有匹配的运行…已选运行仍会保留"; per-cell spinner; per-row "读取失败". | IX:473-482, 536-541, 657-662, 666-669, 692-699 | RUN |
| P01-I1 | I | Hardcoded ZH, `<html lang="en">`. | IX:2, 434-494 | carry-over: RUN |
| P01-SEC | SEC | Task and name are escaped; formatted stats are written with `innerHTML` (numbers only). | IX:609-617, 636-651, 727, 823-827 | carry-over: RUN |
| D-P01a | D | "loadVisibleStats" loads stats for **all** runs in batches of 4, not just visible rows. | IX:671-680 | carry-over: RUN |
| D-P01b | D | `total_runtime_seconds` is formatted but has no column, so a lost field. | IX:722 | RUN (restore) |

### P02 Compare — `analysis/compare.html`

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| P02-Q1 | Q | How do the selected runs compare by generation? Cumulative best valid score, program count or correct count. | CP:353-357, 539-585, 637 | CON / RUN — OD-5 |
| P02-Q2 | Q | How do they compare against cumulative cost? Same metrics vs cost in timestamp order; downsampled to 200 points. | CP:587-630, 667 | CON / RUN — OD-5 |
| P02-Q3 | Q | Legend table: run, best valid score, max generation, programs, cost; a row opens the run's analysis. | CP:376-391, 469-505 | RUN |
| P02-Q4 | Q | What is each run's value at a given generation or spend? Chart.js index-mode tooltips list every run's value at the pointed position (scores to 2 decimals). | CP:681-692, 711-723 | RUN |
| P02-S1 | S | URL `selection` only; the metric choice is not persisted. Inside the app it is mirrored as `compare=`. | CP:409-427; SRC/AnalysisWorkspace.tsx:28-30, 64-68 | RUN |
| P02-E1 | E | Loading "正在读取比较数据。"; "没有可比较的运行：请从全部运行选择至少两个运行。"; "读取失败：<msg>". | CP:338-347, 412-441 | RUN |
| P02-I1 | I | Hardcoded ZH. | CP:332-385, 516-524 | carry-over |
| D-P02 | D | Cost excludes prompt-evo cost (`getProgramCost`), while the run list includes it (IX:705-708), so totals differ for the same run. Series styles repeat after 15 runs (3 tokens × 5 dashes). | CP:507-511; RT:89-94 | carry-over (D-2) |

### P03 App shell + TopBar + WorkspaceNav (React)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| P03-Q1 | Q | Direction header: trials, budget (caution at ≥80%, critical at ≥100%), lockbox state (未设/已开封/封存), with term tooltips. On narrow screens it drops to its own strip. | SRC/TopBar.tsx:21-62 | RPT (header) |
| P03-C1 | C | Theme toggle (dark/light, `localStorage['shinka-theme']`, propagated into the iframe). | SRC/TopBar.tsx:56; SRC/App.tsx:12-45; SRC/AnalysisWorkspace.tsx:42-46 | no fitting destination — OD-17 |
| P03-C2 | C | Nav: 研究方向 (`/`), 全部运行 (`/?view=runs`), 资料库 (`/?view=library`) with a pending-count tag. | SRC/WorkspaceNav.tsx:5-14 | RPT / RUN / LIB |
| P03-C3 | C | Breadcrumb 工作区 › 返回研究 · direction · hypothesis › 实验分析. | SRC/App.tsx:91-99 | RPT/RUN |
| P03-C4 | C | Refresh button (re-fetches the snapshot). | SRC/App.tsx:105-107 | OD-17 |
| P03-S1 | S | URL `view`, `direction`, `hypothesis`, `analysis`, `compare`. | SRC/App.tsx:29-33, 79-84; SRC/AnalysisWorkspace.tsx:21-35, 57-71 | all |
| P03-E1 | E | "读取失败：… 确认 shinka_research ui 还在运行"; "还没有假设：用 shinka_research hypothesis add…"; "载入中…"; analysis iframe errors ("分析链接无效", "分析资源未加载", "分析页面离开了工作区"). | SRC/App.tsx:102-138; SRC/AnalysisWorkspace.tsx:16, 26, 54, 60 | RPT/RUN |
| P03-Q2 | Q | Disclaimer on runs: raw `combined_score` is diagnostic only and not comparable across evaluators; conclusions live on the direction page. | SRC/AnalysisWorkspace.tsx:100 | RUN (must keep) |
| P03-I1 | I | All ZH; no language toggle (VT has one). | SRC/App.tsx, SRC/TopBar.tsx | OD-13 |
| P03-A | D | TopBar uses Astro `rux-global-status-bar` and `rux-monitoring-icon`; Astro is being removed. | SRC/TopBar.tsx:44-58 | OD-15 |

### P04 WorkspaceTable (workspace home `/`)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| P04-Q1 | Q | Which research directions exist? Per direction: hypotheses, trials, spend/budget, lockbox state; folder id shown. | SRC/WorkspaceTable.tsx:14-61; UI:254-298 | RPT (index) — OD-14 |
| P04-C1 | C | Direction link → `/?direction=<id>`. | SRC/WorkspaceTable.tsx:31-36 | RPT |
| P04-E1 | E | No directions: "还没有研究方向——用 shinka_research init…"; unreadable direction: "读取失败：研究记录无法读取（Type）". | SRC/WorkspaceTable.tsx:6-12, 53-55; UI:286-293 | RPT |
| P04-I1 | I | ZH. | — | carry-over |

### P05 ResearchTab (direction page `/?direction=`)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| P05-Q1 | Q | Layout of the direction report: hypotheses table, trial chart, experiment timeline, spec card, events. Single column below 900px (order: table, side, chart, timeline, events). | SRC/ResearchTab.tsx:10-42; SRC/app.css:38-41 | RPT |
| P05-E1 | E | No trials: "还没有进化出的程序（只有 N 次试探，试探不画点）" or "还没有试验——用 shinka_research run…". | SRC/ResearchTab.tsx:21-25 | RPT |

### P06 HypothesisTable

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| P06-Q1 | Q | Hypotheses with id, title, status (10 states), experiments, trials, best in-sample score, validation Sharpe. **Validation shows "—" until a verdict, with tooltip "出结论后显示".** | SRC/HypothesisTable.tsx:11-27, 85-104; SRC/format.ts:5-16, 46-51; SRC/terms.ts:13 | RPT |
| P06-C1 | C | Sortable id/title/status/experiments (nulls last); row click or Enter/Space selects (URL `hypothesis`). | SRC/HypothesisTable.tsx:29-34, 71-84; SRC/App.tsx:79-84 | RPT |
| P06-A | D | Status uses Astro `rux-status`. | SRC/HypothesisTable.tsx:89 | OD-15 |

### P07 TrialChart

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| P07-Q1 | Q | Across the whole direction (trial order): in-sample scores of the selected vs other hypotheses, new records, step frontier, and the **noise line** (expected max under null including probes, dashed). | SRC/TrialChart.tsx:8-41, 43-71, 92-98; UI:112-144 | RPT |
| P07-Q2 | Q | Counts: N trials, failed (not drawn), probes (not drawn but counted in the noise line). | SRC/TrialChart.tsx:21, 94-97 | RPT |
| P07-C1 | C | Carbon ComboChart hover; the legend is not clickable; **no click-through to a program** (no program id in the payload). | SRC/TrialChart.tsx:49; UI:133-143 | RPT — OD-18 |
| D-P07 | D | The noise line's dash is applied by matching DOM stroke colour (fragile). | SRC/TrialChart.tsx:73-90 | carry-over: RPT |

### P08 Timeline

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| P08-Q1 | Q | When did each hypothesis run? One track per hypothesis from first to last trial, coloured by status. Auto-fits minute/hour/day units. | SRC/Timeline.tsx:7-39, 52-81 | RPT |
| P08-C1 | C | Click a region to select the hypothesis. | SRC/Timeline.tsx:68-74 | RPT |
| P08-E1 | E | "还没有试验，画不出时间线…". | SRC/Timeline.tsx:53 | RPT |
| P08-A | D | Built entirely on Astro `rux-timeline`, `rux-track` and `rux-time-region`. | SRC/Timeline.tsx:62-79 | OD-15 |

### P09 SpecCard

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| P09-Q1 | Q | Hypothesis spec: id, status, created, experiments, trials, spend; markdown body. | SRC/SpecCard.tsx:148-171 | RPT |
| P09-Q2 | Q | Links per experiment, "查看实验分析", into run analysis. | SRC/SpecCard.tsx:126-144; UI:63-76 | RPT → LIN/PRG/… |
| P09-Q3 | Q | Before a verdict: progress = trials (failed), best in-sample score vs current noise line, plain-language reading, plus "还没出结论：验证段结果出结论后显示". | SRC/SpecCard.tsx:102-124, 175-187 | RPT |
| P09-Q4 | Q | After a verdict: check ledger with 5 rules (validation Sharpe > 0, beats buy & hold, DSR ≥ thr, subperiods ≥ thr, PBO ≤ thr; PBO counts as pass when it cannot be computed), in-sample Sharpe, noise, trials, reason. | SRC/SpecCard.tsx:48-81; SRC/format.ts:56-84 | RPT |
| P09-Q5 | Q | Lockbox confirmation results (labelled keys). | SRC/SpecCard.tsx:20-46, 189 | RPT |
| P09-Q6 | Q | Next step when nothing has run (scaffold/run commands). | SRC/SpecCard.tsx:86-99 | RPT |
| P09-SEC | SEC | Hypothesis body uses markdown-it `html:false` through `dangerouslySetInnerHTML`; external links get `target=_blank rel=noopener`; front matter is stripped. | SRC/Markdown.tsx:4-17; SRC/renderMarkdown.ts:4-36 | carry-over: RPT |
| P09-A | D | Ledger and status use Astro `rux-status`. | SRC/SpecCard.tsx:59, 161 | OD-15 |

### P10 EventStream

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| P10-Q1 | Q | What happened, newest first (50 latest): `hypothesis/experiment · event · note`, status-coloured (verdict, skeptic, run, run-failed, confirm). Skeptic and verdict notes hidden until a verdict. | SRC/EventStream.tsx:4-11; SRC/format.ts:18-24; UI:88-109, 213 | RPT |
| P10-E1 | E | "还没有事件。" | SRC/EventStream.tsx:5 | RPT |
| P10-Q2 | Q | Filter the events by text: rows whose `hypothesis/experiment · event · note` contains the filter (case-insensitive) stay, with a notice of how many are hidden. Built into `rux-log`, so the source never names it. | npm `@astrouxds/astro-web-components@8.0.0` (the version `app/package.json` pins) › `dist/collection/components/rux-log/rux-log.js:31-42` | RPT |
| P10-A | D | Built on Astro `rux-log`. | SRC/EventStream.tsx:11 | OD-15 |

### P11 LibraryTab (`/?view=library`)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| P11-Q1 | Q | What is in the shared library? Index, log and wiki pages (markdown). | SRC/LibraryTab.tsx:18-55; UI:151-173 | LIB |
| P11-Q2 | Q | How many raw sources are pending curation? (warning lists the first 5; nav tag count) | SRC/LibraryTab.tsx:24-36; SRC/WorkspaceNav.tsx:10-11 | LIB |
| P11-C1 | C | Tree navigation (目录/日志/页面(n)); `[[wiki]]` links jump in-app; external links open a new tab. | SRC/LibraryTab.tsx:38-54; SRC/Markdown.tsx:10-15; SRC/renderMarkdown.ts:6-31 | LIB |
| P11-S1 | S | Selected page **not in the URL**. | SRC/LibraryTab.tsx:10 | LIB |
| P11-E1 | E | No library: "还没有资料库——用 shinka_research library add…"; "wiki 里还没有页面…"; "这一页是空的。" | SRC/LibraryTab.tsx:11-16, 51, 54 | LIB |

### P12 AnalysisWorkspace (iframe host for P01, P02 and VT)

| ID | Kind | Item | Cite | Destination |
|---|---|---|---|---|
| P12-C1 | C | Only `/analysis/index.html`, `/viz_tree.html` and `/compare.html` are allowed; the compare selection is validated (≥2 `{path,label}`). | SRC/AnalysisWorkspace.tsx:4-35 | RUN (contract for native rebuild) |
| P12-C2 | C | Theme is pushed into the frame; the frame's history is projected into the parent URL; "/" links inside the frame are rerouted to the run list. | SRC/AnalysisWorkspace.tsx:42-96 | RUN |
| P12-N | Q | Becomes obsolete once the surfaces are native, but its URL contract (`analysis=`, `compare=`, plus `analysis_url` from UI:63-76) is what deep links depend on. | as cited | RUN — OD-11 |

---

## 6. Proposed mapping by destination (summary of §2–§5)

| Destination | Receives (row IDs) |
|---|---|
| **RPT Report** | P03-Q1, P04 (OD-14), P05, P06, P07, P08, P09, P10; experiment deep links P09-Q2. |
| **LIN Lineage** | L01 (tree), L02 (prompt evolution), L07 (islands), L09 (path), R09-Q4 (mutation size, OD-6), R09-Q11 (lineage fertility). |
| **PRG Progress** | L04 (metrics), L08-Q1 (posterior trajectory, OD-4), L10-Q1/Q2/Q4 (cost), R01 (overview, five charts), R09-Q1/Q2/Q3/Q5..Q10 (one diagnostics block, OD-6, which also shows R09-Q4/Q11 from Lineage so that it holds all 11 plots), R10 (throughput, all 8). |
| **SOL Solution space** | L05 (embed), L06 (clusters 2D+3D, OD-1), L11 (MAPs), L13 (diversity). |
| **CON Contest** | R02 (Pareto), L12 (battle, model vs model), L08 (ensembling, OD-4; the per-generation trajectory is in PRG), L10-Q3 (model cost-efficiency, OD-4), L03 (program ranking, OD-2), S-06 (top-N ZIP), P02 (multi-run compare, OD-5). |
| **INS Program inspector** | S-14 header, R04 Node, R05 Code (+copy/download), R06 Diff (parent/initial), R07 Evaluation (stdout/stderr/plots), R08 LLM. It is the click target of L01-C4, L02-Q4, L03-C4, L04-C2, L05-C3, L06-C1, L07-C2, L09-C1, L11-C3, R01-C2, R02-C2. Hover previews L01-C5. |
| **RUN Runs list/compare** | P01, P02 (OD-5), P03-Q2 disclaimer, P12 URL contract, S-01, S-07..S-10, S-13. |
| **LIB Library** | P11, P03-C2 pending tag. |
| **No fitting destination** | S-02 language (OD-13), S-03/S-04/S-05/P03-C1/P03-C4/P01-C5 refresh, auto-refresh, status and theme chrome (OD-17), R03 Scratchpad (OD-3), R03-X1 PDF (OD-12), dead capabilities (OD-7, OD-19). |

---

## 7. Duplicates and overlaps (with evidence)

| # | Overlap | Evidence | Keep which? (proposal, owner to confirm) |
|---|---|---|---|
| D-1 | **Score over time, 7 renderings**: Meta "Performance Score Across Program Evaluations"; Metrics `combined_score` plot; Islands progression per island; Path score curve; Report TrialChart (direction-wide, with noise line); Compare chart 1; run-list best score. | VT:6839-6922; VT:13005-13301; VT:9360-9676; VT:10712-10767; SRC/TrialChart.tsx:23-41; CP:539-585, 637; IX:646 | One Progress score chart (Meta + Metrics merged; per-island series as a facet). The Path curve stays in Lineage. TrialChart stays in the Report because it is direction-scoped with the noise line. Compare stays in Contest/Runs. |
| D-2 | **Cost totals disagree**: Meta includes prompt-evo cost; Cost tab excludes it; run list includes it; compare excludes it; per-hypothesis spend comes from `rundb.total_cost`. | VT:5597-5603 vs 11763-11766; IX:705-708 vs CP:507-511; UI:60, 276-280 | One cost definition, OD-9. |
| D-3 | **"Best program" computed in 8+ places with different rules**: only the Tree uses strict `correct === true/'true'/1` (VT:4760-4762, 5103). Path, MAPs, Battle, Meta best, Embed best, Clusters, the Programs ranking and ZIP all use truthy `d.correct`, with differing null/NaN checks and tie-breaks (Clusters includes all correct programs with non-null score; Programs drops correct-but-null). | VT:4760-4767; 10631-10634; 11045; 12010-12013; 5566-5578; 8552-8557; 12352-12377; 8116-8126; 13610 | Single shared selector in the rebuild (carry-over constraint). |
| D-4 | **Runtime**: Meta score-vs-runtime and cumulative runtime charts; Throughput runtime timeline and occupancy; Cost zero-cost mode (compute time). | VT:7348, 7734-7800; 17389-17424; 11778-11790 | Progress: keep Throughput as the runtime home; fold Meta runtime charts into it. |
| D-5 | **Model comparisons in 5 places**: Battle (score distribution, win rate), Ensembling (posteriors, success, cost), Analysis patch/novelty/resample success per LLM, Cost $/Δscore per model, Analysis cost heatmap model × patch. | VT:11991-12340; 11289-11751; 14783, 15156, 15527; 11940-11987; 16219 | Contest "models" lens for battle, ensembling and cost per model, OD-4; the per-LLM success rates and the model × patch cost heatmap stay with the other Analysis plots in the Progress diagnostics block, OD-6. |
| D-6 | **Embedding geometry 4 ways**: Embed cosine matrix; Clusters PCA 2D/3D; Diversity PCA-2D per generation plus pairwise distance; MAPs behaviour grid. | VT:8498-9079; 12342-12536; 13307-13588; 10929-11287 | Solution space with modes; OD-1 for 3D. |
| D-7 | **Generation scrubbers 4 ways**: MAPs generation slider; Diversity generation slider; Scratchpad meta-update slider; Tree animate. | VT:2317, 11004-11011; 13468-13555; 2490, 3077-3096; 2187, 5191-5283 | `[INFERENCE]` One shared generation scrubber per surface; OD-8. |
| D-8 | **Diffs 3 ways**: Diff tab parent/initial; Path diff chain; prompt diff vs parent. | VT:6503-6603; 10769-10830; 18407-18560 | Inspector owns the program diff; Lineage path reuses the inspector diff component; prompt diff stays in Lineage. |
| D-9 | **Program lists 4 ways**: Programs table; prompt → generated programs table; ZIP top-N; Pareto points. | VT:8102-8346; 18099-18405; 13592-13726; 12609-12940 | Contest ranking (OD-2) plus filter-by-prompt from Lineage. |
| D-10 | **Run and direction indexes 3 ways**: run list (all program DBs); WorkspaceTable (directions); SpecCard experiment links. | IX:485-653; SRC/WorkspaceTable.tsx:14-61; SRC/SpecCard.tsx:126-144 | RUN for runs, RPT for directions; OD-14. |
| D-11 | **Status and liveness 3 ways**: VT scan-status; run-list auto-refresh; React EventStream and refresh. | VT:2152, 4421; IX:420-422, 834-858; SRC/EventStream.tsx; SRC/App.tsx:105-107 | OD-17. |
| D-12 | **Correct/valid counts 4 ways**: Meta "Correct %"; run list "有效程序"; compare "correct_count"; TrialChart "failed". | VT:5729-5730; IX:643; CP:356, 564; SRC/TrialChart.tsx:21 | Progress headline, linked from Runs. |
| D-13 | **Selection side-effects differ by origin**: Pareto, the Meta charts and migration links switch to Tree **and** Code (default `selectNodeById`); Clusters and Metrics switch only the right pane to Code; MAPs, Path and the Embed cell click switch neither. | VT:8348-8370; 12938, 6899, 10594 vs 12432, 13290 vs 11193, 10673, 8876 | OD-11: inspector updates in place, with no surface switch. |

---

## 8. Risks — capabilities at risk of being lost

| # | Risk | Evidence | Why it matters |
|---|---|---|---|
| R-1 | **Hover-only information disappears on phones.** Tooltips in Tree, Embed, MAPs, Ensembling, Battle, Diversity and Cost have no tap or keyboard equivalent; VT has no touch handlers. | VT:5028-5074, 8836-8934, 11171-11191, 11586-11668, 12173-12262, 13537-13548, 11831; grep of VT for touch/pointer: none | The owner requires the phone to be fully readable. |
| R-2 | **Astro components must be rebuilt natively**: status chips, timeline, event log, global status bar, monitoring icons. | SRC/HypothesisTable.tsx:89; SRC/SpecCard.tsx:59, 161; SRC/Timeline.tsx:62-79; SRC/EventStream.tsx:11; SRC/TopBar.tsx:44-58 | Removing Astro without a replacement silently loses the Timeline and EventStream (OD-15). |
| R-3 | **Already-dead capabilities** (code present, unreachable): min/max objective switch; novelty improvement plots; cost-efficiency plot; search dynamics by cluster; meta-update diff highlighting. | VT:2880, 7877-8010; 15282, 15467, 16110, 17426, 14720; 3295, 3420, 3586 | If not decided now, they vanish permanently during the rebuild (OD-7, OD-19). |
| R-4 | **Silent empty views when data fields are missing**: Throughput needs `timeline_lane_mode='pool_slots'` and stage timestamps; Ensembling needs `model_posteriors`; Clusters needs PCA and cluster id; Embed hangs unless every program has an embedding; Cost blanks on any unknown cost. | VT:16460-16512, 16633-16636; 11299-11341; 12352-12363; 8483-8535; 11817-11821 | Demo data must populate these, or design reviews will judge empty states. See §10. |
| R-5 | **Failed-evaluation visibility is inconsistent**: visible in Tree (red), Programs (opt-in), Pareto (red ×), Diversity scatter and TrialChart counts; invisible in Clusters, Islands, MAPs, Metrics, Battle and Path (null treated as 0). | VT:5089; 8110-8117; 12682-12685; 13538; SRC/TrialChart.tsx:21, 95 vs 12352-12356; 9958-9960; 11045; 13018; 12010-12013; 10690 | Regrouping can hide failures entirely (OD-10). |
| R-6 | **Untrusted text relies on one build-time wrapper.** LLM-authored patch names, errors, prompt names/descriptions, metric keys and meta notes are interpolated raw and are safe only via ShinkaHTML. Plotly hover text bypasses ShinkaHTML. | VT:5823, 5827, 5843, 5852, 18228, 18232, 3341-3418, 5049, 5060, 8261; Plotly 13164, 15734-15737, 12640-12657 | A native rebuild must render these as text nodes; Plotly/Chart labels must be escaped explicitly. |
| R-7 | **Report has no link from a trial to a program**: `trial_points` lack a program id. | UI:129-143; SRC/TrialChart.tsx:49 | "Linked surfaces" cannot connect Report to Inspector without an API addition (OD-18). |
| R-8 | **Deep links**: today's URLs (`db_path`, `left_tab`, `right_tab`, `meta_section`, `selected_node`; `view`, `direction`, `hypothesis`, `analysis`, `compare`; IX `selected`, `q`, `sort`; CP `selection`) and `analysis_url` from the server. | VT:9094-9122; SRC/App.tsx:29-33; SRC/AnalysisWorkspace.tsx:57-71; IX:806-815; CP:411; UI:63-76 | Old links break unless they are mapped. Prompt selection (L02-S1), library page (P11-S1), Pareto axes (R02-S1) and the compare metric are not in the URL today. |
| R-9 | **Single run per workbench**: the analysis surfaces are scoped to one `db_path` (one experiment). Only Compare spans runs, and the Report spans the direction. | VT:9110; CP:409-433; UI:112-144 | Owner should confirm whether Lineage, Progress, Solution space and Contest stay single-run. |
| R-10 | **Language parity**: VT has an EN/中 toggle with partial coverage; the run list, compare and React app are ZH-only; many VT strings are hardcoded EN. | VT:18579-18623; IX:2, 434-494; CP:332-385; SRC/terms.ts | OD-13. |
| R-11 | **Fields shown nowhere today**: `total_runtime_seconds` in run stats; `iter_id`; `children_count`. | IX:722; VT:4509-4513 | Minor; listed for completeness. `[INFERENCE]` `children_count` is not referenced in VT. |
| R-12 | **Correct programs with a null score vanish from the Programs table.** | VT:8120-8121 | Data loss in the ranking if it is carried over as-is. |

---

## 9. Owner decisions (each a question with options)

| OD | Question | Options |
|---|---|---|
| OD-1 | 3D PCA clusters: primary view of Solution space, or a mode? | (a) 2D primary with a 3D mode toggle (phone: 2D only); (b) 3D primary on desktop; (c) drop 3D. Today: side by side 50/50 (VT:2348-2353). |
| OD-2 | Where does the sortable program ranking (L03) live? | (a) Contest leaderboard; (b) a list docked beside the Program inspector, reachable from every surface; (c) both, as one shared component. |
| OD-3 | Where does the Scratchpad (meta-LLM notes per meta update, R03) live? | (a) Report (run-level narrative); (b) Lineage (next to prompt evolution); (c) a Program inspector tab for program summaries plus a Lineage panel for global insights. |
| OD-3b | Does the persistent inspector also inspect **prompts** (L02-Q2), or only programs? | (a) Programs only; prompt detail stays inside Lineage; (b) the inspector is polymorphic (program or prompt). |
| OD-4 | Model-level analyses (Battle, Ensembling, Cost per model, Analysis per-LLM/per-patch success) | (a) all in Contest under a "models" lens; (b) posteriors and success rates in Progress, battle in Contest; (c) a separate lens inside Progress. |
| OD-5 | Multi-run compare: Contest or Runs list/compare? | (a) Contest (per owner note); (b) Runs (the selection originates there); (c) selection in Runs, rendering in Contest. |
| OD-6 | The 11 Analysis sub-plots (R09) are split per row as in §6. | (a) approve the per-plot split; (b) keep them as one collapsible "diagnostics" block inside Progress; (c) drop some (list them). |
| OD-7 | Objective direction (maximize/minimize). The min/max radio is dead; Metrics has a per-plot cumulative min/max toggle; Pareto has per-axis objectives. | (a) restore one run-level objective setting that drives best/ranking everywhere; (b) maximize only (formally drop the dead code); (c) per-chart toggles as today. |
| OD-8 | Tree "Animate" (replay by depth, loops forever). | (a) keep as a generation scrubber/replay shared with D-7; (b) drop. |
| OD-9 | Cost definition. | (a) always include prompt-evolution cost; (b) show both "model calls" and "prompt evolution" separately everywhere; (c) as today (inconsistent). |
| OD-10 | Failed evaluations on each surface. | (a) always visible and marked; (b) hidden by default with one global "show failed" toggle; (c) per-surface as today. |
| OD-11 | Selection behaviour and layout. | (a) selecting a program anywhere only updates the persistent inspector and never switches surface; (b) as today (jump to Tree + Code from some views). Also: is the inspector a right rail on desktop and a bottom sheet on phone? |
| OD-12 | PDF export of meta notes (disabled, server 501). | (a) keep the disabled affordance with its note; (b) remove until isolated conversion exists; (c) replace with markdown/text download. |
| OD-13 | Language. | (a) bilingual toggle across all surfaces; (b) ZH-only; (c) ZH UI with EN technical labels. |
| OD-14 | Workspace home (directions table). | (a) Report index; (b) merged with the Runs list as one "workspace" page. |
| OD-15 | Astro replacements for Timeline, EventStream, status chips and status bar. | (a) rebuild natively with equivalent capability; (b) simplify (timeline becomes a sortable table, events a plain list); (c) drop Timeline. |
| OD-16 | Phone interaction for hover-only details. | (a) tap shows a preview in the inspector sheet; (b) tap opens the inspector directly; (c) keep hover (desktop only). |
| OD-17 | Global chrome: refresh, auto-refresh (3s in VT, 10s in the run list), scan status, theme, language. | (a) one app-level header shared by all surfaces with a single liveness indicator; (b) per-surface controls as today. |
| OD-18 | Report → program linkage. | (a) add a program id to `trial_points` (API change in `shinka/research/ui.py`) so trial dots open the inspector; (b) keep the Report unlinked. |
| OD-19 | Dead renderers (novelty improvement, cost efficiency per generation, search dynamics by cluster, meta-update diff). | (a) revive and assign destinations (PRG, PRG, SOL, OD-3); (b) formally drop. |

---

## 10. Appendix — data each surface needs (for the demo workspace)

| Surface/view | Required fields (all cited) |
|---|---|
| Throughput (R10) | `metadata.{pipeline_started_at, sampling_started_at, sampling_finished_at, evaluation_started_at, evaluation_finished_at, postprocess_started_at, postprocess_finished_at}`, `metadata.timeline_lane_mode='pool_slots'`, `metadata.{sampling,evaluation,postprocess}_worker_{id,capacity}`, `metadata.source_job_id` — VT:16460-16512, 16633-16636 |
| Ensembling (L08) | `metadata.llm_result.model_posteriors` or `metadata.model_posteriors`; `metadata.model_name` — VT:11299-11341, 11380 |
| Clusters, Diversity (L06, L13) | `embedding_pca_2d` (len 2), `embedding_pca_3d` (len 3), `embedding_cluster_id` on correct programs — VT:12352-12357, 13321 |
| Embed (L05) | `embedding` on **every** program — VT:8483-8496 |
| MAPs (L11) | `metadata.code_analysis_metrics.lines_of_code`, `metadata.compute_time` (default axes) — VT:10972-10987 |
| Cost / Meta (L10, R01) | Numeric ≥0 `metadata.{api_costs, embed_cost, novelty_cost, meta_cost}`; prompts `metadata.llm.cost` — VT:5597-5603, 11763-11766; AN:19-34, 297-299 |
| Islands (L07) | `island_idx`, `migration_history` — VT:10004, 10562-10597 |
| Tree, Analysis (L01, R09) | `metadata.patch_type ∈ {init, full, diff, cross, fix}`, `metadata.patch_name`, `metadata.{novelty,resample,patch}_attempt`, `parent_id`, `correct`, `combined_score` — VT:4916-4923, 14320, 8167-8178 |
| Inspector (R04–R08) | `code`, `code_diff`, `public_metrics`, `text_feedback`, `metadata.{stdout_log, stderr_log, llm_result, thought}`, `gen_N/results/plots/*.png` — VT:5838-6270; AN:323-342 |
| PromptEvo (L02) | `prompts.db` (`system_prompts`, `prompt_archive`, `prompt_metadata_store`) — AN:264-287 |
| Scratchpad (R03) | `<run>/meta/meta_<N>.txt` with `# INDIVIDUAL PROGRAM SUMMARIES` / `# GLOBAL INSIGHTS SCRATCHPAD` / `# META RECOMMENDATIONS` — AN:143-152; VT:3297, 3501-3542 |
| Report (P05–P10) | notebook hypotheses/experiments, `RESULTS_LOG` events, trial times via `rundb.trials` — UI:42-215 |

---

## 11. Coverage checklist (closure evidence)

| Inventory | Found | Mapped | Evidence |
|---|---|---|---|
| Left tabs | 13 | 13 (L01–L13) | VT:2164-2176 |
| Right tabs | 10 | 10 (R01–R10) | VT:2450-2459 |
| Nested tabs / sub-views | Scratchpad 3 sub-tabs; Analysis 15 rendered plots (11 rows: success/effectiveness pairs grouped) + 4 dead renderers; Throughput 8; prompt dialog; node summary; tree legend | all mapped (R03, R09, DEAD-R09, R10, L02-Q3, S-14, L01-C6) | VT:2500-2503, 17521-17535, 17416-17423 |
| Hidden or compat DOM | task/result selects, refresh-files, file-list | S-13 | VT:2157-2162 |
| Pages | run list, compare, App/TopBar/Nav, WorkspaceTable, ResearchTab, HypothesisTable, TrialChart, Timeline, SpecCard, EventStream, LibraryTab, AnalysisWorkspace | 12 (P01–P12) | §5 |
| Server routes | 17 (14 ANALYSIS_PATHS + `/plot_file/` + `/api/workspace` + `/api/snapshot`) | 16 consumed; **1 orphan**: `/download_meta_pdf` (deliberately 501) → OD-12 | AN:38-53, 168-190; UI:322-350 |
| Client network call sites | VT 16 (`fetch`/XHR at 3144, 3258, 3631, 3857, 4081, 4126, 4127, 4201, 4386, 4390, 6181, 6350, 6580, 9333, 17585, plus i18n 18586) + RT:26; IX 2 (524, 687); CP 1 (445); App 1 (SRC/App.tsx:55) | all attributed in §1 | grep of `fetch(`/`XMLHttpRequest` |
| URL/state keys | VT 5 URL + 3 localStorage (`shinka_preferences`, `shinka-lang`, `shinka-theme`); IX 3 URL + `collapsed-task-*`; CP 1; App 5 | all in S-07..S-10, P01-S1, P02-S1, P03-S1 | §2, §5 |
| Exports | copy code, download code, ZIP top-N, PDF (disabled) | R05-X1, S-06, R03-X1 | VT:2822-2876, 13592-13726, 2495 |
| Controls | every `<select>`/`<input>`/button/click handler in the cited markup and renderers | each in a C row | §2–§5 |
| Empty destination cells | 0 | — | every row has a destination, `carry-over`, or `OD-n` |

Not verified (static read only, the app was not run): exact visual rendering, whether ShinkaHTML strips any VT-generated markup that the views rely on, and actual phone behaviour. Rows marked `[INFERENCE]` are derived rather than read.
