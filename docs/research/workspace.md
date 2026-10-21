# 研究与分析工作区

工作区页对应 [GitHub #12](https://github.com/2233admin/ShinkaEvolve/issues/12)，统一运行与分析对应 [#27](https://github.com/2233admin/ShinkaEvolve/issues/27)。本文定义维护契约；技术验证不代替所有者视觉验收。研究方向仍沿用 `design.md` §7.2–7.3 的目录约定，不新增工作区数据库或方向注册表。

## 范围与存储

```text
<workspace>/
  library/                  所有方向共享的资料库
  <direction>/notebook.json  一个方向；目录名是导航 ID，标题不要求唯一
```

工作区只发现直接子目录中的有效方向。普通目录忽略；方向目录及 notebook 的符号链接不作为方向发现或读取，方向目录的 junction 等改变规范身份的别名同样排除，避免别名和资料库归属混淆。有 notebook 但内容或结果库损坏的目录保留为不可进入的错误条目，不能伪造零花费、零试验或封存状态。空工作区合法，也可以有共享资料库。

## 启动与 HTTP 契约

`shinka_research ui <workspace> [--runs-root <ordinary-runs>] [--port 8765] [--no-browser]` 启动同源只读工作区。主参数必须是工作区；方向参数仍明确报错并提示父目录。可选 `--runs-root` 显式授权另一处普通运行历史，不从当前目录自动扩大权限。其他研究 CLI 仍接收方向目录。

服务只回应 `Host` 为 `127.0.0.1:<端口>` 或 `localhost:<端口>` 的请求，其他一律 421：把自己的域名指向本机的网页（DNS 重绑定）发来的仍是它自己的域名，读不到笔记本与代码。服务只接受 GET（HEAD 回 405，其他方法 501），没有写接口。

开发环境先在 `shinka/research/webui/app/` 运行 `pnpm install --frozen-lockfile`，再单独运行 `pnpm build`，不要关闭安装器的供应链校验。若安装器要求依赖脚本审批，先审阅再决定；本轮验收显式不执行依赖安装脚本，记录见 `qa.md`。然后在仓库根目录运行 `uv run shinka_research ui <workspace>`。前端构建产物仍不进入源码库。

| 入口 | 输出或行为 |
| --- | --- |
| `GET /api/workspace` | `{name, directions, library, events}`；摘要读 Notebook、各实验运行库的试验与费用、事件日志和共享库，证据数字与方向快照同一算法；每个方向另有 `rounds: {state, pending}`（下一行 `rounds` 的状态与待决事项数）；不返回假设正文或验证段、封存段数字 |
| `GET /api/snapshot?direction=<id>` | 既有方向快照形状和服务端可见性过滤，`trial_points` 每点带程序编号 `program`；必须恰有一个非空方向参数。`rounds` 是 Agent 的轮次（research-agent 3.2，`shinka/research/rounds.py`，`brief` 读同一份）：`grants`（每份授权的条款、已取尝试数与每次尝试的运行：`running`、`starting` 尝试已写而租约还没写（`run` 先写尝试，60 秒内算正在启动）、`unrecorded` 进程已无而结束未记、`ended` 带结束事件、`lost` 尝试已取而运行一直没登记）、`running`（进行中的运行，`env` 为 agent 或 manual）、`approvals`（封存段批准，`used` 为它打开的开封时间，所有者手动开封后未用的记为 `moot`）、`pending`（`unrecorded`、`lost`（这次尝试没有结果，也不挡本轮结束）、`round-over` 一份授权的尝试都已取完、运行都已记下或已丢失且之后没有涉及同一假设的新授权、`stuck` Agent 用不了一份尚有余量的授权的原因：`budget`、`budget-unknown`、`open-ended`、`settings`、`lockbox-unfinished` 附开封所用的批准、`ledger` 与 `unreadable`），每项带一条 `do` 说明该做什么，目录写作 `<dir>`；`state` 依次取 `unknown`（账本或方向文件读不了）、`blocked`、`running`（有 Agent 的运行在跑或正在启动）、`waiting`（有 `round-over`）、`open`、`idle`。只给列出的字段：没有进程号、命令、路径，也没有租约为恢复状态保存的旧判词证据。Agent 自己的申请还没有记录处（research-agent design.md） |
| `/?view=directions`（`/` 同） | 研究方向首页（award-redesign 4.1）：数据句标题、合计小字、可排序方向表与合计行、动态栏；只读 `/api/workspace` 与 `/list_databases`，不逐方向请求 `/api/snapshot` |
| `/?view=direction&direction=<id>&hypothesis=<hid>` | 方向研究报告（award-redesign 4.2）：假设表、所选假设的主张与全文、试验与噪声线、检查台账、判词与封存段确认与实验、时间线和可筛选的事件；只读所开方向的 `/api/snapshot` |
| `/?view=runs&q=…&kind=…&sort=…&select=<run>…` | 全部运行（award-redesign 5.1）：按评估器分组——研究实验（`workspace/` 下属于某个方向的运行）按方向、普通运行按任务；`kind` 为 `ordinary` 或 `experiment` 时只列该类，`sort` 为 `name`、`score` 或 `cost`，缺省按最近更新；每个勾选的运行一个 `select`，被检索或类型隐藏的运行仍保持勾选。统计逐个读 `/get_database_stats`（每批 4 个），随外壳每 10 秒的自动刷新重读；分组折叠沿用旧列表的 `localStorage['collapsed-task-<任务>']` |
| `/?view=compare&runs=<run>&runs=<run>…&metric=…` | 多运行比较（award-redesign 5.2），至少 2 个运行：所选指标（缺省为累计最佳，`metric` 为 `program_count` 或 `correct_count` 时为程序数或有效程序数）按代与按累计花费的阶梯曲线，指向或用 ← → 读取各运行在该处的数值；跨运行排行榜（每个运行至多 4 个有效程序，合起来取前 8 名）与运行级指标。曲线与旧比较页逐点一致：花费按时间顺序累计程序与提示词进化费用，遇到未计价的调用后曲线止于此，超过 200 点时抽样。运行分属多个任务时，得分曲线、排行榜与最佳只在同一任务内比较，计数与费用仍放在一起。每个运行读 `/get_programs_summary`、`/get_database_stats`，有提示词进化时再读 `/get_system_prompts`，随外壳每 30 秒的自动刷新重读 |
| `/?view=run&run=<run>&tab=…&program=…&pane=…&gen=…` | 实验分析；研究入口另带 `direction`、`hypothesis` 作为返回上下文。原生视图（award-redesign 6.1）完成前另带恢复页的面板 `left_tab/right_tab/meta_section`，`tab/pane/gen` 只随链接保留 |
| `/?view=library&page=log` 或 `page=wiki/<name>` | 共享资料库（award-redesign 4.3）：索引、日志与每个 wiki 页面都是可分享的链接，待整理的原始资料在页首说明；无 `page` 为索引，不存在的页面显示索引并注明缺页。正文是 markdown-it（`html: false`）渲染的净化 Markdown，`[[页面]]` 在工作台任何正文里都链接到这一路由，其他链接在新标签页打开 |
| `/list_databases`、`/get_database_stats?db_path=…` | 授权范围内的真实程序库与只读统计；费用记录不全时 `total_cost=null`（或 `prompt_evo_cost=null`），不能当成免费；`recorded_cost` 是该运行已记录的全部费用，两部分都已知时即其合计，否则为下限，跨运行的合计按它求和并标 `≥` |
| `/get_system_prompts?db_path=…` | 不存在提示库时返回 `[]`；已有提示库损坏或缺表时返回明确的 500 JSON 错误，不能伪装为无提示数据 |
| `/analysis/index.html`、`/compare.html`、`/viz_tree.html` | 自研运行浏览、比较及完整分析；由研究层维护构建产物。运行列表与比较已由工作台原生绘制，旧页面按 award-redesign 7.1 保留到能力组删除 |

成功摘要字段为 `id/direction/created/hypotheses/trials/spend_usd/unknown_cost_runs/budget_usd/lockbox_opened/evidence/error`，其中 `error=null`；`budget_usd=null` 表示未设预算，`lockbox_opened=null` 表示未设置数据切分，`unknown_cost_runs>0` 时 `spend_usd` 是下限。`trials` 是 `evidence.evaluated_programs` 加方向全部试探次数，与质疑检验计 DSR 的试验相同（空方向为 0，质疑检验在那里取下限 1），运行中的实验随运行计入，不读运行结束时才写入的 notebook 计数；`/api/snapshot` 的 `trials` 与各实验的 `n_evaluated`、`best_is_score` 同样按运行库计数，同一响应中的计数、最佳分数与试验点来自对每个运行库的同一次读取。`evidence` 含 `hypothesis_status/best_score/noise_bar/evaluated_programs/failed_programs/last_activity`，与同一方向快照的 `trial_points` 计算一致；没有试验时分数、噪声线与最近活动为 `null`。错误条目只有 `id/direction/error`，`error` 是失败类别（如 `ValueError`），页面按读者语言措辞。两种响应中的 `library` 都是同一工作区资料库；读页面不调用 `library.ensure`，不写研究数据。

`events` 是首页动态栏的数据：各可读方向最近的事件按时间由新到旧合并，至多 `WORKSPACE_EVENTS`（30）条，每条与快照事件同形并多一个 `direction`（方向 ID）。说明的隐藏规则与 `/api/snapshot` 相同：所属假设尚无判词时，质疑检验与判词事件的说明为 `null` 且 `withheld=true`，页面按读者语言写占位；不可读方向不贡献事件。

首页的合计与数据句只统计可读方向；不可读方向以错误条目列在表中（目录 ID、失败类别、“不计入合计”），不可进入。表格各行共用一把证据刻度（1、2、5 × 10ⁿ 的整刻度），排序只改本页状态，不写入 URL。动态栏在 1280 宽及以上是贴住视口、自行滚动的右栏，更窄时排在表格下方，手机上每个方向是一张卡片，最后更新时间移到数据句下方。

方向报告按论证顺序排列：假设表 → 主张 → 试验与噪声线 → 检查台账，1440 宽及以上判词、封存段确认结果与实验在右栏，更窄时排在台账之后，再接时间线与事件。试验图横轴是整个方向的已评估程序序号，噪声线与失败带由图表自己画；点按试验点，或聚焦图表后用左右方向键选中再按 Enter，打开该实验分析并显示该程序代码。实验的“进入分析”打开判词所依据的程序（`skeptic.program_id`，或判词证据中属于该实验的 `program_id`），尚无检查时打开本实验得分最高的程序。验证段数字只在判词之后出现，封存段数字只在确认之后出现。排序、正文展开与事件筛选按方向记在本次访问的内存里：从分析页后退、或在同一页切回该方向时原样恢复，刷新后重置；滚动位置由浏览器恢复，所以图表高度不依赖测量结果。

ID 是一个真实目录名，不限 ASCII。服务端只解码查询一次，拒绝斜杠、反斜杠、点路径、绝对路径及驱动器形式，并校验规范路径仍是工作区的直接子目录。中文、空格、`+`、`%`、`#`、`&` 等合法名字可以往返；未知或越界 ID 返回 404，缺失、空值、重复参数返回 400，存在但无法读取的数据返回明确的 500 错误。例外是 `rounds`：账本与运行日志对不上、账本或租约读不了时，快照照常返回，`rounds` 报 `state: unknown` 与一条 `ledger` 或 `unreadable` 待决事项，只给原因代码，不转述含路径的拒绝文字。

UI 在原始工作区/方向授权根下校验规范文件目标：资料库根不能搬到工作区外，库内正文、清单与原始资料不能读出资料库；方向的 notebook、假设正文、事件和实验结果不能读入相邻方向。实验根先相对方向校验，结果目录及数据库叶子再相对实验根校验。分析适配层同时校验查询派生的提示库、meta、plot 和失败提案目标。静态越界链接明确失败；该预检查不防御并发替换链接的恶意本机进程。

## 实现取舍

| 方案 | 状态与复杂度 | 代价与边界 | 选择 |
| --- | --- | --- | --- |
| 查询参数路由（pushState，award-redesign 2.3） | 一个外壳；站内链接、前进后退不整页重载，每个页面的状态都在 URL | 迟到请求由 `AbortController` 取消；切换方向时不显示上一方向的快照 | 采用 |
| 原生文档导航 | 每次跳转整页重载 | 外壳、命令面板与自动刷新的状态每页重建 | 不再采用 |
| 工作区注册表或持久 UUID | 增加索引、迁移和失效规则 | 现有数据没有该模型 | 不采用；重命名后旧链接自然 404 |

外壳（`app/src/shell/`）在页面可见时每 30 秒刷新 `/api/workspace`、所开方向的 `/api/snapshot` 与 `/list_databases`，保持当前选择和滚动位置；刷新失败时保留已读数据，顶栏标注“连接中断”和最后更新时间，点按立即重试。手动刷新同时重载托管的分析页，自动刷新不动它。

统一分析由 `shinka/research/analysis.py` 复用上游只读 API、缓存和文件读取器，不启动额外树端口。客户端数据库身份仅为 `workspace/<relative>` 或显式授权的 `runs/<relative>`；规范路径必须仍在相应根内，且文件必须有真实 Shinka 程序表。原始资料库、方向数据、封存目录、无效 SQLite 和越界别名不进入发现列表。程序接口获得规范绝对路径作为内部缓存键，客户端不获得任意绝对路径权限。派生提示库和 meta 限定在所选运行目录内；图片仅限真实运行的 `gen_N/results/plots/` 图像。服务强制 `show_private=False`，不返回宽松 CORS 头。

分析源来自本 fork 已自研重做的 `b31db504f9ca9616b5d4c8ca523cdcfdab26d8ab:shinka/webui/`，保存在 `shinka/research/webui/analysis/`；上游 `shinka/webui/` 不改。第二个 Vite 构建打包固定版本的本地图表、代码高亮、Markdown、ZIP、字体和净化器，并将动态 HTML 写入编译到受控净化边界。动态操作只允许有限的已声明函数调用语法，不执行字符串代码。维护者应修改源文件后完整运行 `pnpm build`，不能手改 `dist`。

本地打包资源的完整许可告知随构建输出为 `/analysis/THIRD_PARTY_NOTICES.txt`，来源和补充清单保存在 `analysis/licenses/`。补充清单覆盖 Plotly 内嵌组件，包括实际使用的 `@plotly/mapbox-gl` fork；历史 `native-promise-only` 版本只提供作者 MIT 链接，因此保留其原归属和当前作者端点的完整许可，不伪造同时代快照。

React 外壳保留研究导航与原方向返回入口，分析文档保留自身 DOM、页签和图表状态。iframe 的原生导航拥有历史；外层只用 `replaceState` 把分析页的位置投影成公开路由（`app/src/legacyFrame.ts`）：运行列表的检索、排序和勾选成为 `q/sort/select`，比较页的选择成为 `runs`，程序页的 `db_path/selected_node` 成为 `run/program`，其面板 `left_tab/right_tab/meta_section` 原样保留，只有代码页签 `right_tab=agent-code` 与公开的 `pane=code` 互换，因此研究报告给出的程序链接在 6.1 原生检查器取代恢复页后仍然有效；比较页的标签由 `/list_databases` 的运行名还原。因此刷新、直接链接及原生前进/后退不依赖最近一次全局存储值。原方向参数只是来源上下文，分析页转到另一个运行时随即去掉，不证明随后浏览的普通运行属于该方向。尺寸变化只重排已读取数据，不重载运行或清除程序选择；迟到的跨运行响应及延后渲染不取得当前运行身份。

通用分析显示原始 `combined_score`；不同评估器的分数未必同量纲，不能直接作研究结论。研究方向页继续展示全方向试验数及对应噪声基准，通用代数/费用图不套用其横轴或给普通任务虚构 Sharpe 校正。Tree、Path、比较、Cost、Battle 和 Diversity 的有效分数统计排除评估失败的有限占位分数；失败仍计入程序与已记录资源。缺失费用、少于两个得分的样本标准差和少于两个嵌入的距离保持缺失，不能伪装成免费或测得零多样性；明确记录的零成本继续显示为零，混有未知成本的累计值不重新变成已知。

程序 ZIP 导出在浏览器本地生成。研究统一界面的服务端 PDF 导出当前停用；合法请求返回 501 JSON，前端禁用按钮并显示原因，不进入上游转换器，也不将正文文本冒充 PDF。仅安装 `wkhtmltopdf`、`pandoc` 或 PDF 引擎不能安全启用：现有转换路径没有经验证的本地文件与网络资源隔离，必须先提供并验证受限转换实现。草稿阅读和程序 ZIP 导出不受影响；独立上游查看器未改，本次交付不声称完整 PDF 能力对等。

## 验收与触达资产

| 资产 | 变更与风险 | 验证 | 回退 |
| --- | --- | --- | --- |
| `shinka/research/ui.py`、`analysis.py` | 方向读取、显式运行根、同源分析授权与派生文件边界；风险是越界、串缓存和缺失数值伪装 | 真实 HTTP：两方向、无效库、原始资料、路径别名、私有指标及费用缺失回归 | 回退本 PR；不改变 notebook 或研究数据格式 |
| `shinka/research/cli.py` | `ui` 主参数为工作区，可选 `--runs-root` | 实际 CLI 启动及错误目录提示 | 与服务端一起回退 |
| `shinka/research/webui/` | 自研完整分析与研究导航整合、URL/主题/键盘与窄屏适配 | 实际浏览器：研究→代码→刷新→返回、独立运行→比较→分析→后退，以及负数/失败/缺失/零值 | 与服务端一起回退；不回写研究数据 |

实现者负责技术验证与证据；所有者视觉验收独立记录，不能由测试结果自动授予。验收夹具必须通过正常存储或研究接口按对应模型建立并标为夹具，不得称为 AU-5090 真实研究；普通运行夹具不需要伪造研究方向。

`shinka/research/webui/browser-automation/unified-analysis.mjs` 提供可复用、只读的 Puppeteer 回归入口：`verifyPromptProgramSelection(page, options)` 覆盖提示节点的键盘选择、代码与缩放重绘后的树高亮；`verifyPathDeepLink(page, options)` 覆盖 Path 直接链接、刷新、选择及切回树的身份一致性。调用方提供已授权运行的路径、实际提示标签、程序 ID 和代码标记，不依赖临时目录或全局浏览器状态。

方向页保留既有假设正文和结论台账，不增加独立 `REPORT.md` 阅读器。工作区原始验收见 [`research-workspace/qa.md`](../../shinka/research/webui/design/changes/research-workspace/qa.md)，本次完整分析、源码恢复与品质迭代见 [`unified-analysis/qa.md`](../../shinka/research/webui/design/changes/unified-analysis/qa.md)。

## 其余七项的边界

- 第 2 项需要 au-5090 的真实运行、数据与证据，不能用本项夹具替代。
- 第 3–5 项（事件流日期与中文表头、时间线状态配色、噪声虚线）的旧组件已由 award-redesign 4.2 的原生报告取代：事件按 `YYYY-MM-DD HH:MM` UTC 显示且没有英文表头，时间线以形状加文字标状态，噪声虚线由试验图自己画。
- 第 6 项需要核对原验收对应版本及证据，再通过 design-pipeline 补记，不能借本项写为所有者已验收。
- 第 7 项仅在此 fork 发布 PyPI 包时把前端纳入安装包。
- 第 8 项先提出可替换领域评估接口设计，经所有者确认后再实现。