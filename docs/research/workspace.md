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

开发环境先在 `shinka/research/webui/app/` 运行 `pnpm install --frozen-lockfile`，再单独运行 `pnpm build`，不要关闭安装器的供应链校验。若安装器要求依赖脚本审批，先审阅再决定；本轮验收显式不执行依赖安装脚本，记录见 `qa.md`。然后在仓库根目录运行 `uv run shinka_research ui <workspace>`。前端构建产物仍不进入源码库。

| 入口 | 输出或行为 |
| --- | --- |
| `GET /api/workspace` | `{name, directions, library}`；摘要读 Notebook、各实验运行库的试验与费用、事件日志末行和共享库，证据数字与方向快照同一算法；不返回假设正文、事件说明或验证段、封存段数字 |
| `GET /api/snapshot?direction=<id>` | 既有方向快照形状和服务端可见性过滤，`trial_points` 每点带程序编号 `program`；必须恰有一个非空方向参数 |
| `/?view=directions`（`/` 同） | 研究方向首页 |
| `/?view=direction&direction=<id>&hypothesis=<hid>` | 方向及所选假设 |
| `/?view=runs&q=…&sort=…&select=<run>…` | 全部运行；每个勾选的运行一个 `select` |
| `/?view=compare&runs=<run>&runs=<run>…` | 多运行比较，至少 2 个运行 |
| `/?view=run&run=<run>&tab=…&program=…&pane=…&gen=…` | 实验分析；研究入口另带 `direction`、`hypothesis` 作为返回上下文。原生视图（award-redesign 6.1）完成前另带恢复页的面板 `left_tab/right_tab/meta_section`，`tab/pane/gen` 只随链接保留 |
| `/?view=library&page=log` 或 `page=wiki/<name>` | 共享资料库；无 `page` 为索引 |
| `/list_databases`、`/get_database_stats?db_path=…` | 授权范围内的真实程序库与只读统计；费用记录不全时 `total_cost=null`，不能当成免费 |
| `/get_system_prompts?db_path=…` | 不存在提示库时返回 `[]`；已有提示库损坏或缺表时返回明确的 500 JSON 错误，不能伪装为无提示数据 |
| `/analysis/index.html`、`/compare.html`、`/viz_tree.html` | 自研运行浏览、比较及完整分析；由研究层维护构建产物 |

成功摘要字段为 `id/direction/created/hypotheses/trials/spend_usd/unknown_cost_runs/budget_usd/lockbox_opened/evidence/error`，其中 `error=null`；`budget_usd=null` 表示未设预算，`lockbox_opened=null` 表示未设置数据切分，`unknown_cost_runs>0` 时 `spend_usd` 是下限。`evidence` 含 `hypothesis_status/best_score/noise_bar/evaluated_programs/failed_programs/last_activity`，与同一方向快照的 `trial_points` 计算一致；没有试验时分数、噪声线与最近活动为 `null`。错误条目只有 `id/direction/error`，`error` 是失败类别（如 `ValueError`），页面按读者语言措辞。两种响应中的 `library` 都是同一工作区资料库；读页面不调用 `library.ensure`，不写研究数据。

ID 是一个真实目录名，不限 ASCII。服务端只解码查询一次，拒绝斜杠、反斜杠、点路径、绝对路径及驱动器形式，并校验规范路径仍是工作区的直接子目录。中文、空格、`+`、`%`、`#`、`&` 等合法名字可以往返；未知或越界 ID 返回 404，缺失、空值、重复参数返回 400，存在但无法读取的数据返回明确的 500 错误。

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

React 外壳保留研究导航与原方向返回入口，分析文档保留自身 DOM、页签和图表状态。iframe 的原生导航拥有历史；外层只用 `replaceState` 把分析页的位置投影成公开路由（`app/src/legacyFrame.ts`）：运行列表的检索、排序和勾选成为 `q/sort/select`，比较页的选择成为 `runs`，程序页的 `db_path/selected_node` 成为 `run/program`，其面板 `left_tab/right_tab/meta_section` 原样保留；比较页的标签由 `/list_databases` 的运行名还原。因此刷新、直接链接及原生前进/后退不依赖最近一次全局存储值。原方向参数只是来源上下文，分析页转到另一个运行时随即去掉，不证明随后浏览的普通运行属于该方向。尺寸变化只重排已读取数据，不重载运行或清除程序选择；迟到的跨运行响应及延后渲染不取得当前运行身份。

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
- 第 3–5 项分别是事件流日期/中文表头、时间线状态配色、噪声虚线的原生配置或自动检查，不混入本项。
- 第 6 项需要核对原验收对应版本及证据，再通过 design-pipeline 补记，不能借本项写为所有者已验收。
- 第 7 项仅在此 fork 发布 PyPI 包时把前端纳入安装包。
- 第 8 项先提出可替换领域评估接口设计，经所有者确认后再实现。